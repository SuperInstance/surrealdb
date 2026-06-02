//! Query pattern analysis and budget tracking.
//!
//! Models query traffic over the record graph:
//! - Track which queries touch which nodes/edges
//! - Compute cost distributions per namespace/database
//! - Detect query pattern shifts using Jensen-Shannon divergence
//! - Enforce query conservation budgets

use std::collections::HashMap;

use crate::spectral::jensen_shannon_divergence;

/// A record of a single query touching graph records.
#[derive(Clone, Debug)]
pub struct QueryAccess {
    /// Namespace.
    pub ns: String,
    /// Database.
    pub db: String,
    /// The SurrealQL query text (summarized).
    pub query: String,
    /// Record IDs touched by the query.
    pub records_touched: Vec<String>,
    /// Estimated cost (proportional to graph traversal depth).
    pub cost: f64,
    /// Timestamp (seconds since epoch).
    pub timestamp: f64,
}

/// Distribution of query patterns across a time window.
#[derive(Clone, Debug)]
pub struct QueryDistribution {
    /// Map from query pattern key to relative frequency.
    pub patterns: HashMap<String, f64>,
    /// Total count of queries in this window.
    pub total: u64,
}

impl QueryDistribution {
    /// Compute JS divergence from another distribution.
    pub fn divergence_from(&self, other: &QueryDistribution) -> f64 {
        // Build union of keys.
        let mut keys: Vec<String> =
            self.patterns.keys().cloned().chain(other.patterns.keys().cloned()).collect();
        keys.sort();
        keys.dedup();

        let p: Vec<f64> = keys
            .iter()
            .map(|k| self.patterns.get(k).copied().unwrap_or(0.0))
            .collect();
        let q: Vec<f64> = keys
            .iter()
            .map(|k| other.patterns.get(k).copied().unwrap_or(0.0))
            .collect();

        jensen_shannon_divergence(&p, &q)
    }
}

/// Budget tracker for expensive graph traversals.
#[derive(Clone, Debug)]
pub struct QueryBudget {
    /// Maximum allowed cost per time window (by namespace).
    ns_budgets: HashMap<String, f64>,
    /// Current accumulated cost per namespace.
    ns_costs: HashMap<String, f64>,
    /// Maximum allowed cost per database.
    db_budgets: HashMap<String, f64>,
    /// Current accumulated cost per database.
    db_costs: HashMap<String, f64>,
    /// Window size in seconds.
    window_seconds: f64,
    /// Start of current window.
    window_start: f64,
}

impl QueryBudget {
    /// Create a new query budget tracker.
    pub fn new(window_seconds: f64) -> Self {
        QueryBudget {
            ns_budgets: HashMap::new(),
            ns_costs: HashMap::new(),
            db_budgets: HashMap::new(),
            db_costs: HashMap::new(),
            window_seconds,
            window_start: std::time::UNIX_EPOCH
                .elapsed()
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0),
        }
    }

    /// Set the budget for a namespace.
    pub fn set_ns_budget(&mut self, ns: &str, budget: f64) {
        self.ns_budgets.insert(ns.to_string(), budget);
    }

    /// Set the budget for a database.
    pub fn set_db_budget(&mut self, db: &str, budget: f64) {
        self.db_budgets.insert(db.to_string(), budget);
    }

    /// Record query cost and return the conservation score for this access.
    pub fn record(&mut self, ns: &str, db: &str, cost: f64) -> f64 {
        let now = std::time::UNIX_EPOCH
            .elapsed()
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        // Reset window if expired.
        if now - self.window_start > self.window_seconds {
            self.ns_costs.clear();
            self.db_costs.clear();
            self.window_start = now;
        }

        *self.ns_costs.entry(ns.to_string()).or_insert(0.0) += cost;
        *self.db_costs.entry(db.to_string()).or_insert(0.0) += cost;

        let ns_budget = self.ns_budgets.get(ns).copied().unwrap_or(f64::MAX);
        let db_budget = self.db_budgets.get(db).copied().unwrap_or(f64::MAX);

        let ns_cost = self.ns_costs.get(ns).copied().unwrap_or(0.0);
        let db_cost = self.db_costs.get(db).copied().unwrap_or(0.0);

        // Conservation score is the minimum of namespace and database scores.
        let ns_score = if ns_budget > 0.0 {
            (1.0 - ns_cost / ns_budget).max(0.0)
        } else {
            0.0
        };
        let db_score = if db_budget > 0.0 {
            (1.0 - db_cost / db_budget).max(0.0)
        } else {
            0.0
        };

        ns_score.min(db_score)
    }

    /// Get the conservation report for the current window.
    pub fn report(&self) -> HashMap<String, f64> {
        let mut report = HashMap::new();
        for (ns, budget) in &self.ns_budgets {
            let cost = self.ns_costs.get(ns).copied().unwrap_or(0.0);
            let score = if *budget > 0.0 {
                (1.0 - cost / budget).max(0.0)
            } else {
                0.0
            };
            report.insert(format!("ns:{}", ns), score);
        }
        for (db, budget) in &self.db_budgets {
            let cost = self.db_costs.get(db).copied().unwrap_or(0.0);
            let score = if *budget > 0.0 {
                (1.0 - cost / budget).max(0.0)
            } else {
                0.0
            };
            report.insert(format!("db:{}", db), score);
        }
        report
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_distribution_divergence() {
        let mut d1 = QueryDistribution {
            patterns: HashMap::new(),
            total: 10,
        };
        d1.patterns.insert("SELECT * FROM user".into(), 0.5);
        d1.patterns.insert("RELATE user->knows->friend".into(), 0.5);

        let mut d2 = QueryDistribution {
            patterns: HashMap::new(),
            total: 10,
        };
        d2.patterns.insert("SELECT * FROM user".into(), 0.9);
        d2.patterns.insert("RELATE user->knows->friend".into(), 0.1);

        let js = d1.divergence_from(&d2);
        assert!(js > 0.0);
    }

    #[test]
    fn test_query_budget() {
        let mut budget = QueryBudget::new(3600.0);
        budget.set_ns_budget("test_ns", 100.0);
        budget.set_db_budget("test_db", 50.0);

        let s1 = budget.record("test_ns", "test_db", 30.0);
        assert!(s1 > 0.0);
        assert!(s1 <= 1.0);

        let s2 = budget.record("test_ns", "test_db", 30.0);
        // Remaining budget: ns: 100-60=40, db: 50-60=0 → db score = 0
        assert!((s2).abs() < 1e-10);
    }
}
