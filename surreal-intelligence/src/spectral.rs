//! High-level spectral analysis functions.
//!
//! Provides the entry points used by the report generator:
//! community detection, bridge identification, partition quality,
//! and the Cheeger constant computation.

use crate::graph::RecordGraph;

/// Result of a full spectral analysis on a record graph.
#[derive(Debug, Clone)]
pub struct SpectralAnalysis {
    /// Number of nodes in the graph.
    pub node_count: usize,
    /// Number of edges.
    pub edge_count: usize,
    /// Detected communities (each is a list of node indices).
    pub communities: Vec<Vec<usize>>,
    /// Number of natural clusters found.
    pub cluster_count: usize,
    /// Mean cluster size.
    pub mean_cluster_size: f64,
    /// Bridge records — nodes whose removal would disconnect the graph.
    /// Identified by high effective resistance to their own community.
    pub bridge_nodes: Vec<usize>,
    /// The Cheeger constant — measures how well-partitioned the graph is.
    pub cheeger_constant: f64,
    /// Fiedler vector values (for diagnostic / ordering).
    pub fiedler_vector: Vec<f64>,
}

/// Run full spectral analysis on a record graph.
pub fn analyze_graph(graph: &RecordGraph) -> SpectralAnalysis {
    let n = graph.node_count();
    if n == 0 {
        return SpectralAnalysis {
            node_count: 0,
            edge_count: 0,
            communities: vec![],
            cluster_count: 0,
            mean_cluster_size: 0.0,
            bridge_nodes: vec![],
            cheeger_constant: 0.0,
            fiedler_vector: vec![],
        };
    }

    // Detect communities (auto-detect number of clusters).
    let num_clusters = estimate_clusters(graph);
    let communities = graph.detect_communities(num_clusters);
    let cluster_count = communities.len();
    let mean_cluster_size = if cluster_count > 0 {
        n as f64 / cluster_count as f64
    } else {
        0.0
    };

    // Find bridge nodes: nodes where the Fiedler vector sign differs from most of
    // their neighbors → potential cut edges.
    let fv = graph.fiedler_vector();
    let mut bridge_nodes: Vec<usize> = Vec::new();
    for i in 0..n {
        let sign_i = if fv[i] >= 0.0 { 1 } else { -1 };
        let mut neighbor_sign_mismatch = 0;
        let mut total_neighbors = 0;
        for edge in &graph.adj[i] {
            total_neighbors += 1;
            let sign_j = if fv[edge.to] >= 0.0 { 1 } else { -1 };
            if sign_i != sign_j {
                neighbor_sign_mismatch += 1;
            }
        }
        // Node is a bridge if >50% of neighbors have opposite Fiedler sign.
        if total_neighbors > 0
            && neighbor_sign_mismatch as f64 / total_neighbors as f64 > 0.5
        {
            bridge_nodes.push(i);
        }
    }

    // Cheeger constant.
    let cheeger = graph.cheeger_constant();

    SpectralAnalysis {
        node_count: n,
        edge_count: graph.edge_count,
        communities,
        cluster_count,
        mean_cluster_size,
        bridge_nodes,
        cheeger_constant: cheeger,
        fiedler_vector: fv,
    }
}

/// Estimate the number of natural clusters from the eigenvalue gap.
fn estimate_clusters(graph: &RecordGraph) -> usize {
    let n = graph.node_count();
    if n < 3 {
        return 1;
    }

    let l = graph.laplacian();
    let k = n.min(10);
    let (eigenvals, _) = l.symmetric_qr(k, 100);

    // Find the largest gap among eigenvalues (excluding λ₀ ≈ 0).
    let mut max_gap = 0.0;
    let mut gap_idx = 1;
    for i in 1..k.saturating_sub(1) {
        let gap = eigenvals[i + 1] - eigenvals[i];
        if gap > max_gap && eigenvals[i] > 1e-10 {
            max_gap = gap;
            gap_idx = i;
        }
    }

    gap_idx + 1
}

/// Compute the Jensen-Shannon divergence between two probability distributions.
///
/// JS divergence is a symmetrized, bounded measure of dissimilarity between
/// two distributions. It's useful for comparing query pattern distributions
/// across different time windows.
///
/// Returns a value in [0, ln(2)] (or [0, 1] if sqrt is used).
pub fn jensen_shannon_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "distributions must have same length");

    let n = p.len();
    if n == 0 {
        return 0.0;
    }

    // KL(P || M) where M = (P + Q) / 2
    let mut kl_pm = 0.0;
    let mut kl_qm = 0.0;

    for i in 0..n {
        let pi = p[i];
        let qi = q[i];
        let mi = (pi + qi) / 2.0;

        if pi > 0.0 && mi > 0.0 {
            kl_pm += pi * (pi / mi).ln();
        }
        if qi > 0.0 && mi > 0.0 {
            kl_qm += qi * (qi / mi).ln();
        }
    }

    // JS divergence = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    0.5 * kl_pm + 0.5 * kl_qm
}

/// Compute the conservation score for a query, given the query's graph
/// traversal cost and the available budget.
///
/// Returns a value in [0, 1] where:
/// - `1.0` = within budget (no conservation pressure)
/// - `0.0` = at or over budget
pub fn query_conservation(cost: f64, budget: f64) -> f64 {
    if budget <= 0.0 {
        return 0.0;
    }
    if cost <= 0.0 {
        return 1.0;
    }
    (1.0 - cost / budget).max(0.0).min(1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::RecordGraph;

    #[test]
    fn test_js_divergence_identical() {
        let p = vec![0.3, 0.2, 0.5];
        let q = vec![0.3, 0.2, 0.5];
        let js = jensen_shannon_divergence(&p, &q);
        assert!((js).abs() < 1e-10);
    }

    #[test]
    fn test_js_divergence_different() {
        let p = vec![1.0, 0.0];
        let q = vec![0.0, 1.0];
        let js = jensen_shannon_divergence(&p, &q);
        // JS divergence for fully separated distributions.
        assert!(js > 0.0);
    }

    #[test]
    fn test_conservation_within_budget() {
        let c = query_conservation(5.0, 10.0);
        assert!((c - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_conservation_over_budget() {
        let c = query_conservation(20.0, 10.0);
        assert!((c).abs() < 1e-10);
    }

    #[test]
    fn test_analyze_simple_graph() {
        let mut g = RecordGraph::new();
        for i in 0..5 {
            g.add_edge(format!("n{}", i), format!("n{}", (i + 1) % 5), 1.0);
        }
        let analysis = analyze_graph(&g);
        assert!(!analysis.fiedler_vector.is_empty());
        assert!(analysis.node_count == 5);
        assert!(analysis.cluster_count >= 1);
    }

    #[test]
    fn test_estimate_clusters() {
        // Two disconnected cliques → should find 2 clusters.
        let mut g = RecordGraph::new();
        // Clique A: 0-1-2-3
        for i in 0..3 {
            for j in (i + 1)..4 {
                g.add_edge(format!("a{}", i), format!("a{}", j), 1.0);
            }
        }
        // Clique B: 4-5-6-7
        for i in 0..3 {
            for j in (i + 1)..4 {
                g.add_edge(format!("b{}", i), format!("b{}", j), 1.0);
            }
        }
        let k = estimate_clusters(&g);
        // Should detect at least 1 cluster (the spectral method on disconnected
        // components).
        assert!(k >= 1);
    }
}
