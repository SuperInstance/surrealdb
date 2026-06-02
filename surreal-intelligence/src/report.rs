//! Query intelligence report generation.
//!
//! Builds the full `QueryIntelligenceReport` that is returned by
//! `ANALYZE::INTELLIGENCE::REPORT()`. The report contains:
//!
//! - Graph topology summary (nodes, edges, clusters, bridges)
//! - Spectral analysis results (Fiedler vector, Cheeger constant)
//! - Query pattern analysis (distribution shifts, conservation scores)
//! - Actionable recommendations

use serde::{Deserialize, Serialize};

use crate::graph::RecordGraph;
use crate::spectral::analyze_graph;

/// The complete query intelligence report.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryIntelligenceReport {
    /// Graph topology summary.
    pub graph_topology: GraphTopology,
    /// Spectral analysis results.
    pub spectral: SpectralSummary,
    /// Bridge records.
    pub bridge_records: Vec<BridgeRecord>,
    /// Community structure.
    pub communities: Vec<Community>,
    /// Partition metrics.
    pub partition_quality: PartitionQuality,
    /// Query pattern analysis.
    pub query_analysis: QueryAnalysis,
    /// Recommendations.
    pub recommendations: Vec<String>,
}

/// Summary of graph topology.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphTopology {
    /// Total number of record nodes.
    pub total_nodes: usize,
    /// Total number of edges.
    pub total_edges: usize,
    /// Number of detected clusters.
    pub natural_clusters: usize,
    /// Number of bridge records.
    pub bridge_count: usize,
    /// Mean cluster size.
    pub mean_cluster_size: f64,
}

/// Summary of spectral analysis.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpectralSummary {
    /// The Fiedler value (second smallest eigenvalue of the Laplacian).
    pub fiedler_value: f64,
    /// The Cheeger constant.
    pub cheeger_constant: f64,
    /// Whether the partition is "good" (Cheeger < 0.5).
    pub well_partitioned: bool,
    /// Number of eigenvalues computed.
    pub eigenvalue_count: usize,
}

/// A bridge record that connects clusters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BridgeRecord {
    /// Index in the graph.
    pub node_index: usize,
    /// Node ID (record ID string).
    pub node_id: String,
    /// Number of neighbors in different communities.
    pub cross_cluster_edges: usize,
    /// Effective resistance to the rest of its community.
    pub effective_resistance: f64,
}

/// A detected community.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Community {
    /// Community index.
    pub index: usize,
    /// Number of nodes in this community.
    pub size: usize,
    /// Internal edge density (total internal weight / max possible).
    pub internal_density: f64,
    /// Node IDs in this community.
    pub member_ids: Vec<String>,
}

/// Partition quality metrics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartitionQuality {
    /// Cheeger constant.
    pub cheeger_constant: f64,
    /// Interpretation of the constant.
    pub interpretation: String,
}

/// Query pattern analysis.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryAnalysis {
    /// Conservation scores by namespace.
    pub conservation_scores: Vec<ConservationEntry>,
    /// Query distribution shift warning.
    pub distribution_shift_warning: Option<String>,
}

/// A conservation score entry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConservationEntry {
    /// The scope (namespace or database).
    pub scope: String,
    /// Conservation score in [0, 1].
    pub score: f64,
}

impl QueryIntelligenceReport {
    /// Generate a full report from a record graph.
    pub fn generate(graph: &RecordGraph) -> Self {
        if graph.node_count() == 0 {
            return Self::empty();
        }

        let analysis = analyze_graph(graph);

        // Build topology summary.
        let topology = GraphTopology {
            total_nodes: analysis.node_count,
            total_edges: analysis.edge_count,
            natural_clusters: analysis.cluster_count,
            bridge_count: analysis.bridge_nodes.len(),
            mean_cluster_size: analysis.mean_cluster_size,
        };

        // Spectral summary.
        let cheeger = analysis.cheeger_constant;
        let fiedler_stat = if analysis.fiedler_vector.len() >= 2 {
            // Rough estimate of Fiedler value from Rayleigh quotient.
            let l = graph.laplacian();
            let fv = &analysis.fiedler_vector;
            let lfv = l.vec_mul(fv);
            let lambda = crate::matrix::DenseMatrix::dot(fv, &lfv)
                / crate::matrix::DenseMatrix::dot(fv, fv).max(1e-15);
            lambda
        } else {
            0.0
        };

        let spectral = SpectralSummary {
            fiedler_value: fiedler_stat,
            cheeger_constant: cheeger,
            well_partitioned: cheeger < 0.5,
            eigenvalue_count: graph.node_count().min(10),
        };

        // Bridge records.
        let mut bridge_records = Vec::new();
        for &node_idx in &analysis.bridge_nodes {
            let node_id = graph.nodes[node_idx].clone();
            let mut cross = 0;
            let fv_sign = if analysis.fiedler_vector[node_idx] >= 0.0 {
                1
            } else {
                -1
            };
            for edge in &graph.adj[node_idx] {
                let sign_j = if analysis.fiedler_vector[edge.to] >= 0.0 {
                    1
                } else {
                    -1
                };
                if fv_sign != sign_j {
                    cross += 1;
                }
            }
            let er = graph.effective_resistance(node_idx, (node_idx + 1) % graph.node_count());
            bridge_records.push(BridgeRecord {
                node_index: node_idx,
                node_id,
                cross_cluster_edges: cross,
                effective_resistance: er,
            });
        }

        // Communities.
        let mut communities = Vec::new();
        for (idx, community) in analysis.communities.iter().enumerate() {
            let size = community.len();
            // Internal density.
            let max_edges = if size > 1 {
                size * (size - 1) / 2
            } else {
                1
            };
            let mut internal_edges = 0;
            for &node in community {
                for edge in &graph.adj[node] {
                    if community.contains(&edge.to) {
                        internal_edges += 1;
                    }
                }
            }
            let density = if size > 0 {
                (internal_edges as f64) / (2.0 * max_edges as f64)
            } else {
                0.0
            };
            let member_ids: Vec<String> =
                community.iter().map(|&i| graph.nodes[i].clone()).collect();
            communities.push(Community {
                index: idx,
                size,
                internal_density: density,
                member_ids,
            });
        }

        // Partition quality.
        let interpretation = if cheeger < 0.1 {
            "Excellent — data is highly modular and naturally partitioned."
        } else if cheeger < 0.3 {
            "Good — data has clear modular structure."
        } else if cheeger < 0.5 {
            "Moderate — some structure exists but clusters are loosely connected."
        } else if cheeger < 0.8 {
            "Weak — the graph is nearly a single cluster."
        } else {
            "Uniform — no significant cluster structure detected."
        };
        let quality = PartitionQuality {
            cheeger_constant: cheeger,
            interpretation: interpretation.to_string(),
        };

        // Recommendations.
        let mut recommendations = Vec::new();
        if !bridge_records.is_empty() {
            recommendations.push(format!(
                "Your graph has {} bridge records that connect clusters. \
                 Queries traversing these bridges are the most expensive. \
                 Consider pre-loading them in application context or adding direct relations \
                 to reduce cross-cluster traversal.",
                bridge_records.len()
            ));
        }
        if cheeger < 0.3 {
            recommendations.push(
                "Data is well-partitioned — consider sharding by detected communities for \
                 optimal query locality."
                    .to_string(),
            );
        }
        if communities.len() > 1 {
            recommendations.push(format!(
                "Found {} natural data clusters with average size {:.0}. \
                 Co-locating cluster members on the same shard reduces distributed query overhead.",
                communities.len(),
                analysis.mean_cluster_size
            ));
        }
        if !recommendations.is_empty() {
            recommendations.push(
                "Run ANALYZE::INTELLIGENCE::REPORT() periodically to track how your \
                 graph structure evolves with data."
                    .to_string(),
            );
        }

        QueryIntelligenceReport {
            graph_topology: topology,
            spectral,
            bridge_records,
            communities,
            partition_quality: quality,
            query_analysis: QueryAnalysis {
                conservation_scores: vec![],
                distribution_shift_warning: None,
            },
            recommendations,
        }
    }

    /// Generate an empty report (for empty graphs).
    pub fn empty() -> Self {
        QueryIntelligenceReport {
            graph_topology: GraphTopology {
                total_nodes: 0,
                total_edges: 0,
                natural_clusters: 0,
                bridge_count: 0,
                mean_cluster_size: 0.0,
            },
            spectral: SpectralSummary {
                fiedler_value: 0.0,
                cheeger_constant: 1.0,
                well_partitioned: false,
                eigenvalue_count: 0,
            },
            bridge_records: vec![],
            communities: vec![],
            partition_quality: PartitionQuality {
                cheeger_constant: 1.0,
                interpretation: "No data to analyze.".to_string(),
            },
            query_analysis: QueryAnalysis {
                conservation_scores: vec![],
                distribution_shift_warning: None,
            },
            recommendations: vec![
                "No records found. Add some data to begin graph analysis.".to_string(),
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_report() {
        let g = RecordGraph::new();
        let report = QueryIntelligenceReport::generate(&g);
        assert_eq!(report.graph_topology.total_nodes, 0);
    }

    #[test]
    fn test_small_graph_report() {
        let mut g = RecordGraph::new();
        // Small cycle graph.
        for i in 0..10 {
            g.add_edge(format!("r{}", i), format!("r{}", (i + 1) % 10), 1.0);
        }
        let report = QueryIntelligenceReport::generate(&g);
        assert_eq!(report.graph_topology.total_nodes, 10);
        assert!(!report.recommendations.is_empty());
    }
}
