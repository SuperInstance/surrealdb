# Surreal Intelligence (`surrealdb-intelligence`)

> **Your graph database has structure you can't see. Spectral analysis reveals it. Same SurrealDB. Self-aware SurrealDB.**

Spectral graph analysis for SurrealDB — finds hidden structure, critical bridges, and natural clusters in your record graph.

## What it does

- **Fiedler vector** — optimal sharding/partitioning of records across nodes
- **Community detection** — which records form natural clusters (should be co-located or sharded together)
- **Effective resistance** — identify critical bridge records that connect clusters
- **Cheeger constant** — measure how well-partitioned your data naturally is
- **Jensen-Shannon divergence** — compare query pattern distributions across time windows
- **Query conservation** — budget tracking for expensive graph traversals

## Usage

### Via SurrealQL

```surql
-- Build a graph from your records and relations:
LET $nodes = (SELECT VALUE id FROM user);
LET $edges = (SELECT VALUE { from: in, to: out, weight: 1 } FROM ->knows->user);

-- Full intelligence report:
ANALYZE::INTELLIGENCE::REPORT($nodes, $edges);

-- Individual analyses:
ANALYZE::INTELLIGENCE::FIEDLER($nodes, $edges);
ANALYZE::INTELLIGENCE::COMMUNITIES($nodes, $edges);
ANALYZE::INTELLIGENCE::CHEEGER($nodes, $edges);
```

### Example output

```json
{
    "graph_topology": {
        "total_nodes": 10000,
        "total_edges": 45000,
        "natural_clusters": 4,
        "bridge_count": 12,
        "mean_cluster_size": 2500
    },
    "spectral": {
        "fiedler_value": 0.182,
        "cheeger_constant": 0.235,
        "well_partitioned": true,
        "eigenvalue_count": 10
    },
    "bridge_records": [
        { "node_index": 7, "node_id": "user:⟨carlos⟩", "cross_cluster_edges": 3, "effective_resistance": 1.47 },
        { "node_index": 142, "node_id": "post:⟨announcements⟩", "cross_cluster_edges": 5, "effective_resistance": 2.13 }
    ],
    "recommendations": [
        "Your 10K record graph has 4 natural clusters. 12 records are bridges.",
        "12 bridge records connect clusters — pre-fetch them during high-traffic traversals.",
        "Most expensive traversals always cross bridge records. Consider caching bridge data.",
        "Shard by detected communities for optimal performance."
    ]
}
```

## Design

This crate implements spectral graph analysis using pure-Rust numerical methods:

1. **Matrix construction**: Build adjacency and Laplacian matrices from the record graph
2. **Eigen-decomposition**: Jacobi eigenvalue algorithm for symmetric matrices
3. **Spectral embedding**: Project records into the Fiedler eigenspace
4. **Community detection**: Spectral clustering via k-means on the embedding
5. **Bridge analysis**: Effective resistance computation from pseudoinverse Laplacian
6. **Query intelligence**: Track query patterns, measure distribution shift, enforce budgets

## License

Same as SurrealDB.
