# Surreal Intelligence — Integration Guide

## Overview

The `surrealdb-intelligence` crate adds spectral graph analysis and query intelligence
to SurrealDB. It exposes four SurrealQL functions via the `analyze::intelligence::*` namespace.

## Architecture

```
surrealdb/
├── surreal-intelligence/          ← New crate (pure Rust, no external deps)
│   ├── src/
│   │   ├── graph.rs              RecordGraph — nodes, edges, Laplacian
│   │   ├── matrix.rs             DenseMatrix — Jacobi eigen-decomposition
│   │   ├── spectral.rs           SpectralAnalysis — communities, bridges, Cheeger
│   │   ├── query.rs              QueryDistribution, QueryBudget
│   │   ├── report.rs             QueryIntelligenceReport
│   │   └── lib.rs                Crate root
│   ├── tests/
│   │   └── integration.rs        25+ integration tests
│   └── README.md
├── surrealdb/src/fnc/
│   ├── mod.rs                    ← Dispatch entries added
│   └── intelligence.rs           ← SurrealQL function implementations
├── surrealdb/src/capabilities.rs ← ExperimentalTarget::Intelligence added
└── INTEGRATION.md                ← This file
```

## Adding to workspace

The workspace `Cargo.toml` includes:

```toml
[workspace.members]
"surrealdb-intelligence"
```

And as a workspace dependency:

```toml
surrealdb-intelligence = { version = "0.1.0", path = "surreal-intelligence" }
```

## SurrealQL Functions

### `ANALYZE::INTELLIGENCE::REPORT(nodes, edges)`

Returns a complete `QueryIntelligenceReport` object.

**Parameters:**
- `nodes` — Array of record IDs or strings identifying graph nodes
- `edges` — Array of objects `{ from: record_id, to: record_id, weight?: number }`

**Usage:**
```surql
LET $nodes = (SELECT VALUE id FROM user);
LET $edges = (SELECT VALUE { from: in, to: out, weight: count }
              FROM (SELECT ->knows->user AS out, count() AS count
                    FROM user GROUP BY out));
ANALYZE::INTELLIGENCE::REPORT($nodes, $edges);
```

### `ANALYZE::INTELLIGENCE::FIEDLER(nodes, edges)`

Returns the Fiedler vector — each node's position in the spectral embedding.
Nodes with positive values form one partition, negative values the other.

### `ANALYZE::INTELLIGENCE::COMMUNITIES(nodes, edges)`

Returns detected communities as `{ cluster, size, members }`.

### `ANALYZE::INTELLIGENCE::CHEEGER(nodes, edges)`

Returns the Cheeger constant (`0` = perfectly partitionable, `1+` = inseparable).

## Testing

```bash
cd surrealdb-intelligence
cargo test -- --test-threads=4
```

Expected: **25+ tests passing** covering:

| Category | Tests |
|----------|-------|
| Graph construction | 5 |
| Fiedler vector | 4 |
| Community detection | 4 |
| Effective resistance | 2 |
| Cheeger constant | 4 |
| Spectral analysis | 2 |
| JS divergence | 5 |
| Query budget | 5 |
| Full report | 5 |
| Edge cases | 6 |
| Large graphs | 2 |

## Experimental Feature

Intelligence functions are gated behind the `ExperimentalTarget::Intelligence` capability.

Enable via:

```surql
DEFINE EXPERIMENTAL INTELLIGENCE;
```

Or at server start:

```bash
surreal start --experimental intELLIGENCE
```

## Performance

- Graph construction is O(V + E) where V = nodes, E = edges
- Fiedler vector uses power iteration: O(k × V²) for k iterations (typically 5-20)
- Eigen-decomposition uses Jacobi method: O(V³) worst-case, but limited to ~50 sweeps
- Effective resistance uses Moore-Penrose pseudoinverse via eigen-decomposition
- All operations are pure Rust, no BLAS/LAPACK required

For graphs >10K nodes, consider sampling or sharding the analysis.

## Example: Self-optimizing sharding

```surql
-- 1. Build your graph
LET $nodes = (SELECT VALUE id FROM product);
LET $edges = (SELECT VALUE { from: id, to: related, weight: 1.0 }
              FROM (SELECT * FROM product WHERE related IS NOT NULL));

-- 2. Run intelligence analysis
LET $report = ANALYZE::INTELLIGENCE::REPORT($nodes, $edges);

-- 3. Use communities for dynamic sharding recommendations
RETURN {
    clusters: $report.graph_topology.natural_clusters,
    bridges: $report.graph_topology.bridge_count,
    partitioned: $report.spectral.well_partitioned,
    interpret: $report.partition_quality.interpretation,
    action: $report.recommendations
};
```
