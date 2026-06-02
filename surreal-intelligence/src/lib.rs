//! # SurrealDB Intelligence — Spectral Graph Analysis & Query Intelligence
//!
//! This crate provides tools for **spectral graph analysis** of SurrealDB's
//! record graph. It builds a weighted undirected graph from records (nodes)
//! and their relations (edges), then performs spectral decomposition to
//! uncover latent structure:
//!
//! - **Fiedler vector / spectral partitioning** — find optimal sharding boundaries
//! - **Community detection** — discover natural record clusters
//! - **Effective resistance** — identify critical bridge records
//! - **Cheeger constant** — measure how well-partitioned the graph is
//! - **Jensen–Shannon divergence** — compare query pattern distributions
//! - **Query budget / conservation** — track expensive graph traversals
//!
//! ## SurrealQL integration
//!
//! ```surql
//! ANALYZE::INTELLIGENCE::REPORT()
//! ```
//!
//! Returns a full `QueryIntelligenceReport` as a SurrealDB object.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod graph;
pub mod matrix;
pub mod spectral;
pub mod query;
pub mod report;

pub use graph::RecordGraph;
pub use report::QueryIntelligenceReport;
