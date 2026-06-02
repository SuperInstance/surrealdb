//! `analyze::intelligence` functions — spectral graph analysis and query intelligence.
//!
//! These functions expose the `surrealdb-intelligence` crate through SurrealQL,
//! providing spectral analysis of the record graph.
//!
//! # Available functions
//!
//! - `analyze::intelligence::report()` — Full query intelligence report
//!   Accepts a graph defined by nodes and edges arrays.
//! - `analyze::intelligence::fiedler(nodes, edges)` — Fiedler vector
//! - `analyze::intelligence::communities(nodes, edges)` — Detected communities
//! - `analyze::intelligence::cheeger(nodes, edges)` — Cheeger constant
//!
//! # Usage
//!
//! ```surql
//! -- Build your graph from records and relations:
//! LET $nodes = (SELECT VALUE id FROM user);
//! LET $edges = (SELECT VALUE
//!     { from: in, to: out, weight: 1 }
//!     FROM ->knows->user
//! );
//!
//! -- Get the full intelligence report:
//! ANALYZE::INTELLIGENCE::REPORT($nodes, $edges);
//!
//! -- Or individual spectral analyses:
//! ANALYZE::INTELLIGENCE::FIEDLER($nodes, $edges);
//! ANALYZE::INTELLIGENCE::COMMUNITIES($nodes, $edges);
//! ANALYZE::INTELLIGENCE::CHEEGER($nodes, $edges);
//! ```

use anyhow::Result;
use reblessive::tree::Stk;

use crate::ctx::FrozenContext;
use crate::dbs::Options;
use crate::doc::CursorDoc;
use crate::val::{Array, Number, Object, Value};

use surrealdb_intelligence::graph::RecordGraph;
use surrealdb_intelligence::report::QueryIntelligenceReport;
use surrealdb_intelligence::spectral;

/// Helper: extract a RecordId string from a Value.
fn rid_to_string(v: &Value) -> Option<String> {
    match v {
        Value::RecordId(rid) => {
            // Construct the SurrealQL-style record ID string.
            Some(format!(
                "{}:{}",
                rid.table,
                match &rid.key {
                    crate::val::RecordIdKey::String(s) => format!("⟨{}⟩", s),
                    crate::val::RecordIdKey::Number(n) => n.to_string(),
                    crate::val::RecordIdKey::Uuid(u) => u.to_string(),
                }
            ))
        }
        Value::String(s) => Some(s.to_string()),
        _ => None,
    }
}

/// Helper: build a `RecordGraph` from SurrealQL arrays of nodes and edges.
///
/// `nodes`: array of record IDs or strings identifying graph nodes.
/// `edges`: array of objects with `from`, `to`, and optional `weight` (default 1.0).
fn build_graph_from_values(nodes: &Value, edges: &Value) -> Result<RecordGraph> {
    let mut graph = RecordGraph::new();

    // Process nodes.
    if let Value::Array(nodes_arr) = nodes {
        for node_val in nodes_arr.iter() {
            if let Some(id) = rid_to_string(node_val) {
                graph.add_node(id);
            }
        }
    }

    // Process edges.
    if let Value::Array(edges_arr) = edges {
        for edge_val in edges_arr.iter() {
            if let Value::Object(edge_obj) = edge_val {
                let from = edge_obj
                    .get("from")
                    .and_then(rid_to_string)
                    .unwrap_or_default();
                let to = edge_obj
                    .get("to")
                    .and_then(rid_to_string)
                    .unwrap_or_default();
                let weight = edge_obj
                    .get("weight")
                    .and_then(|v| match v {
                        Value::Number(n) => Some(n.as_float()),
                        _ => None,
                    })
                    .unwrap_or(1.0);

                if !from.is_empty() && !to.is_empty() {
                    graph.add_edge(from, to, weight);
                }
            }
        }
    }

    Ok(graph)
}

/// `analyze::intelligence::report(nodes, edges)` — Full query intelligence report.
///
/// Accepts:
/// - `nodes`: Array of record IDs or strings identifying graph nodes.
/// - `edges`: Array of objects `{ from: record_id, to: record_id, weight?: number }`.
///
/// Returns a detailed `QueryIntelligenceReport` as a SurrealDB object.
pub async fn report(
    (_stk, _ctx, _opt): (&mut Stk, &FrozenContext, Option<&Options>),
    (nodes, edges): (Value, Value),
) -> Result<Value> {
    let graph = build_graph_from_values(&nodes, &edges)?;
    let report = QueryIntelligenceReport::generate(&graph);
    Ok(serialize_report(report))
}

/// `analyze::intelligence::fiedler(nodes, edges)` — Return the Fiedler vector.
///
/// Returns an array of objects `{ node: string, fiedler: float, partition: "A" | "B" }`.
pub async fn fiedler(
    (_stk, _ctx, _opt): (&mut Stk, &FrozenContext, Option<&Options>),
    (nodes, edges): (Value, Value),
) -> Result<Value> {
    let graph = build_graph_from_values(&nodes, &edges)?;

    if graph.node_count() == 0 {
        return Ok(Value::None);
    }

    let fv = graph.fiedler_vector();
    let mut result = Array::with_capacity(fv.len());
    for (i, &val) in fv.iter().enumerate() {
        let mut obj = Object::default();
        obj.insert("node".to_string(), Value::from(graph.nodes[i].as_str()));
        obj.insert("fiedler".to_string(), Value::Number(Number::Float(val)));
        obj.insert(
            "partition".to_string(),
            Value::from(if val >= 0.0 { "A" } else { "B" }),
        );
        result.push(Value::Object(obj));
    }
    Ok(Value::Array(result))
}

/// `analyze::intelligence::communities(nodes, edges)` — Return detected communities.
///
/// Returns an array of `{ cluster: int, size: int, members: [string] }`.
pub async fn communities(
    (_stk, _ctx, _opt): (&mut Stk, &FrozenContext, Option<&Options>),
    (nodes, edges): (Value, Value),
) -> Result<Value> {
    let graph = build_graph_from_values(&nodes, &edges)?;

    if graph.node_count() == 0 {
        return Ok(Value::None);
    }

    let analysis = spectral::analyze_graph(&graph);
    let mut result = Array::with_capacity(analysis.communities.len());

    for (idx, community) in analysis.communities.iter().enumerate() {
        let mut obj = Object::default();
        obj.insert("cluster".to_string(), Value::from(idx as i64));
        obj.insert("size".to_string(), Value::from(community.len() as i64));
        let members: Array = community
            .iter()
            .map(|&i| Value::from(graph.nodes[i].as_str()))
            .collect();
        obj.insert("members".to_string(), Value::Array(members));
        result.push(Value::Object(obj));
    }

    Ok(Value::Array(result))
}

/// `analyze::intelligence::cheeger(nodes, edges)` — Return the Cheeger constant.
///
/// The Cheeger constant measures how well-partitioned the graph is
/// (0 = perfectly partitioned, 1+ = no meaningful partition).
pub async fn cheeger(
    (_stk, _ctx, _opt): (&mut Stk, &FrozenContext, Option<&Options>),
    (nodes, edges): (Value, Value),
) -> Result<Value> {
    let graph = build_graph_from_values(&nodes, &edges)?;

    if graph.node_count() < 2 {
        return Ok(Value::Number(Number::Float(1.0)));
    }

    let h = graph.cheeger_constant();
    Ok(Value::Number(Number::Float(h)))
}

/// Convert the `QueryIntelligenceReport` into a SurrealDB `Value::Object`.
fn serialize_report(report: QueryIntelligenceReport) -> Value {
    let mut obj = Object::default();

    // Graph topology.
    let mut topo = Object::default();
    topo.insert(
        "total_nodes".into(),
        Value::from(report.graph_topology.total_nodes as i64),
    );
    topo.insert(
        "total_edges".into(),
        Value::from(report.graph_topology.total_edges as i64),
    );
    topo.insert(
        "natural_clusters".into(),
        Value::from(report.graph_topology.natural_clusters as i64),
    );
    topo.insert(
        "bridge_count".into(),
        Value::from(report.graph_topology.bridge_count as i64),
    );
    topo.insert(
        "mean_cluster_size".into(),
        Value::Number(Number::Float(report.graph_topology.mean_cluster_size)),
    );
    obj.insert("graph_topology".into(), Value::Object(topo));

    // Spectral summary.
    let mut spec = Object::default();
    spec.insert(
        "fiedler_value".into(),
        Value::Number(Number::Float(report.spectral.fiedler_value)),
    );
    spec.insert(
        "cheeger_constant".into(),
        Value::Number(Number::Float(report.spectral.cheeger_constant)),
    );
    spec.insert(
        "well_partitioned".into(),
        Value::from(report.spectral.well_partitioned),
    );
    spec.insert(
        "eigenvalue_count".into(),
        Value::from(report.spectral.eigenvalue_count as i64),
    );
    obj.insert("spectral".into(), Value::Object(spec));

    // Bridge records.
    let bridges: Array = report
        .bridge_records
        .iter()
        .map(|b| {
            let mut bo = Object::default();
            bo.insert("node_index".into(), Value::from(b.node_index as i64));
            bo.insert("node_id".into(), Value::from(b.node_id.as_str()));
            bo.insert(
                "cross_cluster_edges".into(),
                Value::from(b.cross_cluster_edges as i64),
            );
            bo.insert(
                "effective_resistance".into(),
                Value::Number(Number::Float(b.effective_resistance)),
            );
            Value::Object(bo)
        })
        .collect();
    obj.insert("bridge_records".into(), Value::Array(bridges));

    // Communities.
    let comms: Array = report
        .communities
        .iter()
        .map(|c| {
            let mut co = Object::default();
            co.insert("index".into(), Value::from(c.index as i64));
            co.insert("size".into(), Value::from(c.size as i64));
            co.insert(
                "internal_density".into(),
                Value::Number(Number::Float(c.internal_density)),
            );
            let members: Array = c
                .member_ids
                .iter()
                .map(|m| Value::from(m.as_str()))
                .collect();
            co.insert("members".into(), Value::Array(members));
            Value::Object(co)
        })
        .collect();
    obj.insert("communities".into(), Value::Array(comms));

    // Partition quality.
    let mut pq = Object::default();
    pq.insert(
        "cheeger_constant".into(),
        Value::Number(Number::Float(report.partition_quality.cheeger_constant)),
    );
    pq.insert(
        "interpretation".into(),
        Value::from(report.partition_quality.interpretation.as_str()),
    );
    obj.insert("partition_quality".into(), Value::Object(pq));

    // Query analysis.
    let mut qa = Object::default();
    let scores: Array = report
        .query_analysis
        .conservation_scores
        .iter()
        .map(|e| {
            let mut so = Object::default();
            so.insert("scope".into(), Value::from(e.scope.as_str()));
            so.insert("score".into(), Value::Number(Number::Float(e.score)));
            Value::Object(so)
        })
        .collect();
    qa.insert("conservation_scores".into(), Value::Array(scores));
    qa.insert(
        "distribution_shift_warning".into(),
        report
            .query_analysis
            .distribution_shift_warning
            .map(|s| Value::from(s))
            .unwrap_or(Value::None),
    );
    obj.insert("query_analysis".into(), Value::Object(qa));

    // Recommendations.
    let recs: Array = report
        .recommendations
        .iter()
        .map(|r| Value::from(r.as_str()))
        .collect();
    obj.insert("recommendations".into(), Value::Array(recs));

    Value::Object(obj)
}
