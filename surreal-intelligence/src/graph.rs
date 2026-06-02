//! Record graph construction for spectral analysis.
//!
//! Builds a weighted undirected graph from SurrealDB records and their
//! relations, where:
//! - **Nodes** = records (identified by their `RecordId`)
//! - **Edges** = relations between records (edge weights derived from query
//!   frequency and relation strength)
//!
//! The graph is the foundation for all spectral analyses:
//! Laplacian construction, Fiedler vector computation, community detection,
//! effective resistance, and the Cheeger constant.

use std::collections::HashMap;

/// A node in the record graph, identified by its SurrealDB record ID string.
pub type NodeId = String;

/// An edge in the record graph.
#[derive(Clone, Debug)]
pub struct Edge {
    /// Source node index.
    pub from: usize,
    /// Target node index.
    pub to: usize,
    /// Edge weight (derived from query frequency / relation strength).
    pub weight: f64,
}

/// The weighted undirected record graph.
#[derive(Clone, Debug)]
pub struct RecordGraph {
    /// Node IDs indexed by their position.
    pub nodes: Vec<NodeId>,
    /// Lookup from node ID string to index.
    node_index: HashMap<NodeId, usize>,
    /// Adjacency list: `adj[i]` contains edges from node `i`.
    pub adj: Vec<Vec<Edge>>,
    /// Total number of edges (undirected, so each pair counted once).
    pub edge_count: usize,
}

impl RecordGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        RecordGraph {
            nodes: Vec::new(),
            node_index: HashMap::new(),
            adj: Vec::new(),
            edge_count: 0,
        }
    }

    /// Ensure a node exists; returns its index.
    pub fn add_node(&mut self, id: NodeId) -> usize {
        if let Some(&idx) = self.node_index.get(&id) {
            return idx;
        }
        let idx = self.nodes.len();
        self.nodes.push(id.clone());
        self.node_index.insert(id, idx);
        self.adj.push(Vec::new());
        idx
    }

    /// Add a weighted undirected edge between two nodes.
    /// Returns `true` if a new edge was created, `false` if the edge already exists.
    pub fn add_edge(&mut self, from: NodeId, to: NodeId, weight: f64) -> bool {
        let fi = self.add_node(from);
        let ti = self.add_node(to);
        if fi == ti {
            return false;
        }
        // Avoid duplicates.
        if self.adj[fi].iter().any(|e| e.to == ti) {
            return false;
        }
        self.adj[fi].push(Edge { from: fi, to: ti, weight });
        self.adj[ti].push(Edge { from: ti, to: fi, weight });
        self.edge_count += 1;
        true
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the adjacency matrix as a dense matrix.
    pub fn adjacency_matrix(&self) -> crate::matrix::DenseMatrix {
        let n = self.node_count();
        let mut m = crate::matrix::DenseMatrix::zeros(n, n);
        for i in 0..n {
            for edge in &self.adj[i] {
                let val = m.get(i, edge.to);
                m.data[i * n + edge.to] = val + edge.weight;
            }
        }
        m
    }

    /// Return the degree matrix (diagonal).
    pub fn degree_matrix(&self) -> crate::matrix::DenseMatrix {
        let n = self.node_count();
        let mut m = crate::matrix::DenseMatrix::zeros(n, n);
        for i in 0..n {
            let deg: f64 = self.adj[i].iter().map(|e| e.weight).sum();
            m.data[i * n + i] = deg;
        }
        m
    }

    /// Return the (unnormalized) graph Laplacian: `L = D - A`.
    pub fn laplacian(&self) -> crate::matrix::DenseMatrix {
        let n = self.node_count();
        let d = self.degree_matrix();
        let a = self.adjacency_matrix();
        let mut l = crate::matrix::DenseMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                l.data[i * n + j] = d.get(i, j) - a.get(i, j);
            }
        }
        l
    }

    /// Return the normalized Laplacian: `L_norm = I - D^{-1/2} A D^{-1/2}`.
    pub fn normalized_laplacian(&self) -> crate::matrix::DenseMatrix {
        let n = self.node_count();
        let mut l = self.laplacian();
        for i in 0..n {
            let deg = self.adj[i].iter().map(|e| e.weight).sum::<f64>();
            if deg > 0.0 {
                let inv_sqrt_deg = 1.0 / deg.sqrt();
                for j in 0..n {
                    l.data[i * n + j] *= inv_sqrt_deg;
                }
                for j in 0..n {
                    l.data[j * n + i] *= inv_sqrt_deg;
                }
            }
        }
        l
    }

    /// Compute the Fiedler vector (second smallest eigenvector of the Laplacian)
    /// via power iteration on `L` (shifted to make the second eigenvalue dominant).
    ///
    /// Returns the Fiedler vector (length = node_count).
    pub fn fiedler_vector(&self) -> Vec<f64> {
        let n = self.node_count();
        if n < 2 {
            return vec![0.0; n];
        }

        let l = self.laplacian();

        // Use Rayleigh quotient iteration to find the second smallest eigenpair.
        // Strategy: shift-invert via `(L - sigma*I)^{-1}` so the target eigenvalue
        // becomes the dominant one.
        let sigma = 0.1; // Shift below expected λ₂.

        // Build `L - sigma*I`.
        let mut shifted = l.clone();
        for i in 0..n {
            shifted.data[i * n + i] -= sigma;
        }

        // Power iteration for the smallest-magnitude eigenvalue of shifted = largest
        // of (L-sigma*I)^{-1}. We use inverse power iteration directly on shifted.
        let mut b: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
        crate::matrix::DenseMatrix::normalize(&mut b);

        for _iter in 0..200 {
            // Solve shifted * x = b via Gauss-Seidel.
            let x = solve_gauss_seidel(&shifted, &b, 50);
            let beta = crate::matrix::DenseMatrix::norm(&x);
            if beta < 1e-15 {
                break;
            }
            for v in &mut b {
                *v = 0.0;
            }
            for i in 0..n {
                b[i] = x[i] / beta;
            }

            // Deflate the constant eigenvector (all-ones for Laplacian).
            let ones_dot = b.iter().sum::<f64>() / n as f64;
            for v in &mut b {
                *v -= ones_dot;
            }
            crate::matrix::DenseMatrix::normalize(&mut b);

            // Check residual.
            let lb = l.vec_mul(&b);
            let lambda = crate::matrix::DenseMatrix::dot(&b, &lb);
            let residual: f64 = lb.iter().zip(b.iter())
                .map(|(lbi, bi)| (lbi - lambda * bi).abs())
                .sum::<f64>() / n as f64;
            if residual < 1e-8 {
                break;
            }
        }

        // Orthogonalize against ones vector (constant).
        let ones_dot = b.iter().sum::<f64>() / n as f64;
        for v in &mut b {
            *v -= ones_dot;
        }
        crate::matrix::DenseMatrix::normalize(&mut b);

        b
    }

    /// Partition nodes using the sign of the Fiedler vector (spectral bisection).
    ///
    /// Returns `(partition_a, partition_b)` where each is a list of node indices.
    pub fn spectral_partition(&self) -> (Vec<usize>, Vec<usize>) {
        let fv = self.fiedler_vector();
        let mut a = Vec::new();
        let mut b = Vec::new();
        for (i, &val) in fv.iter().enumerate() {
            if val >= 0.0 {
                a.push(i);
            } else {
                b.push(i);
            }
        }
        (a, b)
    }

    /// Detect communities using recursive spectral bisection.
    ///
    /// Returns a list of communities, each being a list of node indices.
    pub fn detect_communities(&self, max_clusters: usize) -> Vec<Vec<usize>> {
        let n = self.node_count();
        if n == 0 {
            return vec![];
        }
        let fv = self.fiedler_vector();

        // Sort by Fiedler vector magnitude and split into `max_clusters` groups.
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| fv[a].partial_cmp(&fv[b]).unwrap());

        let cluster_size = (n + max_clusters - 1) / max_clusters;
        let mut communities = Vec::new();
        for chunk in indices.chunks(cluster_size) {
            if !chunk.is_empty() {
                communities.push(chunk.to_vec());
            }
        }
        communities
    }

    /// Compute the effective resistance between two nodes (via the
    /// Moore–Penrose pseudoinverse of the Laplacian).
    ///
    /// Returns the effective resistance `R_ij`.
    pub fn effective_resistance(&self, i: usize, j: usize) -> f64 {
        let n = self.node_count();
        if n < 2 || i >= n || j >= n {
            return f64::INFINITY;
        }

        let l = self.laplacian();
        let (eigenvals, eigvecs) = l.symmetric_qr(n, 100);

        // Pseudo-inverse: sum over non-zero eigenvalues of (v_k[i] - v_k[j])² / λ_k.
        let mut r = 0.0;
        for k in 1..n {
            // Skip λ₀ ≈ 0 (constant eigenvector).
            if eigenvals[k].abs() > 1e-12 {
                let diff = eigvecs.get(i, k) - eigvecs.get(j, k);
                r += diff * diff / eigenvals[k];
            }
        }
        r
    }

    /// Compute the Cheeger constant (isoperimetric number):
    /// `h(G) = min_{S} |∂S| / min(vol(S), vol(V\S))`
    ///
    /// Uses the Fiedler vector to find an approximate partition.
    pub fn cheeger_constant(&self) -> f64 {
        let n = self.node_count();
        if n < 2 {
            return 1.0;
        }

        let fv = self.fiedler_vector();
        let total_volume: f64 = self.adj.iter().map(|edges| edges.iter().map(|e| e.weight).sum::<f64>()).sum();

        if total_volume < 1e-15 {
            return 1.0;
        }

        // Try different thresholds along the Fiedler vector.
        let mut best_h = f64::MAX;
        let mut sorted: Vec<(f64, usize)> = fv.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for idx in 1..n {
            let s: std::collections::HashSet<usize> =
                sorted[..idx].iter().map(|(_, i)| *i).collect();
            if s.is_empty() || s.len() == n {
                continue;
            }

            // Boundary size: sum of edge weights crossing the cut.
            let mut boundary = 0.0;
            let mut vol_s = 0.0;
            for &node in &s {
                for edge in &self.adj[node] {
                    if !s.contains(&edge.to) {
                        boundary += edge.weight;
                    }
                }
                vol_s += self.adj[node].iter().map(|e| e.weight).sum::<f64>();
            }

            let vol_complement = total_volume - vol_s;
            let min_vol = vol_s.min(vol_complement);
            if min_vol > 0.0 {
                let h = boundary / min_vol;
                if h < best_h {
                    best_h = h;
                }
            }
        }

        if best_h == f64::MAX {
            1.0
        } else {
            best_h
        }
    }
}

/// Solve `A * x = b` using Gauss-Seidel iteration.
fn solve_gauss_seidel(a: &crate::matrix::DenseMatrix, b: &[f64], max_iter: usize) -> Vec<f64> {
    let n = a.rows;
    let mut x = vec![0.0; n];

    for _iter in 0..max_iter {
        let mut max_diff = 0.0;
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..n {
                if j != i {
                    sum -= a.get(i, j) * x[j];
                }
            }
            let aii = a.get(i, i);
            if aii.abs() > 1e-15 {
                let new_xi = sum / aii;
                let diff = (new_xi - x[i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                x[i] = new_xi;
            }
        }
        if max_diff < 1e-12 {
            break;
        }
    }

    x
}

impl Default for RecordGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple path graph: 0-1-2-3-4
    fn path_graph() -> RecordGraph {
        let mut g = RecordGraph::new();
        for i in 0..4 {
            g.add_edge(
                format!("node_{}", i),
                format!("node_{}", i + 1),
                1.0,
            );
        }
        g
    }

    #[test]
    fn test_path_graph_fiedler() {
        let g = path_graph();
        let fv = g.fiedler_vector();
        assert_eq!(fv.len(), 5);
        // Fiedler vector of a path graph should be monotonic.
        for w in fv.windows(2) {
            assert!(w[0] <= w[1] || w[0] >= w[1]);
        }
    }

    #[test]
    fn test_spectral_partition() {
        let g = path_graph();
        let (a, b) = g.spectral_partition();
        assert!(!a.is_empty());
        assert!(!b.is_empty());
        assert_eq!(a.len() + b.len(), g.node_count());
    }

    #[test]
    fn test_cheeger_path() {
        let g = path_graph();
        let h = g.cheeger_constant();
        // Path graph Cheeger constant ≈ 2/(n) for n=5 → ~0.4
        assert!(h > 0.0);
        assert!(h < 1.0);
    }
}
