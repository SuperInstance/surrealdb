//! Minimal dense matrix implementation for spectral graph analysis.
//!
//! This module provides a lightweight `DenseMatrix` type with the operations
//! needed for Laplacian eigen-decomposition: multiplication, transposition,
//! power iteration, and QR decomposition — all in pure Rust with no native
//! BLAS/LAPACK dependencies.

use std::fmt;

/// A dense `f64` matrix stored in row-major order.
#[derive(Clone, Debug)]
pub struct DenseMatrix {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Flattened row-major data.
    pub data: Vec<f64>,
}

impl DenseMatrix {
    /// Create a new matrix with all elements set to zero.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        DenseMatrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Create a matrix from a flat slice (row-major).
    pub fn from_row_major(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        DenseMatrix { rows, cols, data }
    }

    /// Create an identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    /// Index into the flat data.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        debug_assert!(row < self.rows && col < self.cols);
        self.data[row * self.cols + col]
    }

    /// Mutable reference into the flat data.
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut f64 {
        debug_assert!(row < self.rows && col < self.cols);
        &mut self.data[row * self.cols + col]
    }

    /// Matrix multiplication: `self * other`.
    pub fn mat_mul(&self, other: &DenseMatrix) -> DenseMatrix {
        assert_eq!(self.cols, other.rows);
        let mut result = DenseMatrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let aik = self.data[i * self.cols + k];
                if aik.abs() > 1e-15 {
                    for j in 0..other.cols {
                        result.data[i * other.cols + j] += aik * other.data[k * other.cols + j];
                    }
                }
            }
        }
        result
    }

    /// Multiply by a vector (`self * vec`).
    pub fn vec_mul(&self, vec: &[f64]) -> Vec<f64> {
        assert_eq!(self.cols, vec.len());
        let mut result = vec![0.0; self.rows];
        for i in 0..self.rows {
            let mut sum = 0.0;
            for j in 0..self.cols {
                sum += self.data[i * self.cols + j] * vec[j];
            }
            result[i] = sum;
        }
        result
    }

    /// Transpose.
    pub fn transpose(&self) -> DenseMatrix {
        let mut result = DenseMatrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        result
    }

    /// Extract a column as a slice.
    pub fn col(&self, col: usize) -> Vec<f64> {
        (0..self.rows).map(|r| self.get(r, col)).collect()
    }

    /// Dot product of two vectors.
    pub fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// Euclidean norm.
    pub fn norm(v: &[f64]) -> f64 {
        DenseMatrix::dot(v, v).sqrt()
    }

    /// Normalize a vector in-place.
    pub fn normalize(v: &mut [f64]) {
        let n = DenseMatrix::norm(v);
        if n > 1e-15 {
            for x in v.iter_mut() {
                *x /= n;
            }
        }
    }

    /// Symmetric QR iteration for eigenvalues/vectors of a symmetric matrix.
    ///
    /// Returns `(eigenvalues, eigenvectors)` where eigenvectors are stored
    /// column-wise (i.e., the i-th column is the eigenvector for eigenvalue[i]).
    /// Only returns the `k` smallest-magnitude eigenvalues and their vectors.
    pub fn symmetric_qr(&self, k: usize, max_iter: usize) -> (Vec<f64>, DenseMatrix) {
        assert_eq!(self.rows, self.cols, "QR requires a square matrix");
        let n = self.rows;
        let k = k.min(n);

        // Work on a copy.
        let mut a = self.clone();

        // QR iteration with explicit shifts (Francis double-shift style simplified).
        for _iter in 0..max_iter {
            // Check for convergence: off-diagonal below tolerance.
            let mut converged = true;
            for i in 0..n {
                for j in 0..n {
                    if i != j && a.get(i, j).abs() > 1e-12 {
                        converged = false;
                        break;
                    }
                }
                if !converged {
                    break;
                }
            }
            if converged {
                break;
            }

            // Simplified: one Jacobi sweep.
            a = jacobi_sweep(&a);
        }

        // Extract eigenvalues from diagonal.
        let eigenvals: Vec<f64> = (0..n).map(|i| a.get(i, i)).collect();

        // Sort by absolute value (smallest first → needed for Fiedler vector).
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| eigenvals[i].abs().partial_cmp(&eigenvals[j].abs()).unwrap());

        // For eigenvectors, we need to reapply the Jacobi rotations.
        // Jacobi with eigenvector accumulation.
        let (eigenvals_full, eigvecs) = jacobi_with_eigenvectors(self, max_iter);

        let mut sorted_vals2: Vec<(f64, usize)> =
            eigenvals_full.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();
        sorted_vals2.sort_by(|a, b| a.0.abs().partial_cmp(&b.0.abs()).unwrap());

        let mut result_vals = Vec::with_capacity(k);
        let mut result_vecs = DenseMatrix::zeros(n, k);
        for (idx, &(_val, col_idx)) in sorted_vals2.iter().enumerate().take(k) {
            result_vals.push(eigenvals_full[col_idx]);
            for row in 0..n {
                result_vecs.data[row * k + idx] = eigvecs.get(row, col_idx);
            }
        }

        (result_vals, result_vecs)
    }
}

/// Perform one Jacobi sweep over all off-diagonal pairs.
fn jacobi_sweep(a: &DenseMatrix) -> DenseMatrix {
    let n = a.rows;
    let mut result = a.clone();

    for p in 0..n {
        for q in (p + 1)..n {
            let app = result.get(p, p);
            let aqq = result.get(q, q);
            let apq = result.get(p, q);

            if apq.abs() < 1e-15 {
                continue;
            }

            // Compute Jacobi rotation angle.
            let tau = (aqq - app) / (2.0 * apq);
            let t = if tau >= 0.0 {
                1.0 / (tau + (1.0 + tau * tau).sqrt())
            } else {
                -1.0 / (-tau + (1.0 + tau * tau).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;

            // Apply rotation to rows p and q.
            for j in 0..n {
                if j == p || j == q {
                    continue;
                }
                let rpj = result.get(p, j);
                let rqj = result.get(q, j);
                result.data[p * n + j] = c * rpj - s * rqj;
                result.data[q * n + j] = s * rpj + c * rqj;
                // Maintain symmetry.
                result.data[j * n + p] = result.data[p * n + j];
                result.data[j * n + q] = result.data[q * n + j];
            }

            // Update diagonal and off-diagonal.
            result.data[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
            result.data[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
            result.data[p * n + q] = 0.0;
            result.data[q * n + p] = 0.0;
        }
    }

    result
}

/// Full Jacobi eigen-decomposition with eigenvector accumulation.
fn jacobi_with_eigenvectors(a: &DenseMatrix, max_iter: usize) -> (Vec<f64>, DenseMatrix) {
    let n = a.rows;
    let mut a_cur = a.clone();
    let mut v = DenseMatrix::identity(n);

    for _iter in 0..max_iter {
        // Find largest off-diagonal element in magnitude.
        let mut max_off = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = a_cur.get(i, j).abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_off < 1e-15 {
            break;
        }

        let app = a_cur.get(p, p);
        let aqq = a_cur.get(q, q);
        let apq = a_cur.get(p, q);

        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        // Apply rotation to A.
        for j in 0..n {
            if j == p || j == q {
                continue;
            }
            let rpj = a_cur.get(p, j);
            let rqj = a_cur.get(q, j);
            a_cur.data[p * n + j] = c * rpj - s * rqj;
            a_cur.data[q * n + j] = s * rpj + c * rqj;
            a_cur.data[j * n + p] = a_cur.data[p * n + j];
            a_cur.data[j * n + q] = a_cur.data[q * n + j];
        }

        a_cur.data[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a_cur.data[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a_cur.data[p * n + q] = 0.0;
        a_cur.data[q * n + p] = 0.0;

        // Apply rotation to V (eigenvectors accumulate on right).
        for i in 0..n {
            let vip = v.get(i, p);
            let viq = v.get(i, q);
            v.data[i * n + p] = c * vip - s * viq;
            v.data[i * n + q] = s * vip + c * viq;
        }
    }

    let eigenvals: Vec<f64> = (0..n).map(|i| a_cur.get(i, i)).collect();
    (eigenvals, v)
}

impl fmt::Display for DenseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, "{:10.6} ", self.get(i, j))?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_mul() {
        let i = DenseMatrix::identity(4);
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let r = i.vec_mul(&v);
        assert_eq!(r, v);
    }

    #[test]
    fn test_symmetric_3x3() {
        // Simple 3x3 symmetric matrix.
        let m = DenseMatrix::from_row_major(
            3,
            3,
            vec![2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0],
        );
        let (vals, _vecs) = m.symmetric_qr(3, 100);
        // Known eigenvalues of this path graph Laplacian: 2-√2, 2, 2+√2 approx.
        assert!((vals[0] - (2.0 - std::f64::consts::SQRT_2)).abs() < 0.1);
        assert!((vals[1] - 2.0).abs() < 0.1);
        assert!((vals[2] - (2.0 + std::f64::consts::SQRT_2)).abs() < 0.1);
    }
}
