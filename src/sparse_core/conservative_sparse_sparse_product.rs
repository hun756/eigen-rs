use num_traits::{One, Zero};
use std::marker::PhantomData;
use std::ops::{Add, Mul};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StorageOrder {
    RowMajor,
    ColMajor,
}

pub trait Scalar: Copy + Add<Output = Self> + Mul<Output = Self> + Zero + One + PartialEq {}
impl<T> Scalar for T where T: Copy + Add<Output = T> + Mul<Output = T> + Zero + One + PartialEq {}

#[derive(Debug, Clone)]
pub struct SparseMatrix<T: Scalar> {
    rows: usize,
    cols: usize,
    values: Vec<T>,
    inner_indices: Vec<usize>,
    outer_starts: Vec<usize>,
    storage_order: StorageOrder,
}

pub struct SparseMatrixIterator<'a, T: Scalar> {
    matrix: &'a SparseMatrix<T>,
    outer_idx: usize,
    inner_pos: usize,
}

impl<T: Scalar> SparseMatrix<T> {
    pub fn new(rows: usize, cols: usize, storage_order: StorageOrder) -> Self {
        let outer_size = match storage_order {
            StorageOrder::RowMajor => rows,
            StorageOrder::ColMajor => cols,
        };

        SparseMatrix {
            rows,
            cols,
            values: Vec::new(),
            inner_indices: Vec::new(),
            outer_starts: vec![0; outer_size + 1],
            storage_order,
        }
    }

    pub fn reserve(&mut self, nnz: usize) {
        self.values.reserve(nnz);
        self.inner_indices.reserve(nnz);
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn insert(&mut self, row: usize, col: usize, value: T) {
        if row >= self.rows || col >= self.cols {
            panic!("Index out of bounds");
        }

        let (outer_idx, inner_idx) = match self.storage_order {
            StorageOrder::RowMajor => (row, col),
            StorageOrder::ColMajor => (col, row),
        };

        let pos = self.find_insert_position(outer_idx, inner_idx);
        self.values.insert(pos, value);
        self.inner_indices.insert(pos, inner_idx);

        for i in (outer_idx + 1)..self.outer_starts.len() {
            self.outer_starts[i] += 1;
        }
    }

    fn find_insert_position(&self, outer_idx: usize, inner_idx: usize) -> usize {
        let start = self.outer_starts[outer_idx];
        let end = self.outer_starts[outer_idx + 1];

        if start == end {
            return start;
        }

        match self.inner_indices[start..end].binary_search(&inner_idx) {
            Ok(pos) => start + pos,
            Err(pos) => start + pos,
        }
    }

    pub fn iter_outer(&self, idx: usize) -> SparseMatrixIterator<T> {
        assert!(
            idx < if self.storage_order == StorageOrder::RowMajor {
                self.rows
            } else {
                self.cols
            }
        );

        SparseMatrixIterator {
            matrix: self,
            outer_idx: idx,
            inner_pos: self.outer_starts[idx],
        }
    }
}

impl<'a, T: Scalar> Iterator for SparseMatrixIterator<'a, T> {
    type Item = (usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.inner_pos >= self.matrix.outer_starts[self.outer_idx + 1] {
            return None;
        }

        let inner_idx = self.matrix.inner_indices[self.inner_pos];
        let value = self.matrix.values[self.inner_pos];
        self.inner_pos += 1;

        Some((inner_idx, value))
    }
}

pub struct ConservativeSparseSparseProduct<T: Scalar> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar> ConservativeSparseSparseProduct<T> {
    pub fn multiply(
        lhs: &SparseMatrix<T>,
        rhs: &SparseMatrix<T>,
    ) -> Result<SparseMatrix<T>, String> {
        let (lhs_rows, lhs_cols) = lhs.dimensions();
        let (rhs_rows, rhs_cols) = rhs.dimensions();

        if lhs_cols != rhs_rows {
            return Err(format!(
                "Matrix dimensions do not match for multiplication: {}x{} * {}x{}",
                lhs_rows, lhs_cols, rhs_rows, rhs_cols
            ));
        }

        let mut result = SparseMatrix::new(lhs_rows, rhs_cols, StorageOrder::ColMajor);
        let estimated_nnz = lhs.values.len() + rhs.values.len();
        result.reserve(estimated_nnz);

        let mut mask = vec![false; lhs_rows];
        let mut values = vec![T::zero(); lhs_rows];
        let mut indices = Vec::with_capacity(lhs_rows);

        for j in 0..rhs_cols {
            indices.clear();

            for (k, rhs_val) in rhs.iter_outer(j) {
                for (i, lhs_val) in lhs.iter_outer(k) {
                    let product = lhs_val * rhs_val;

                    if !mask[i] {
                        mask[i] = true;
                        values[i] = product;
                        indices.push(i);
                    } else {
                        values[i] = values[i] + product;
                    }
                }
            }

            indices.sort_unstable();
            for &i in &indices {
                result.insert(i, j, values[i]);
                mask[i] = false;
            }
        }

        Ok(result)
    }
}

pub trait SparseMatrixConversion<T: Scalar> {
    fn to_storage_order(&self, order: StorageOrder) -> SparseMatrix<T>;
}

impl<T: Scalar> SparseMatrixConversion<T> for SparseMatrix<T> {
    fn to_storage_order(&self, order: StorageOrder) -> SparseMatrix<T> {
        if self.storage_order == order {
            return self.clone();
        }

        let mut result = SparseMatrix::new(self.rows, self.cols, order);
        result.reserve(self.values.len());

        for i in 0..self.rows {
            for (j, val) in self.iter_outer(i) {
                result.insert(i, j, val);
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_matrix_creation() {
        let matrix: SparseMatrix<f64> = SparseMatrix::new(3, 3, StorageOrder::ColMajor);
        assert_eq!(matrix.dimensions(), (3, 3));
        assert_eq!(matrix.values.len(), 0);
    }

    #[test]
    fn test_sparse_matrix_insertion() {
        let mut matrix = SparseMatrix::new(3, 3, StorageOrder::ColMajor);
        matrix.insert(0, 0, 1.0);
        matrix.insert(1, 1, 2.0);
        matrix.insert(2, 2, 3.0);

        assert_eq!(matrix.values.len(), 3);
        assert_eq!(matrix.values[0], 1.0);
        assert_eq!(matrix.values[1], 2.0);
        assert_eq!(matrix.values[2], 3.0);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_sparse_matrix_out_of_bounds_insertion() {
        let mut matrix = SparseMatrix::new(2, 2, StorageOrder::ColMajor);
        matrix.insert(2, 2, 1.0);
    }

    #[test]
    fn test_sparse_matrix_multiplication() {
        let mut lhs = SparseMatrix::new(2, 2, StorageOrder::ColMajor);
        lhs.insert(0, 0, 1.0);
        lhs.insert(1, 1, 2.0);

        let mut rhs = SparseMatrix::new(2, 2, StorageOrder::ColMajor);
        rhs.insert(0, 0, 3.0);
        rhs.insert(1, 1, 4.0);

        let result = ConservativeSparseSparseProduct::multiply(&lhs, &rhs).unwrap();

        assert_eq!(result.dimensions(), (2, 2));

        let mut found_values = Vec::new();
        for i in 0..2 {
            for (j, val) in result.iter_outer(i) {
                found_values.push((i, j, val));
            }
        }

        assert_eq!(found_values.len(), 2);
        assert!(found_values.contains(&(0, 0, 3.0)));
        assert!(found_values.contains(&(1, 1, 8.0)));
    }

    #[test]
    fn test_storage_order_conversion() {
        let mut matrix = SparseMatrix::new(2, 2, StorageOrder::ColMajor);
        matrix.insert(0, 0, 1.0);
        matrix.insert(1, 1, 2.0);

        let converted = matrix.to_storage_order(StorageOrder::RowMajor);
        assert_eq!(converted.storage_order, StorageOrder::RowMajor);

        let mut found_values = Vec::new();
        for i in 0..2 {
            for (j, val) in converted.iter_outer(i) {
                found_values.push((i, j, val));
            }
        }

        assert_eq!(found_values.len(), 2);
        assert!(found_values.contains(&(0, 0, 1.0)));
        assert!(found_values.contains(&(1, 1, 2.0)));
    }

    #[test]
    fn test_multiplication_dimension_mismatch() {
        let lhs = SparseMatrix::<f64>::new(2, 3, StorageOrder::ColMajor);
        let rhs = SparseMatrix::<f64>::new(2, 2, StorageOrder::ColMajor);

        let result = ConservativeSparseSparseProduct::multiply(&lhs, &rhs);
        assert!(result.is_err());
    }
}
