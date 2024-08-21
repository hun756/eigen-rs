//! # AmbiVector
//!
//! `AmbiVector` is a versatile vector implementation that can dynamically switch between
//! dense and sparse storage modes based on the vector's content density. This makes it
//! particularly useful for scenarios where the sparsity of the vector may change over time
//! or is not known in advance.
//!
//! ## Features
//!
//! - Dynamic switching between dense and sparse storage modes
//! - Efficient element access and modification
//! - Iterator support for non-zero elements
//! - Customizable epsilon for zero-comparison in floating-point types
//!
//! ## Usage
//!
//! Here's a quick example of how to use `AmbiVector`:
//!
//! ```rust
//!  use eigen_rs::sparse_core::ambi_vector::AmbiVector;
//!
//! // Create a new AmbiVector with 10 elements
//! let mut vec: AmbiVector<f64, u32> = AmbiVector::new(10);
//!
//! // Initialize the vector with an estimated density
//! vec.init(0.5); // This will set it to dense mode
//!
//! // Set some values
//! *vec.coeff_ref(2) = 3.14;
//! *vec.coeff_ref(5) = 2.718;
//!
//! // Iterate over non-zero elements
//! for (index, value) in vec.iter(1e-10) {
//!     println!("Index: {}, Value: {}", index, value);
//! }
//! ```
//!
//! ## Choosing Between Dense and Sparse Modes
//!
//! - Use `init(estimated_density)` to automatically choose the mode based on the estimated density.
//! - Use `init_mode(Mode::Dense)` or `init_mode(Mode::Sparse)` to explicitly set the mode.
//!
//! Dense mode is generally more efficient when more than 10% of the elements are non-zero.
//! Sparse mode is more memory-efficient for vectors with few non-zero elements.
//!
//! ## Performance Considerations
//!
//! - Dense mode provides O(1) access time but uses memory for all elements.
//! - Sparse mode provides O(log n) access time (where n is the number of non-zero elements) but only stores non-zero elements.
//! - The `resize()` method may trigger reallocation, which can be costly for large vectors.
//!
//! ## Examples
//!
//! ### Creating and Populating an AmbiVector
//!
//! ```rust
//! use eigen_rs::sparse_core::ambi_vector::{AmbiVector, Mode};
//!
//! let mut vec: AmbiVector<f64, u32> = AmbiVector::new(5);
//! vec.init_mode(Mode::Sparse);
//!
//! *vec.coeff_ref(1) = 2.0;
//! *vec.coeff_ref(3) = 4.0;
//!
//! assert_eq!(vec.non_zeros(), 2);
//! ```
//!
//! ### Using the Iterator
//!
//! ```rust
//!  use eigen_rs::sparse_core::ambi_vector::AmbiVector;
//!
//! let mut vec: AmbiVector<f64, u32> = AmbiVector::new(5);
//! *vec.coeff_ref(1) = 2.0;
//! *vec.coeff_ref(3) = 4.0;
//!
//! let sum: f64 = vec.iter(1e-10).map(|(_, value)| value).sum();
//! assert_eq!(sum, 6.0);
//! ```
//!
//! ### Resizing and Changing Bounds
//!
//! ```rust
//!  use eigen_rs::sparse_core::ambi_vector::AmbiVector;
//!
//! let mut vec: AmbiVector<f64, u32> = AmbiVector::new(5);
//! vec.resize(10);
//! vec.set_bounds(2, 8);
//!
//! // Now, only indices 2 through 7 (inclusive) are valid
//! ```
use num_traits::{Float, PrimInt, Unsigned, Zero};
use std::marker::PhantomData;
use std::mem;

/// A versatile vector that can switch between dense and sparse storage modes.
///
/// # Type Parameters
///
/// - `Scalar`: The type of the values stored in the vector. Must be a floating-point type.
/// - `StorageIndex`: The type used for indexing. Must be an unsigned integer type.
///
/// # Examples
///
/// ```
/// use eigen_rs::sparse_core::ambi_vector::AmbiVector;
///
/// let mut vec: AmbiVector<f64, u32> = AmbiVector::new(10);
/// vec.init(0.2); // Initialize with 20% estimated density
/// *vec.coeff_ref(3) = 3.14;
/// assert_eq!(*vec.coeff_ref(3), 3.14);
/// ```
#[derive(Debug)]
pub struct AmbiVector<Scalar, StorageIndex>
where
    Scalar: Zero + Clone + PartialOrd,
    StorageIndex: PrimInt + Unsigned,
{
    buffer: Vec<u8>,
    zero: Scalar,
    size: StorageIndex,
    start: StorageIndex,
    end: StorageIndex,
    allocated_elements: StorageIndex,
    mode: Mode,
    ll_start: StorageIndex,
    ll_current: StorageIndex,
    ll_size: StorageIndex,
    _marker: PhantomData<(Scalar, StorageIndex)>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    Dense,
    Sparse,
}

#[derive(Debug, Clone)]
struct ListEl<Scalar, StorageIndex> {
    next: StorageIndex,
    index: StorageIndex,
    value: Scalar,
}

impl<Scalar, StorageIndex> AmbiVector<Scalar, StorageIndex>
where
    Scalar: Zero + Clone + PartialOrd + Float,
    StorageIndex: PrimInt + Unsigned,
{
    /// Creates a new `AmbiVector` with the specified size.
    ///
    /// # Arguments
    ///
    /// * `size` - The number of elements in the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use eigen_rs::sparse_core::ambi_vector::AmbiVector;
    ///
    /// let vec: AmbiVector<f64, u32> = AmbiVector::new(10);
    /// ```
    pub fn new(size: usize) -> Self {
        let mut vec = Self {
            buffer: Vec::new(),
            zero: Scalar::zero(),
            size: StorageIndex::zero(),
            start: StorageIndex::zero(),
            end: StorageIndex::zero(),
            allocated_elements: StorageIndex::zero(),
            mode: Mode::Dense,
            ll_start: StorageIndex::max_value(),
            ll_current: StorageIndex::zero(),
            ll_size: StorageIndex::zero(),
            _marker: PhantomData,
        };
        vec.resize(size);
        vec
    }

    /// Initializes the vector with an estimated density.
    ///
    /// This method chooses between dense and sparse modes based on the estimated density.
    ///
    /// # Arguments
    ///
    /// * `estimated_density` - The estimated fraction of non-zero elements (0.0 to 1.0).
    ///
    /// # Examples
    ///
    /// ```
    /// use eigen_rs::sparse_core::ambi_vector::AmbiVector;
    ///
    /// let mut vec: AmbiVector<f64, u32> = AmbiVector::new(100);
    /// vec.init(0.05); // Likely to choose sparse mode
    /// ```
    pub fn init(&mut self, estimated_density: f64) {
        if estimated_density > 0.1 {
            self.init_mode(Mode::Dense);
        } else {
            self.init_mode(Mode::Sparse);
        }
    }

    /// Initializes the vector with the specified mode.
    /// 
    /// # Arguments
    /// * `mode` - The mode to initialize the vector with.
    /// 
    /// # Examples
    /// ```
    /// use eigen_rs::sparse_core::ambi_vector::{AmbiVector, Mode};
    /// 
    /// let mut vec: AmbiVector<f64, u32> = AmbiVector::new(100);
    /// vec.init_mode(Mode::Sparse);
    /// ```
    pub fn init_mode(&mut self, mode: Mode) {
        self.mode = mode;
        if self.mode == Mode::Sparse {
            self.ll_size = StorageIndex::zero();
            self.ll_start = StorageIndex::max_value();
        }
    }

    pub fn resize(&mut self, size: usize) {
        let size_index = StorageIndex::from(size).unwrap();
        if self.allocated_elements < size_index {
            self.reallocate(size);
        }
        self.size = size_index;
    }

    fn reallocate(&mut self, size: usize) {
        let size_index = StorageIndex::from(size).unwrap();
        if size < 1000 {
            let alloc_size = (size * mem::size_of::<ListEl<Scalar, StorageIndex>>()
                + mem::size_of::<Scalar>()
                - 1)
                / mem::size_of::<Scalar>();
            self.allocated_elements = StorageIndex::from(
                alloc_size * mem::size_of::<Scalar>()
                    / mem::size_of::<ListEl<Scalar, StorageIndex>>(),
            )
            .unwrap();
            self.buffer.resize(alloc_size * mem::size_of::<Scalar>(), 0);
        } else {
            self.allocated_elements = StorageIndex::from(
                size * mem::size_of::<Scalar>() / mem::size_of::<ListEl<Scalar, StorageIndex>>(),
            )
            .unwrap();
            self.buffer.resize(size * mem::size_of::<Scalar>(), 0);
        }
        self.size = size_index;
        self.start = StorageIndex::zero();
        self.end = self.size;
    }

    pub fn set_bounds(&mut self, start: usize, end: usize) {
        self.start = StorageIndex::from(start).unwrap();
        self.end = StorageIndex::from(end).unwrap();
    }

    pub fn set_zero(&mut self) {
        match self.mode {
            Mode::Dense => {
                for i in self.start.to_usize().unwrap()..self.end.to_usize().unwrap() {
                    unsafe {
                        let ptr = self.buffer.as_mut_ptr().add(i * mem::size_of::<Scalar>())
                            as *mut Scalar;
                        *ptr = Scalar::zero();
                    }
                }
            }
            Mode::Sparse => {
                self.ll_size = StorageIndex::zero();
                self.ll_start = StorageIndex::max_value();
            }
        }
    }

    pub fn restart(&mut self) {
        self.ll_current = self.ll_start;
    }

    /// Returns a mutable reference to the element at the specified index.
    ///
    /// # Arguments
    ///
    /// * `i` - The index of the element.
    ///
    /// # Examples
    ///
    /// ```
    /// use eigen_rs::sparse_core::ambi_vector::AmbiVector;
    ///
    /// let mut vec: AmbiVector<f64, u32> = AmbiVector::new(5);
    /// *vec.coeff_ref(2) = 3.14;
    /// assert_eq!(*vec.coeff_ref(2), 3.14);
    /// ```
    pub fn coeff_ref(&mut self, i: usize) -> &mut Scalar {
        let i_index = StorageIndex::from(i).unwrap();
        match self.mode {
            Mode::Dense => unsafe {
                let ptr = self.buffer.as_mut_ptr().add(i * mem::size_of::<Scalar>()) as *mut Scalar;
                &mut *ptr
            },
            Mode::Sparse => {
                let mut elements = self.buffer.as_mut_ptr() as *mut ListEl<Scalar, StorageIndex>;
                unsafe {
                    if self.ll_size == StorageIndex::zero() {
                        self.ll_start = StorageIndex::zero();
                        self.ll_current = StorageIndex::zero();
                        self.ll_size = StorageIndex::one();
                        let el = &mut *elements.add(0);
                        el.value = Scalar::zero();
                        el.index = i_index;
                        el.next = StorageIndex::max_value();
                        &mut el.value
                    } else if i_index < (*elements.add(self.ll_start.to_usize().unwrap())).index {
                        let el = &mut *elements.add(self.ll_size.to_usize().unwrap());
                        el.value = Scalar::zero();
                        el.index = i_index;
                        el.next = self.ll_start;
                        self.ll_start = self.ll_size;
                        self.ll_size = self.ll_size + StorageIndex::one();
                        self.ll_current = self.ll_start;
                        &mut el.value
                    } else {
                        let mut next_el = (*elements.add(self.ll_current.to_usize().unwrap())).next;
                        while next_el < StorageIndex::max_value()
                            && (*elements.add(next_el.to_usize().unwrap())).index <= i_index
                        {
                            self.ll_current = next_el;
                            next_el = (*elements.add(next_el.to_usize().unwrap())).next;
                        }

                        if (*elements.add(self.ll_current.to_usize().unwrap())).index == i_index {
                            &mut (*elements.add(self.ll_current.to_usize().unwrap())).value
                        } else {
                            if self.ll_size >= self.allocated_elements {
                                self.reallocate_sparse();
                                elements =
                                    self.buffer.as_mut_ptr() as *mut ListEl<Scalar, StorageIndex>;
                            }
                            let el = &mut *elements.add(self.ll_size.to_usize().unwrap());
                            el.value = Scalar::zero();
                            el.index = i_index;
                            el.next = (*elements.add(self.ll_current.to_usize().unwrap())).next;
                            (*elements.add(self.ll_current.to_usize().unwrap())).next =
                                self.ll_size;
                            self.ll_size = self.ll_size + StorageIndex::one();
                            &mut el.value
                        }
                    }
                }
            }
        }
    }

    fn reallocate_sparse(&mut self) {
        let copy_elements = self.allocated_elements;
        self.allocated_elements = StorageIndex::min(
            self.allocated_elements * StorageIndex::from(3).unwrap()
                / StorageIndex::from(2).unwrap(),
            self.size,
        );
        let alloc_size = self.allocated_elements.to_usize().unwrap()
            * mem::size_of::<ListEl<Scalar, StorageIndex>>();
        let alloc_size = (alloc_size + mem::size_of::<Scalar>() - 1) / mem::size_of::<Scalar>();
        let mut new_buffer = vec![0u8; alloc_size * mem::size_of::<Scalar>()];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.buffer.as_ptr(),
                new_buffer.as_mut_ptr(),
                copy_elements.to_usize().unwrap() * mem::size_of::<ListEl<Scalar, StorageIndex>>(),
            );
        }
        self.buffer = new_buffer;
    }
    
    /// Returns an iterator over the non-zero elements of the vector.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use eigen_rs::sparse_core::ambi_vector::{AmbiVector, Mode};
    /// 
    /// let mut vec: AmbiVector<f64, u32> = AmbiVector::new(5);
    /// vec.init_mode(Mode::Sparse);
    /// *vec.coeff_ref(2) = 3.0;
    /// *vec.coeff_ref(4) = 5.0;
    /// vec.set_zero();
    /// assert_eq!(vec.non_zeros(), 0);
    /// ```
    pub fn non_zeros(&self) -> usize {
        match self.mode {
            Mode::Sparse => self.ll_size.to_usize().unwrap(),
            Mode::Dense => (self.end - self.start).to_usize().unwrap(),
        }
    }
}

pub struct AmbiVectorIterator<'a, Scalar, StorageIndex>
where
    Scalar: Zero + Clone + PartialOrd + Float,
    StorageIndex: PrimInt + Unsigned,
{
    vector: &'a AmbiVector<Scalar, StorageIndex>,
    current_el: StorageIndex,
    epsilon: Scalar,
    cached_index: StorageIndex,
    cached_value: Scalar,
    is_dense: bool,
}

impl<'a, Scalar, StorageIndex> AmbiVectorIterator<'a, Scalar, StorageIndex>
where
    Scalar: Zero + Clone + PartialOrd + Float,
    StorageIndex: PrimInt + Unsigned,
{
    pub fn new(vector: &'a AmbiVector<Scalar, StorageIndex>, epsilon: Scalar) -> Self {
        let mut iter = Self {
            vector,
            current_el: StorageIndex::zero(),
            epsilon,
            cached_index: StorageIndex::from(vector.start)
                .unwrap()
                .saturating_sub(StorageIndex::one()),
            cached_value: Scalar::zero(),
            is_dense: vector.mode == Mode::Dense,
        };
        iter.find_next();
        iter
    }

    fn find_next(&mut self) {
        if self.is_dense {
            loop {
                if let Some(next_index) = self.cached_index.checked_add(&StorageIndex::one()) {
                    self.cached_index = next_index;
                    if self.cached_index >= self.vector.end {
                        self.cached_index = StorageIndex::max_value();
                        break;
                    }
                    unsafe {
                        let ptr =
                            self.vector.buffer.as_ptr().add(
                                self.cached_index.to_usize().unwrap() * mem::size_of::<Scalar>(),
                            ) as *const Scalar;
                        self.cached_value = (*ptr).clone();
                    }
                    if self.cached_value.abs() > self.epsilon {
                        break;
                    }
                } else {
                    self.cached_index = StorageIndex::max_value();
                    break;
                }
            }
        } else {
            let elements = self.vector.buffer.as_ptr() as *const ListEl<Scalar, StorageIndex>;
            loop {
                if self.current_el == StorageIndex::max_value() {
                    self.cached_index = StorageIndex::max_value();
                    break;
                }
                unsafe {
                    let el = &*elements.add(self.current_el.to_usize().unwrap());
                    self.cached_index = el.index;
                    self.cached_value = el.value.clone();
                    self.current_el = el.next;
                }
                if self.cached_value.abs() > self.epsilon {
                    break;
                }
            }
        }
    }
}

impl<'a, Scalar, StorageIndex> Iterator for AmbiVectorIterator<'a, Scalar, StorageIndex>
where
    Scalar: Zero + Clone + PartialOrd + Float,
    StorageIndex: PrimInt + Unsigned,
{
    type Item = (StorageIndex, Scalar);

    fn next(&mut self) -> Option<Self::Item> {
        if self.cached_index == StorageIndex::max_value() {
            None
        } else {
            let result = Some((self.cached_index, self.cached_value.clone()));
            self.find_next();
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_new_ambi_vector() {
        let vec: AmbiVector<f64, u32> = AmbiVector::new(10);
        assert_eq!(vec.size, 10);
        assert_eq!(vec.mode, Mode::Dense);
    }

    #[test]
    fn test_init_mode() {
        let mut vec: AmbiVector<f64, u32> = AmbiVector::new(10);
        vec.init(0.05);
        assert_eq!(vec.mode, Mode::Sparse);

        vec.init(0.2);
        assert_eq!(vec.mode, Mode::Dense);
    }

    #[test]
    fn test_resize() {
        let mut vec: AmbiVector<f64, u32> = AmbiVector::new(10);
        vec.resize(20);
        assert_eq!(vec.size, 20);
    }

    #[test]
    fn test_set_bounds() {
        let mut vec: AmbiVector<f64, u32> = AmbiVector::new(10);
        vec.set_bounds(2, 8);
        assert_eq!(vec.start, 2);
        assert_eq!(vec.end, 8);
    }

    #[test]
    fn test_set_zero_dense() {
        let mut vec: AmbiVector<f64, u32> = AmbiVector::new(5);
        vec.init_mode(Mode::Dense);
        for i in 0..5 {
            *vec.coeff_ref(i) = (i + 1) as f64;
        }
        vec.set_zero();
        for i in 0..5 {
            assert_relative_eq!(*vec.coeff_ref(i), 0.0);
        }
    }

    #[test]
    fn test_set_zero_sparse() {
        let mut vec: AmbiVector<f64, u32> = AmbiVector::new(5);
        vec.init_mode(Mode::Sparse);
        *vec.coeff_ref(2) = 3.0;
        *vec.coeff_ref(4) = 5.0;
        vec.set_zero();
        assert_eq!(vec.non_zeros(), 0);
    }

    #[test]
    fn test_coeff_ref_dense() {
        let mut vec: AmbiVector<f64, u32> = AmbiVector::new(5);
        vec.init_mode(Mode::Dense);
        *vec.coeff_ref(2) = 3.0;
        assert_relative_eq!(*vec.coeff_ref(2), 3.0);
    }

    #[test]
    fn test_coeff_ref_sparse() {
        let mut vec: AmbiVector<f64, u32> = AmbiVector::new(5);
        vec.init_mode(Mode::Sparse);
        *vec.coeff_ref(2) = 3.0;
        *vec.coeff_ref(4) = 5.0;
        assert_relative_eq!(*vec.coeff_ref(2), 3.0);
        assert_relative_eq!(*vec.coeff_ref(4), 5.0);
        assert_eq!(vec.non_zeros(), 2);
    }

    #[test]
    fn test_iterator_dense() {
        let mut vec: AmbiVector<f64, u32> = AmbiVector::new(5);
        vec.init_mode(Mode::Dense);
        *vec.coeff_ref(1) = 2.0;
        *vec.coeff_ref(3) = 4.0;

        let iter = AmbiVectorIterator::new(&vec, 1e-10);
        let result: Vec<(u32, f64)> = iter.collect();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (1, 2.0));
        assert_eq!(result[1], (3, 4.0));
    }

    #[test]
    fn test_iterator_sparse() {
        let mut vec: AmbiVector<f64, u32> = AmbiVector::new(5);
        vec.init_mode(Mode::Sparse);
        *vec.coeff_ref(1) = 2.0;
        *vec.coeff_ref(3) = 4.0;

        let iter = AmbiVectorIterator::new(&vec, 1e-10);
        let result: Vec<(u32, f64)> = iter.collect();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (1, 2.0));
        assert_eq!(result[1], (3, 4.0));
    }
}
