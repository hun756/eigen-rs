use std::marker::PhantomData;
use std::ptr;
use std::slice;

pub trait Scalar: Copy + Default {}
impl<T: Copy + Default> Scalar for T {}

pub trait StorageIndex: Copy + Default + Ord + Into<usize> {}
impl<T: Copy + Default + Ord + Into<usize>> StorageIndex for T {}

pub struct CompressedStorage<S: Scalar, I: StorageIndex> {
    values: *mut S,
    indices: *mut I,
    size: usize,
    allocated_size: usize,
    _marker: PhantomData<(S, I)>,
}

impl<S: Scalar, I: StorageIndex> CompressedStorage<S, I> {
    pub fn new() -> Self {
        CompressedStorage {
            values: ptr::null_mut(),
            indices: ptr::null_mut(),
            size: 0,
            allocated_size: 0,
            _marker: PhantomData,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let mut storage = Self::new();
        storage.reserve(capacity);
        storage
    }

    pub fn reserve(&mut self, additional: usize) {
        let new_capacity = self.size + additional;
        if new_capacity > self.allocated_size {
            self.reallocate(new_capacity);
        }
    }

    pub fn squeeze(&mut self) {
        if self.allocated_size > self.size {
            self.reallocate(self.size);
        }
    }

    pub fn resize(&mut self, new_size: usize, reserve_size_factor: f64) {
        if self.allocated_size < new_size {
            let realloc_size = (new_size as f64 * (1.0 + reserve_size_factor)) as usize;
            self.reallocate(realloc_size.max(new_size));
        }
        self.size = new_size;
    }

    pub fn append(&mut self, value: S, index: I) {
        let id = self.size;
        self.resize(self.size + 1, 1.0);
        unsafe {
            *self.values.add(id) = value;
            *self.indices.add(id) = index;
        }
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn capacity(&self) -> usize {
        self.allocated_size
    }

    pub fn clear(&mut self) {
        self.size = 0;
    }

    pub fn values(&self) -> &[S] {
        unsafe { slice::from_raw_parts(self.values, self.size) }
    }

    pub fn values_mut(&mut self) -> &mut [S] {
        unsafe { slice::from_raw_parts_mut(self.values, self.size) }
    }

    pub fn indices(&self) -> &[I] {
        unsafe { slice::from_raw_parts(self.indices, self.size) }
    }

    pub fn indices_mut(&mut self) -> &mut [I] {
        unsafe { slice::from_raw_parts_mut(self.indices, self.size) }
    }

    pub fn search_lower_index(&self, key: I) -> usize {
        self.indices()[..self.size].partition_point(|&x| x < key)
    }

    pub fn at(&self, key: I) -> Option<&S> {
        let id = self.search_lower_index(key);
        if id < self.size && unsafe { *self.indices.add(id) } == key {
            Some(unsafe { &*self.values.add(id) })
        } else {
            None
        }
    }

    pub fn at_mut(&mut self, key: I) -> Option<&mut S> {
        let id = self.search_lower_index(key);
        if id < self.size && unsafe { *self.indices.add(id) } == key {
            Some(unsafe { &mut *self.values.add(id) })
        } else {
            None
        }
    }

    pub fn at_with_insertion(&mut self, key: I, default_value: S) -> &mut S {
        let id = self.search_lower_index(key);
        if id >= self.size || unsafe { *self.indices.add(id) } != key {
            self.insert(id, key, default_value);
        }
        unsafe { &mut *self.values.add(id) }
    }

    fn insert(&mut self, id: usize, key: I, value: S) {
        if self.size == self.allocated_size {
            let new_capacity = (self.size + 1).next_power_of_two();
            self.reallocate(new_capacity);
        }

        unsafe {
            ptr::copy(self.values.add(id), self.values.add(id + 1), self.size - id);
            ptr::copy(
                self.indices.add(id),
                self.indices.add(id + 1),
                self.size - id,
            );
            *self.values.add(id) = value;
            *self.indices.add(id) = key;
        }

        self.size += 1;
    }

    fn reallocate(&mut self, new_capacity: usize) {
        let new_values = Self::alloc_values(new_capacity);
        let new_indices = Self::alloc_indices(new_capacity);

        if !self.values.is_null() {
            unsafe {
                ptr::copy_nonoverlapping(self.values, new_values, self.size);
                ptr::copy_nonoverlapping(self.indices, new_indices, self.size);
                Self::dealloc_values(self.values, self.allocated_size);
                Self::dealloc_indices(self.indices, self.allocated_size);
            }
        }

        self.values = new_values;
        self.indices = new_indices;
        self.allocated_size = new_capacity;
    }

    fn alloc_values(capacity: usize) -> *mut S {
        let layout = std::alloc::Layout::array::<S>(capacity).unwrap();
        unsafe { std::alloc::alloc(layout) as *mut S }
    }

    fn dealloc_values(ptr: *mut S, capacity: usize) {
        if !ptr.is_null() {
            let layout = std::alloc::Layout::array::<S>(capacity).unwrap();
            unsafe { std::alloc::dealloc(ptr as *mut u8, layout) };
        }
    }

    fn alloc_indices(capacity: usize) -> *mut I {
        let layout = std::alloc::Layout::array::<I>(capacity).unwrap();
        unsafe { std::alloc::alloc(layout) as *mut I }
    }

    fn dealloc_indices(ptr: *mut I, capacity: usize) {
        if !ptr.is_null() {
            let layout = std::alloc::Layout::array::<I>(capacity).unwrap();
            unsafe { std::alloc::dealloc(ptr as *mut u8, layout) };
        }
    }
}

impl<S: Scalar, I: StorageIndex> Drop for CompressedStorage<S, I> {
    fn drop(&mut self) {
        Self::dealloc_values(self.values, self.allocated_size);
        Self::dealloc_indices(self.indices, self.allocated_size);
    }
}

impl<S: Scalar, I: StorageIndex> Default for CompressedStorage<S, I> {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<S: Scalar, I: StorageIndex> Send for CompressedStorage<S, I> {}
unsafe impl<S: Scalar, I: StorageIndex> Sync for CompressedStorage<S, I> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressed_storage() {
        let mut storage = CompressedStorage::<f64, usize>::new();
        storage.append(1.0, 0);
        storage.append(2.0, 2);
        storage.append(3.0, 4);

        assert_eq!(storage.len(), 3);
        assert_eq!(storage.values(), &[1.0, 2.0, 3.0]);
        assert_eq!(storage.indices(), &[0, 2, 4]);

        assert_eq!(storage.at(2), Some(&2.0));
        assert_eq!(storage.at(3), None);

        *storage.at_with_insertion(3, 4.0) = 5.0;
        assert_eq!(storage.len(), 4);
        assert_eq!(storage.values(), &[1.0, 2.0, 5.0, 3.0]);
        assert_eq!(storage.indices(), &[0, 2, 3, 4]);
    }
}
