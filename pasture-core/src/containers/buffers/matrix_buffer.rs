use std::{
    iter::FromIterator,
    ops::{Index, IndexMut, Range},
};

use itertools::Itertools;

use crate::{
    layout::{PointAttributeMember, PointLayout, PointType},
    math::Alignable,
};

use super::{
    BorrowedBuffer, BorrowedMutBuffer, ColumnarBuffer, ColumnarBufferMut, MakeBufferFromLayout,
    OwningBuffer,
};

pub trait BinaryStorage: Index<Range<usize>, Output = [u8]> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait BinaryStorageMut: BinaryStorage + IndexMut<Range<usize>, Output = [u8]> {}

pub trait BinaryStorageOwned: BinaryStorageMut {
    fn resize(&mut self, num_bytes: usize);
    fn new(num_bytes: usize) -> Self;
}

impl BinaryStorage for Vec<u8> {
    fn len(&self) -> usize {
        self.len()
    }
}

impl BinaryStorageMut for Vec<u8> {}

impl BinaryStorageOwned for Vec<u8> {
    fn resize(&mut self, num_bytes: usize) {
        self.resize(num_bytes, 0);
    }

    fn new(num_bytes: usize) -> Self {
        vec![0; num_bytes]
    }
}

impl BinaryStorage for [u8] {
    fn len(&self) -> usize {
        self.len()
    }
}

impl BinaryStorageMut for [u8] {}

impl<const SIZE: usize> BinaryStorage for [u8; SIZE] {
    fn len(&self) -> usize {
        SIZE
    }
}

impl<const SIZE: usize> BinaryStorageMut for [u8; SIZE] {}

/// A columnar buffer that stores point data in a single 1D array similar to a dense matrix.
///
/// # Data layout
///
/// The point data is stored in row-major order in a single 1D array, like this:
/// ```md
/// a1 a2 a3 ... aN
/// b1 b2 b3 ... bN
/// ...
/// k1 k2 k3 ... kN
/// ```
/// flattened to:
/// ```md
/// [a1, a2, a3, ..., aN, b1, b2, b3, ..., bN, ..., k1, k2, k3, ..., kN]
/// ```
///
/// Where `a`, `b`, `c` etc. are the different point attributes, such as `POSITION_3D`, `INTENSITY` etc.
/// and `ax` refers to attribute `a` of point `x`. Compared to the `HashMapBuffer`, the memory layout of
/// this buffer is more efficient for read/write operations, requiring no hash lookups, but resizing is
/// much more costly, as the whole buffer has to be reallocated and restructured upon every insert/remove.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatrixBufferBase<S: BinaryStorage> {
    storage: S,
    point_layout: PointLayout,
    point_count: usize,
}

impl<S: BinaryStorage> MatrixBufferBase<S> {
    fn offset_to_attribute(
        point_count: usize,
        point_index: usize,
        attribute: &PointAttributeMember,
    ) -> usize {
        let required_alignment = attribute.datatype().min_alignment() as usize;
        (point_count * attribute.offset() as usize).align_to(required_alignment)
            + (point_index * attribute.size() as usize)
    }
}

impl<S: BinaryStorage> BorrowedBuffer<'_> for MatrixBufferBase<S> {
    fn len(&self) -> usize {
        self.point_count
    }

    fn point_layout(&self) -> &PointLayout {
        &self.point_layout
    }

    fn get_point(&self, index: usize, data: &mut [u8]) {
        for attribute in self.point_layout.attributes() {
            let offset = attribute.offset() as usize;
            let size = attribute.size() as usize;
            let dst_slice = &mut data[offset..(offset + size)];
            // Safe because attribute comes from this buffer's point layout
            unsafe {
                self.get_attribute_unchecked(attribute, index, dst_slice);
            }
        }
    }

    fn get_point_range(&self, range: Range<usize>, data: &mut [u8]) {
        let point_size = self.point_layout.size_of_point_entry() as usize;
        for attribute in self.point_layout.attributes() {
            let offset = attribute.offset() as usize;
            let size = attribute.size() as usize;
            for point_index in range.clone() {
                let dst_slice = &mut data[(point_index * point_size)..][offset..(offset + size)];
                // Safe because attribute comes from this buffer's point layout
                unsafe {
                    self.get_attribute_unchecked(attribute, point_index, dst_slice);
                }
            }
        }
    }

    unsafe fn get_attribute_unchecked(
        &self,
        attribute_member: &crate::layout::PointAttributeMember,
        index: usize,
        data: &mut [u8],
    ) {
        let attribute_size = attribute_member.size() as usize;
        let offset = Self::offset_to_attribute(self.point_count, index, attribute_member);
        data.copy_from_slice(&self.storage[offset..(offset + attribute_size)])
    }
}

impl<S: BinaryStorageMut> BorrowedMutBuffer<'_> for MatrixBufferBase<S> {
    unsafe fn set_point(&mut self, index: usize, point_data: &[u8]) {
        for attribute in self.point_layout.attributes() {
            let attribute_size = attribute.size() as usize;
            let attribute_offset = attribute.offset() as usize;
            let dst_offset = Self::offset_to_attribute(self.point_count, index, attribute);
            let dst_slice = &mut self.storage[dst_offset..(dst_offset + attribute_size)];
            let src_slice = &point_data[attribute_offset..(attribute_offset + attribute_size)];
            dst_slice.copy_from_slice(src_slice);
        }
    }

    unsafe fn set_point_range(&mut self, point_range: Range<usize>, point_data: &[u8]) {
        let point_size = self.point_layout.size_of_point_entry() as usize;
        let base_index = point_range.start;
        for attribute in self.point_layout.attributes() {
            let attribute_size = attribute.size() as usize;
            let attribute_offset = attribute.offset() as usize;
            for index in point_range.clone() {
                let src_offset = ((index - base_index) * point_size) + attribute_offset;
                let dst_offset = Self::offset_to_attribute(self.point_count, index, attribute);
                let src_slice = &point_data[src_offset..(src_offset + attribute_size)];
                let dst_slice = &mut self.storage[dst_offset..(dst_offset + attribute_size)];
                dst_slice.copy_from_slice(src_slice);
            }
        }
    }

    unsafe fn set_attribute(
        &mut self,
        attribute: &crate::layout::PointAttributeDefinition,
        index: usize,
        attribute_data: &[u8],
    ) {
        self.get_attribute_mut(attribute, index)
            .copy_from_slice(attribute_data)
    }

    unsafe fn set_attribute_range(
        &mut self,
        attribute: &crate::layout::PointAttributeDefinition,
        point_range: Range<usize>,
        attribute_data: &[u8],
    ) {
        self.get_attribute_range_mut(attribute, point_range)
            .copy_from_slice(attribute_data)
    }

    fn swap(&mut self, from_index: usize, to_index: usize) {
        if from_index == to_index {
            return;
        }
        let len = self.storage.len();
        for attribute in self.point_layout.attributes() {
            unsafe {
                let base_ptr = self.storage[0..len].as_mut_ptr();
                let from_ptr = base_ptr.add(Self::offset_to_attribute(
                    self.point_count,
                    from_index,
                    attribute,
                ));
                let to_ptr = base_ptr.add(Self::offset_to_attribute(
                    self.point_count,
                    to_index,
                    attribute,
                ));
                std::ptr::copy_nonoverlapping(from_ptr, to_ptr, attribute.size() as usize);
            }
        }
    }
}

impl<S: BinaryStorageOwned> OwningBuffer<'_> for MatrixBufferBase<S> {
    unsafe fn push_points(&mut self, point_bytes: &[u8]) {
        let old_count = self.point_count;
        let new_count =
            old_count + (point_bytes.len() / self.point_layout.size_of_point_entry() as usize);
        self.resize(new_count);
        self.set_point_range(old_count..new_count, point_bytes);
    }

    fn resize(&mut self, count: usize) {
        let alignment = self
            .point_layout
            .attributes()
            .map(|a| a.datatype().min_alignment() as usize)
            .max()
            .unwrap_or(1);
        let new_bytes =
            (count * self.point_layout.size_of_point_entry() as usize).align_to(alignment);
        let mut new_storage = S::new(new_bytes);

        // Copy all attribute blocks to their corresponding positions in the new storage
        let old_count = self.point_count;
        let min_count = old_count.min(count);
        for attribute in self.point_layout.attributes() {
            let old_range =
                self.get_attribute_range_ref(attribute.attribute_definition(), 0..min_count);

            let new_attribute_offset_start = Self::offset_to_attribute(count, 0, attribute);
            let new_attribute_offset_end = Self::offset_to_attribute(count, min_count, attribute);
            let new_range = &mut new_storage[new_attribute_offset_start..new_attribute_offset_end];
            new_range.copy_from_slice(old_range);
        }

        self.storage = new_storage;
        self.point_count = count;
    }

    fn clear(&mut self) {
        self.point_count = 0;
        self.storage.resize(0);
    }
}

impl<'a, S: BinaryStorage> ColumnarBuffer<'a> for MatrixBufferBase<S> {
    fn get_attribute_ref<'b>(
        &'b self,
        attribute: &crate::layout::PointAttributeDefinition,
        index: usize,
    ) -> &'b [u8]
    where
        'a: 'b,
    {
        let attribute_member = self
            .point_layout
            .get_attribute(attribute)
            .expect("Attribute not found in PointLayout");
        let attribute_size = attribute.size() as usize;
        let offset = Self::offset_to_attribute(self.point_count, index, attribute_member);
        &self.storage[offset..(offset + attribute_size)]
    }

    fn get_attribute_range_ref<'b>(
        &'b self,
        attribute: &crate::layout::PointAttributeDefinition,
        range: Range<usize>,
    ) -> &'b [u8]
    where
        'a: 'b,
    {
        let attribute_member = self
            .point_layout
            .get_attribute(attribute)
            .expect("Attribute not found in PointLayout");
        let offset_start =
            Self::offset_to_attribute(self.point_count, range.start, attribute_member);
        let offset_end = Self::offset_to_attribute(self.point_count, range.end, attribute_member);
        &self.storage[offset_start..offset_end]
    }
}

impl<'a, S: BinaryStorageMut> ColumnarBufferMut<'a> for MatrixBufferBase<S> {
    fn get_attribute_mut<'b>(
        &'b mut self,
        attribute: &crate::layout::PointAttributeDefinition,
        index: usize,
    ) -> &'b mut [u8]
    where
        'a: 'b,
    {
        let attribute_member = self
            .point_layout
            .get_attribute(attribute)
            .expect("Attribute not found in PointLayout");
        let attribute_size = attribute.size() as usize;
        let offset =
            (self.point_count * attribute_member.offset() as usize) + (index * attribute_size);
        &mut self.storage[offset..(offset + attribute_size)]
    }

    fn get_attribute_range_mut<'b>(
        &'b mut self,
        attribute: &crate::layout::PointAttributeDefinition,
        range: Range<usize>,
    ) -> &'b mut [u8]
    where
        'a: 'b,
    {
        let attribute_member = self
            .point_layout
            .get_attribute(attribute)
            .expect("Attribute not found in PointLayout");
        let attribute_size = attribute.size() as usize;
        let offset_start = (self.point_count * attribute_member.offset() as usize)
            + (range.start * attribute_size);
        let offset_end =
            (self.point_count * attribute_member.offset() as usize) + (range.end * attribute_size);
        &mut self.storage[offset_start..offset_end]
    }
}

impl MakeBufferFromLayout<'_> for MatrixBufferBase<Vec<u8>> {
    fn new_from_layout(point_layout: PointLayout) -> Self {
        Self {
            storage: vec![],
            point_layout,
            point_count: 0,
        }
    }
}

impl<const NUM_BYTES: usize> MakeBufferFromLayout<'_> for MatrixBufferStatic<NUM_BYTES> {
    fn new_from_layout(point_layout: PointLayout) -> Self {
        let aligned_size = (point_layout.size_of_point_entry() as usize)
            .align_to(point_layout.min_alignment_for_rust_layout() as usize);
        let max_count = NUM_BYTES / aligned_size;
        Self {
            point_count: max_count,
            point_layout,
            storage: [0; NUM_BYTES],
        }
    }
}

impl<'a, T: PointType, S: BinaryStorageOwned> FromIterator<T> for MatrixBufferBase<S>
where
    Self: MakeBufferFromLayout<'a>,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let point_layout = T::layout();
        let mut buffer = Self::new_from_layout(point_layout);
        let iter = iter.into_iter();
        let (_, maybe_known_length) = iter.size_hint();
        if let Some(known_length) = maybe_known_length {
            buffer.resize(known_length);
            iter.enumerate().for_each(|(index, point)| {
                let point_bytes = bytemuck::bytes_of(&point);
                // Safe because we created `buffer` from `T::layout()`, so we know the layouts match
                unsafe {
                    buffer.set_point(index, point_bytes);
                }
            });
            buffer
        } else {
            // Since resizing is so expensive for MatrixBuffer, it is faster to first collect the points into
            // a vec so that we know the actual size
            let points = iter.collect_vec();
            buffer.resize(points.len());
            points.into_iter().enumerate().for_each(|(index, point)| {
                let point_bytes = bytemuck::bytes_of(&point);
                // Safe because we created `buffer` from `T::layout()`, so we know the layouts match
                unsafe {
                    buffer.set_point(index, point_bytes);
                }
            });
            buffer
        }
    }
}

pub type MatrixBuffer = MatrixBufferBase<Vec<u8>>;
pub type MatrixBufferBorrowed<'a> = MatrixBufferBase<&'a [u8]>;
pub type MatrixBufferBorrowedMut<'a> = MatrixBufferBase<&'a mut [u8]>;
pub type MatrixBufferStatic<const NUM_BYTES: usize> = MatrixBufferBase<[u8; NUM_BYTES]>;

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::{distributions::Distribution, thread_rng, Rng};

    use crate::{
        containers::{BorrowedBufferExt, BorrowedMutBufferExt, SliceBuffer, VectorBuffer},
        layout::PointType,
        test_utils::{
            compare_attributes, CustomPointTypeBig, CustomPointTypeSmall, DefaultPointDistribution,
        },
    };

    use super::*;

    fn test_matrix_buffer_with_point_type<T: PointType + PartialEq + std::fmt::Debug>()
    where
        DefaultPointDistribution: Distribution<T>,
    {
        {
            let default_buffer = MatrixBuffer::new_from_layout(T::layout());
            assert_eq!(0, default_buffer.len());
            assert!(default_buffer.is_empty());
            assert_eq!(T::layout(), *default_buffer.point_layout());
        }

        const COUNT: usize = 15;
        let test_points = thread_rng()
            .sample_iter::<T, _>(DefaultPointDistribution)
            .take(COUNT)
            .collect_vec();
        let test_points_vec_buffer: VectorBuffer = test_points.iter().copied().collect();

        {
            let buffer: MatrixBuffer = test_points.iter().copied().collect();
            assert_eq!(COUNT, buffer.len());
            let points_from_buffer = buffer.view::<T>().into_iter().collect_vec();
            assert_eq!(test_points, points_from_buffer);

            for attribute in buffer.point_layout().attributes() {
                compare_attributes(
                    &buffer,
                    attribute.attribute_definition(),
                    &test_points_vec_buffer,
                );
            }
        }

        {
            let mut buffer: MatrixBuffer = test_points.iter().copied().collect();
            buffer.resize(4);
            assert_eq!(4, buffer.len());
            let points_from_buffer = buffer.view::<T>().into_iter().collect_vec();
            assert_eq!(&test_points[0..4], points_from_buffer);

            for attribute in buffer.point_layout().attributes() {
                compare_attributes(
                    &buffer,
                    attribute.attribute_definition(),
                    &test_points_vec_buffer.slice(0..4),
                );
            }
        }

        {
            let mut buffer = MatrixBuffer::new_from_layout(T::layout());
            let mut view = buffer.view_mut::<T>();
            for test_point in &test_points {
                view.push_point(*test_point);
            }

            assert_eq!(COUNT, buffer.len());
            let points_from_buffer = buffer.view::<T>().into_iter().collect_vec();
            assert_eq!(test_points, points_from_buffer);

            for attribute in buffer.point_layout().attributes() {
                compare_attributes(
                    &buffer,
                    attribute.attribute_definition(),
                    &test_points_vec_buffer,
                );
            }
        }
    }

    #[test]
    fn test_matrix_buffer() {
        test_matrix_buffer_with_point_type::<CustomPointTypeSmall>();
        test_matrix_buffer_with_point_type::<CustomPointTypeBig>();
    }
}
