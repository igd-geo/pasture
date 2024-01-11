use std::{iter::FromIterator, ops::Range};

use crate::{
    containers::{BufferSliceInterleaved, BufferSliceInterleavedMut, SliceBuffer, SliceBufferMut},
    layout::{PointAttributeDefinition, PointAttributeMember, PointLayout, PointType},
};

use super::{
    BorrowedBuffer, BorrowedMutBuffer, InterleavedBuffer, InterleavedBufferMut,
    MakeBufferFromLayout, OwningBuffer,
};

/// A point buffer that uses a `Vec<u8>` as its underlying storage. It stores point data in interleaved memory
/// layout and generally behaves like an untyped vector.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VectorBuffer {
    storage: Vec<u8>,
    point_layout: PointLayout,
    length: usize,
}

impl VectorBuffer {
    /// Creates a new `VectorBuffer` with the given `capacity` and `point_layout`. This preallocates enough memory
    /// to store at least `capacity` points
    pub fn with_capacity(capacity: usize, point_layout: PointLayout) -> Self {
        let required_bytes = capacity * point_layout.size_of_point_entry() as usize;
        Self {
            point_layout,
            storage: Vec::with_capacity(required_bytes),
            length: 0,
        }
    }

    fn get_byte_range_of_point(&self, point_index: usize) -> Range<usize> {
        let size_of_point = self.point_layout.size_of_point_entry() as usize;
        (point_index * size_of_point)..((point_index + 1) * size_of_point)
    }

    fn get_byte_range_of_points(&self, points_range: Range<usize>) -> Range<usize> {
        let size_of_point = self.point_layout.size_of_point_entry() as usize;
        (points_range.start * size_of_point)..(points_range.end * size_of_point)
    }

    fn get_byte_range_of_attribute(
        &self,
        point_index: usize,
        attribute: &PointAttributeMember,
    ) -> Range<usize> {
        let start_byte = (point_index * self.point_layout.size_of_point_entry() as usize)
            + attribute.offset() as usize;
        let end_byte = start_byte + attribute.size() as usize;
        start_byte..end_byte
    }
}

impl<'a> MakeBufferFromLayout<'a> for VectorBuffer {
    fn new_from_layout(point_layout: PointLayout) -> Self {
        Self {
            point_layout,
            storage: Default::default(),
            length: 0,
        }
    }
}

impl<'a> BorrowedBuffer<'a> for VectorBuffer
where
    VectorBuffer: 'a,
{
    fn len(&self) -> usize {
        self.length
    }

    fn point_layout(&self) -> &PointLayout {
        &self.point_layout
    }

    fn get_point(&self, index: usize, data: &mut [u8]) {
        let point_ref = self.get_point_ref(index);
        data.copy_from_slice(point_ref);
    }

    fn get_point_range(&self, range: Range<usize>, data: &mut [u8]) {
        let points_ref = self.get_point_range_ref(range);
        data.copy_from_slice(points_ref);
    }

    unsafe fn get_attribute_unchecked(
        &self,
        attribute_member: &PointAttributeMember,
        index: usize,
        data: &mut [u8],
    ) {
        let byte_range = self.get_byte_range_of_attribute(index, attribute_member);
        data.copy_from_slice(&self.storage[byte_range]);
    }

    fn as_interleaved(&self) -> Option<&dyn InterleavedBuffer<'a>> {
        Some(self)
    }
}

impl<'a> BorrowedMutBuffer<'a> for VectorBuffer
where
    VectorBuffer: 'a,
{
    unsafe fn set_point(&mut self, index: usize, point_data: &[u8]) {
        let point_bytes = self.get_point_mut(index);
        point_bytes.copy_from_slice(point_data);
    }

    unsafe fn set_attribute(
        &mut self,
        attribute: &PointAttributeDefinition,
        index: usize,
        attribute_data: &[u8],
    ) {
        let attribute_member = self
            .point_layout
            .get_attribute(attribute)
            .expect("Attribute not found in PointLayout of this buffer");
        let attribute_byte_range = self.get_byte_range_of_attribute(index, attribute_member);
        let attribute_bytes = &mut self.storage[attribute_byte_range];
        attribute_bytes.copy_from_slice(attribute_data);
    }

    fn swap(&mut self, from_index: usize, to_index: usize) {
        assert!(from_index < self.len());
        assert!(to_index < self.len());
        if from_index == to_index {
            return;
        }
        let size_of_point = self.point_layout.size_of_point_entry() as usize;
        // Is safe as long as 'from_index' and 'to_index' are not out of bounds, which is asserted
        unsafe {
            let from_ptr = self.storage.as_mut_ptr().add(from_index * size_of_point);
            let to_ptr = self.storage.as_mut_ptr().add(to_index * size_of_point);
            std::ptr::swap_nonoverlapping(from_ptr, to_ptr, size_of_point);
        }
    }

    unsafe fn set_point_range(&mut self, point_range: Range<usize>, point_data: &[u8]) {
        let point_bytes = self.get_point_range_mut(point_range);
        point_bytes.copy_from_slice(point_data);
    }

    unsafe fn set_attribute_range(
        &mut self,
        attribute: &PointAttributeDefinition,
        point_range: Range<usize>,
        attribute_data: &[u8],
    ) {
        let attribute_member = self
            .point_layout
            .get_attribute(attribute)
            .expect("Attribute not found in PointLayout of this buffer");
        let attribute_size = attribute_member.size() as usize;
        let first_point = point_range.start;
        for point_index in point_range {
            let zero_based_index = point_index - first_point;
            let src_slice = &attribute_data
                [(zero_based_index * attribute_size)..((zero_based_index + 1) * attribute_size)];
            let attribute_byte_range =
                self.get_byte_range_of_attribute(point_index, attribute_member);
            let attribute_bytes = &mut self.storage[attribute_byte_range];
            attribute_bytes.copy_from_slice(src_slice);
        }
    }

    fn as_interleaved_mut(&mut self) -> Option<&mut dyn InterleavedBufferMut<'a>> {
        Some(self)
    }
}

impl<'a> OwningBuffer<'a> for VectorBuffer
where
    VectorBuffer: 'a,
{
    unsafe fn push_points(&mut self, point_bytes: &[u8]) {
        let size_of_point = self.point_layout.size_of_point_entry() as usize;
        if size_of_point == 0 {
            assert_eq!(0, point_bytes.len());
        } else {
            assert_eq!(point_bytes.len() % size_of_point, 0);
            self.storage.extend_from_slice(point_bytes);
            self.length += point_bytes.len() / size_of_point;
        }
    }

    fn resize(&mut self, count: usize) {
        let size_of_point = self.point_layout.size_of_point_entry() as usize;
        self.storage.resize(count * size_of_point, 0);
        self.length = count;
    }

    fn clear(&mut self) {
        self.storage.clear();
        self.length = 0;
    }
}

impl<'a> InterleavedBuffer<'a> for VectorBuffer
where
    VectorBuffer: 'a,
{
    fn get_point_ref<'b>(&'b self, index: usize) -> &'b [u8]
    where
        'a: 'b,
    {
        &self.storage[self.get_byte_range_of_point(index)]
    }

    fn get_point_range_ref<'b>(&'b self, range: Range<usize>) -> &'b [u8]
    where
        'a: 'b,
    {
        &self.storage[self.get_byte_range_of_points(range)]
    }
}

impl<'a> InterleavedBufferMut<'a> for VectorBuffer
where
    VectorBuffer: 'a,
{
    fn get_point_mut<'b>(&'b mut self, index: usize) -> &'b mut [u8]
    where
        'a: 'b,
    {
        let byte_range = self.get_byte_range_of_point(index);
        &mut self.storage[byte_range]
    }

    fn get_point_range_mut<'b>(&'b mut self, range: Range<usize>) -> &'b mut [u8]
    where
        'a: 'b,
    {
        let byte_range = self.get_byte_range_of_points(range);
        &mut self.storage[byte_range]
    }
}

impl<'a> SliceBuffer<'a> for VectorBuffer
where
    Self: 'a,
{
    type SliceType = BufferSliceInterleaved<'a, Self>;

    fn slice(&'a self, range: Range<usize>) -> Self::SliceType {
        BufferSliceInterleaved::new(self, range)
    }
}

impl<'a> SliceBufferMut<'a> for VectorBuffer
where
    Self: 'a,
{
    type SliceTypeMut = BufferSliceInterleavedMut<'a, Self>;

    fn slice_mut(&'a mut self, range: Range<usize>) -> Self::SliceTypeMut {
        BufferSliceInterleavedMut::new(self, range)
    }
}

impl<T: PointType> FromIterator<T> for VectorBuffer {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let point_layout = T::layout();
        let iter = iter.into_iter();
        let (_, maybe_known_length) = iter.size_hint();
        if let Some(known_length) = maybe_known_length {
            let num_bytes = known_length * point_layout.size_of_point_entry() as usize;
            let storage = vec![0; num_bytes];
            let mut buffer = Self {
                point_layout,
                storage,
                length: known_length,
            };
            // Overwrite the preallocated memory of the buffer with the points in the iterator:
            iter.enumerate().for_each(|(index, point)| {
                let point_bytes = bytemuck::bytes_of(&point);
                // Safe because we created `buffer` from `T::layout()`, so we know the layouts match
                unsafe {
                    buffer.set_point(index, point_bytes);
                }
            });
            buffer
        } else {
            let mut buffer = Self {
                point_layout,
                storage: Default::default(),
                length: 0,
            };
            iter.for_each(|point| {
                let point_bytes = bytemuck::bytes_of(&point);
                // Safe because we know that `buffer` has the same layout as `T`
                unsafe {
                    buffer.push_points(point_bytes);
                }
            });
            buffer
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{distributions::Distribution, thread_rng, Rng};

    use crate::{
        containers::{BorrowedBufferExt, BorrowedMutBufferExt},
        test_utils::{
            compare_attributes, CustomPointTypeBig, CustomPointTypeSmall, DefaultPointDistribution,
        },
    };

    use super::*;

    fn test_vector_buffer_with_type<T: PointType + std::fmt::Debug + PartialEq + Copy + Clone>()
    where
        DefaultPointDistribution: Distribution<T>,
    {
        const COUNT: usize = 16;
        let test_data: Vec<T> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();
        let overwrite_data: Vec<T> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();

        let test_data_as_buffer = test_data.iter().copied().collect::<VectorBuffer>();

        {
            let mut buffer = VectorBuffer::new_from_layout(T::layout());
            assert_eq!(0, buffer.len());
            assert_eq!(T::layout(), *buffer.point_layout());
            assert_eq!(0, buffer.view::<T>().into_iter().count());

            for (idx, point) in test_data.iter().enumerate() {
                buffer.view_mut().push_point(*point);
                assert_eq!(idx + 1, buffer.len());
                assert_eq!(*point, buffer.view().at(idx));
            }

            let mut collected_points = buffer.view().into_iter().collect::<Vec<_>>();
            assert_eq!(test_data, collected_points);

            let collected_points_by_ref = buffer.view().iter().copied().collect::<Vec<_>>();
            assert_eq!(test_data, collected_points_by_ref);

            for attribute in buffer.point_layout().attributes() {
                compare_attributes(
                    &buffer,
                    attribute.attribute_definition(),
                    &test_data_as_buffer,
                );
            }

            let slice = buffer.slice(1..2);
            assert_eq!(test_data[1], slice.view().at(0));

            for (idx, point) in overwrite_data.iter().enumerate() {
                *buffer.view_mut().at_mut(idx) = *point;
            }
            collected_points = buffer.view().iter().copied().collect();
            assert_eq!(overwrite_data, collected_points);
        }
    }

    #[test]
    fn test_vector_buffer() {
        test_vector_buffer_with_type::<CustomPointTypeSmall>();
        test_vector_buffer_with_type::<CustomPointTypeBig>();
    }
}
