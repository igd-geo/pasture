use std::ops::Range;

use crate::{
    containers::{BufferSliceInterleaved, BufferSliceInterleavedMut, SliceBuffer, SliceBufferMut},
    layout::{PointAttributeDefinition, PointAttributeMember, PointLayout},
};

use super::{BorrowedBuffer, BorrowedMutBuffer, InterleavedBuffer, InterleavedBufferMut};

/// A point buffer that stores point data in interleaved memory layout in an externally borrowed memory resource.
/// This can be any type that is convertible to a `&[u8]`. If `T` also is convertible to a `&mut [u8]`, this buffer
/// also implements [`BorrowedMutBuffer`]
pub struct ExternalMemoryBuffer<T: AsRef<[u8]>> {
    external_memory: T,
    point_layout: PointLayout,
    length: usize,
}

impl<T: AsRef<[u8]>> ExternalMemoryBuffer<T> {
    /// Creates a new `ExternalMemoryBuffer` from the given `external_memory` resource and the given `PointLayout`
    pub fn new(external_memory: T, point_layout: PointLayout) -> Self {
        let length = match point_layout.size_of_point_entry() {
            0 => {
                assert_eq!(0, external_memory.as_ref().len());
                0
            }
            point_size => {
                assert!(external_memory.as_ref().len() % point_size as usize == 0);
                external_memory.as_ref().len() / point_size as usize
            }
        };
        Self {
            external_memory,
            point_layout,
            length,
        }
    }

    fn get_byte_range_for_point(&self, point_index: usize) -> Range<usize> {
        let size_of_point = self.point_layout.size_of_point_entry() as usize;
        (point_index * size_of_point)..((point_index + 1) * size_of_point)
    }

    fn get_byte_range_for_point_range(&self, point_range: Range<usize>) -> Range<usize> {
        let size_of_point = self.point_layout.size_of_point_entry() as usize;
        (point_range.start * size_of_point)..(point_range.end * size_of_point)
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

impl<'a, T: AsRef<[u8]>> BorrowedBuffer<'a> for ExternalMemoryBuffer<T>
where
    ExternalMemoryBuffer<T>: 'a,
{
    fn len(&self) -> usize {
        self.length
    }

    fn point_layout(&self) -> &PointLayout {
        &self.point_layout
    }

    fn get_point(&self, index: usize, data: &mut [u8]) {
        let point_bytes = &self.external_memory.as_ref()[self.get_byte_range_for_point(index)];
        data.copy_from_slice(point_bytes);
    }

    fn get_point_range(&self, range: Range<usize>, data: &mut [u8]) {
        let point_bytes =
            &self.external_memory.as_ref()[self.get_byte_range_for_point_range(range)];
        data.copy_from_slice(point_bytes);
    }

    unsafe fn get_attribute_unchecked(
        &self,
        attribute_member: &PointAttributeMember,
        index: usize,
        data: &mut [u8],
    ) {
        let attribute_bytes_range = self.get_byte_range_of_attribute(index, attribute_member);
        let attribute_bytes = &self.external_memory.as_ref()[attribute_bytes_range];
        data.copy_from_slice(attribute_bytes);
    }

    fn as_interleaved(&self) -> Option<&dyn InterleavedBuffer<'a>> {
        Some(self)
    }
}

impl<'a, T: AsMut<[u8]> + AsRef<[u8]>> BorrowedMutBuffer<'a> for ExternalMemoryBuffer<T>
where
    ExternalMemoryBuffer<T>: 'a,
{
    unsafe fn set_point(&mut self, index: usize, point_data: &[u8]) {
        let point_byte_range = self.get_byte_range_for_point(index);
        let point_memory = &mut self.external_memory.as_mut()[point_byte_range];
        point_memory.copy_from_slice(point_data);
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
        let attribute_bytes = &mut self.external_memory.as_mut()[attribute_byte_range];
        attribute_bytes.copy_from_slice(attribute_data);
    }

    fn swap(&mut self, from_index: usize, to_index: usize) {
        assert!(from_index < self.len());
        assert!(to_index < self.len());
        if from_index == to_index {
            return;
        }
        let size_of_point = self.point_layout.size_of_point_entry() as usize;
        // Is safe if neither `from_index` nor `to_index` is out of bounds, which is asserted
        unsafe {
            let from_ptr = self
                .external_memory
                .as_mut()
                .as_mut_ptr()
                .add(from_index * size_of_point);
            let to_ptr = self
                .external_memory
                .as_mut()
                .as_mut_ptr()
                .add(to_index * size_of_point);
            std::ptr::swap_nonoverlapping(from_ptr, to_ptr, size_of_point);
        }
    }

    unsafe fn set_point_range(&mut self, point_range: Range<usize>, point_data: &[u8]) {
        let point_byte_range = self.get_byte_range_for_point_range(point_range);
        let point_memory = &mut self.external_memory.as_mut()[point_byte_range];
        point_memory.copy_from_slice(point_data);
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
            let attribute_bytes = &mut self.external_memory.as_mut()[attribute_byte_range];
            attribute_bytes.copy_from_slice(src_slice);
        }
    }

    fn as_interleaved_mut(&mut self) -> Option<&mut dyn InterleavedBufferMut<'a>> {
        Some(self)
    }
}

impl<'a, T: AsRef<[u8]>> InterleavedBuffer<'a> for ExternalMemoryBuffer<T>
where
    ExternalMemoryBuffer<T>: 'a,
{
    fn get_point_ref<'b>(&'b self, index: usize) -> &'b [u8]
    where
        'a: 'b,
    {
        let memory = self.external_memory.as_ref();
        &memory[self.get_byte_range_for_point(index)]
    }

    fn get_point_range_ref<'b>(&'b self, range: Range<usize>) -> &'b [u8]
    where
        'a: 'b,
    {
        let memory = self.external_memory.as_ref();
        &memory[self.get_byte_range_for_point_range(range)]
    }
}

impl<'a, T: AsRef<[u8]> + AsMut<[u8]>> InterleavedBufferMut<'a> for ExternalMemoryBuffer<T>
where
    ExternalMemoryBuffer<T>: 'a,
{
    fn get_point_mut<'b>(&'b mut self, index: usize) -> &'b mut [u8]
    where
        'a: 'b,
    {
        let byte_range = self.get_byte_range_for_point(index);
        let memory = self.external_memory.as_mut();
        &mut memory[byte_range]
    }

    fn get_point_range_mut<'b>(&'b mut self, range: Range<usize>) -> &'b mut [u8]
    where
        'a: 'b,
    {
        let byte_range = self.get_byte_range_for_point_range(range);
        let memory = self.external_memory.as_mut();
        &mut memory[byte_range]
    }
}

impl<'a, T: AsRef<[u8]> + 'a> SliceBuffer<'a> for ExternalMemoryBuffer<T>
where
    Self: 'a,
{
    type SliceType = BufferSliceInterleaved<'a, Self>;

    fn slice(&'a self, range: Range<usize>) -> Self::SliceType {
        BufferSliceInterleaved::new(self, range)
    }
}

impl<'a, T: AsRef<[u8]> + AsMut<[u8]> + 'a> SliceBufferMut<'a> for ExternalMemoryBuffer<T> {
    type SliceTypeMut = BufferSliceInterleavedMut<'a, Self>;

    fn slice_mut(&'a mut self, range: Range<usize>) -> Self::SliceTypeMut {
        BufferSliceInterleavedMut::new(self, range)
    }
}

#[cfg(test)]
mod tests {
    use rand::{distributions::Distribution, thread_rng, Rng};

    use crate::{
        containers::{BorrowedBufferExt, BorrowedMutBufferExt, MakeBufferFromLayout, VectorBuffer},
        layout::PointType,
        test_utils::{
            compare_attributes, CustomPointTypeBig, CustomPointTypeSmall, DefaultPointDistribution,
        },
    };

    use super::*;

    fn test_external_memory_buffer_with_type<
        T: PointType + std::fmt::Debug + PartialEq + Copy + Clone,
    >()
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

        let mut memory_of_buffer: Vec<u8> =
            vec![0; COUNT * T::layout().size_of_point_entry() as usize];
        let mut test_data_as_buffer = ExternalMemoryBuffer::new(&mut memory_of_buffer, T::layout());
        for (idx, point) in test_data.iter().copied().enumerate() {
            test_data_as_buffer.view_mut().set_at(idx, point);
        }

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
    fn test_external_memory_buffer() {
        test_external_memory_buffer_with_type::<CustomPointTypeSmall>();
        test_external_memory_buffer_with_type::<CustomPointTypeBig>();
    }
}
