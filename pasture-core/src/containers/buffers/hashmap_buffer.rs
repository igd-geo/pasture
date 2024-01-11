use std::{collections::HashMap, iter::FromIterator, ops::Range};

use crate::{
    containers::{BufferSliceColumnar, BufferSliceColumnarMut, SliceBuffer, SliceBufferMut},
    layout::{
        PointAttributeDefinition, PointAttributeMember, PointLayout, PointType, PrimitiveType,
    },
};

use super::{
    BorrowedBuffer, BorrowedMutBuffer, ColumnarBuffer, ColumnarBufferMut, MakeBufferFromLayout,
    OwningBuffer,
};

/// Helper struct to push point data into a `HashMapBuffer` attribute by attribute. This allows constructing a point buffer
/// from multiple ranges of attribute data, since regular point buffers do not allow pushing just a single attribute into
/// the buffer, as buffers always have to store complete points (even with columnar memory layout)
pub struct HashMapBufferAttributePusher<'a> {
    attributes_storage: HashMap<PointAttributeDefinition, Vec<u8>>,
    num_new_points: Option<usize>,
    buffer: &'a mut HashMapBuffer,
}

impl<'a> HashMapBufferAttributePusher<'a> {
    pub(crate) fn new(buffer: &'a mut HashMapBuffer) -> Self {
        let attributes_storage = buffer
            .point_layout()
            .attributes()
            .map(|attribute| (attribute.attribute_definition().clone(), vec![]))
            .collect();
        Self {
            attributes_storage,
            num_new_points: None,
            buffer,
        }
    }

    /// Push a range of values for the given `attribute` into the underlying buffer. The first range of values that
    /// is pushed in this way determines the expected number of points that will be added to the buffer. Consecutive
    /// calls to `push_attribute_range` will assert that `data.len()` matches the expected count.
    ///
    /// # Panics
    ///
    /// If `attribute` is not part of the `PointLayout` of the underlying buffer.<br>
    /// If `T::data_type()` does not match `attribute.datatype()`.<br>
    /// If this is not the first call to `push_attribute_range`, and `data.len()` does not match the length of the
    /// data that was passed to the first invocation of `push_attribute_range`
    pub fn push_attribute_range<T: PrimitiveType>(
        &mut self,
        attribute: &PointAttributeDefinition,
        data: &[T],
    ) {
        assert_eq!(T::data_type(), attribute.datatype());
        let storage = self
            .attributes_storage
            .get_mut(attribute)
            .expect("Attribute not found in PointLayout of this buffer");
        if let Some(point_count) = self.num_new_points {
            assert_eq!(point_count, data.len());
        } else {
            self.num_new_points = Some(data.len());
        }
        storage.extend_from_slice(bytemuck::cast_slice(data));
    }

    /// Commit all pushed data into the underlying buffer. This function checks that there is the correct amount
    /// of data for all expected attributes in the `PointLayout` of the underlying buffer and will panic otherwise
    ///
    /// # Panics
    ///
    /// If there is missing data for at least one of the attributes in the `PointLayout` of the underlying buffer,
    /// i.e. if `push_attribute_range` was not called for at least one of these attributes.
    pub fn done(self) {
        let num_new_points = self.num_new_points.unwrap_or(0);
        if num_new_points == 0 {
            return;
        }

        // Check that all attributes are complete! We don't have to check the exact size of the vectors,
        // as this is checked in `push_attribute_range`, it is sufficient to verify that no vector is empty
        assert!(self
            .attributes_storage
            .values()
            .all(|vector| !vector.is_empty()));

        for (attribute, mut data) in self.attributes_storage {
            // Can safely unwrap, because self.attributes_storage was initialized from the `PointLayout` of the buffer!
            let buffer_storage = self.buffer.attributes_storage.get_mut(&attribute).unwrap();
            buffer_storage.append(&mut data);
        }

        self.buffer.length += num_new_points;
    }
}

/// A point buffer that stores point data in columnar memory layout, using a `HashMap<PointAttributeDefinition, Vec<u8>>` as
/// its underlying storage
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashMapBuffer {
    attributes_storage: HashMap<PointAttributeDefinition, Vec<u8>>,
    point_layout: PointLayout,
    length: usize,
}

impl HashMapBuffer {
    /// Creates a new `HashMapBuffer` with the given `capacity` and `point_layout`. It preallocates enough memory to store
    /// at least `capacity` points
    pub fn with_capacity(capacity: usize, point_layout: PointLayout) -> Self {
        let attributes_storage = point_layout
            .attributes()
            .map(|attribute| {
                let bytes_for_attribute = capacity * attribute.size() as usize;
                (
                    attribute.attribute_definition().clone(),
                    Vec::with_capacity(bytes_for_attribute),
                )
            })
            .collect();
        Self {
            attributes_storage,
            point_layout,
            length: 0,
        }
    }

    /// Create a new helper object through which ranges of attribute data can be pushed into this buffer
    pub fn begin_push_attributes(&mut self) -> HashMapBufferAttributePusher<'_> {
        HashMapBufferAttributePusher::new(self)
    }

    /// Like `Iterator::filter`, but filters into a point buffer of type `B`
    pub fn filter<
        B: for<'a> OwningBuffer<'a> + for<'a> MakeBufferFromLayout<'a>,
        F: Fn(usize) -> bool,
    >(
        &self,
        predicate: F,
    ) -> B {
        let num_matches = (0..self.len()).filter(|idx| predicate(*idx)).count();
        let mut filtered_points = B::new_from_layout(self.point_layout.clone());
        filtered_points.resize(num_matches);
        self.filter_into(&mut filtered_points, predicate, Some(num_matches));
        filtered_points
    }

    /// Like `filter`, but writes the filtered points into the given `buffer`
    ///
    /// # panics
    ///
    /// If `buffer.len()` is less than the number of matching points according to `predicate`
    /// If the `PointLayout` of `buffer` does not match the `PointLayout` of `self`
    pub fn filter_into<B: for<'a> BorrowedMutBuffer<'a>, F: Fn(usize) -> bool>(
        &self,
        buffer: &mut B,
        predicate: F,
        num_matches_hint: Option<usize>,
    ) {
        if buffer.point_layout() != self.point_layout() {
            panic!("PointLayouts must match");
        }
        let num_matches = num_matches_hint
            .unwrap_or_else(|| (0..self.len()).filter(|idx| predicate(*idx)).count());
        if buffer.len() < num_matches {
            panic!("buffer.len() must be at least as large as the number of predicate matches");
        }
        if let Some(columnar_buffer) = buffer.as_columnar_mut() {
            for attribute in self.point_layout.attributes() {
                let src_attribute_data =
                    self.get_attribute_range_ref(attribute.attribute_definition(), 0..self.len());
                let dst_attribute_data = columnar_buffer
                    .get_attribute_range_mut(attribute.attribute_definition(), 0..num_matches);
                let stride = attribute.size() as usize;
                for (dst_index, src_index) in
                    (0..self.len()).filter(|idx| predicate(*idx)).enumerate()
                {
                    dst_attribute_data[(dst_index * stride)..((dst_index + 1) * stride)]
                        .copy_from_slice(
                            &src_attribute_data[(src_index * stride)..((src_index + 1) * stride)],
                        );
                }
            }
        } else if let Some(interleaved_buffer) = buffer.as_interleaved_mut() {
            let dst_data = interleaved_buffer.get_point_range_mut(0..num_matches);
            for attribute in self.point_layout.attributes() {
                let src_attribute_data =
                    self.get_attribute_range_ref(attribute.attribute_definition(), 0..self.len());
                let src_stride = attribute.size() as usize;
                let dst_offset = attribute.offset() as usize;
                let dst_stride = self.point_layout.size_of_point_entry() as usize;
                for (dst_index, src_index) in
                    (0..self.len()).filter(|idx| predicate(*idx)).enumerate()
                {
                    let dst_attribute_start = dst_offset + (dst_index * dst_stride);
                    let dst_point_range = dst_attribute_start..(dst_attribute_start + src_stride);
                    dst_data[dst_point_range].copy_from_slice(
                        &src_attribute_data
                            [(src_index * src_stride)..((src_index + 1) * src_stride)],
                    );
                }
            }
        } else {
            unimplemented!()
        }
    }

    fn get_byte_range_for_attribute(
        point_index: usize,
        attribute: &PointAttributeDefinition,
    ) -> Range<usize> {
        let attribute_size = attribute.size() as usize;
        (point_index * attribute_size)..((point_index + 1) * attribute_size)
    }

    fn get_byte_range_for_attributes(
        points_range: Range<usize>,
        attribute: &PointAttributeDefinition,
    ) -> Range<usize> {
        let attribute_size = attribute.size() as usize;
        (points_range.start * attribute_size)..(points_range.end * attribute_size)
    }
}

impl<'a> MakeBufferFromLayout<'a> for HashMapBuffer {
    fn new_from_layout(point_layout: PointLayout) -> Self {
        let attributes_storage = point_layout
            .attributes()
            .map(|attribute| (attribute.attribute_definition().clone(), Vec::default()))
            .collect();
        Self {
            attributes_storage,
            point_layout,
            length: 0,
        }
    }
}

impl<'a> BorrowedBuffer<'a> for HashMapBuffer
where
    HashMapBuffer: 'a,
{
    fn len(&self) -> usize {
        self.length
    }

    fn point_layout(&self) -> &PointLayout {
        &self.point_layout
    }

    fn get_point(&self, index: usize, data: &mut [u8]) {
        for attribute in self.point_layout.attributes() {
            let attribute_storage = self
                .attributes_storage
                .get(attribute.attribute_definition())
                .expect("Attribute not found within storage of this PointBuffer");
            let src_slice = &attribute_storage
                [Self::get_byte_range_for_attribute(index, attribute.attribute_definition())];
            let dst_slice = &mut data[attribute.byte_range_within_point()];
            dst_slice.copy_from_slice(src_slice);
        }
    }

    fn get_point_range(&self, range: Range<usize>, data: &mut [u8]) {
        let size_of_point = self.point_layout.size_of_point_entry() as usize;
        for attribute in self.point_layout.attributes() {
            let attribute_storage = self
                .attributes_storage
                .get(attribute.attribute_definition())
                .expect("Attribute not found within storage of this PointBuffer");
            for point_index in range.clone() {
                let src_slice = &attribute_storage[Self::get_byte_range_for_attribute(
                    point_index,
                    attribute.attribute_definition(),
                )];
                let dst_point_slice = &mut data[((point_index - range.start) * size_of_point)..];
                let dst_slice = &mut dst_point_slice[attribute.byte_range_within_point()];
                dst_slice.copy_from_slice(src_slice);
            }
        }
    }

    fn get_attribute(&self, attribute: &PointAttributeDefinition, index: usize, data: &mut [u8]) {
        let memory = self
            .attributes_storage
            .get(attribute)
            .expect("Attribute not found in PointLayout of this buffer");
        let attribute_byte_range = Self::get_byte_range_for_attribute(index, attribute);
        data.copy_from_slice(&memory[attribute_byte_range]);
    }

    unsafe fn get_attribute_unchecked(
        &self,
        attribute_member: &PointAttributeMember,
        index: usize,
        data: &mut [u8],
    ) {
        let memory = self
            .attributes_storage
            .get(attribute_member.attribute_definition())
            .expect("Attribute not found in PointLayout of this buffer");
        let attribute_byte_range =
            Self::get_byte_range_for_attribute(index, attribute_member.attribute_definition());
        data.copy_from_slice(&memory[attribute_byte_range]);
    }

    fn as_columnar(&self) -> Option<&dyn ColumnarBuffer<'a>> {
        Some(self)
    }
}

impl<'a> BorrowedMutBuffer<'a> for HashMapBuffer
where
    HashMapBuffer: 'a,
{
    unsafe fn set_point(&mut self, index: usize, point_data: &[u8]) {
        for attribute in self.point_layout.attributes() {
            let attribute_definition = attribute.attribute_definition();
            let attribute_byte_range =
                Self::get_byte_range_for_attribute(index, attribute_definition);
            let attribute_storage = self
                .attributes_storage
                .get_mut(attribute_definition)
                .expect("Attribute not found within storage of this PointBuffer");
            let dst_slice = &mut attribute_storage[attribute_byte_range];
            let src_slice = &point_data[attribute.byte_range_within_point()];
            dst_slice.copy_from_slice(src_slice);
        }
    }

    unsafe fn set_attribute(
        &mut self,
        attribute: &PointAttributeDefinition,
        index: usize,
        attribute_data: &[u8],
    ) {
        let attribute_byte_range = Self::get_byte_range_for_attribute(index, attribute);
        let attribute_storage = self
            .attributes_storage
            .get_mut(attribute)
            .expect("Attribute not found in PointLayout of this buffer");
        let attribute_bytes = &mut attribute_storage[attribute_byte_range];
        attribute_bytes.copy_from_slice(attribute_data);
    }

    fn swap(&mut self, from_index: usize, to_index: usize) {
        assert!(from_index < self.len());
        assert!(to_index < self.len());
        if from_index == to_index {
            return;
        }
        for (attribute, storage) in self.attributes_storage.iter_mut() {
            let src_byte_range = Self::get_byte_range_for_attribute(from_index, attribute);
            let dst_byte_range = Self::get_byte_range_for_attribute(to_index, attribute);
            // Is safe as long as 'from_index' and 'to_index' are not out of bounds, which is asserted
            unsafe {
                let src_ptr = storage.as_mut_ptr().add(src_byte_range.start);
                let dst_ptr = storage.as_mut_ptr().add(dst_byte_range.start);
                std::ptr::swap_nonoverlapping(src_ptr, dst_ptr, attribute.size() as usize);
            }
        }
    }

    unsafe fn set_point_range(&mut self, point_range: Range<usize>, point_data: &[u8]) {
        let size_of_point = self.point_layout.size_of_point_entry() as usize;
        let first_point = point_range.start;
        for attribute in self.point_layout.attributes() {
            let attribute_definition = attribute.attribute_definition();
            let attribute_storage = self
                .attributes_storage
                .get_mut(attribute_definition)
                .expect("Attribute not found within storage of this PointBuffer");
            for point_index in point_range.clone() {
                let zero_based_index = point_index - first_point;
                let attribute_byte_range =
                    Self::get_byte_range_for_attribute(point_index, attribute_definition);

                let dst_slice = &mut attribute_storage[attribute_byte_range];
                let src_point_slice = &point_data
                    [(zero_based_index * size_of_point)..((zero_based_index + 1) * size_of_point)];
                let src_slice = &src_point_slice[attribute.byte_range_within_point()];
                dst_slice.copy_from_slice(src_slice);
            }
        }
    }

    unsafe fn set_attribute_range(
        &mut self,
        attribute: &PointAttributeDefinition,
        point_range: Range<usize>,
        attribute_data: &[u8],
    ) {
        let attribute_range = self.get_attribute_range_mut(attribute, point_range);
        attribute_range.copy_from_slice(attribute_data);
    }

    fn as_columnar_mut(&mut self) -> Option<&mut dyn ColumnarBufferMut<'a>> {
        Some(self)
    }
}

impl<'a> OwningBuffer<'a> for HashMapBuffer
where
    HashMapBuffer: 'a,
{
    unsafe fn push_points(&mut self, point_bytes: &[u8]) {
        let point_size = self.point_layout.size_of_point_entry() as usize;
        assert_eq!(point_bytes.len() % point_size, 0);
        let num_points_added = point_bytes.len() / point_size;
        for attribute in self.point_layout.attributes() {
            let storage = self
                .attributes_storage
                .get_mut(attribute.attribute_definition())
                .expect("Attribute not found in storage of this buffer");
            for index in 0..num_points_added {
                let point_bytes = &point_bytes[(index * point_size)..((index + 1) * point_size)];
                let attribute_bytes = &point_bytes[attribute.byte_range_within_point()];
                storage.extend_from_slice(attribute_bytes);
            }
        }
        self.length += num_points_added;
    }

    fn resize(&mut self, count: usize) {
        for (attribute, storage) in self.attributes_storage.iter_mut() {
            let new_num_bytes = count * attribute.size() as usize;
            storage.resize(new_num_bytes, 0);
        }
        self.length = count;
    }

    fn clear(&mut self) {
        for storage in self.attributes_storage.values_mut() {
            storage.clear();
        }
        self.length = 0;
    }
}

impl<'a> ColumnarBuffer<'a> for HashMapBuffer
where
    HashMapBuffer: 'a,
{
    fn get_attribute_ref<'b>(
        &'b self,
        attribute: &PointAttributeDefinition,
        index: usize,
    ) -> &'b [u8]
    where
        'a: 'b,
    {
        let storage_of_attribute = self
            .attributes_storage
            .get(attribute)
            .expect("Attribute not found in PointLayout of this buffer");
        &storage_of_attribute[Self::get_byte_range_for_attribute(index, attribute)]
    }

    fn get_attribute_range_ref<'b>(
        &'b self,
        attribute: &PointAttributeDefinition,
        range: Range<usize>,
    ) -> &'b [u8]
    where
        'a: 'b,
    {
        let storage_of_attribute = self
            .attributes_storage
            .get(attribute)
            .expect("Attribute not found in PointLayout of this buffer");
        &storage_of_attribute[Self::get_byte_range_for_attributes(range, attribute)]
    }
}

impl<'a> ColumnarBufferMut<'a> for HashMapBuffer
where
    HashMapBuffer: 'a,
{
    fn get_attribute_mut<'b>(
        &'b mut self,
        attribute: &PointAttributeDefinition,
        index: usize,
    ) -> &'b mut [u8]
    where
        'a: 'b,
    {
        let byte_range = Self::get_byte_range_for_attribute(index, attribute);
        let storage_of_attribute = self
            .attributes_storage
            .get_mut(attribute)
            .expect("Attribute not found in PointLayout of this buffer");
        &mut storage_of_attribute[byte_range]
    }

    fn get_attribute_range_mut<'b>(
        &'b mut self,
        attribute: &PointAttributeDefinition,
        range: Range<usize>,
    ) -> &'b mut [u8]
    where
        'a: 'b,
    {
        let byte_range = Self::get_byte_range_for_attributes(range, attribute);
        let storage_of_attribute = self
            .attributes_storage
            .get_mut(attribute)
            .expect("Attribute not found in PointLayout of this buffer");
        &mut storage_of_attribute[byte_range]
    }
}

impl<'a> SliceBuffer<'a> for HashMapBuffer
where
    Self: 'a,
{
    type SliceType = BufferSliceColumnar<'a, Self>;

    fn slice(&'a self, range: Range<usize>) -> Self::SliceType {
        BufferSliceColumnar::new(self, range)
    }
}

impl<'a> SliceBufferMut<'a> for HashMapBuffer {
    type SliceTypeMut = BufferSliceColumnarMut<'a, Self>;

    fn slice_mut(&'a mut self, range: Range<usize>) -> Self::SliceTypeMut {
        BufferSliceColumnarMut::new(self, range)
    }
}

impl<T: PointType> FromIterator<T> for HashMapBuffer {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let point_layout = T::layout();
        let mut buffer = Self::new_from_layout(point_layout);
        iter.into_iter().for_each(|point| {
            let point_bytes = bytemuck::bytes_of(&point);
            // Safe because we know that `buffer` has the same `PointLayout` as `T`
            unsafe {
                buffer.push_points(point_bytes);
            }
        });
        buffer
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use nalgebra::Vector3;
    use rand::{distributions::Distribution, thread_rng, Rng};

    use crate::{
        containers::{BorrowedBufferExt, BorrowedMutBufferExt, VectorBuffer},
        layout::attributes::POSITION_3D,
        test_utils::{
            compare_attributes, CustomPointTypeBig, CustomPointTypeSmall, DefaultPointDistribution,
        },
    };

    use super::*;

    fn test_hashmap_buffer_with_type<T: PointType + std::fmt::Debug + PartialEq + Copy + Clone>()
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

        let test_data_as_buffer = test_data.iter().copied().collect::<HashMapBuffer>();

        {
            let mut buffer = HashMapBuffer::new_from_layout(T::layout());
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
                buffer.view_mut().set_at(idx, *point);
            }
            collected_points = buffer.view().into_iter().collect();
            assert_eq!(overwrite_data, collected_points);
        }
    }

    #[test]
    fn test_hash_map_buffer() {
        test_hashmap_buffer_with_type::<CustomPointTypeSmall>();
        test_hashmap_buffer_with_type::<CustomPointTypeBig>();
    }

    #[test]
    fn test_hash_map_buffer_mutate_attribute() {
        const COUNT: usize = 16;
        let test_data: Vec<CustomPointTypeBig> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();
        let overwrite_data: Vec<CustomPointTypeBig> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();

        let mut buffer = test_data.iter().copied().collect::<HashMapBuffer>();

        for (idx, attribute) in buffer
            .view_attribute_mut::<Vector3<f64>>(&POSITION_3D)
            .iter_mut()
            .enumerate()
        {
            *attribute = overwrite_data[idx].position;
        }

        let expected_positions = overwrite_data
            .iter()
            .map(|point| point.position)
            .collect::<Vec<_>>();
        let actual_positions = buffer
            .view_attribute(&POSITION_3D)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(expected_positions, actual_positions);
    }

    #[test]
    fn test_hash_map_buffer_filter() {
        const COUNT: usize = 16;
        let test_data: Vec<CustomPointTypeBig> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();
        let even_points = test_data
            .iter()
            .enumerate()
            .filter_map(
                |(idx, point)| {
                    if idx % 2 == 0 {
                        Some(*point)
                    } else {
                        None
                    }
                },
            )
            .collect_vec();

        let src_buffer = test_data.iter().copied().collect::<HashMapBuffer>();

        let even_points_columnar = src_buffer.filter::<HashMapBuffer, _>(|idx| idx % 2 == 0);
        assert_eq!(
            even_points_columnar,
            even_points.iter().copied().collect::<HashMapBuffer>()
        );

        let even_points_interleaved = src_buffer.filter::<VectorBuffer, _>(|idx| idx % 2 == 0);
        assert_eq!(
            even_points_interleaved,
            even_points.iter().copied().collect::<VectorBuffer>()
        );
    }
}
