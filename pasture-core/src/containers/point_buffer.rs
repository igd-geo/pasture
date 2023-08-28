use std::{collections::HashMap, iter::FromIterator, ops::Range};

use crate::layout::{
    PointAttributeDefinition, PointAttributeMember, PointLayout, PointType, PrimitiveType,
};

use super::buffer_views::{AttributeView, AttributeViewMut, PointView, PointViewMut};

/// Trait for buffers that support slicing, similar to the builtin slice type
///
/// # Note
///
/// If there would be better support for custom DSTs, the `BufferSlice` type could be a DST and
/// we could use the `Index` trait instead.
pub trait SliceBuffer<'a> {
    /// The type of the underlying buffer of the slice. For non-sliced buffers, this will typically just be `Self`,
    /// but `BufferSlice<'a, T>` has `T` as the `UnderlyingBuffer` instead of `Self` so that nested slicing does
    /// not result in nested types (i.e. `BufferSlice<'a, BufferSlice<'a, ...>>`)
    type UnderlyingBuffer: BorrowedBuffer<'a> + Sized;

    /// Take a immutable slice to this buffer using the given `range` of points
    ///
    /// # Panics
    ///
    /// May panic if `range` is out of bounds
    fn slice<'b>(&'b self, range: Range<usize>) -> BufferSlice<'a, Self::UnderlyingBuffer>
    where
        'b: 'a;
}

/// Trait for buffers that support mutable slicing
pub trait SliceBufferMut<'a>: SliceBuffer<'a>
where
    Self::UnderlyingBuffer: BorrowedMutBuffer<'a>,
{
    /// Take a mutable slice to this buffer using the given `range` of points
    ///
    /// # Panics
    ///
    /// May panic if `range` is out of bounds
    fn slice_mut<'b>(
        &'b mut self,
        range: Range<usize>,
    ) -> BufferSliceMut<'a, Self::UnderlyingBuffer>
    where
        'b: 'a;
}

pub trait BorrowedBuffer<'a> {
    fn len(&self) -> usize;
    fn point_layout(&self) -> &PointLayout;
    fn get_point(&self, index: usize, data: &mut [u8]);
    fn get_attribute(&self, attribute: &PointAttributeDefinition, index: usize, data: &mut [u8]) {
        let attribute_member = self
            .point_layout()
            .get_attribute(attribute)
            .expect("Attribute not found in PointLayout of this buffer");
        // Is safe because we get the `attribute_member` from this buffer's point layout
        unsafe {
            self.get_attribute_unchecked(attribute_member, index, data);
        }
    }
    /// Like `get_attribute`, but performs no check whether the attribute actually is part of this buffers `PointLayout`
    /// or not. Because of this, this function accepts a `PointAttributeMember` instead of a `PointAttributeDefinition`,
    /// and this `PointAttributeMember` must come from the `PointLayout` of this buffer! The benefit over `get_attribute`
    /// is that this function skips the include checks and thus will be faster if you repeatedly want to get data for a
    /// single attribute
    ///
    /// # Safety
    ///
    /// Requires `attribute_member` to be a part of this buffer's `PointLayout`
    unsafe fn get_attribute_unchecked(
        &self,
        attribute_member: &PointAttributeMember,
        index: usize,
        data: &mut [u8],
    );

    fn view<T: PointType>(&'a self) -> PointView<'a, Self, T>
    where
        Self: Sized,
    {
        PointView::new(self)
    }

    fn view_attribute<T: PrimitiveType>(
        &'a self,
        attribute: &PointAttributeDefinition,
    ) -> AttributeView<'a, Self, T>
    where
        Self: Sized,
    {
        AttributeView::new(self, attribute)
    }
}

pub trait BorrowedMutBuffer<'a>: BorrowedBuffer<'a> {
    fn set_point(&mut self, index: usize, point_data: &[u8]);
    fn set_attribute(
        &mut self,
        attribute: &PointAttributeDefinition,
        index: usize,
        attribute_data: &[u8],
    );
    fn swap(&mut self, from_index: usize, to_index: usize);

    fn view_mut<T: PointType>(&'a mut self) -> PointViewMut<'a, Self, T>
    where
        Self: Sized,
    {
        PointViewMut::new(self)
    }

    fn view_attribute_mut<T: PrimitiveType>(
        &'a mut self,
        attribute: &PointAttributeDefinition,
    ) -> AttributeViewMut<'a, Self, T>
    where
        Self: Sized,
    {
        AttributeViewMut::new(self, attribute)
    }
}

pub trait OwningBuffer<'a>: BorrowedMutBuffer<'a> + Sized {
    fn push_point(&mut self, point_bytes: &[u8]);
    fn extend(&mut self, many_point_bytes: &[u8]);
    fn resize(&mut self, count: usize);
    fn clear(&mut self);
}

pub trait InterleavedBuffer<'a>: BorrowedBuffer<'a> {
    fn get_point_ref(&'a self, index: usize) -> &'a [u8];
    fn get_point_range_ref(&'a self, range: Range<usize>) -> &'a [u8];
}

pub trait InterleavedBufferMut<'a>: InterleavedBuffer<'a> + BorrowedMutBuffer<'a> {
    fn get_point_mut(&'a mut self, index: usize) -> &'a mut [u8];
    fn get_point_range_mut(&'a mut self, range: Range<usize>) -> &'a mut [u8];
}

pub trait ColumnarBuffer<'a>: BorrowedBuffer<'a> {
    fn get_attribute_ref(&'a self, attribute: &PointAttributeDefinition, index: usize) -> &'a [u8];
    fn get_attribute_range_ref(
        &'a self,
        attribute: &PointAttributeDefinition,
        range: Range<usize>,
    ) -> &'a [u8];
}

pub trait ColumnarBufferMut<'a>: ColumnarBuffer<'a> + BorrowedMutBuffer<'a> {
    fn get_attribute_mut(
        &'a mut self,
        attribute: &PointAttributeDefinition,
        index: usize,
    ) -> &'a mut [u8];
    fn get_attribute_range_mut(
        &'a mut self,
        attribute: &PointAttributeDefinition,
        range: Range<usize>,
    ) -> &'a mut [u8];
}

/**
 * Now we have a handful of specific buffer types:
 * - VectorBuffer (which is Owning + Interleaved)
 * - HashMapBuffer (which is Owning + Columnar)
 * - ExternalMemoryBuffer (?)
 * - Slice<'a, T: BorrowedBuffer<'a>> (which is Borrowed and has dependent implementations if T is interleaved and/or columnar)
 * - SliceMut<'a, T: BorrowedMutBuffer<'a>> (which is BorrowedMut and again has dependent implementations)
 *
 * And some view types for typed access to the data:
 * - PointView<'a, T: PointType, U: BorrowedBuffer<'a>> (which is NOT a point buffer but instead a typed view with specific methods)
 * - PointViewRef<'a, T: PointType, U: InterleavedBuffer<'a>>
 * - PointViewMut<'a, T: PointType: U: InterleavedBuffer<'a> + BorrowedMutBuffer<'a>>
 * - AttributeView<'a, T: PrimitiveType, U: BorrowedBuffer<'a>> (which is NOT a point buffer but instead a typed view of a single specific attribute)
 * - AttributeViewRef<'a, T: PrimitiveType, U: ColumnarBuffer<'a>>
 * - AttributeViewMut<'a, T: PrimitiveType, U: ColumnarBuffer<'a> + BorrowedMutBuffer<'a>>
 */

pub struct VectorBuffer {
    storage: Vec<u8>,
    point_layout: PointLayout,
}

impl VectorBuffer {
    pub fn new(point_layout: PointLayout) -> Self {
        Self {
            point_layout,
            storage: Default::default(),
        }
    }

    pub fn with_capacity(capacity: usize, point_layout: PointLayout) -> Self {
        let required_bytes = capacity * point_layout.size_of_point_entry() as usize;
        Self {
            point_layout,
            storage: Vec::with_capacity(required_bytes),
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

impl<'a> BorrowedBuffer<'a> for VectorBuffer
where
    VectorBuffer: 'a,
{
    fn len(&self) -> usize {
        self.storage.len() / self.point_layout.size_of_point_entry() as usize
    }

    fn point_layout(&self) -> &PointLayout {
        &self.point_layout
    }

    fn get_point(&self, index: usize, data: &mut [u8]) {
        let point_ref = self.get_point_ref(index);
        data.copy_from_slice(point_ref);
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
}

impl<'a> BorrowedMutBuffer<'a> for VectorBuffer
where
    VectorBuffer: 'a,
{
    fn set_point(&mut self, index: usize, point_data: &[u8]) {
        let point_bytes = self.get_point_mut(index);
        point_bytes.copy_from_slice(point_data);
    }

    fn set_attribute(
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
}

impl<'a> OwningBuffer<'a> for VectorBuffer
where
    VectorBuffer: 'a,
{
    fn push_point(&mut self, point_bytes: &[u8]) {
        assert_eq!(
            point_bytes.len(),
            self.point_layout.size_of_point_entry() as usize
        );
        self.storage.extend_from_slice(point_bytes);
    }

    fn extend(&mut self, many_point_bytes: &[u8]) {
        assert!(many_point_bytes.len() % self.point_layout.size_of_point_entry() as usize == 0);
        self.storage.extend_from_slice(many_point_bytes);
    }

    fn resize(&mut self, count: usize) {
        let size_of_point = self.point_layout.size_of_point_entry() as usize;
        self.storage.resize(count * size_of_point, 0);
    }

    fn clear(&mut self) {
        self.storage.clear();
    }
}

impl<'a> InterleavedBuffer<'a> for VectorBuffer
where
    VectorBuffer: 'a,
{
    fn get_point_ref(&'a self, index: usize) -> &'a [u8] {
        &self.storage[self.get_byte_range_of_point(index)]
    }

    fn get_point_range_ref(&'a self, range: Range<usize>) -> &'a [u8] {
        &self.storage[self.get_byte_range_of_points(range)]
    }
}

impl<'a> InterleavedBufferMut<'a> for VectorBuffer
where
    VectorBuffer: 'a,
{
    fn get_point_mut(&'a mut self, index: usize) -> &'a mut [u8] {
        let byte_range = self.get_byte_range_of_point(index);
        &mut self.storage[byte_range]
    }

    fn get_point_range_mut(&'a mut self, range: Range<usize>) -> &'a mut [u8] {
        let byte_range = self.get_byte_range_of_points(range);
        &mut self.storage[byte_range]
    }
}

impl<'a> SliceBuffer<'a> for VectorBuffer
where
    VectorBuffer: 'a,
{
    type UnderlyingBuffer = Self;

    fn slice<'b>(&'b self, range: Range<usize>) -> BufferSlice<'a, Self::UnderlyingBuffer>
    where
        'b: 'a,
    {
        BufferSlice {
            buffer: self,
            point_range: range,
        }
    }
}

impl<'a> SliceBufferMut<'a> for VectorBuffer
where
    VectorBuffer: 'a,
{
    fn slice_mut<'b>(
        &'b mut self,
        range: Range<usize>,
    ) -> BufferSliceMut<'a, Self::UnderlyingBuffer>
    where
        'b: 'a,
    {
        BufferSliceMut {
            buffer: self,
            point_range: range,
        }
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
            };
            // Overwrite the preallocated memory of the buffer with the points in the iterator:
            iter.enumerate().for_each(|(index, point)| {
                let point_bytes = bytemuck::bytes_of(&point);
                buffer.set_point(index, point_bytes);
            });
            buffer
        } else {
            let mut buffer = Self {
                point_layout,
                storage: Default::default(),
            };
            iter.for_each(|point| {
                let point_bytes = bytemuck::bytes_of(&point);
                buffer.push_point(point_bytes);
            });
            buffer
        }
    }
}

pub struct HashMapBuffer {
    attributes_storage: HashMap<PointAttributeDefinition, Vec<u8>>,
    point_layout: PointLayout,
    length: usize,
}

impl HashMapBuffer {
    /// Creates a new empty `HashMapBuffer` from the given `PointLayout`
    pub fn new(point_layout: PointLayout) -> Self {
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
}

impl<'a> BorrowedMutBuffer<'a> for HashMapBuffer
where
    HashMapBuffer: 'a,
{
    fn set_point(&mut self, index: usize, point_data: &[u8]) {
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

    fn set_attribute(
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
}

impl<'a> OwningBuffer<'a> for HashMapBuffer
where
    HashMapBuffer: 'a,
{
    fn push_point(&mut self, point_bytes: &[u8]) {
        for attribute in self.point_layout.attributes() {
            let attribute_bytes = &point_bytes[attribute.byte_range_within_point()];
            let storage = self
                .attributes_storage
                .get_mut(attribute.attribute_definition())
                .expect("Attribute not found in storage of this buffer");
            storage.extend_from_slice(attribute_bytes);
        }
        self.length += 1;
    }

    fn extend(&mut self, many_point_bytes: &[u8]) {
        let point_size = self.point_layout.size_of_point_entry() as usize;
        let num_points_added = many_point_bytes.len() / point_size;
        for attribute in self.point_layout.attributes() {
            let storage = self
                .attributes_storage
                .get_mut(attribute.attribute_definition())
                .expect("Attribute not found in storage of this buffer");
            for index in 0..num_points_added {
                let point_bytes =
                    &many_point_bytes[(index * point_size)..((index + 1) * point_size)];
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
    fn get_attribute_ref(&'a self, attribute: &PointAttributeDefinition, index: usize) -> &'a [u8] {
        let storage_of_attribute = self
            .attributes_storage
            .get(attribute)
            .expect("Attribute not found in PointLayout of this buffer");
        &storage_of_attribute[Self::get_byte_range_for_attribute(index, attribute)]
    }

    fn get_attribute_range_ref(
        &'a self,
        attribute: &PointAttributeDefinition,
        range: Range<usize>,
    ) -> &'a [u8] {
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
    fn get_attribute_mut(
        &'a mut self,
        attribute: &PointAttributeDefinition,
        index: usize,
    ) -> &'a mut [u8] {
        let byte_range = Self::get_byte_range_for_attribute(index, attribute);
        let storage_of_attribute = self
            .attributes_storage
            .get_mut(attribute)
            .expect("Attribute not found in PointLayout of this buffer");
        &mut storage_of_attribute[byte_range]
    }

    fn get_attribute_range_mut(
        &'a mut self,
        attribute: &PointAttributeDefinition,
        range: Range<usize>,
    ) -> &'a mut [u8] {
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
    HashMapBuffer: 'a,
{
    type UnderlyingBuffer = Self;

    fn slice<'b>(&'b self, range: Range<usize>) -> BufferSlice<'a, Self::UnderlyingBuffer>
    where
        'b: 'a,
    {
        BufferSlice {
            buffer: self,
            point_range: range,
        }
    }
}

impl<'a> SliceBufferMut<'a> for HashMapBuffer
where
    HashMapBuffer: 'a,
{
    fn slice_mut<'b>(
        &'b mut self,
        range: Range<usize>,
    ) -> BufferSliceMut<'a, Self::UnderlyingBuffer>
    where
        'b: 'a,
    {
        BufferSliceMut {
            buffer: self,
            point_range: range,
        }
    }
}

impl<T: PointType> FromIterator<T> for HashMapBuffer {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let point_layout = T::layout();
        let mut buffer = Self::new(point_layout);
        iter.into_iter().for_each(|point| {
            let point_bytes = bytemuck::bytes_of(&point);
            buffer.push_point(point_bytes);
        });
        buffer
    }
}

pub struct ExternalMemoryBuffer<T: AsRef<[u8]>> {
    external_memory: T,
    point_layout: PointLayout,
}

impl<T: AsRef<[u8]>> ExternalMemoryBuffer<T> {
    pub fn new(external_memory: T, point_layout: PointLayout) -> Self {
        Self {
            external_memory,
            point_layout,
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
        self.external_memory.as_ref().len() / self.point_layout.size_of_point_entry() as usize
    }

    fn point_layout(&self) -> &PointLayout {
        &self.point_layout
    }

    fn get_point(&self, index: usize, data: &mut [u8]) {
        let point_bytes = &self.external_memory.as_ref()[self.get_byte_range_for_point(index)];
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
}

impl<'a, T: AsMut<[u8]> + AsRef<[u8]>> BorrowedMutBuffer<'a> for ExternalMemoryBuffer<T>
where
    ExternalMemoryBuffer<T>: 'a,
{
    fn set_point(&mut self, index: usize, point_data: &[u8]) {
        let point_byte_range = self.get_byte_range_for_point(index);
        let point_memory = &mut self.external_memory.as_mut()[point_byte_range];
        point_memory.copy_from_slice(point_data);
    }

    fn set_attribute(
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
}

impl<'a, T: AsRef<[u8]>> InterleavedBuffer<'a> for ExternalMemoryBuffer<T>
where
    ExternalMemoryBuffer<T>: 'a,
{
    fn get_point_ref(&'a self, index: usize) -> &'a [u8] {
        let memory = self.external_memory.as_ref();
        &memory[self.get_byte_range_for_point(index)]
    }

    fn get_point_range_ref(&'a self, range: Range<usize>) -> &'a [u8] {
        let memory = self.external_memory.as_ref();
        &memory[self.get_byte_range_for_point_range(range)]
    }
}

impl<'a, T: AsRef<[u8]> + AsMut<[u8]>> InterleavedBufferMut<'a> for ExternalMemoryBuffer<T>
where
    ExternalMemoryBuffer<T>: 'a,
{
    fn get_point_mut(&'a mut self, index: usize) -> &'a mut [u8] {
        let byte_range = self.get_byte_range_for_point(index);
        let memory = self.external_memory.as_mut();
        &mut memory[byte_range]
    }

    fn get_point_range_mut(&'a mut self, range: Range<usize>) -> &'a mut [u8] {
        let byte_range = self.get_byte_range_for_point_range(range);
        let memory = self.external_memory.as_mut();
        &mut memory[byte_range]
    }
}

impl<'a, T: AsRef<[u8]>> SliceBuffer<'a> for ExternalMemoryBuffer<T>
where
    ExternalMemoryBuffer<T>: 'a,
{
    type UnderlyingBuffer = Self;

    fn slice<'b>(&'b self, range: Range<usize>) -> BufferSlice<'a, Self::UnderlyingBuffer>
    where
        'b: 'a,
    {
        BufferSlice {
            buffer: self,
            point_range: range,
        }
    }
}

impl<'a, T: AsRef<[u8]> + AsMut<[u8]>> SliceBufferMut<'a> for ExternalMemoryBuffer<T>
where
    ExternalMemoryBuffer<T>: 'a,
{
    fn slice_mut<'b>(
        &'b mut self,
        range: Range<usize>,
    ) -> BufferSliceMut<'a, Self::UnderlyingBuffer>
    where
        'b: 'a,
    {
        BufferSliceMut {
            buffer: self,
            point_range: range,
        }
    }
}

pub struct BufferSlice<'a, T: BorrowedBuffer<'a>> {
    buffer: &'a T,
    point_range: Range<usize>,
}

impl<'a, T: BorrowedBuffer<'a>> BufferSlice<'a, T> {
    fn get_and_check_global_point_index(&self, local_index: usize) -> usize {
        assert!(local_index < self.point_range.end);
        local_index + self.point_range.start
    }

    fn get_and_check_global_point_range(&self, local_range: Range<usize>) -> Range<usize> {
        assert!(local_range.end <= self.point_range.end);
        if local_range.start >= local_range.end {
            // Doesn't matter what range we return, as long as it is empty
            self.point_range.end..self.point_range.end
        } else {
            (local_range.start + self.point_range.start)..(local_range.end + self.point_range.start)
        }
    }
}

impl<'a, T: BorrowedBuffer<'a>> BorrowedBuffer<'a> for BufferSlice<'a, T> {
    fn len(&self) -> usize {
        self.point_range.end - self.point_range.start
    }

    fn point_layout(&self) -> &PointLayout {
        self.buffer.point_layout()
    }

    fn get_point(&self, index: usize, data: &mut [u8]) {
        self.buffer
            .get_point(self.get_and_check_global_point_index(index), data)
    }

    fn get_attribute(&self, attribute: &PointAttributeDefinition, index: usize, data: &mut [u8]) {
        self.buffer.get_attribute(
            attribute,
            self.get_and_check_global_point_index(index),
            data,
        )
    }

    unsafe fn get_attribute_unchecked(
        &self,
        attribute_member: &PointAttributeMember,
        index: usize,
        data: &mut [u8],
    ) {
        self.buffer.get_attribute_unchecked(
            attribute_member,
            self.get_and_check_global_point_index(index),
            data,
        )
    }
}

impl<'a, T: InterleavedBuffer<'a>> InterleavedBuffer<'a> for BufferSlice<'a, T> {
    fn get_point_ref(&'a self, index: usize) -> &'a [u8] {
        self.buffer
            .get_point_ref(self.get_and_check_global_point_index(index))
    }

    fn get_point_range_ref(&'a self, range: Range<usize>) -> &'a [u8] {
        self.buffer
            .get_point_range_ref(self.get_and_check_global_point_range(range))
    }
}

impl<'a, T: ColumnarBuffer<'a>> ColumnarBuffer<'a> for BufferSlice<'a, T> {
    fn get_attribute_ref(&'a self, attribute: &PointAttributeDefinition, index: usize) -> &'a [u8] {
        self.buffer
            .get_attribute_ref(attribute, self.get_and_check_global_point_index(index))
    }

    fn get_attribute_range_ref(
        &'a self,
        attribute: &PointAttributeDefinition,
        range: Range<usize>,
    ) -> &'a [u8] {
        self.buffer
            .get_attribute_range_ref(attribute, self.get_and_check_global_point_range(range))
    }
}

impl<'a, T: BorrowedBuffer<'a> + Sized> SliceBuffer<'a> for BufferSlice<'a, T> {
    type UnderlyingBuffer = T;

    fn slice<'b>(&'b self, range: Range<usize>) -> BufferSlice<'a, Self::UnderlyingBuffer>
    where
        'b: 'a,
    {
        assert!(range.start < self.len());
        assert!(range.end < self.len());
        let global_range = if range.start > range.end {
            (self.point_range.start + range.end)..(self.point_range.start + range.end)
        } else {
            (self.point_range.start + range.start)..(self.point_range.start + range.end)
        };
        BufferSlice {
            buffer: self.buffer,
            point_range: global_range,
        }
    }
}

pub struct BufferSliceMut<'a, T: BorrowedMutBuffer<'a>> {
    buffer: &'a mut T,
    point_range: Range<usize>,
}

impl<'a, T: BorrowedMutBuffer<'a>> BufferSliceMut<'a, T> {
    fn get_and_check_global_point_index(&self, local_index: usize) -> usize {
        assert!(local_index < self.point_range.end);
        local_index + self.point_range.start
    }

    fn get_and_check_global_point_range(&self, local_range: Range<usize>) -> Range<usize> {
        assert!(local_range.end <= self.point_range.end);
        if local_range.start >= local_range.end {
            // Doesn't matter what range we return, as long as it is empty
            self.point_range.end..self.point_range.end
        } else {
            (local_range.start + self.point_range.start)..(local_range.end + self.point_range.start)
        }
    }
}

impl<'a, T: BorrowedMutBuffer<'a>> BorrowedBuffer<'a> for BufferSliceMut<'a, T> {
    fn len(&self) -> usize {
        self.point_range.end - self.point_range.start
    }

    fn point_layout(&self) -> &PointLayout {
        self.buffer.point_layout()
    }

    fn get_point(&self, index: usize, data: &mut [u8]) {
        self.buffer
            .get_point(self.get_and_check_global_point_index(index), data)
    }

    fn get_attribute(&self, attribute: &PointAttributeDefinition, index: usize, data: &mut [u8]) {
        self.buffer.get_attribute(
            attribute,
            self.get_and_check_global_point_index(index),
            data,
        )
    }

    unsafe fn get_attribute_unchecked(
        &self,
        attribute_member: &PointAttributeMember,
        index: usize,
        data: &mut [u8],
    ) {
        self.buffer.get_attribute_unchecked(
            attribute_member,
            self.get_and_check_global_point_index(index),
            data,
        )
    }
}

impl<'a, T: BorrowedMutBuffer<'a>> BorrowedMutBuffer<'a> for BufferSliceMut<'a, T> {
    fn set_point(&mut self, index: usize, point_data: &[u8]) {
        self.buffer
            .set_point(self.get_and_check_global_point_index(index), point_data)
    }

    fn set_attribute(
        &mut self,
        attribute: &PointAttributeDefinition,
        index: usize,
        attribute_data: &[u8],
    ) {
        self.buffer.set_attribute(
            attribute,
            self.get_and_check_global_point_index(index),
            attribute_data,
        )
    }

    fn swap(&mut self, from_index: usize, to_index: usize) {
        self.buffer.swap(
            self.get_and_check_global_point_index(from_index),
            self.get_and_check_global_point_index(to_index),
        )
    }
}

impl<'a, T: InterleavedBuffer<'a> + BorrowedMutBuffer<'a>> InterleavedBuffer<'a>
    for BufferSliceMut<'a, T>
{
    fn get_point_ref(&'a self, index: usize) -> &'a [u8] {
        self.buffer
            .get_point_ref(self.get_and_check_global_point_index(index))
    }

    fn get_point_range_ref(&'a self, range: Range<usize>) -> &'a [u8] {
        self.buffer
            .get_point_range_ref(self.get_and_check_global_point_range(range))
    }
}

impl<'a, T: InterleavedBufferMut<'a> + BorrowedMutBuffer<'a>> InterleavedBufferMut<'a>
    for BufferSliceMut<'a, T>
{
    fn get_point_mut(&'a mut self, index: usize) -> &'a mut [u8] {
        self.buffer
            .get_point_mut(self.get_and_check_global_point_index(index))
    }

    fn get_point_range_mut(&'a mut self, range: Range<usize>) -> &'a mut [u8] {
        self.buffer
            .get_point_range_mut(self.get_and_check_global_point_range(range))
    }
}

impl<'a, T: ColumnarBuffer<'a> + BorrowedMutBuffer<'a>> ColumnarBuffer<'a>
    for BufferSliceMut<'a, T>
{
    fn get_attribute_ref(&'a self, attribute: &PointAttributeDefinition, index: usize) -> &'a [u8] {
        self.buffer
            .get_attribute_ref(attribute, self.get_and_check_global_point_index(index))
    }

    fn get_attribute_range_ref(
        &'a self,
        attribute: &PointAttributeDefinition,
        range: Range<usize>,
    ) -> &'a [u8] {
        self.buffer
            .get_attribute_range_ref(attribute, self.get_and_check_global_point_range(range))
    }
}

impl<'a, T: ColumnarBufferMut<'a> + BorrowedMutBuffer<'a>> ColumnarBufferMut<'a>
    for BufferSliceMut<'a, T>
{
    fn get_attribute_mut(
        &'a mut self,
        attribute: &PointAttributeDefinition,
        index: usize,
    ) -> &'a mut [u8] {
        self.buffer
            .get_attribute_mut(attribute, self.get_and_check_global_point_index(index))
    }

    fn get_attribute_range_mut(
        &'a mut self,
        attribute: &PointAttributeDefinition,
        range: Range<usize>,
    ) -> &'a mut [u8] {
        self.buffer
            .get_attribute_range_mut(attribute, self.get_and_check_global_point_range(range))
    }
}

impl<'a, T: BorrowedBuffer<'a> + BorrowedMutBuffer<'a> + Sized> SliceBuffer<'a>
    for BufferSliceMut<'a, T>
{
    type UnderlyingBuffer = T;

    fn slice<'b>(&'b self, range: Range<usize>) -> BufferSlice<'a, Self::UnderlyingBuffer>
    where
        'b: 'a,
    {
        assert!(range.start < self.len());
        assert!(range.end < self.len());
        let global_range = if range.start > range.end {
            (self.point_range.start + range.end)..(self.point_range.start + range.end)
        } else {
            (self.point_range.start + range.start)..(self.point_range.start + range.end)
        };
        BufferSlice {
            buffer: self.buffer,
            point_range: global_range,
        }
    }
}

impl<'a, T: BorrowedBuffer<'a> + BorrowedMutBuffer<'a> + Sized> SliceBufferMut<'a>
    for BufferSliceMut<'a, T>
{
    fn slice_mut<'b>(
        &'b mut self,
        range: Range<usize>,
    ) -> BufferSliceMut<'a, Self::UnderlyingBuffer>
    where
        'b: 'a,
    {
        assert!(range.start < self.len());
        assert!(range.end < self.len());
        let global_range = if range.start > range.end {
            (self.point_range.start + range.end)..(self.point_range.start + range.end)
        } else {
            (self.point_range.start + range.start)..(self.point_range.start + range.end)
        };
        BufferSliceMut {
            buffer: self.buffer,
            point_range: global_range,
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;
    use pasture_derive::PointType;
    use rand::{prelude::Distribution, thread_rng, Rng};

    use crate::layout::{attributes::POSITION_3D, PointAttributeDataType};

    use super::*;

    #[derive(
        PointType,
        Default,
        Copy,
        Clone,
        PartialEq,
        Debug,
        bytemuck::AnyBitPattern,
        bytemuck::NoUninit,
    )]
    #[repr(C, packed)]
    struct CustomPointTypeSmall {
        #[pasture(BUILTIN_POSITION_3D)]
        pub position: Vector3<f64>,
        #[pasture(BUILTIN_CLASSIFICATION)]
        pub classification: u8,
    }

    #[derive(
        PointType,
        Default,
        Copy,
        Clone,
        PartialEq,
        Debug,
        bytemuck::AnyBitPattern,
        bytemuck::NoUninit,
    )]
    #[repr(C, packed)]
    struct CustomPointTypeBig {
        #[pasture(BUILTIN_GPS_TIME)]
        pub gps_time: f64,
        #[pasture(BUILTIN_COLOR_RGB)]
        pub color: Vector3<u16>,
        #[pasture(BUILTIN_POSITION_3D)]
        pub position: Vector3<f64>,
        #[pasture(BUILTIN_CLASSIFICATION)]
        pub classification: u8,
        #[pasture(BUILTIN_INTENSITY)]
        pub intensity: i16,
    }

    struct DefaultPointDistribution;

    impl Distribution<CustomPointTypeSmall> for DefaultPointDistribution {
        fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> CustomPointTypeSmall {
            CustomPointTypeSmall {
                classification: rng.gen(),
                position: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
            }
        }
    }

    impl Distribution<CustomPointTypeBig> for DefaultPointDistribution {
        fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> CustomPointTypeBig {
            CustomPointTypeBig {
                classification: rng.gen(),
                position: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
                color: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
                gps_time: rng.gen(),
                intensity: rng.gen(),
            }
        }
    }

    fn compare_attributes_typed<
        'a,
        T: PointType + std::fmt::Debug + PartialEq + Copy + Clone,
        U: PrimitiveType + std::fmt::Debug + PartialEq,
    >(
        buffer: &'a impl BorrowedBuffer<'a>,
        attribute: &PointAttributeDefinition,
        expected_points: &'a impl BorrowedBuffer<'a>,
    ) {
        let collected_values = buffer
            .view_attribute::<U>(attribute)
            .into_iter()
            .collect::<Vec<_>>();
        let expected_values = expected_points
            .view_attribute::<U>(attribute)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(expected_values, collected_values);
    }

    /// Compare the given point attribute using the static type corresponding to the attribute's `PointAttributeDataType`
    fn compare_attributes<'a, T: PointType + std::fmt::Debug + PartialEq + Copy + Clone>(
        buffer: &'a impl BorrowedBuffer<'a>,
        attribute: &PointAttributeDefinition,
        expected_points: &'a impl BorrowedBuffer<'a>,
    ) {
        match attribute.datatype() {
            PointAttributeDataType::F32 => {
                compare_attributes_typed::<T, f32>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::F64 => {
                compare_attributes_typed::<T, f64>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::I16 => {
                compare_attributes_typed::<T, i16>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::I32 => {
                compare_attributes_typed::<T, i32>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::I64 => {
                compare_attributes_typed::<T, i64>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::I8 => {
                compare_attributes_typed::<T, i8>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::U16 => {
                compare_attributes_typed::<T, u16>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::U32 => {
                compare_attributes_typed::<T, u32>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::U64 => {
                compare_attributes_typed::<T, u64>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::U8 => {
                compare_attributes_typed::<T, u8>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::Vec3f32 => {
                compare_attributes_typed::<T, Vector3<f32>>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::Vec3f64 => {
                compare_attributes_typed::<T, Vector3<f64>>(buffer, attribute, expected_points);
            }
            PointAttributeDataType::Vec3i32 => {
                compare_attributes_typed::<T, Vector3<i32>>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::Vec3u16 => {
                compare_attributes_typed::<T, Vector3<u16>>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::Vec3u8 => {
                compare_attributes_typed::<T, Vector3<u8>>(buffer, attribute, expected_points)
            }
            _ => unimplemented!(),
        }
    }

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
            let mut buffer = VectorBuffer::new(T::layout());
            assert_eq!(0, buffer.len());
            assert_eq!(T::layout(), *buffer.point_layout());
            assert_eq!(0, buffer.view::<T>().into_iter().count());

            for idx in 0..COUNT {
                buffer.view_mut().push_point(test_data[idx]);
                assert_eq!(idx + 1, buffer.len());
                assert_eq!(test_data[idx], buffer.view().at(idx));
            }

            let mut collected_points = buffer.view().into_iter().collect::<Vec<_>>();
            assert_eq!(test_data, collected_points);

            let collected_points_by_ref = buffer.view().iter().copied().collect::<Vec<_>>();
            assert_eq!(test_data, collected_points_by_ref);

            for attribute in buffer.point_layout().attributes() {
                compare_attributes::<T>(
                    &buffer,
                    attribute.attribute_definition(),
                    &test_data_as_buffer,
                );
            }

            let slice = buffer.slice(1..2);
            assert_eq!(test_data[1], slice.view().at(0));

            for idx in 0..COUNT {
                *buffer.view_mut().at_mut(idx) = overwrite_data[idx];
            }
            collected_points = buffer.view().iter().copied().collect();
            assert_eq!(overwrite_data, collected_points);
        }
    }

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
            let mut buffer = HashMapBuffer::new(T::layout());
            assert_eq!(0, buffer.len());
            assert_eq!(T::layout(), *buffer.point_layout());
            assert_eq!(0, buffer.view::<T>().into_iter().count());

            for idx in 0..COUNT {
                buffer.view_mut().push_point(test_data[idx]);
                assert_eq!(idx + 1, buffer.len());
                assert_eq!(test_data[idx], buffer.view().at(idx));
            }

            let mut collected_points = buffer.view().into_iter().collect::<Vec<_>>();
            assert_eq!(test_data, collected_points);

            for attribute in buffer.point_layout().attributes() {
                compare_attributes::<T>(
                    &buffer,
                    attribute.attribute_definition(),
                    &test_data_as_buffer,
                );
            }

            let slice = buffer.slice(1..2);
            assert_eq!(test_data[1], slice.view().at(0));

            for idx in 0..COUNT {
                buffer.view_mut().set_at(idx, overwrite_data[idx]);
            }
            collected_points = buffer.view().into_iter().collect();
            assert_eq!(overwrite_data, collected_points);
        }
    }

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
            let mut buffer = VectorBuffer::new(T::layout());
            assert_eq!(0, buffer.len());
            assert_eq!(T::layout(), *buffer.point_layout());
            assert_eq!(0, buffer.view::<T>().into_iter().count());

            for idx in 0..COUNT {
                buffer.view_mut().push_point(test_data[idx]);
                assert_eq!(idx + 1, buffer.len());
                assert_eq!(test_data[idx], buffer.view().at(idx));
            }

            let mut collected_points = buffer.view().into_iter().collect::<Vec<_>>();
            assert_eq!(test_data, collected_points);

            let collected_points_by_ref = buffer.view().iter().copied().collect::<Vec<_>>();
            assert_eq!(test_data, collected_points_by_ref);

            for attribute in buffer.point_layout().attributes() {
                compare_attributes::<T>(
                    &buffer,
                    attribute.attribute_definition(),
                    &test_data_as_buffer,
                );
            }

            let slice = buffer.slice(1..2);
            assert_eq!(test_data[1], slice.view().at(0));

            for idx in 0..COUNT {
                *buffer.view_mut().at_mut(idx) = overwrite_data[idx];
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
    fn test_external_memory_buffer() {
        test_external_memory_buffer_with_type::<CustomPointTypeSmall>();
        test_external_memory_buffer_with_type::<CustomPointTypeBig>();
    }
}
