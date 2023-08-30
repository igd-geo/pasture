use anyhow::Result;
use std::{collections::HashMap, iter::FromIterator, ops::Range};

use crate::layout::{
    PointAttributeDefinition, PointAttributeMember, PointLayout, PointType, PrimitiveType,
};

use super::{
    buffer_views::{AttributeView, AttributeViewMut, PointView, PointViewMut},
    AttributeViewConverting,
};

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
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn point_layout(&self) -> &PointLayout;
    fn get_point(&self, index: usize, data: &mut [u8]);
    fn get_point_range(&self, range: Range<usize>, data: &mut [u8]);
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

    fn view<'b, T: PointType>(&'b self) -> PointView<'a, 'b, Self, T>
    where
        Self: Sized,
        'a: 'b,
    {
        PointView::new(self)
    }

    fn view_attribute<'b, T: PrimitiveType>(
        &'b self,
        attribute: &PointAttributeDefinition,
    ) -> AttributeView<'a, 'b, Self, T>
    where
        Self: Sized,
        'a: 'b,
    {
        AttributeView::new(self, attribute)
    }

    fn view_attribute_with_conversion<'b, T: PrimitiveType>(
        &'b self,
        attribute: &PointAttributeDefinition,
    ) -> Result<AttributeViewConverting<'a, 'b, Self, T>>
    where
        Self: Sized,
        'a: 'b,
    {
        AttributeViewConverting::new(self, attribute)
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

    /// Apply a transformation function to the given `attribute` of all points within this buffer. This function is
    /// helpful if you want to modify a single attribute of a buffer in-place and works for buffers of all memory
    /// layouts. For columnar buffers, prefer using `get_attribute_range_mut` to modify attribute data in-place.
    ///
    /// This function does not support attribute type conversion, so the type `T` must match the `PointAttributeDataType`
    /// of `attribute`!
    ///
    /// The conversion function takes two attributes: The index of the point and the value for the attribute of that
    /// point. This is a bit more flexible than just passing the attribute, as the index allows accessing further
    /// information for that point from within the conversion function.
    ///
    /// # Panics
    ///
    /// If `attribute` is not part of the `PointLayout` of this buffer
    fn transform_attribute<'b, T: PrimitiveType, F: Fn(usize, T) -> T>(
        &'b mut self,
        attribute: &PointAttributeDefinition,
        func: F,
    ) where
        Self: Sized,
        'a: 'b,
    {
        let num_points = self.len();
        let mut attribute_view = self.view_attribute_mut(attribute);
        for point_index in 0..num_points {
            let attribute_value = attribute_view.at(point_index);
            attribute_view.set_at(point_index, func(point_index, attribute_value));
        }
    }

    fn view_mut<'b, T: PointType>(&'b mut self) -> PointViewMut<'a, 'b, Self, T>
    where
        Self: Sized,
        'a: 'b,
    {
        PointViewMut::new(self)
    }

    fn view_attribute_mut<'b, T: PrimitiveType>(
        &'b mut self,
        attribute: &PointAttributeDefinition,
    ) -> AttributeViewMut<'a, 'b, Self, T>
    where
        Self: Sized,
        'a: 'b,
    {
        AttributeViewMut::new(self, attribute)
    }
}

pub trait OwningBuffer<'a>: BorrowedMutBuffer<'a> + Sized {
    unsafe fn push_points(&mut self, point_bytes: &[u8]);
    /// Appends data from the given buffer to the end of this buffer. Makes no assumptions about the memory
    /// layout of `other`
    ///
    /// # Panics
    ///
    /// If `self.point_layout()` does not equal `other.point_layout()`
    fn append<'b, B: BorrowedBuffer<'b>>(&mut self, other: &'_ B) {
        assert_eq!(self.point_layout(), other.point_layout());
        let old_self_len = self.len();
        self.resize(old_self_len + other.len());
        let mut point_buffer = vec![0; self.point_layout().size_of_point_entry() as usize];
        for point_index in 0..other.len() {
            other.get_point(point_index, &mut point_buffer);
            self.set_point(old_self_len + point_index, &point_buffer);
        }
    }
    /// Appends data from the given interleaved buffer to the end of this buffer
    ///
    /// # Note
    ///
    /// Why is there no general `append` function? As far as I understand the currently Rust rules, we can't
    /// state that two traits are mutually exclusive. So in principle there could be some point buffer type
    /// that implements both `InterleavedBuffer` and `ColumnarBuffer`. So we can't auto-detect from the type
    /// `B` whether we should use an implementation that assumes interleaved memory layout, or one that assumes
    /// columnar memory layout. We could always be conservative and assume neither layout and use the `get_point`
    /// and `set_point` API, but this is pessimistic and has suboptimal performance. So instead, we provide
    /// two independent functions that allow more optimal implementations if the memory layouts of `Self` and
    /// `B` match.
    ///
    /// # Panics
    ///
    /// If `self.point_layout()` does not equal `other.point_layout()`
    fn append_interleaved<'b, B: InterleavedBuffer<'b>>(&mut self, other: &'_ B);
    /// Appends data from the given columnar buffer to the end of this buffer
    ///
    /// # Panics
    ///
    /// If `self.point_layout()` does not equal `other.point_layout()`
    fn append_columnar<'b, B: ColumnarBuffer<'b>>(&mut self, other: &'_ B);
    fn resize(&mut self, count: usize);
    fn clear(&mut self);
}

pub trait MakeBufferFromLayout<'a>: BorrowedBuffer<'a> + Sized {
    fn new_from_layout(point_layout: PointLayout) -> Self;
}

pub trait InterleavedBuffer<'a>: BorrowedBuffer<'a> {
    fn get_point_ref<'b>(&'b self, index: usize) -> &'b [u8]
    where
        'a: 'b;
    fn get_point_range_ref<'b>(&'b self, range: Range<usize>) -> &'b [u8]
    where
        'a: 'b;
}

pub trait InterleavedBufferMut<'a>: InterleavedBuffer<'a> + BorrowedMutBuffer<'a> {
    fn get_point_mut<'b>(&'b mut self, index: usize) -> &'b mut [u8]
    where
        'a: 'b;
    fn get_point_range_mut<'b>(&'b mut self, range: Range<usize>) -> &'b mut [u8]
    where
        'a: 'b;
}

pub trait ColumnarBuffer<'a>: BorrowedBuffer<'a> {
    fn get_attribute_ref<'b>(
        &'b self,
        attribute: &PointAttributeDefinition,
        index: usize,
    ) -> &'b [u8]
    where
        'a: 'b;
    fn get_attribute_range_ref<'b>(
        &'b self,
        attribute: &PointAttributeDefinition,
        range: Range<usize>,
    ) -> &'b [u8]
    where
        'a: 'b;
}

pub trait ColumnarBufferMut<'a>: ColumnarBuffer<'a> + BorrowedMutBuffer<'a> {
    fn get_attribute_mut<'b>(
        &'b mut self,
        attribute: &PointAttributeDefinition,
        index: usize,
    ) -> &'b mut [u8]
    where
        'a: 'b;
    fn get_attribute_range_mut<'b>(
        &'b mut self,
        attribute: &PointAttributeDefinition,
        range: Range<usize>,
    ) -> &'b mut [u8]
    where
        'a: 'b;
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VectorBuffer {
    storage: Vec<u8>,
    point_layout: PointLayout,
}

impl VectorBuffer {
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

impl<'a> MakeBufferFromLayout<'a> for VectorBuffer {
    fn new_from_layout(point_layout: PointLayout) -> Self {
        Self {
            point_layout,
            storage: Default::default(),
        }
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
    unsafe fn push_points(&mut self, point_bytes: &[u8]) {
        assert_eq!(
            (point_bytes.len() % self.point_layout.size_of_point_entry() as usize),
            0
        );
        self.storage.extend_from_slice(point_bytes);
    }

    fn resize(&mut self, count: usize) {
        let size_of_point = self.point_layout.size_of_point_entry() as usize;
        self.storage.resize(count * size_of_point, 0);
    }

    fn clear(&mut self) {
        self.storage.clear();
    }

    fn append_interleaved<'b, B: InterleavedBuffer<'b>>(&mut self, other: &'_ B) {
        assert_eq!(self.point_layout(), other.point_layout());
        // Is safe because we checked that the two `PointLayout`s match
        unsafe {
            self.push_points(other.get_point_range_ref(0..other.len()));
        }
    }

    fn append_columnar<'b, B: ColumnarBuffer<'b>>(&mut self, other: &'_ B) {
        assert_eq!(self.point_layout(), other.point_layout());
        let previous_self_len = self.len();
        self.resize(previous_self_len + other.len());
        for point_index in 0..other.len() {
            let self_memory = self.get_point_mut(previous_self_len + point_index);
            other.get_point(point_index, self_memory);
        }
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
                // Safe because we know that `buffer` has the same layout as `T`
                unsafe {
                    buffer.push_points(point_bytes);
                }
            });
            buffer
        }
    }
}

/// Helper struct to push point data into a `HashMapBuffer` attribute by attribute
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashMapBuffer {
    attributes_storage: HashMap<PointAttributeDefinition, Vec<u8>>,
    point_layout: PointLayout,
    length: usize,
}

impl HashMapBuffer {
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

    pub fn begin_push_attributes(&mut self) -> HashMapBufferAttributePusher<'_> {
        HashMapBufferAttributePusher::new(self)
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
                let dst_point_slice = &mut data[(point_index * size_of_point)..];
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

    fn append_interleaved<'b, B: InterleavedBuffer<'b>>(&mut self, other: &'_ B) {
        assert_eq!(self.point_layout(), other.point_layout());
        // Safe because we checked that the point layouts match
        unsafe {
            self.push_points(other.get_point_range_ref(0..other.len()));
        }
    }

    fn append_columnar<'b, B: ColumnarBuffer<'b>>(&mut self, other: &'_ B) {
        assert_eq!(self.point_layout(), other.point_layout());
        for attribute in self.point_layout.attributes() {
            let storage = self
                .attributes_storage
                .get_mut(attribute.attribute_definition())
                .expect("Attribute not found in storage of this buffer");
            storage.extend_from_slice(
                other.get_attribute_range_ref(attribute.attribute_definition(), 0..other.len()),
            );
        }
        self.length += other.len();
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

    fn get_point_range(&self, range: Range<usize>, data: &mut [u8]) {
        self.buffer
            .get_point_range(self.get_and_check_global_point_range(range), data)
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
    fn get_point_ref<'b>(&'b self, index: usize) -> &'b [u8]
    where
        'a: 'b,
    {
        self.buffer
            .get_point_ref(self.get_and_check_global_point_index(index))
    }

    fn get_point_range_ref<'b>(&'b self, range: Range<usize>) -> &'b [u8]
    where
        'a: 'b,
    {
        self.buffer
            .get_point_range_ref(self.get_and_check_global_point_range(range))
    }
}

impl<'a, T: ColumnarBuffer<'a>> ColumnarBuffer<'a> for BufferSlice<'a, T> {
    fn get_attribute_ref<'b>(
        &'b self,
        attribute: &PointAttributeDefinition,
        index: usize,
    ) -> &'b [u8]
    where
        'a: 'b,
    {
        self.buffer
            .get_attribute_ref(attribute, self.get_and_check_global_point_index(index))
    }

    fn get_attribute_range_ref<'b>(
        &'b self,
        attribute: &PointAttributeDefinition,
        range: Range<usize>,
    ) -> &'b [u8]
    where
        'a: 'b,
    {
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

    fn get_point_range(&self, range: Range<usize>, data: &mut [u8]) {
        self.buffer
            .get_point_range(self.get_and_check_global_point_range(range), data)
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
    fn get_point_ref<'b>(&'b self, index: usize) -> &'b [u8]
    where
        'a: 'b,
    {
        self.buffer
            .get_point_ref(self.get_and_check_global_point_index(index))
    }

    fn get_point_range_ref<'b>(&'b self, range: Range<usize>) -> &'b [u8]
    where
        'a: 'b,
    {
        self.buffer
            .get_point_range_ref(self.get_and_check_global_point_range(range))
    }
}

impl<'a, T: InterleavedBufferMut<'a> + BorrowedMutBuffer<'a>> InterleavedBufferMut<'a>
    for BufferSliceMut<'a, T>
{
    fn get_point_mut<'b>(&'b mut self, index: usize) -> &'b mut [u8]
    where
        'a: 'b,
    {
        self.buffer
            .get_point_mut(self.get_and_check_global_point_index(index))
    }

    fn get_point_range_mut<'b>(&'b mut self, range: Range<usize>) -> &'b mut [u8]
    where
        'a: 'b,
    {
        self.buffer
            .get_point_range_mut(self.get_and_check_global_point_range(range))
    }
}

impl<'a, T: ColumnarBuffer<'a> + BorrowedMutBuffer<'a>> ColumnarBuffer<'a>
    for BufferSliceMut<'a, T>
{
    fn get_attribute_ref<'b>(
        &'b self,
        attribute: &PointAttributeDefinition,
        index: usize,
    ) -> &'b [u8]
    where
        'a: 'b,
    {
        self.buffer
            .get_attribute_ref(attribute, self.get_and_check_global_point_index(index))
    }

    fn get_attribute_range_ref<'b>(
        &'b self,
        attribute: &PointAttributeDefinition,
        range: Range<usize>,
    ) -> &'b [u8]
    where
        'a: 'b,
    {
        self.buffer
            .get_attribute_range_ref(attribute, self.get_and_check_global_point_range(range))
    }
}

impl<'a, T: ColumnarBufferMut<'a> + BorrowedMutBuffer<'a>> ColumnarBufferMut<'a>
    for BufferSliceMut<'a, T>
{
    fn get_attribute_mut<'b>(
        &'b mut self,
        attribute: &PointAttributeDefinition,
        index: usize,
    ) -> &'b mut [u8]
    where
        'a: 'b,
    {
        self.buffer
            .get_attribute_mut(attribute, self.get_and_check_global_point_index(index))
    }

    fn get_attribute_range_mut<'b>(
        &'b mut self,
        attribute: &PointAttributeDefinition,
        range: Range<usize>,
    ) -> &'b mut [u8]
    where
        'a: 'b,
    {
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
            let mut buffer = VectorBuffer::new_from_layout(T::layout());
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
            let mut buffer = HashMapBuffer::new_from_layout(T::layout());
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
            let mut buffer = VectorBuffer::new_from_layout(T::layout());
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

    fn test_transform_attribute_generic<
        'a,
        B: BorrowedMutBuffer<'a> + FromIterator<CustomPointTypeBig> + 'a,
    >() {
        const COUNT: usize = 16;
        let test_data: Vec<CustomPointTypeBig> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();
        let overwrite_data: Vec<CustomPointTypeBig> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();

        let mut buffer = test_data.iter().copied().collect::<B>();
        // Overwrite the positions with the positions in `overwrite_data` using the `transform_attribute` function
        buffer.transform_attribute(&POSITION_3D, |index, _| -> Vector3<f64> {
            overwrite_data[index].position
        });

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
    fn test_transform_attribute() {
        test_transform_attribute_generic::<VectorBuffer>();
        test_transform_attribute_generic::<HashMapBuffer>();
    }

    #[test]
    fn test_append() {
        const COUNT: usize = 16;
        let test_data: Vec<CustomPointTypeBig> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();

        let expected_buffer_interleaved = test_data.iter().copied().collect::<VectorBuffer>();
        let expected_buffer_columnar = test_data.iter().copied().collect::<HashMapBuffer>();

        {
            let mut vector_buffer = VectorBuffer::new_from_layout(CustomPointTypeBig::layout());
            vector_buffer.append(&expected_buffer_interleaved);
            assert_eq!(expected_buffer_interleaved, vector_buffer);
        }
        {
            let mut vector_buffer = VectorBuffer::new_from_layout(CustomPointTypeBig::layout());
            vector_buffer.append(&expected_buffer_columnar);
            assert_eq!(expected_buffer_interleaved, vector_buffer);
        }
        {
            let mut vector_buffer = VectorBuffer::new_from_layout(CustomPointTypeBig::layout());
            vector_buffer.append_interleaved(&expected_buffer_interleaved);
            assert_eq!(expected_buffer_interleaved, vector_buffer);
        }
        {
            let mut vector_buffer = VectorBuffer::new_from_layout(CustomPointTypeBig::layout());
            vector_buffer.append_columnar(&expected_buffer_columnar);
            assert_eq!(expected_buffer_interleaved, vector_buffer);
        }
        {
            let mut hashmap_buffer = HashMapBuffer::new_from_layout(CustomPointTypeBig::layout());
            hashmap_buffer.append(&expected_buffer_columnar);
            assert_eq!(expected_buffer_columnar, hashmap_buffer);
        }
        {
            let mut hashmap_buffer = HashMapBuffer::new_from_layout(CustomPointTypeBig::layout());
            hashmap_buffer.append(&expected_buffer_interleaved);
            assert_eq!(expected_buffer_columnar, hashmap_buffer);
        }
        {
            let mut hashmap_buffer = HashMapBuffer::new_from_layout(CustomPointTypeBig::layout());
            hashmap_buffer.append_columnar(&expected_buffer_columnar);
            assert_eq!(expected_buffer_columnar, hashmap_buffer);
        }
        {
            let mut hashmap_buffer: HashMapBuffer =
                HashMapBuffer::new_from_layout(CustomPointTypeBig::layout());
            hashmap_buffer.append_interleaved(&expected_buffer_interleaved);
            assert_eq!(expected_buffer_columnar, hashmap_buffer);
        }
    }
}
