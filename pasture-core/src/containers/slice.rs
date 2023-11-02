use std::ops::Range;

use crate::layout::{PointAttributeDefinition, PointAttributeMember, PointLayout};

use super::{
    BorrowedBuffer, BorrowedMutBuffer, ColumnarBuffer, ColumnarBufferMut, InterleavedBuffer,
    InterleavedBufferMut,
};

/// Trait for buffers that support slicing, similar to the builtin slice type
///
/// # Note
///
/// If there would be better support for custom DSTs, the `BufferSlice` type could be a DST and
/// we could use the `Index` trait instead.
pub trait SliceBuffer<'a>
where
    Self: 'a,
{
    /// The slice type
    type SliceType: BorrowedBuffer<'a> + Sized;

    /// Take a immutable slice to this buffer using the given `range` of points
    ///
    /// # Panics
    ///
    /// May panic if `range` is out of bounds
    fn slice(&'a self, range: Range<usize>) -> Self::SliceType;
}

/// Trait for buffers that support mutable slicing
pub trait SliceBufferMut<'a>: SliceBuffer<'a>
where
    Self: 'a,
{
    type SliceTypeMut: BorrowedMutBuffer<'a> + Sized;
    /// Take a mutable slice to this buffer using the given `range` of points
    ///
    /// # Panics
    ///
    /// May panic if `range` is out of bounds
    fn slice_mut(&'a mut self, range: Range<usize>) -> Self::SliceTypeMut;
}

/// Converts a local point index to a global point index given the `point_range` of a slice. Performs
/// bounds checks while converting
///
/// # Panics
///
/// If `local_index` is out of bounds of `point_range`
fn get_and_check_global_point_index(local_index: usize, point_range: &Range<usize>) -> usize {
    assert!(local_index < point_range.len());
    local_index + point_range.start
}

/// Converts a local range of points ot a global range of points given the `point_range` of a slice.
/// Performs bounds checks while converting
///
/// # Panics
///
/// If `local_range` is out of bounds of `point_range`
fn get_and_check_global_point_range(
    local_range: Range<usize>,
    point_range: &Range<usize>,
) -> Range<usize> {
    assert!(local_range.end <= point_range.len());
    if local_range.start >= local_range.end {
        // Doesn't matter what range we return, as long as it is empty
        point_range.end..point_range.end
    } else {
        (local_range.start + point_range.start)..(local_range.end + point_range.start)
    }
}

/// An immutable slice to a point buffer. In terms of memory layout, the slice will have the same
/// capabilities as the underlying buffer, i.e. if `T` implements `InterleavedBuffer`, so does this
/// slice, and similar for the other memory layout traits.
pub struct BufferSlice<'a, T: BorrowedBuffer<'a>> {
    buffer: &'a T,
    point_range: Range<usize>,
}

impl<'a, T: BorrowedBuffer<'a>> BufferSlice<'a, T> {
    /// Creates a new `BufferSlice` for the given `point_range` in the given `buffer`
    pub fn new(buffer: &'a T, point_range: Range<usize>) -> Self {
        Self {
            buffer,
            point_range,
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
        self.buffer.get_point(
            get_and_check_global_point_index(index, &self.point_range),
            data,
        )
    }

    fn get_point_range(&self, range: Range<usize>, data: &mut [u8]) {
        self.buffer.get_point_range(
            get_and_check_global_point_range(range, &self.point_range),
            data,
        )
    }

    fn get_attribute(&self, attribute: &PointAttributeDefinition, index: usize, data: &mut [u8]) {
        self.buffer.get_attribute(
            attribute,
            get_and_check_global_point_index(index, &self.point_range),
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
            get_and_check_global_point_index(index, &self.point_range),
            data,
        )
    }
}

impl<'a, T: InterleavedBuffer<'a>> InterleavedBuffer<'a> for BufferSlice<'a, T> {
    fn get_point_ref<'c>(&'c self, index: usize) -> &'c [u8]
    where
        'a: 'c,
    {
        self.buffer
            .get_point_ref(get_and_check_global_point_index(index, &self.point_range))
    }

    fn get_point_range_ref<'c>(&'c self, range: Range<usize>) -> &'c [u8]
    where
        'a: 'c,
    {
        self.buffer
            .get_point_range_ref(get_and_check_global_point_range(range, &self.point_range))
    }
}

impl<'a, T: ColumnarBuffer<'a>> ColumnarBuffer<'a> for BufferSlice<'a, T> {
    fn get_attribute_ref<'c>(
        &'c self,
        attribute: &PointAttributeDefinition,
        index: usize,
    ) -> &'c [u8]
    where
        'a: 'c,
    {
        self.buffer.get_attribute_ref(
            attribute,
            get_and_check_global_point_index(index, &self.point_range),
        )
    }

    fn get_attribute_range_ref<'c>(
        &'c self,
        attribute: &PointAttributeDefinition,
        range: Range<usize>,
    ) -> &'c [u8]
    where
        'a: 'c,
    {
        self.buffer.get_attribute_range_ref(
            attribute,
            get_and_check_global_point_range(range, &self.point_range),
        )
    }
}

impl<'a, T: BorrowedBuffer<'a> + Sized> SliceBuffer<'a> for BufferSlice<'a, T> {
    type SliceType = BufferSlice<'a, T>;

    fn slice(&self, range: Range<usize>) -> BufferSlice<'a, T> {
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

/// A mutable slice to a point buffer. Works like [`BufferSlice`], but allows mutable access to the
/// underlying buffer. This type conditionally implements the [`InterleavedBufferMut`] and [`ColumnarBufferMut`]
/// traits if `T` implements them
pub struct BufferSliceMut<'a, T: BorrowedMutBuffer<'a>> {
    buffer: &'a mut T,
    point_range: Range<usize>,
}

impl<'a, T: BorrowedMutBuffer<'a>> BufferSliceMut<'a, T> {
    /// Creates a new `BufferSliceMut` for the given `point_range` in the given `buffer`
    pub fn new(buffer: &'a mut T, point_range: Range<usize>) -> Self {
        Self {
            buffer,
            point_range,
        }
    }

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
    unsafe fn set_point(&mut self, index: usize, point_data: &[u8]) {
        self.buffer
            .set_point(self.get_and_check_global_point_index(index), point_data)
    }

    unsafe fn set_attribute(
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

    unsafe fn set_point_range(&mut self, point_range: Range<usize>, point_data: &[u8]) {
        self.buffer.set_point_range(
            self.get_and_check_global_point_range(point_range),
            point_data,
        )
    }

    unsafe fn set_attribute_range(
        &mut self,
        attribute: &PointAttributeDefinition,
        point_range: Range<usize>,
        attribute_data: &[u8],
    ) {
        self.buffer.set_attribute_range(
            attribute,
            self.get_and_check_global_point_range(point_range),
            attribute_data,
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
    type SliceType = BufferSlice<'a, T>;

    fn slice(&'a self, range: Range<usize>) -> Self::SliceType {
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
    type SliceTypeMut = BufferSliceMut<'a, T>;

    fn slice_mut(&'a mut self, range: Range<usize>) -> Self::SliceTypeMut {
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

/// A buffer slice for an interleaved buffer
pub struct BufferSliceInterleaved<'a, T: InterleavedBuffer<'a>>(BufferSlice<'a, T>);

impl<'a, T: InterleavedBuffer<'a>> BufferSliceInterleaved<'a, T> {
    pub fn new(buffer: &'a T, point_range: Range<usize>) -> Self {
        Self(BufferSlice::new(buffer, point_range))
    }
}

impl<'a, T: InterleavedBuffer<'a>> BorrowedBuffer<'a> for BufferSliceInterleaved<'a, T> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn point_layout(&self) -> &PointLayout {
        self.0.point_layout()
    }

    fn get_point(&self, index: usize, data: &mut [u8]) {
        self.0.get_point(index, data)
    }

    fn get_point_range(&self, range: Range<usize>, data: &mut [u8]) {
        self.0.get_point_range(range, data)
    }

    unsafe fn get_attribute_unchecked(
        &self,
        attribute_member: &PointAttributeMember,
        index: usize,
        data: &mut [u8],
    ) {
        self.0
            .get_attribute_unchecked(attribute_member, index, data)
    }

    fn as_interleaved(&self) -> Option<&dyn InterleavedBuffer<'a>> {
        Some(self)
    }
}

impl<'a, T: InterleavedBuffer<'a>> InterleavedBuffer<'a> for BufferSliceInterleaved<'a, T> {
    fn get_point_ref<'c>(&'c self, index: usize) -> &'c [u8]
    where
        'a: 'c,
    {
        self.0.get_point_ref(index)
    }

    fn get_point_range_ref<'c>(&'c self, range: Range<usize>) -> &'c [u8]
    where
        'a: 'c,
    {
        self.0.get_point_range_ref(range)
    }
}

impl<'a, T: InterleavedBuffer<'a>> SliceBuffer<'a> for BufferSliceInterleaved<'a, T> {
    type SliceType = Self;

    fn slice(&'a self, range: Range<usize>) -> Self::SliceType {
        Self(self.0.slice(range))
    }
}

/// A mutable buffer slice for an interleaved buffer
pub struct BufferSliceInterleavedMut<'a, T: InterleavedBufferMut<'a>>(BufferSliceMut<'a, T>);

impl<'a, T: InterleavedBufferMut<'a>> BufferSliceInterleavedMut<'a, T> {
    pub fn new(buffer: &'a mut T, point_range: Range<usize>) -> Self {
        Self(BufferSliceMut::new(buffer, point_range))
    }
}

impl<'a, T: InterleavedBufferMut<'a>> BorrowedBuffer<'a> for BufferSliceInterleavedMut<'a, T> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn point_layout(&self) -> &PointLayout {
        self.0.point_layout()
    }

    fn get_point(&self, index: usize, data: &mut [u8]) {
        self.0.get_point(index, data)
    }

    fn get_point_range(&self, range: Range<usize>, data: &mut [u8]) {
        self.0.get_point_range(range, data)
    }

    unsafe fn get_attribute_unchecked(
        &self,
        attribute_member: &PointAttributeMember,
        index: usize,
        data: &mut [u8],
    ) {
        self.0
            .get_attribute_unchecked(attribute_member, index, data)
    }

    fn as_interleaved(&self) -> Option<&dyn InterleavedBuffer<'a>> {
        Some(self)
    }
}

impl<'a, T: InterleavedBufferMut<'a>> BorrowedMutBuffer<'a> for BufferSliceInterleavedMut<'a, T> {
    unsafe fn set_point(&mut self, index: usize, point_data: &[u8]) {
        self.0.set_point(index, point_data)
    }

    unsafe fn set_point_range(&mut self, point_range: Range<usize>, point_data: &[u8]) {
        self.0.set_point_range(point_range, point_data)
    }

    unsafe fn set_attribute(
        &mut self,
        attribute: &PointAttributeDefinition,
        index: usize,
        attribute_data: &[u8],
    ) {
        self.0.set_attribute(attribute, index, attribute_data)
    }

    unsafe fn set_attribute_range(
        &mut self,
        attribute: &PointAttributeDefinition,
        point_range: Range<usize>,
        attribute_data: &[u8],
    ) {
        self.0
            .set_attribute_range(attribute, point_range, attribute_data)
    }

    fn swap(&mut self, from_index: usize, to_index: usize) {
        self.0.swap(from_index, to_index)
    }

    fn as_interleaved_mut(&mut self) -> Option<&mut dyn InterleavedBufferMut<'a>> {
        Some(self)
    }
}

impl<'a, T: InterleavedBufferMut<'a>> InterleavedBuffer<'a> for BufferSliceInterleavedMut<'a, T> {
    fn get_point_ref<'b>(&'b self, index: usize) -> &'b [u8]
    where
        'a: 'b,
    {
        self.0.get_point_ref(index)
    }

    fn get_point_range_ref<'b>(&'b self, range: Range<usize>) -> &'b [u8]
    where
        'a: 'b,
    {
        self.0.get_point_range_ref(range)
    }
}

impl<'a, T: InterleavedBufferMut<'a>> InterleavedBufferMut<'a>
    for BufferSliceInterleavedMut<'a, T>
{
    fn get_point_mut<'b>(&'b mut self, index: usize) -> &'b mut [u8]
    where
        'a: 'b,
    {
        self.0.get_point_mut(index)
    }

    fn get_point_range_mut<'b>(&'b mut self, range: Range<usize>) -> &'b mut [u8]
    where
        'a: 'b,
    {
        self.0.get_point_range_mut(range)
    }
}

impl<'a, T: InterleavedBufferMut<'a>> SliceBuffer<'a> for BufferSliceInterleavedMut<'a, T> {
    type SliceType = BufferSliceInterleaved<'a, T>;

    fn slice(&'a self, range: Range<usize>) -> Self::SliceType {
        BufferSliceInterleaved(self.0.slice(range))
    }
}

impl<'a, T: InterleavedBufferMut<'a>> SliceBufferMut<'a> for BufferSliceInterleavedMut<'a, T> {
    type SliceTypeMut = Self;

    fn slice_mut(&'a mut self, range: Range<usize>) -> Self::SliceTypeMut {
        Self(self.0.slice_mut(range))
    }
}

/// A buffer slice for a columnar buffer
pub struct BufferSliceColumnar<'a, T: ColumnarBuffer<'a>>(BufferSlice<'a, T>);

impl<'a, T: ColumnarBuffer<'a>> BufferSliceColumnar<'a, T> {
    pub fn new(buffer: &'a T, point_range: Range<usize>) -> Self {
        Self(BufferSlice::new(buffer, point_range))
    }
}

impl<'a, T: ColumnarBuffer<'a>> BorrowedBuffer<'a> for BufferSliceColumnar<'a, T> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn point_layout(&self) -> &PointLayout {
        self.0.point_layout()
    }

    fn get_point(&self, index: usize, data: &mut [u8]) {
        self.0.get_point(index, data)
    }

    fn get_point_range(&self, range: Range<usize>, data: &mut [u8]) {
        self.0.get_point_range(range, data)
    }

    unsafe fn get_attribute_unchecked(
        &self,
        attribute_member: &PointAttributeMember,
        index: usize,
        data: &mut [u8],
    ) {
        self.0
            .get_attribute_unchecked(attribute_member, index, data)
    }

    fn as_columnar(&self) -> Option<&dyn ColumnarBuffer<'a>> {
        Some(self)
    }
}

impl<'a, T: ColumnarBuffer<'a>> ColumnarBuffer<'a> for BufferSliceColumnar<'a, T> {
    fn get_attribute_ref<'c>(
        &'c self,
        attribute: &PointAttributeDefinition,
        index: usize,
    ) -> &'c [u8]
    where
        'a: 'c,
    {
        self.0.get_attribute_ref(attribute, index)
    }

    fn get_attribute_range_ref<'c>(
        &'c self,
        attribute: &PointAttributeDefinition,
        range: Range<usize>,
    ) -> &'c [u8]
    where
        'a: 'c,
    {
        self.0.get_attribute_range_ref(attribute, range)
    }
}

/// A mutable buffer slice for a columnar buffer
pub struct BufferSliceColumnarMut<'a, T: ColumnarBufferMut<'a>>(BufferSliceMut<'a, T>);

impl<'a, T: ColumnarBufferMut<'a>> BufferSliceColumnarMut<'a, T> {
    pub fn new(buffer: &'a mut T, point_range: Range<usize>) -> Self {
        Self(BufferSliceMut::new(buffer, point_range))
    }
}

impl<'a, T: ColumnarBufferMut<'a>> BorrowedBuffer<'a> for BufferSliceColumnarMut<'a, T> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn point_layout(&self) -> &PointLayout {
        self.0.point_layout()
    }

    fn get_point(&self, index: usize, data: &mut [u8]) {
        self.0.get_point(index, data)
    }

    fn get_point_range(&self, range: Range<usize>, data: &mut [u8]) {
        self.0.get_point_range(range, data)
    }

    unsafe fn get_attribute_unchecked(
        &self,
        attribute_member: &PointAttributeMember,
        index: usize,
        data: &mut [u8],
    ) {
        self.0
            .get_attribute_unchecked(attribute_member, index, data)
    }

    fn as_columnar(&self) -> Option<&dyn ColumnarBuffer<'a>> {
        Some(self)
    }
}

impl<'a, T: ColumnarBufferMut<'a>> BorrowedMutBuffer<'a> for BufferSliceColumnarMut<'a, T> {
    unsafe fn set_point(&mut self, index: usize, point_data: &[u8]) {
        self.0.set_point(index, point_data)
    }

    unsafe fn set_point_range(&mut self, point_range: Range<usize>, point_data: &[u8]) {
        self.0.set_point_range(point_range, point_data)
    }

    unsafe fn set_attribute(
        &mut self,
        attribute: &PointAttributeDefinition,
        index: usize,
        attribute_data: &[u8],
    ) {
        self.0.set_attribute(attribute, index, attribute_data)
    }

    unsafe fn set_attribute_range(
        &mut self,
        attribute: &PointAttributeDefinition,
        point_range: Range<usize>,
        attribute_data: &[u8],
    ) {
        self.0
            .set_attribute_range(attribute, point_range, attribute_data)
    }

    fn swap(&mut self, from_index: usize, to_index: usize) {
        self.0.swap(from_index, to_index)
    }

    fn as_columnar_mut(&mut self) -> Option<&mut dyn ColumnarBufferMut<'a>> {
        Some(self)
    }
}

impl<'a, T: ColumnarBufferMut<'a>> ColumnarBuffer<'a> for BufferSliceColumnarMut<'a, T> {
    fn get_attribute_ref<'b>(
        &'b self,
        attribute: &PointAttributeDefinition,
        index: usize,
    ) -> &'b [u8]
    where
        'a: 'b,
    {
        self.0.get_attribute_ref(attribute, index)
    }

    fn get_attribute_range_ref<'b>(
        &'b self,
        attribute: &PointAttributeDefinition,
        range: Range<usize>,
    ) -> &'b [u8]
    where
        'a: 'b,
    {
        self.0.get_attribute_range_ref(attribute, range)
    }
}

impl<'a, T: ColumnarBufferMut<'a>> ColumnarBufferMut<'a> for BufferSliceColumnarMut<'a, T> {
    fn get_attribute_mut<'b>(
        &'b mut self,
        attribute: &PointAttributeDefinition,
        index: usize,
    ) -> &'b mut [u8]
    where
        'a: 'b,
    {
        self.0.get_attribute_mut(attribute, index)
    }

    fn get_attribute_range_mut<'b>(
        &'b mut self,
        attribute: &PointAttributeDefinition,
        range: Range<usize>,
    ) -> &'b mut [u8]
    where
        'a: 'b,
    {
        self.0.get_attribute_range_mut(attribute, range)
    }
}
