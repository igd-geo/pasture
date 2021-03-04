use std::ops::Range;

use crate::layout::{PointAttributeDefinition, PointLayout};

use super::{
    InterleavedPointBuffer, PerAttributePointBuffer, PerAttributePointBufferMut, PointBuffer,
};

/// Non-owning, read-only slice of the data of an `InterleavedPointBuffer`
pub struct InterleavedPointBufferSlice<'p> {
    buffer: &'p dyn InterleavedPointBuffer,
    range_in_buffer: Range<usize>,
}

impl<'p> InterleavedPointBufferSlice<'p> {
    /// Creates a new `InterleavedPointBufferSlice` pointing to the given range within the given buffer
    ///
    /// # Panics
    ///
    /// Panics if the end of `range_in_buffer` is larger than `buffer.len()`
    pub fn new(buffer: &'p dyn InterleavedPointBuffer, range_in_buffer: Range<usize>) -> Self {
        if range_in_buffer.end > buffer.len() {
            panic!(
                "InterleavedPointBufferSlice::new: Range {:?} is out of bounds!",
                range_in_buffer
            );
        }
        Self {
            buffer,
            range_in_buffer,
        }
    }
}

impl<'p> PointBuffer for InterleavedPointBufferSlice<'p> {
    fn get_point_by_copy(&self, point_index: usize, buf: &mut [u8]) {
        let point_index_in_buffer = point_index + self.range_in_buffer.start;
        self.buffer.get_point_by_copy(point_index_in_buffer, buf);
    }

    fn get_attribute_by_copy(
        &self,
        point_index: usize,
        attribute: &PointAttributeDefinition,
        buf: &mut [u8],
    ) {
        let point_index_in_buffer = point_index + self.range_in_buffer.start;
        self.buffer
            .get_attribute_by_copy(point_index_in_buffer, attribute, buf);
    }

    fn get_points_by_copy(&self, index_range: Range<usize>, buf: &mut [u8]) {
        let range_in_buffer = index_range.start + self.range_in_buffer.start
            ..index_range.end + self.range_in_buffer.start;
        self.buffer.get_points_by_copy(range_in_buffer, buf);
    }

    fn get_attribute_range_by_copy(
        &self,
        index_range: Range<usize>,
        attribute: &PointAttributeDefinition,
        buf: &mut [u8],
    ) {
        let range_in_buffer = index_range.start + self.range_in_buffer.start
            ..index_range.end + self.range_in_buffer.start;
        self.buffer
            .get_attribute_range_by_copy(range_in_buffer, attribute, buf);
    }

    fn len(&self) -> usize {
        self.range_in_buffer.end - self.range_in_buffer.start
    }

    fn point_layout(&self) -> &PointLayout {
        self.buffer.point_layout()
    }

    fn as_interleaved(&self) -> Option<&dyn InterleavedPointBuffer> {
        Some(self)
    }
}

impl<'p> InterleavedPointBuffer for InterleavedPointBufferSlice<'p> {
    fn get_point_ref(&self, point_index: usize) -> &[u8] {
        let point_index_in_buffer = point_index + self.range_in_buffer.start;
        self.buffer.get_point_ref(point_index_in_buffer)
    }

    fn get_points_ref(&self, index_range: Range<usize>) -> &[u8] {
        let range_in_buffer = index_range.start + self.range_in_buffer.start
            ..index_range.end + self.range_in_buffer.start;
        self.buffer.get_points_ref(range_in_buffer)
    }
}

/// Non-owning, read-only slice of the data of a `PerAttributePointBuffer`
pub struct PerAttributePointBufferSlice<'p> {
    buffer: &'p dyn PerAttributePointBuffer,
    range_in_buffer: Range<usize>,
}

impl<'p> PerAttributePointBufferSlice<'p> {
    /// Creates a new `PerAttributePointBufferSlice` pointing to the given range within the given buffer
    ///
    /// # Panics
    ///
    /// Panics if the end of `range_in_buffer` is larger than `buffer.len()`
    pub fn new(buffer: &'p dyn PerAttributePointBuffer, range_in_buffer: Range<usize>) -> Self {
        if range_in_buffer.end > buffer.len() {
            panic!(
                "PerAttributePointBufferSlice::new: Range {:?} is out of bounds!",
                range_in_buffer
            );
        }
        Self {
            buffer,
            range_in_buffer,
        }
    }
}

impl<'p> PointBuffer for PerAttributePointBufferSlice<'p> {
    fn get_point_by_copy(&self, point_index: usize, buf: &mut [u8]) {
        let point_index_in_buffer = point_index + self.range_in_buffer.start;
        self.buffer.get_point_by_copy(point_index_in_buffer, buf);
    }

    fn get_attribute_by_copy(
        &self,
        point_index: usize,
        attribute: &PointAttributeDefinition,
        buf: &mut [u8],
    ) {
        let point_index_in_buffer = point_index + self.range_in_buffer.start;
        self.buffer
            .get_attribute_by_copy(point_index_in_buffer, attribute, buf);
    }

    fn get_points_by_copy(&self, index_range: Range<usize>, buf: &mut [u8]) {
        let range_in_buffer = index_range.start + self.range_in_buffer.start
            ..index_range.end + self.range_in_buffer.start;
        self.buffer.get_points_by_copy(range_in_buffer, buf);
    }

    fn get_attribute_range_by_copy(
        &self,
        index_range: Range<usize>,
        attribute: &PointAttributeDefinition,
        buf: &mut [u8],
    ) {
        let range_in_buffer = index_range.start + self.range_in_buffer.start
            ..index_range.end + self.range_in_buffer.start;
        self.buffer
            .get_attribute_range_by_copy(range_in_buffer, attribute, buf);
    }

    fn len(&self) -> usize {
        self.range_in_buffer.end - self.range_in_buffer.start
    }

    fn point_layout(&self) -> &PointLayout {
        self.buffer.point_layout()
    }

    fn as_per_attribute(&self) -> Option<&dyn PerAttributePointBuffer> {
        Some(self)
    }
}

impl<'p> PerAttributePointBuffer for PerAttributePointBufferSlice<'p> {
    fn get_attribute_ref(&self, point_index: usize, attribute: &PointAttributeDefinition) -> &[u8] {
        let point_index_in_buffer = point_index + self.range_in_buffer.start;
        self.buffer
            .get_attribute_ref(point_index_in_buffer, attribute)
    }

    fn get_attribute_range_ref(
        &self,
        index_range: Range<usize>,
        attribute: &PointAttributeDefinition,
    ) -> &[u8] {
        let range_in_buffer = index_range.start + self.range_in_buffer.start
            ..index_range.end + self.range_in_buffer.start;
        self.buffer
            .get_attribute_range_ref(range_in_buffer, attribute)
    }

    fn slice(&self, range: Range<usize>) -> PerAttributePointBufferSlice<'_> {
        PerAttributePointBufferSlice::new(self, range)
    }
}

/// Non-owning, mutable slice of the data of a `PerAttributePointBufferMut`
pub struct PerAttributePointBufferSliceMut<'p> {
    buffer: &'p mut (dyn PerAttributePointBufferMut<'p> + 'p),
    range_in_buffer: Range<usize>,
}

unsafe impl<'a> Send for PerAttributePointBufferSliceMut<'a> {}

impl<'p> PerAttributePointBufferSliceMut<'p> {
    /// Creates a new `PerAttributePointBufferSlice` pointing to the given range within the given buffer
    ///
    /// # Panics
    ///
    /// Panics if the end of `range_in_buffer` is larger than `buffer.len()`
    pub fn new(
        buffer: &'p mut dyn PerAttributePointBufferMut<'p>,
        range_in_buffer: Range<usize>,
    ) -> Self {
        if range_in_buffer.end > buffer.len() {
            panic!(
                "PerAttributePointBufferSliceMut::new: Range {:?} is out of bounds!",
                range_in_buffer
            );
        }
        Self {
            buffer,
            range_in_buffer,
        }
    }

    pub(crate) fn from_raw_ptr(
        buffer: *mut dyn PerAttributePointBufferMut<'p>,
        range_in_buffer: Range<usize>,
    ) -> Self {
        unsafe {
            Self {
                buffer: &mut *buffer,
                range_in_buffer,
            }
        }
    }

    fn from_raw_slice(
        slice: *mut PerAttributePointBufferSliceMut<'p>,
        range_in_buffer: Range<usize>,
    ) -> Self {
        unsafe {
            Self {
                buffer: &mut *slice,
                range_in_buffer,
            }
        }
    }
}

impl<'p> PointBuffer for PerAttributePointBufferSliceMut<'p> {
    fn get_point_by_copy(&self, point_index: usize, buf: &mut [u8]) {
        let point_index_in_buffer = point_index + self.range_in_buffer.start;
        self.buffer.get_point_by_copy(point_index_in_buffer, buf);
    }

    fn get_attribute_by_copy(
        &self,
        point_index: usize,
        attribute: &PointAttributeDefinition,
        buf: &mut [u8],
    ) {
        let point_index_in_buffer = point_index + self.range_in_buffer.start;
        self.buffer
            .get_attribute_by_copy(point_index_in_buffer, attribute, buf);
    }

    fn get_points_by_copy(&self, index_range: Range<usize>, buf: &mut [u8]) {
        let range_in_buffer = index_range.start + self.range_in_buffer.start
            ..index_range.end + self.range_in_buffer.start;
        self.buffer.get_points_by_copy(range_in_buffer, buf);
    }

    fn get_attribute_range_by_copy(
        &self,
        index_range: Range<usize>,
        attribute: &PointAttributeDefinition,
        buf: &mut [u8],
    ) {
        let range_in_buffer = index_range.start + self.range_in_buffer.start
            ..index_range.end + self.range_in_buffer.start;
        self.buffer
            .get_attribute_range_by_copy(range_in_buffer, attribute, buf);
    }

    fn len(&self) -> usize {
        self.range_in_buffer.end - self.range_in_buffer.start
    }

    fn point_layout(&self) -> &PointLayout {
        self.buffer.point_layout()
    }

    fn as_per_attribute(&self) -> Option<&dyn PerAttributePointBuffer> {
        Some(self)
    }
}

impl<'p> PerAttributePointBuffer for PerAttributePointBufferSliceMut<'p> {
    fn get_attribute_ref(&self, point_index: usize, attribute: &PointAttributeDefinition) -> &[u8] {
        let point_index_in_buffer = point_index + self.range_in_buffer.start;
        self.buffer
            .get_attribute_ref(point_index_in_buffer, attribute)
    }

    fn get_attribute_range_ref(
        &self,
        index_range: Range<usize>,
        attribute: &PointAttributeDefinition,
    ) -> &[u8] {
        let range_in_buffer = index_range.start + self.range_in_buffer.start
            ..index_range.end + self.range_in_buffer.start;
        self.buffer
            .get_attribute_range_ref(range_in_buffer, attribute)
    }

    fn slice(&self, range: Range<usize>) -> PerAttributePointBufferSlice<'_> {
        PerAttributePointBufferSlice::new(self, range)
    }
}

impl<'p> PerAttributePointBufferMut<'p> for PerAttributePointBufferSliceMut<'p> {
    fn get_attribute_mut(
        &mut self,
        point_index: usize,
        attribute: &PointAttributeDefinition,
    ) -> &mut [u8] {
        let point_index_in_buffer = point_index + self.range_in_buffer.start;
        self.buffer
            .get_attribute_mut(point_index_in_buffer, attribute)
    }

    fn get_attribute_range_mut(
        &mut self,
        index_range: Range<usize>,
        attribute: &PointAttributeDefinition,
    ) -> &mut [u8] {
        let range_in_buffer = index_range.start + self.range_in_buffer.start
            ..index_range.end + self.range_in_buffer.start;
        self.buffer
            .get_attribute_range_mut(range_in_buffer, attribute)
    }

    fn slice_mut(&'p mut self, range: Range<usize>) -> PerAttributePointBufferSliceMut<'p> {
        PerAttributePointBufferSliceMut::new(self, range)
    }

    fn disjunct_slices_mut<'b>(
        &'b mut self,
        ranges: &[Range<usize>],
    ) -> Vec<PerAttributePointBufferSliceMut<'p>>
    where
        'p: 'b,
    {
        let self_ptr = self as *mut PerAttributePointBufferSliceMut<'p>;

        ranges
            .iter()
            .map(|range| PerAttributePointBufferSliceMut::from_raw_slice(self_ptr, range.clone()))
            .collect()
    }

    fn as_per_attribute_point_buffer(&self) -> &dyn PerAttributePointBuffer {
        self
    }
}
