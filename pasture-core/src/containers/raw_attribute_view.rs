use std::{
    mem,
    ops::{Index, IndexMut},
};

use crate::layout::{PointAttributeDefinition, PointAttributeMember};

use super::{ColumnarBuffer, ColumnarBufferMut, InterleavedBuffer, InterleavedBufferMut};

/// A view over raw memory for a point attribute. This view can be obtained from any buffer that
/// is either interleaved or columnar, and will be more efficient than calling `get_attribute` on
/// the buffer
pub struct RawAttributeView<'a> {
    point_data: &'a [u8],
    offset: usize,
    stride: usize,
    size_of_attribute: usize,
}

impl<'a> RawAttributeView<'a> {
    /// Creates a `RawAttributeView` for the given `attribute_member` from an interleaved point buffer
    pub(crate) fn from_interleaved_buffer<B: InterleavedBuffer + ?Sized>(
        buffer: &'a B,
        attribute_member: &PointAttributeMember,
    ) -> Self {
        let stride = buffer.point_layout().size_of_point_entry() as usize;
        Self {
            offset: attribute_member.offset() as usize,
            point_data: buffer.get_point_range_ref(0..buffer.len()),
            size_of_attribute: attribute_member.size() as usize,
            stride,
        }
    }

    /// Creates a `RawAttributeView` for the given `attribute_definition` from a columnar point buffer
    pub(crate) fn from_columnar_buffer<B: ColumnarBuffer + ?Sized>(
        buffer: &'a B,
        attribute_definition: &PointAttributeDefinition,
    ) -> Self {
        Self {
            offset: 0,
            point_data: buffer.get_attribute_range_ref(attribute_definition, 0..buffer.len()),
            size_of_attribute: attribute_definition.size() as usize,
            stride: attribute_definition.size() as usize,
        }
    }

    /// The length of this 'RawAttributeView` (i.e. the number of points within the view)
    pub fn len(&self) -> usize {
        self.point_data.len() / self.stride
    }

    /// Is this `RawAttributeView` empty?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Index<usize> for RawAttributeView<'_> {
    type Output = [u8];

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let start = self.offset + (self.stride * index);
        let end = start + self.size_of_attribute;
        &self.point_data[start..end]
    }
}

impl<'a> Iterator for RawAttributeView<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((first_point, rest)) = self.point_data.split_at_checked(self.stride) {
            self.point_data = rest;

            let start = self.offset;
            let end = start + self.size_of_attribute;
            let attribute = &first_point[start..end];
            Some(attribute)
        } else {
            None
        }
    }
}

/// Like `RawAttributeView`, but for mutable data
pub struct RawAttributeViewMut<'a> {
    point_data: &'a mut [u8],
    offset: usize,
    stride: usize,
    size_of_attribute: usize,
}

impl<'a> RawAttributeViewMut<'a> {
    /// Creates a `RawAttributeView` for the given `attribute_member` from an interleaved point buffer
    pub(crate) fn from_interleaved_buffer<B: InterleavedBufferMut + ?Sized>(
        buffer: &'a mut B,
        attribute_member: &PointAttributeMember,
    ) -> Self {
        let stride = buffer.point_layout().size_of_point_entry() as usize;
        Self {
            offset: attribute_member.offset() as usize,
            point_data: buffer.get_point_range_mut(0..buffer.len()),
            size_of_attribute: attribute_member.size() as usize,
            stride,
        }
    }

    /// Creates a `RawAttributeView` for the given `attribute_definition` from a columnar point buffer
    pub(crate) fn from_columnar_buffer<B: ColumnarBufferMut + ?Sized>(
        buffer: &'a mut B,
        attribute_definition: &PointAttributeDefinition,
    ) -> Self {
        Self {
            offset: 0,
            point_data: buffer.get_attribute_range_mut(attribute_definition, 0..buffer.len()),
            size_of_attribute: attribute_definition.size() as usize,
            stride: attribute_definition.size() as usize,
        }
    }
}

impl Index<usize> for RawAttributeViewMut<'_> {
    type Output = [u8];

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let start = self.offset + (self.stride * index);
        let end = start + self.size_of_attribute;
        &self.point_data[start..end]
    }
}

impl IndexMut<usize> for RawAttributeViewMut<'_> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let start = self.offset + (self.stride * index);
        let end = start + self.size_of_attribute;
        &mut self.point_data[start..end]
    }
}

impl<'a> Iterator for RawAttributeViewMut<'a> {
    type Item = &'a mut [u8];

    fn next(&mut self) -> Option<Self::Item> {
        let point_data = mem::take(&mut self.point_data);
        if let Some((first_point, rest)) = point_data.split_at_mut_checked(self.stride) {
            self.point_data = rest;

            let start = self.offset;
            let end = start + self.size_of_attribute;
            let attribute = &mut first_point[start..end];
            Some(attribute)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::{Rng, thread_rng};

    use crate::containers::{BorrowedBuffer, HashMapBuffer, VectorBuffer};
    use crate::layout::PointType;
    use crate::test_utils::*;

    #[test]
    fn attribute_view_from_interleaved() {
        const COUNT: usize = 64;
        let mut test_data: VectorBuffer = thread_rng()
            .sample_iter::<CustomPointTypeBig, _>(DefaultPointDistribution)
            .take(COUNT)
            .collect();

        let layout = CustomPointTypeBig::layout();

        for attribute in layout.attributes() {
            let mut buffer = vec![0; attribute.size() as usize];
            let raw_view = RawAttributeView::from_interleaved_buffer(&test_data, attribute);
            let data_from_iter_view: Vec<Vec<u8>> = raw_view.map(|a| a.to_vec()).collect();
            let raw_view_mut =
                RawAttributeViewMut::from_interleaved_buffer(&mut test_data, attribute);
            let data_mut_from_iter_view: Vec<Vec<u8>> = raw_view_mut.map(|a| a.to_vec()).collect();

            for point_idx in 0..COUNT {
                test_data.get_attribute(
                    attribute.attribute_definition(),
                    point_idx,
                    &mut buffer[..],
                );

                assert_eq!(data_from_iter_view[point_idx], buffer);
                assert_eq!(data_mut_from_iter_view[point_idx], buffer);

                // Creating the RawAttributeViewMut in the inner loop because otherwise we couldn't call `get_attribute`
                // on `test_data` since RawAttributeViewMut mutably borrows the buffer

                let raw_view = RawAttributeView::from_interleaved_buffer(&test_data, attribute);
                let data_from_view = &raw_view[point_idx];
                assert_eq!(buffer.as_slice(), data_from_view);

                let mut raw_view_mut =
                    RawAttributeViewMut::from_interleaved_buffer(&mut test_data, attribute);
                let data_mut_from_view = &mut raw_view_mut[point_idx];
                assert_eq!(buffer.as_slice(), data_mut_from_view);
            }
        }
    }

    #[test]
    fn attribute_view_from_columnar() {
        const COUNT: usize = 64;
        let mut test_data: HashMapBuffer = thread_rng()
            .sample_iter::<CustomPointTypeBig, _>(DefaultPointDistribution)
            .take(COUNT)
            .collect();

        let layout = CustomPointTypeBig::layout();

        for attribute in layout.attributes() {
            let mut buffer = vec![0; attribute.size() as usize];
            let raw_view = RawAttributeView::from_columnar_buffer(
                &test_data,
                attribute.attribute_definition(),
            );
            let data_from_iter_view: Vec<Vec<u8>> = raw_view.map(|a| a.to_vec()).collect();
            let raw_view_mut = RawAttributeViewMut::from_columnar_buffer(
                &mut test_data,
                attribute.attribute_definition(),
            );
            let data_mut_from_iter_view: Vec<Vec<u8>> = raw_view_mut.map(|a| a.to_vec()).collect();

            for point_idx in 0..COUNT {
                test_data.get_attribute(
                    attribute.attribute_definition(),
                    point_idx,
                    &mut buffer[..],
                );

                assert_eq!(data_from_iter_view[point_idx], buffer);
                assert_eq!(data_mut_from_iter_view[point_idx], buffer);

                // Creating the RawAttributeViewMut in the inner loop because otherwise we couldn't call `get_attribute`
                // on `test_data` since RawAttributeViewMut mutably borrows the buffer

                let raw_view = RawAttributeView::from_columnar_buffer(
                    &test_data,
                    attribute.attribute_definition(),
                );
                let data_from_view = &raw_view[point_idx];
                assert_eq!(buffer.as_slice(), data_from_view);

                let mut raw_view_mut = RawAttributeViewMut::from_columnar_buffer(
                    &mut test_data,
                    attribute.attribute_definition(),
                );
                let data_mut_from_view = &mut raw_view_mut[point_idx];
                assert_eq!(buffer.as_slice(), data_mut_from_view);
            }
        }
    }
}
