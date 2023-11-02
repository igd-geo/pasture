use std::ops::{Index, IndexMut};

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
    pub(crate) fn from_interleaved_buffer<'b, B: InterleavedBuffer<'b> + ?Sized>(
        buffer: &'a B,
        attribute_member: &PointAttributeMember,
    ) -> Self
    where
        'b: 'a,
    {
        let stride = buffer.point_layout().size_of_point_entry() as usize;
        Self {
            offset: attribute_member.offset() as usize,
            point_data: buffer.get_point_range_ref(0..buffer.len()),
            size_of_attribute: attribute_member.size() as usize,
            stride,
        }
    }

    /// Creates a `RawAttributeView` for the given `attribute_definition` from a columnar point buffer
    pub(crate) fn from_columnar_buffer<'b, B: ColumnarBuffer<'b> + ?Sized>(
        buffer: &'a B,
        attribute_definition: &PointAttributeDefinition,
    ) -> Self
    where
        'b: 'a,
    {
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

impl<'a> Index<usize> for RawAttributeView<'a> {
    type Output = [u8];

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let start = self.offset + (self.stride * index);
        let end = start + self.size_of_attribute;
        &self.point_data[start..end]
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
    pub(crate) fn from_interleaved_buffer<'b, B: InterleavedBufferMut<'b> + ?Sized>(
        buffer: &'a mut B,
        attribute_member: &PointAttributeMember,
    ) -> Self
    where
        'b: 'a,
    {
        let stride = buffer.point_layout().size_of_point_entry() as usize;
        Self {
            offset: attribute_member.offset() as usize,
            point_data: buffer.get_point_range_mut(0..buffer.len()),
            size_of_attribute: attribute_member.size() as usize,
            stride,
        }
    }

    /// Creates a `RawAttributeView` for the given `attribute_definition` from a columnar point buffer
    pub(crate) fn from_columnar_buffer<'b, B: ColumnarBufferMut<'b> + ?Sized>(
        buffer: &'a mut B,
        attribute_definition: &PointAttributeDefinition,
    ) -> Self
    where
        'b: 'a,
    {
        Self {
            offset: 0,
            point_data: buffer.get_attribute_range_mut(attribute_definition, 0..buffer.len()),
            size_of_attribute: attribute_definition.size() as usize,
            stride: attribute_definition.size() as usize,
        }
    }
}

impl<'a> Index<usize> for RawAttributeViewMut<'a> {
    type Output = [u8];

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let start = self.offset + (self.stride * index);
        let end = start + self.size_of_attribute;
        &self.point_data[start..end]
    }
}

impl<'a> IndexMut<usize> for RawAttributeViewMut<'a> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let start = self.offset + (self.stride * index);
        let end = start + self.size_of_attribute;
        &mut self.point_data[start..end]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::{thread_rng, Rng};

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

            for point_idx in 0..COUNT {
                test_data.get_attribute(
                    attribute.attribute_definition(),
                    point_idx,
                    &mut buffer[..],
                );

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

            for point_idx in 0..COUNT {
                test_data.get_attribute(
                    attribute.attribute_definition(),
                    point_idx,
                    &mut buffer[..],
                );

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
