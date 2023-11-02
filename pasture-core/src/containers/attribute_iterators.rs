use std::marker::PhantomData;

use crate::layout::{PointAttributeDefinition, PointAttributeMember, PrimitiveType};

use super::point_buffer::{BorrowedBuffer, ColumnarBuffer, ColumnarBufferMut};

/// An iterator over strongly typed attribute data in a point buffer. Returns attribute data
/// by value and makes assumptions about the memory layout of the underlying buffer
pub struct AttributeIteratorByValue<'a, 'b, T: PrimitiveType, B: BorrowedBuffer<'a>>
where
    'a: 'b,
{
    buffer: &'b B,
    attribute_member: &'b PointAttributeMember,
    current_index: usize,
    _phantom: PhantomData<&'a T>,
}

impl<'a, 'b, T: PrimitiveType, B: BorrowedBuffer<'a>> AttributeIteratorByValue<'a, 'b, T, B> {
    pub(crate) fn new(buffer: &'b B, attribute: &PointAttributeDefinition) -> Self {
        Self {
            attribute_member: buffer
                .point_layout()
                .get_attribute(attribute)
                .expect("Attribute not found in PointLayout of buffer"),
            buffer,
            current_index: 0,
            _phantom: Default::default(),
        }
    }
}

impl<'a, 'b, T: PrimitiveType, B: BorrowedBuffer<'a>> Iterator
    for AttributeIteratorByValue<'a, 'b, T, B>
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index == self.buffer.len() {
            None
        } else {
            let mut attribute = T::zeroed();
            let attribute_bytes = bytemuck::bytes_of_mut(&mut attribute);
            // This is safe because in `new` we obtain the `attribute_member` from the point layout of the buffer
            unsafe {
                self.buffer.get_attribute_unchecked(
                    self.attribute_member,
                    self.current_index,
                    attribute_bytes,
                );
            }
            self.current_index += 1;
            Some(attribute)
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.current_index += n;
        self.next()
    }
}

/// Like [`AttributeIteratorByValue`], but returns attribute data by immutable reference. Can only be
/// constructed from a buffer that implements [`ColumnarBuffer`]
pub struct AttributeIteratorByRef<'a, T: PrimitiveType> {
    attribute_data: &'a [T],
    current_index: usize,
}

impl<'a, T: PrimitiveType> AttributeIteratorByRef<'a, T> {
    pub(crate) fn new<'b, B: ColumnarBuffer<'b>>(
        buffer: &'a B,
        attribute: &PointAttributeDefinition,
    ) -> Self
    where
        'b: 'a,
    {
        let attribute_memory = buffer.get_attribute_range_ref(attribute, 0..buffer.len());
        Self {
            attribute_data: bytemuck::cast_slice(attribute_memory),
            current_index: 0,
        }
    }
}

impl<'a, T: PrimitiveType> Iterator for AttributeIteratorByRef<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index == self.attribute_data.len() {
            None
        } else {
            let ret = &self.attribute_data[self.current_index];
            self.current_index += 1;
            Some(ret)
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.current_index += n;
        self.next()
    }
}

/// Like [`AttributeIteratorByRef`], but returns attribute data by mutable reference, allowing mutation
/// of the attribute data in-place. Can only be constructed from a buffer that implements [`ColumnarBufferMut`]
pub struct AttributeIteratorByMut<'a, T: PrimitiveType> {
    attribute_data: &'a mut [T],
    current_index: usize,
}

impl<'a, T: PrimitiveType> AttributeIteratorByMut<'a, T> {
    pub(crate) fn new<'b, B: ColumnarBufferMut<'b>>(
        buffer: &'a mut B,
        attribute: &PointAttributeDefinition,
    ) -> Self
    where
        'b: 'a,
    {
        let attribute_memory = buffer.get_attribute_range_mut(attribute, 0..buffer.len());
        Self {
            attribute_data: bytemuck::cast_slice_mut(attribute_memory),
            current_index: 0,
        }
    }
}

impl<'a, T: PrimitiveType> Iterator for AttributeIteratorByMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index == self.attribute_data.len() {
            None
        } else {
            unsafe {
                let attribute_ptr = self.attribute_data.as_mut_ptr().add(self.current_index);
                self.current_index += 1;
                Some(&mut *attribute_ptr)
            }
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.current_index += n;
        self.next()
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;
    use rand::{thread_rng, Rng};

    use crate::{
        containers::{BorrowedMutBuffer, HashMapBuffer},
        layout::attributes::POSITION_3D,
        test_utils::{CustomPointTypeSmall, DefaultPointDistribution},
    };

    use super::*;

    #[test]
    #[allow(clippy::iter_nth_zero)]
    fn attribute_iterator_nth() {
        const COUNT: usize = 16;
        let mut points = thread_rng()
            .sample_iter::<CustomPointTypeSmall, _>(DefaultPointDistribution)
            .take(COUNT)
            .collect::<HashMapBuffer>();
        let mut points_clone = points.clone();

        let positions_view = points.view_attribute::<Vector3<f64>>(&POSITION_3D);
        {
            let mut positions_iter = positions_view.clone().into_iter();
            assert_eq!(positions_iter.nth(0), Some(positions_view.at(0)));
        }
        {
            let mut positions_iter = positions_view.clone().into_iter();
            assert_eq!(positions_iter.nth(6), Some(positions_view.at(6)));
        }
        {
            let mut positions_iter = positions_view.clone().into_iter();
            assert_eq!(positions_iter.nth(COUNT), None);
        }
        {
            let mut positions_iter = positions_view.clone().into_iter();
            positions_iter.nth(0);
            assert_eq!(positions_iter.nth(0), Some(positions_view.at(1)));
        }

        {
            let mut positions_iter = positions_view.iter();
            assert_eq!(positions_iter.nth(0), Some(positions_view.at_ref(0)));
        }
        {
            let mut positions_iter = positions_view.iter();
            assert_eq!(positions_iter.nth(6), Some(positions_view.at_ref(6)));
        }
        {
            let mut positions_iter = positions_view.iter();
            assert_eq!(positions_iter.nth(COUNT), None);
        }
        {
            let mut positions_iter = positions_view.iter();
            positions_iter.nth(0);
            assert_eq!(positions_iter.nth(0), Some(positions_view.at_ref(1)));
        }

        {
            let mut positions_view_mut = points.view_attribute_mut::<Vector3<f64>>(&POSITION_3D);
            let mut cloned_positions_view_mut =
                points_clone.view_attribute_mut::<Vector3<f64>>(&POSITION_3D);
            let mut positions_iter = positions_view_mut.iter_mut();
            assert_eq!(
                positions_iter.nth(0),
                Some(cloned_positions_view_mut.at_mut(0))
            );
        }
        {
            let mut positions_view_mut = points.view_attribute_mut::<Vector3<f64>>(&POSITION_3D);
            let mut cloned_positions_view_mut =
                points_clone.view_attribute_mut::<Vector3<f64>>(&POSITION_3D);
            let mut positions_iter = positions_view_mut.iter_mut();
            assert_eq!(
                positions_iter.nth(6),
                Some(cloned_positions_view_mut.at_mut(6))
            );
        }
        {
            let mut positions_view_mut = points.view_attribute_mut::<Vector3<f64>>(&POSITION_3D);
            let mut positions_iter = positions_view_mut.iter_mut();
            assert_eq!(positions_iter.nth(COUNT), None);
        }
        {
            let mut positions_view_mut = points.view_attribute_mut::<Vector3<f64>>(&POSITION_3D);
            let mut cloned_positions_view_mut =
                points_clone.view_attribute_mut::<Vector3<f64>>(&POSITION_3D);
            let mut positions_iter = positions_view_mut.iter_mut();
            positions_iter.nth(0);
            assert_eq!(
                positions_iter.nth(0),
                Some(cloned_positions_view_mut.at_mut(1))
            );
        }
    }
}
