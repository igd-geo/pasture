use std::marker::PhantomData;

use crate::layout::PointType;

use super::point_buffer::{BorrowedBuffer, InterleavedBuffer, InterleavedBufferMut};

pub struct PointIteratorByValue<'a, 'b, T: PointType, B: BorrowedBuffer<'a>>
where
    'a: 'b,
{
    buffer: &'b B,
    current_index: usize,
    _phantom: PhantomData<&'a T>,
}

impl<'a, 'b, T: PointType, B: BorrowedBuffer<'a>> From<&'b B>
    for PointIteratorByValue<'a, 'b, T, B>
{
    fn from(value: &'b B) -> Self {
        Self {
            buffer: value,
            current_index: 0,
            _phantom: Default::default(),
        }
    }
}

impl<'a, 'b, T: PointType, B: BorrowedBuffer<'a>> Iterator for PointIteratorByValue<'a, 'b, T, B> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index == self.buffer.len() {
            None
        } else {
            let mut point = T::zeroed();
            let point_bytes = bytemuck::bytes_of_mut(&mut point);
            self.buffer.get_point(self.current_index, point_bytes);
            self.current_index += 1;
            Some(point)
        }
    }
}

pub struct PointIteratorByRef<'a, T: PointType> {
    point_data: &'a [T],
    current_index: usize,
}

impl<'a, 'b, T: PointType, B: InterleavedBuffer<'b>> From<&'a B> for PointIteratorByRef<'a, T>
where
    'b: 'a,
{
    fn from(value: &'a B) -> Self {
        let points_memory = value.get_point_range_ref(0..value.len());
        Self {
            point_data: bytemuck::cast_slice(points_memory),
            current_index: 0,
        }
    }
}

impl<'a, T: PointType> Iterator for PointIteratorByRef<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index == self.point_data.len() {
            None
        } else {
            let point = &self.point_data[self.current_index];
            self.current_index += 1;
            Some(point)
        }
    }
}

pub struct PointIteratorByMut<'a, T: PointType> {
    point_data: &'a mut [T],
    current_index: usize,
    _phantom: PhantomData<T>,
}

impl<'a, 'b, T: PointType, B: InterleavedBufferMut<'b>> From<&'a mut B>
    for PointIteratorByMut<'a, T>
where
    'b: 'a,
{
    fn from(value: &'a mut B) -> Self {
        let memory_for_all_points = value.get_point_range_mut(0..value.len());
        Self {
            point_data: bytemuck::cast_slice_mut(memory_for_all_points),
            current_index: 0,
            _phantom: Default::default(),
        }
    }
}

impl<'a, T: PointType> Iterator for PointIteratorByMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index == self.point_data.len() {
            None
        } else {
            unsafe {
                let memory = self.point_data.as_mut_ptr().add(self.current_index);
                self.current_index += 1;
                Some(&mut *memory)
            }
        }
    }
}
