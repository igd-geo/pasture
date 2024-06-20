use std::marker::PhantomData;

use crate::layout::PointType;

use super::point_buffer::{BorrowedBuffer, InterleavedBuffer, InterleavedBufferMut};

/// Iterator over strongly typed points by value
pub struct PointIteratorByValue<'a, 'b, T: PointType, B: BorrowedBuffer<'a> + ?Sized>
where
    'a: 'b,
{
    buffer: &'b B,
    current_index: usize,
    _phantom: PhantomData<&'a T>,
}

impl<'a, 'b, T: PointType, B: BorrowedBuffer<'a> + ?Sized> From<&'b B>
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

impl<'a, 'b, T: PointType, B: BorrowedBuffer<'a> + ?Sized> Iterator
    for PointIteratorByValue<'a, 'b, T, B>
{
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

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.current_index += n;
        self.next()
    }
}

/// Iterator over strongly typed points by immutable borrow
pub struct PointIteratorByRef<'a, T: PointType> {
    point_data: &'a [T],
    current_index: usize,
}

impl<'a, 'b, T: PointType, B: InterleavedBuffer<'b> + ?Sized> From<&'a B>
    for PointIteratorByRef<'a, T>
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

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.current_index += n;
        self.next()
    }
}

/// Iterator over strongly typed points by mutable borrow
pub struct PointIteratorByMut<'a, T: PointType> {
    point_data: &'a mut [T],
    current_index: usize,
    _phantom: PhantomData<T>,
}

impl<'a, 'b, T: PointType, B: InterleavedBufferMut<'b> + ?Sized> From<&'a mut B>
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

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.current_index += n;
        self.next()
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};

    use crate::{
        containers::{BorrowedBufferExt, BorrowedMutBufferExt, VectorBuffer},
        test_utils::{CustomPointTypeSmall, DefaultPointDistribution},
    };

    #[test]
    #[allow(clippy::iter_nth_zero)]
    fn point_iterator_nth() {
        const COUNT: usize = 16;
        let mut points = thread_rng()
            .sample_iter::<CustomPointTypeSmall, _>(DefaultPointDistribution)
            .take(COUNT)
            .collect::<VectorBuffer>();
        let mut points_clone = points.clone();

        let points_view = points.view::<CustomPointTypeSmall>();
        {
            let mut points_iter = points_view.clone().into_iter();
            assert_eq!(points_iter.nth(0), Some(points_view.at(0)));
        }
        {
            let mut points_iter = points_view.clone().into_iter();
            assert_eq!(points_iter.nth(6), Some(points_view.at(6)));
        }
        {
            let mut points_iter = points_view.clone().into_iter();
            assert_eq!(points_iter.nth(COUNT), None);
        }
        {
            let mut points_iter = points_view.clone().into_iter();
            points_iter.nth(0);
            assert_eq!(points_iter.nth(0), Some(points_view.at(1)));
        }

        {
            let mut points_iter = points_view.iter();
            assert_eq!(points_iter.nth(0), Some(points_view.at_ref(0)));
        }
        {
            let mut points_iter = points_view.iter();
            assert_eq!(points_iter.nth(6), Some(points_view.at_ref(6)));
        }
        {
            let mut points_iter = points_view.iter();
            assert_eq!(points_iter.nth(COUNT), None);
        }
        {
            let mut points_iter = points_view.iter();
            points_iter.nth(0);
            assert_eq!(points_iter.nth(0), Some(points_view.at_ref(1)));
        }

        let mut points_view_mut = points.view_mut::<CustomPointTypeSmall>();
        let mut cloned_points_view_mut = points_clone.view_mut::<CustomPointTypeSmall>();
        {
            let mut points_iter = points_view_mut.iter_mut();
            assert_eq!(points_iter.nth(0), Some(cloned_points_view_mut.at_mut(0)));
        }
        {
            let mut points_iter = points_view_mut.iter_mut();
            assert_eq!(points_iter.nth(6), Some(cloned_points_view_mut.at_mut(6)));
        }
        {
            let mut points_iter = points_view_mut.iter_mut();
            assert_eq!(points_iter.nth(COUNT), None);
        }
        {
            let mut points_iter = points_view_mut.iter_mut();
            points_iter.nth(0);
            assert_eq!(points_iter.nth(0), Some(cloned_points_view_mut.at_mut(1)));
        }
    }
}
