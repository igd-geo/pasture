use crate::layout::PointType;

use super::{InterleavedPointBuffer, InterleavedPointBufferMut, PointBuffer};

mod iterators {

    //! Contains `Iterator` implementations through which the untyped contents of `PointBuffer` structures
    //! can be accessed in a safe and strongly-typed manner.

    use crate::{
        containers::{InterleavedPointBuffer, InterleavedPointBufferMut, PointBuffer},
        layout::PointType,
    };

    use std::marker::PhantomData;
    use std::mem::MaybeUninit;

    /// Iterator over an arbitrary `PointBuffer` that yields strongly typed points by value
    pub struct PointIteratorByValue<'a, T: PointType> {
        buffer: &'a dyn PointBuffer,
        current_index: usize,
        unused: PhantomData<T>,
    }

    impl<'a, T: PointType> PointIteratorByValue<'a, T> {
        /// Creates a new `DefaultPointIterator` over all points in the given `PointBuffer`
        pub fn new(buffer: &'a dyn PointBuffer) -> Self {
            Self {
                buffer,
                current_index: 0,
                unused: Default::default(),
            }
        }
    }

    impl<'a, T: PointType> Iterator for PointIteratorByValue<'a, T> {
        type Item = T;

        fn next(&mut self) -> Option<Self::Item> {
            if self.current_index == self.buffer.len() {
                return None;
            }

            // Create an uninitialized T which is filled by the call to `buffer.get_point_by_copy`
            let mut point = MaybeUninit::<T>::uninit();
            unsafe {
                let point_byte_slice = std::slice::from_raw_parts_mut(
                    point.as_mut_ptr() as *mut u8,
                    std::mem::size_of::<T>(),
                );
                self.buffer
                    .get_point_by_copy(self.current_index, point_byte_slice);
            }

            self.current_index += 1;

            unsafe { Some(point.assume_init()) }
        }
    }

    /// Iterator over an interleaved `PointBuffer` that yields strongly typed points by reference
    pub struct PointIteratorByRef<'a, T: PointType + 'a> {
        point_data: &'a [T],
        current_index: usize,
    }

    impl<'a, T: PointType + 'a> PointIteratorByRef<'a, T> {
        /// Creates a new `InterleavedPointIterator` over all points in the given `PointBuffer`
        pub fn new<B: InterleavedPointBuffer + ?Sized>(buffer: &'a B) -> Self {
            let buffer_len = buffer.len();
            let point_data = unsafe {
                std::slice::from_raw_parts(
                    buffer.get_points_ref(0..buffer_len).as_ptr() as *const T,
                    buffer_len,
                )
            };
            Self {
                point_data,
                current_index: 0,
            }
        }
    }

    impl<'a, T: PointType + 'a> Iterator for PointIteratorByRef<'a, T> {
        type Item = &'a T;
        fn next(&mut self) -> Option<Self::Item> {
            if self.current_index == self.point_data.len() {
                return None;
            }

            let point = &self.point_data[self.current_index];
            self.current_index += 1;
            Some(point)
        }
    }

    /// Iterator over a `PointBuffer` that yields strongly typed points by mutable reference
    pub struct PointIteratorByMut<'a, T: PointType + 'a> {
        point_data: &'a mut [T],
        current_index: usize,
    }

    impl<'a, T: PointType + 'a> PointIteratorByMut<'a, T> {
        /// Creates a new `PointIteratorByMut` that iterates over the points in the given buffer
        pub fn new<B: InterleavedPointBufferMut + ?Sized>(buffer: &'a mut B) -> Self {
            let buffer_len = buffer.len();
            let point_data = unsafe {
                std::slice::from_raw_parts_mut(
                    buffer.get_points_mut(0..buffer_len).as_mut_ptr() as *mut T,
                    buffer_len,
                )
            };
            Self {
                point_data,
                current_index: 0,
            }
        }
    }

    impl<'a, T: PointType + 'a> Iterator for PointIteratorByMut<'a, T> {
        type Item = &'a mut T;

        fn next(&mut self) -> Option<Self::Item> {
            if self.current_index == self.point_data.len() {
                return None;
            }

            // Seems like an iterator returning mutable references only works with unsafe code :(

            unsafe {
                let ptr = self
                    .point_data
                    .as_mut_ptr()
                    .offset(self.current_index as isize);
                self.current_index += 1;
                Some(&mut *ptr)
            }
        }
    }
}

// TODO points() should be more powerful. It should be able to return the points in any type T that is convertible from
// the underlying PointLayout of the buffer

/// Returns an iterator over all points within the given PointBuffer, strongly typed to the PointType T. Works with any type
/// that implements the `PointBuffer` trait, but returns points by value, potentially copying them. If you want to iterate over
/// (mutable) references, use the `points_ref`/`points_mut` functions, which require an `InterleavedPointBuffer`.
///
/// # Panics
///
/// Panics if the `PointLayout` of `buffer` does not match the default `PointLayout` of type `T`
pub fn points<'a, T: PointType + 'a>(buffer: &'a dyn PointBuffer) -> impl Iterator<Item = T> + 'a {
    let point_layout = T::layout();
    if point_layout != *buffer.point_layout() {
        panic!(
            "points: PointLayouts do not match (type T has layout {}, buffer has layout {})",
            point_layout,
            buffer.point_layout()
        );
    }

    iterators::PointIteratorByValue::new(buffer)
}

/// Returns an iterator over references to all points within the given `PointBuffer`, strongly typed to the `PointType` `T`.
///
/// # Panics
///
/// Panics if the `PointLayout` of `buffer` does not match the default `PointLayout` of type `T`
pub fn points_ref<'a, T: PointType + 'a, B: InterleavedPointBuffer + ?Sized>(
    buffer: &'a B,
) -> iterators::PointIteratorByRef<'a, T> {
    let point_layout = T::layout();
    if point_layout != *buffer.point_layout() {
        panic!(
            "points_ref: PointLayouts do not match (type T has layout {}, buffer has layout {})",
            point_layout,
            buffer.point_layout()
        );
    }

    iterators::PointIteratorByRef::new(buffer)
}

/// Returns an iterator over mutable references to all points within the given `PointBuffer`, strongly typed to the `PointType` `T`.
///
/// # Panics
///
/// Panics if the `PointLayout` of `buffer` does not match the default `PointLayout` of type `T`
pub fn points_mut<'a, T: PointType + 'a, B: InterleavedPointBufferMut + ?Sized>(
    buffer: &'a mut B,
) -> iterators::PointIteratorByMut<'a, T> {
    let point_layout = T::layout();
    if point_layout != *buffer.point_layout() {
        panic!(
            "points_mut: PointLayouts do not match (type T has layout {}, buffer has layout {})",
            point_layout,
            buffer.point_layout()
        );
    }

    iterators::PointIteratorByMut::new(buffer)
}

#[cfg(test)]
mod tests {

    use super::*;

    use crate::containers::{InterleavedVecPointStorage, PerAttributeVecPointStorage};
    use pasture_derive::PointType;

    // We need this, otherwise we can't use the derive(PointType) macro from within pasture_core because the macro
    // doesn't recognize the name 'pasture_core' :/
    use crate as pasture_core;

    #[derive(Debug, Copy, Clone, PartialEq, PointType)]
    #[repr(C)]
    struct TestPointType {
        #[pasture(BUILTIN_INTENSITY)]
        pub intensity: u16,
        #[pasture(BUILTIN_GPS_TIME)]
        pub gps_time: f64,
    }

    #[test]
    fn test_points_iterator_from_interleaved() {
        let reference_points = vec![
            TestPointType {
                intensity: 42,
                gps_time: 0.123,
            },
            TestPointType {
                intensity: 43,
                gps_time: 0.456,
            },
        ];
        let mut storage = InterleavedVecPointStorage::from(reference_points.as_slice());

        {
            let points_by_mut_view = points_mut::<TestPointType, _>(&mut storage);
            points_by_mut_view.for_each(|point| {
                point.intensity *= 2;
                point.gps_time += 1.0;
            });
        }

        let modified_points = vec![
            TestPointType {
                intensity: 84,
                gps_time: 1.123,
            },
            TestPointType {
                intensity: 86,
                gps_time: 1.456,
            },
        ];

        {
            let points_by_val_view = points::<TestPointType>(&storage);
            let points_by_val_collected = points_by_val_view.collect::<Vec<_>>();
            assert_eq!(modified_points, points_by_val_collected);
        }

        {
            let points_by_ref_view = points_ref::<TestPointType, _>(&storage);
            let points_by_ref_collected = points_by_ref_view.map(|r| *r).collect::<Vec<_>>();
            assert_eq!(modified_points, points_by_ref_collected);
        }
    }

    #[test]
    fn test_points_iterator_from_per_attribute() {
        let reference_points = vec![
            TestPointType {
                intensity: 42,
                gps_time: 0.123,
            },
            TestPointType {
                intensity: 43,
                gps_time: 0.456,
            },
        ];
        let storage = PerAttributeVecPointStorage::from(reference_points.as_slice());

        let collected_points = points::<TestPointType>(&storage).collect::<Vec<_>>();

        assert_eq!(reference_points, collected_points);
    }
}
