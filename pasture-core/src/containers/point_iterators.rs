pub mod iterators {

    //! Contains `Iterator` implementations through which the untyped contents of `PointBuffer` structures
    //! can be accessed in a safe and strongly-typed manner.

    use crate::{
        containers::{InterleavedPointBuffer, InterleavedPointBufferMut, PointBuffer},
        layout::PointType,
    };

    use std::marker::PhantomData;
    use std::mem::MaybeUninit;

    /// Iterator over an arbitrary `PointBuffer` that yields strongly typed points by value
    pub struct PointIteratorByValue<'a, T: PointType, B: PointBuffer + ?Sized> {
        buffer: &'a B,
        current_index: usize,
        unused: PhantomData<T>,
    }

    impl<'a, T: PointType, B: PointBuffer + ?Sized> PointIteratorByValue<'a, T, B> {
        /// Creates a new `DefaultPointIterator` over all points in the given `PointBuffer`
        pub fn new(buffer: &'a B) -> Self {
            if *buffer.point_layout() != T::layout() {
                panic!("PointLayout of type T does not match PointLayout of buffer (buffer layout: {}, T layout: {})", buffer.point_layout(), T::layout());
            }
            Self {
                buffer,
                current_index: 0,
                unused: Default::default(),
            }
        }
    }

    impl<'a, T: PointType, B: PointBuffer + ?Sized> Iterator for PointIteratorByValue<'a, T, B> {
        type Item = T;

        fn next(&mut self) -> Option<Self::Item> {
            if self.current_index == self.buffer.len() {
                return None;
            }

            // Create an uninitialized T which is filled by the call to `buffer.get_raw_point`
            let mut point = MaybeUninit::<T>::uninit();
            unsafe {
                let point_byte_slice = std::slice::from_raw_parts_mut(
                    point.as_mut_ptr() as *mut u8,
                    std::mem::size_of::<T>(),
                );
                self.buffer
                    .get_raw_point(self.current_index, point_byte_slice);
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
            if *buffer.point_layout() != T::layout() {
                panic!("PointLayout of type T does not match PointLayout of buffer (buffer layout: {}, T layout: {})", buffer.point_layout(), T::layout());
            }
            let buffer_len = buffer.len();
            let point_data = unsafe {
                std::slice::from_raw_parts(
                    buffer.get_raw_points_ref(0..buffer_len).as_ptr() as *const T,
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
            if *buffer.point_layout() != T::layout() {
                panic!("PointLayout of type T does not match PointLayout of buffer (buffer layout: {}, T layout: {})", buffer.point_layout(), T::layout());
            }
            let buffer_len = buffer.len();
            let point_data = unsafe {
                std::slice::from_raw_parts_mut(
                    buffer.get_raw_points_mut(0..buffer_len).as_mut_ptr() as *mut T,
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

#[cfg(test)]
mod tests {

    use crate::containers::{
        InterleavedPointBufferExt, InterleavedPointBufferMutExt, InterleavedVecPointStorage,
        PerAttributeVecPointStorage, PointBufferExt,
    };
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
            let points_by_mut_view = storage.iter_point_mut::<TestPointType>();
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
            let points_by_val_collected = storage.iter_point::<TestPointType>().collect::<Vec<_>>();
            assert_eq!(modified_points, points_by_val_collected);
        }

        {
            let points_by_ref_view = storage.iter_point_ref::<TestPointType>();
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

        let collected_points = storage.iter_point::<TestPointType>().collect::<Vec<_>>();

        assert_eq!(reference_points, collected_points);
    }
}
