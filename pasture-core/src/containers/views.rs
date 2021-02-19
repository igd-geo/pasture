use crate::containers::InterleavedPointBuffer;
use crate::containers::PerAttributePointBuffer;
use crate::containers::PointBuffer;
use crate::layout::{PointAttributeDefinition, PointType, PrimitiveType};

use super::{attr1, attr2, attr3, attr4, InterleavedPointBufferMut, PerAttributePointBufferMut};

mod iterators {

    //! Contains `Iterator` implementations through which the untyped contents of `PointBuffer` structures
    //! can be accessed in a safe and strongly-typed manner.

    use crate::containers::{InterleavedPointBufferMut, PerAttributePointBufferMut};

    use super::*;
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
        pub fn new(buffer: &'a dyn InterleavedPointBuffer) -> Self {
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
        pub fn new(buffer: &'a mut dyn InterleavedPointBufferMut) -> Self {
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

// TODO points() and attributes() should be macro calls. Inside a macro, we can dispatch on the actual type,
// so in cases where the user has a specific PointBuffer type and calls points(&buf), we can return a specific
// iterator implementation instead of a boxed iterator. This will then be much faster in these cases because it
// alleviates the need for virtual dispatch on every iteration step

/// Returns an iterator over all points within the given PointBuffer, strongly typed to the PointType T. Assumes no
/// internal memory representation for the source buffer, so returns an opaque iterator type that works with arbitrary
/// PointBuffer implementations. If you know the type of your PointBuffer, prefer one of the points_from_... variants
/// as they will yield better performance. Or simply use the points! macro, which selects the best matching candidate.
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

/// Returns an iterator over references to all points within the given PointBuffer, strongly typed to the PointType T.
pub fn points_ref<'a, T: PointType + 'a>(
    buffer: &'a dyn InterleavedPointBuffer,
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

/// Returns an iterator over mutable references to all points within the given PointBuffer, strongly typed to the PointType T.
pub fn points_mut<'a, T: PointType + 'a>(
    buffer: &'a mut dyn InterleavedPointBufferMut,
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

/// Returns an iterator over the specific attribute for all points within the given `PointBuffer`, strongly typed over the `PrimitiveType` `T`
pub fn attribute<'a, T: PrimitiveType + 'a>(
    buffer: &'a dyn PointBuffer,
    attribute: &'a PointAttributeDefinition,
) -> attr1::AttributeIteratorByValue<'a, T> {
    attr1::AttributeIteratorByValue::<T>::new(buffer, attribute)
}

/// Returns an iterator over references to the specific attribute for all points within the given `PointBuffer`, strongly typed over the `PrimitiveType` `T`
pub fn attribute_ref<'a, T: PrimitiveType + 'a>(
    buffer: &'a dyn PerAttributePointBuffer,
    attribute: &'a PointAttributeDefinition,
) -> attr1::AttributeIteratorByRef<'a, T> {
    attr1::AttributeIteratorByRef::<T>::new(buffer, attribute)
}

/// Returns an iterator over mutable references to the specific attribute for all points within the given `PointBuffer`, strongly typed over the `PrimitiveType` `T`
pub fn attribute_mut<'a, T: PrimitiveType + 'a>(
    buffer: &'a mut dyn PerAttributePointBufferMut,
    attribute: &'a PointAttributeDefinition,
) -> attr1::AttributeIteratorByMut<'a, T> {
    attr1::AttributeIteratorByMut::<T>::new(buffer, attribute)
}

macro_rules! attributes {
    ($t1:ty, $t2:ty, $buffer:expr, $attr1:expr, $attr2:expr,) => {
        attr2::AttributeIteratorByValue::<$t1, $t2>::new($buffer, [$attr1, $attr2])
    };
    ($t1:ty, $t2:ty, $t3:ty, $buffer:expr, $attr1:expr, $attr2:expr, $attr3:expr,) => {
        attr3::AttributeIteratorByValue::<$t1, $t2, $t3>::new($buffer, [$attr1, $attr2, $attr3])
    };
    ($t1:ty, $t2:ty, $t3:ty, $t4:ty, $buffer:expr, $attr1:expr, $attr2:expr, $attr3:expr, $attr4:expr,) => {
        attr3::AttributeIteratorByValue::<$t1, $t2, $t3, $t4>::new(
            $buffer,
            [$attr1, $attr2, $attr3, $attr4],
        )
    };
}

macro_rules! attributes_ref {
    ($t1:ty, $t2:ty, $buffer:expr, $attr1:expr, $attr2:expr,) => {
        attr2::AttributeIteratorByRef::<$t1, $t2>::new($buffer, [$attr1, $attr2])
    };
    ($t1:ty, $t2:ty, $t3:ty, $buffer:expr, $attr1:expr, $attr2:expr, $attr3:expr,) => {
        attr3::AttributeIteratorByRef::<$t1, $t2, $t3>::new($buffer, [$attr1, $attr2, $attr3])
    };
    ($t1:ty, $t2:ty, $t3:ty, $t4:ty, $buffer:expr, $attr1:expr, $attr2:expr, $attr3:expr, $attr4:expr,) => {
        attr3::AttributeIteratorByRef::<$t1, $t2, $t3, $t4>::new(
            $buffer,
            [$attr1, $attr2, $attr3, $attr4],
        )
    };
}

macro_rules! attributes_mut {
    ($t1:ty, $t2:ty, $buffer:expr, $attr1:expr, $attr2:expr,) => {
        attr2::AttributeIteratorByMut::<$t1, $t2>::new($buffer, [$attr1, $attr2])
    };
    ($t1:ty, $t2:ty, $t3:ty, $buffer:expr, $attr1:expr, $attr2:expr, $attr3:expr,) => {
        attr3::AttributeIteratorByMut::<$t1, $t2, $t3>::new($buffer, [$attr1, $attr2, $attr3])
    };
    ($t1:ty, $t2:ty, $t3:ty, $t4:ty, $buffer:expr, $attr1:expr, $attr2:expr, $attr3:expr, $attr4:expr,) => {
        attr3::AttributeIteratorByMut::<$t1, $t2, $t3, $t4>::new(
            $buffer,
            [$attr1, $attr2, $attr3, $attr4],
        )
    };
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::layout::attributes;
    use crate::{
        containers::{InterleavedVecPointStorage, PerAttributeVecPointStorage},
        layout::attributes::POSITION_3D,
    };
    use nalgebra::Vector3;
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
    fn test_points_view_from_interleaved() {
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
        let mut storage = InterleavedVecPointStorage::new(TestPointType::layout());
        storage.push_point(reference_points[0]);
        storage.push_point(reference_points[1]);

        {
            let points_by_mut_view = points_mut::<TestPointType>(&mut storage);
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
            let points_by_ref_view = points_ref::<TestPointType>(&storage);
            let points_by_ref_collected = points_by_ref_view.map(|r| *r).collect::<Vec<_>>();
            assert_eq!(modified_points, points_by_ref_collected);
        }
    }

    #[test]
    fn test_single_attribute_view_from_per_attribute() {
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
        let mut storage = PerAttributeVecPointStorage::new(TestPointType::layout());
        storage.push_point(reference_points[0]);
        storage.push_point(reference_points[1]);

        {
            let first_attribute_mut_view =
                attribute_mut::<u16>(&mut storage, &attributes::INTENSITY);
            first_attribute_mut_view.for_each(|a| {
                *a *= 2;
            });
        }

        let modified_intensities = vec![84_u16, 86_u16];

        {
            let attribute_by_val_view = attribute::<u16>(&storage, &attributes::INTENSITY);
            let attribute_by_val_collected = attribute_by_val_view.collect::<Vec<_>>();
            assert_eq!(modified_intensities, attribute_by_val_collected);
        }

        {
            let attribute_by_ref_view = attribute_ref::<u16>(&storage, &attributes::INTENSITY);
            let attribute_by_ref_collected = attribute_by_ref_view.map(|a| *a).collect::<Vec<_>>();
            assert_eq!(modified_intensities, attribute_by_ref_collected);
        }
    }

    #[test]
    fn test_two_attributes_view_from_per_attribute() {
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
        let mut storage = PerAttributeVecPointStorage::new(TestPointType::layout());
        storage.push_point(reference_points[0]);
        storage.push_point(reference_points[1]);

        {
            let attributes_mut_view = attributes_mut!(
                u16,
                f64,
                &mut storage,
                &attributes::INTENSITY,
                &attributes::GPS_TIME,
            );
            attributes_mut_view.for_each(|(intensity, gps_time)| {
                *intensity *= 2;
                *gps_time += 1.0;
            });
        }

        let modified_data = vec![(84_u16, 1.123), (86_u16, 1.456)];

        {
            let attributes_by_val_view = attributes!(
                u16,
                f64,
                &storage,
                &attributes::INTENSITY,
                &attributes::GPS_TIME,
            );
            let attributes_by_val_collected = attributes_by_val_view.collect::<Vec<_>>();
            assert_eq!(modified_data, attributes_by_val_collected);
        }

        {
            let attributes_by_ref_view = attributes_ref!(
                u16,
                f64,
                &storage,
                &attributes::INTENSITY,
                &attributes::GPS_TIME,
            );
            let attributes_by_ref_collected = attributes_by_ref_view
                .map(|(&intensity, &gps_time)| (intensity, gps_time))
                .collect::<Vec<_>>();
            assert_eq!(modified_data, attributes_by_ref_collected);
        }
    }

    #[test]
    #[should_panic(expected = "not contained in PointLayout of buffer")]
    fn test_attributes_with_different_datatype_fails() {
        #[derive(Debug, Copy, Clone, PartialEq, PointType)]
        #[repr(C)]
        struct PositionLowp(#[pasture(BUILTIN_POSITION_3D)] Vector3<f32>);

        let mut storage = InterleavedVecPointStorage::new(PositionLowp::layout());
        storage.push_point(PositionLowp(Default::default()));

        attribute::<Vector3<f64>>(&storage, &POSITION_3D).for_each(drop);
    }

    #[test]
    #[should_panic(expected = "not contained in PointLayout of buffer")]
    fn test_attributes_ref_with_different_datatype_fails() {
        #[derive(Debug, Copy, Clone, PartialEq, PointType)]
        #[repr(C)]
        struct PositionLowp(#[pasture(BUILTIN_POSITION_3D)] Vector3<f32>);

        let mut storage = PerAttributeVecPointStorage::new(PositionLowp::layout());
        storage.push_point(PositionLowp(Default::default()));

        attribute_ref::<Vector3<f64>>(&storage, &POSITION_3D).for_each(drop);
    }

    #[test]
    #[should_panic(expected = "not contained in PointLayout of buffer")]
    fn test_attributes_mut_with_different_datatype_fails() {
        #[derive(Debug, Copy, Clone, PartialEq, PointType)]
        #[repr(C)]
        struct PositionLowp(#[pasture(BUILTIN_POSITION_3D)] Vector3<f32>);

        let mut storage = PerAttributeVecPointStorage::new(PositionLowp::layout());
        storage.push_point(PositionLowp(Default::default()));

        attribute_mut::<Vector3<f64>>(&mut storage, &POSITION_3D).for_each(drop);
    }

    // #[test]
    // fn test_points_with_interleaved_point_storage() {
    //     let mut storage = InterleavedVecPointStorage::new(TestPointType::layout());
    //     storage.push_point(TestPointType(42, 0.123));
    //     storage.push_point(TestPointType(43, 0.345));

    //     let points_view = points::<TestPointType>(&storage);

    //     let points_collected = points_view.collect::<Vec<_>>();

    //     assert_eq!(2, points_collected.len());
    //     assert_eq!(42, { points_collected[0].0 });
    //     assert_eq!(0.123, { points_collected[0].1 });
    //     assert_eq!(43, { points_collected[1].0 });
    //     assert_eq!(0.345, { points_collected[1].1 });
    // }

    // #[test]
    // fn test_points_with_per_attribute_point_storage() {
    //     let mut storage = PerAttributeVecPointStorage::new(TestPointType::layout());
    //     storage.push_point(TestPointType(42, 0.123));
    //     storage.push_point(TestPointType(43, 0.345));

    //     let points_view = points::<TestPointType>(&storage);

    //     let points_collected = points_view.collect::<Vec<_>>();

    //     assert_eq!(2, points_collected.len());
    //     assert_eq!(42, { points_collected[0].0 });
    //     assert_eq!(0.123, { points_collected[0].1 });
    //     assert_eq!(43, { points_collected[1].0 });
    //     assert_eq!(0.345, { points_collected[1].1 });
    // }

    // #[test]
    // fn test_attributes_from_per_attribute_buffer() {
    //     let mut storage = PerAttributeVecPointStorage::new(TestPointType::layout());
    //     storage.push_point(TestPointType(42, 0.123));
    //     storage.push_point(TestPointType(43, 0.345));

    //     let attributes_view =
    //         attributes_from_per_attribute_buffer::<u16>(&storage, &attributes::INTENSITY);

    //     let attributes_collected = attributes_view.collect::<Vec<_>>();

    //     assert_eq!(2, attributes_collected.len());
    //     assert_eq!(42, *attributes_collected[0]);
    //     assert_eq!(43, *attributes_collected[1]);
    // }

    // #[test]
    // fn test_attributes_with_interleaved_point_storage() {
    //     let mut storage = InterleavedVecPointStorage::new(TestPointType::layout());
    //     storage.push_point(TestPointType(42, 0.123));
    //     storage.push_point(TestPointType(43, 0.345));

    //     let attributes_view = attributes::<u16>(&storage, &attributes::INTENSITY);

    //     let attributes_collected = attributes_view.collect::<Vec<_>>();

    //     assert_eq!(2, attributes_collected.len());
    //     assert_eq!(42, attributes_collected[0]);
    //     assert_eq!(43, attributes_collected[1]);
    // }

    // #[test]
    // fn test_attributes_with_per_attribute_point_storage() {
    //     let mut storage = PerAttributeVecPointStorage::new(TestPointType::layout());
    //     storage.push_point(TestPointType(42, 0.123));
    //     storage.push_point(TestPointType(43, 0.345));

    //     let attributes_view = attributes::<u16>(&storage, &attributes::INTENSITY);

    //     let attributes_collected = attributes_view.collect::<Vec<_>>();

    //     assert_eq!(2, attributes_collected.len());
    //     assert_eq!(42, attributes_collected[0]);
    //     assert_eq!(43, attributes_collected[1]);
    // }

    // #[test]
    // fn test_attributes_mut_with_per_attribute_point_storage() {
    //     let mut storage = PerAttributeVecPointStorage::new(TestPointType::layout());
    //     storage.push_point(TestPointType(42, 0.123));
    //     storage.push_point(TestPointType(43, 0.345));

    //     {
    //         let attributes_mut_view = attributes_mut_from_per_attribute_buffer::<u16>(
    //             &mut storage,
    //             &attributes::INTENSITY,
    //         );
    //         for (idx, attribute) in attributes_mut_view.enumerate() {
    //             *attribute = idx as u16;
    //         }
    //     }

    //     let first_attribute_range = storage.get_attribute_range_ref(0..2, &attributes::INTENSITY);
    //     let first_attribute_range_typed =
    //         unsafe { std::slice::from_raw_parts(first_attribute_range.as_ptr() as *const u16, 2) };

    //     assert_eq!(0, first_attribute_range_typed[0]);
    //     assert_eq!(1, first_attribute_range_typed[1]);
    // }
}
