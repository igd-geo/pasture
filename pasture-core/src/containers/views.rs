use crate::containers::InterleavedPointBuffer;
use crate::containers::PerAttributePointBuffer;
use crate::containers::PointBuffer;
use crate::layout::{PointAttributeDefinition, PointType, PrimitiveType};

use super::{attr1, InterleavedPointBufferMut, PerAttributePointBufferMut};

mod iterators {

    //! Contains `Iterator` implementations through which the untyped contents of `PointBuffer` structures
    //! can be accessed in a safe and strongly-typed manner.

    use crate::containers::InterleavedPointBufferMut;

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

/// Returns an iterator over mutable references to all points within the given PointBuffer, strongly typed to the PointType T.
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

/// Returns an iterator over the specific attribute for all points within the given `PointBuffer`, strongly typed over the `PrimitiveType` `T`
pub fn attribute<'a, T: PrimitiveType + 'a>(
    buffer: &'a dyn PointBuffer,
    attribute: &'a PointAttributeDefinition,
) -> attr1::AttributeIteratorByValue<'a, T> {
    attr1::AttributeIteratorByValue::<T>::new(buffer, attribute)
}

/// Returns an iterator over the specific attribute for all points within the given `PointBuffer`, converted to the `PrimitiveType` `T`. Use this function
/// when the `buffer` stores the attribute with a different datatype than `T`
pub fn attribute_as<'a, T: PrimitiveType + 'a>(
    buffer: &'a dyn PointBuffer,
    attribute: &'a PointAttributeDefinition,
) -> attr1::AttributeIteratorByValueWithConversion<'a, T> {
    attr1::AttributeIteratorByValueWithConversion::<T>::new(buffer, attribute)
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

/// Create an iterator over multiple attributes within a `PointBuffer`. This macro uses some special syntax  to determine the attributes
/// and their types:
///
/// `attributes!{ ATTRIBUTE_1_EXPR => ATTRIBUTE_1_TYPE, ATTRIBUTE_2_EXPR => ATTRIBUTE_2_TYPE, ..., buffer }`
///
/// `ATTRIBUTE_X_EXPR` must be an expression that evaluates to a `&PointAttributeDefinition` and `ATTRIBUTE_X_TYPE` must be the Pasture
/// `PrimitiveType` that the attribute will be returned as. The type must match the type that the attribute is stored with inside `buffer`.
/// The iterator will then return tuples of the type:
///
/// `(ATTRIBUTE_1_TYPE, ATTRIBUTE_2_TYPE, ...)`
///
/// *Note:* Currently, a maximum of 4 attributes at the same time are supported.
#[macro_export]
macro_rules! attributes {
    ($attr1:expr => $t1:ty, $attr2:expr => $t2:ty, $buffer:expr) => {
        crate::containers::attr2::AttributeIteratorByValue::<$t1, $t2>::new(
            $buffer,
            [$attr1, $attr2],
        )
    };
    ($attr1:expr => $t1:ty, $attr2:expr => $t2:ty, $attr3:expr => $t3:ty, $buffer:expr) => {
        crate::containers::attr3::AttributeIteratorByValue::<$t1, $t2, $t3>::new(
            $buffer,
            [$attr1, $attr2, $attr3],
        )
    };
    ($attr1:expr => $t1:ty, $attr2:expr => $t2:ty, $attr3:expr => $t3:ty, $attr4:expr => $t4:ty, $buffer:expr) => {
        crate::containers::attr3::AttributeIteratorByValue::<$t1, $t2, $t3, $t4>::new(
            $buffer,
            [$attr1, $attr2, $attr3, $attr4],
        )
    };
}

/// Create an iterator over multiple attributes within a `PointBuffer`, supporting type converisons. This macro uses some special syntax
/// to determine the attributes and their types:
///
/// `attributes_as!{ ATTRIBUTE_1_EXPR => ATTRIBUTE_1_TYPE, ATTRIBUTE_2_EXPR => ATTRIBUTE_2_TYPE, ..., buffer }`
///
/// `ATTRIBUTE_X_EXPR` must be an expression that evaluates to a `&PointAttributeDefinition` and `ATTRIBUTE_X_TYPE` must be the Pasture
/// `PrimitiveType` that the attribute will be returned as. This type must be convertible from the actual type that the attribute
/// is stored with inside `buffer`. The iterator will then return tuples of the form:
///
/// `(ATTRIBUTE_1_TYPE, ATTRIBUTE_2_TYPE, ...)`
///
/// *Note:* Currently, a maximum of 4 attributes at the same time are supported.
#[macro_export]
macro_rules! attributes_as {
    ($attr1:expr => $t1:ty, $attr2:expr => $t2:ty, $buffer:expr) => {
        crate::containers::attr2::AttributeIteratorByValueWithConversion::<$t1, $t2>::new(
            $buffer,
            [$attr1, $attr2],
        )
    };
    ($attr1:expr => $t1:ty, $attr2:expr => $t2:ty, $attr3:expr => $t3:ty, $buffer:expr) => {
        crate::containers::attr3::AttributeIteratorByValueWithConversion::<$t1, $t2, $t3>::new(
            $buffer,
            [$attr1, $attr2, $attr3],
        )
    };
    ($attr1:expr => $t1:ty, $attr2:expr => $t2:ty, $attr3:expr => $t3:ty, $attr4:expr => $t4:ty, $buffer:expr) => {
        crate::containers::attr3::AttributeIteratorByValueWithConversion::<$t1, $t2, $t3, $t4>::new(
            $buffer,
            [$attr1, $attr2, $attr3, $attr4],
        )
    };
}

/// Create an iterator over references to multiple attributes within a `PointBuffer`. Requires that the buffer implements
/// `PerAttributePointBuffer`. This macro uses some special syntax to determine the attributes and their types:
///
/// `attributes_ref!{ ATTRIBUTE_1_EXPR => ATTRIBUTE_1_TYPE, ATTRIBUTE_2_EXPR => ATTRIBUTE_2_TYPE, ..., buffer }`
///
/// `ATTRIBUTE_X_EXPR` must be an expression that evaluates to a `&PointAttributeDefinition` and `ATTRIBUTE_X_TYPE` must be the Pasture
/// `PrimitiveType` that the attribute reference will be returned as. The type must match the type that the attribute is stored with in
/// the `buffer`. The iterator will then return tuples of the form:
///
/// `(&ATTRIBUTE_1_TYPE, &ATTRIBUTE_2_TYPE, ...)`
///
/// *Note:* Currently, a maximum of 4 attributes at the same time are supported.
#[macro_export]
macro_rules! attributes_ref {
    ($attr1:expr => $t1:ty, $attr2:expr => $t2:ty, $buffer:expr) => {
        crate::containers::attr2::AttributeIteratorByRef::<$t1, $t2>::new($buffer, [$attr1, $attr2])
    };
    ($attr1:expr => $t1:ty, $attr2:expr => $t2:ty, $attr3:expr => $t3:ty, $buffer:expr) => {
        crate::containers::attr3::AttributeIteratorByRef::<$t1, $t2, $t3>::new(
            $buffer,
            [$attr1, $attr2, $attr3],
        )
    };
    ($attr1:expr => $t1:ty, $attr2:expr => $t2:ty, $attr3:expr => $t3:ty, $attr4:expr => $t4:ty, $buffer:expr) => {
        crate::containers::attr3::AttributeIteratorByRef::<$t1, $t2, $t3, $t4>::new(
            $buffer,
            [$attr1, $attr2, $attr3, $attr4],
        )
    };
}

/// Create an iterator over mutable references to multiple attributes within a `PointBuffer`. Requires that the buffer implements
/// `PerAttributePointBufferMut`. This macro uses some special syntax to determine the attributes and their types:
///
/// `attributes_mut!{ ATTRIBUTE_1_EXPR => ATTRIBUTE_1_TYPE, ATTRIBUTE_2_EXPR => ATTRIBUTE_2_TYPE, ..., buffer }`
///
/// `ATTRIBUTE_X_EXPR` must be an expression that evaluates to a `&PointAttributeDefinition` and `ATTRIBUTE_X_TYPE` must be the Pasture
/// `PrimitiveType` that the attribute reference will be returned as. The type must match the type that the attribute is stored with in
/// the `buffer`. The iterator will then return tuples of the form:
///
/// `(&mut ATTRIBUTE_1_TYPE, &mut ATTRIBUTE_2_TYPE, ...)`
///
/// *Note:* Currently, a maximum of 4 attributes at the same time are supported.
#[macro_export]
macro_rules! attributes_mut {
    ($attr1:expr => $t1:ty, $attr2:expr => $t2:ty, $buffer:expr) => {
        crate::containers::attr2::AttributeIteratorByMut::<$t1, $t2>::new($buffer, [$attr1, $attr2])
    };
    ($attr1:expr => $t1:ty, $attr2:expr => $t2:ty, $attr3:expr => $t3:ty, $buffer:expr) => {
        crate::containers::attr3::AttributeIteratorByMut::<$t1, $t2, $t3>::new(
            $buffer,
            [$attr1, $attr2, $attr3],
        )
    };
    ($attr1:expr => $t1:ty, $attr2:expr => $t2:ty, $attr3:expr => $t3:ty, $attr4:expr => $t4:ty, $buffer:expr) => {
        crate::containers::attr3::AttributeIteratorByMut::<$t1, $t2, $t3, $t4>::new(
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
                &attributes::INTENSITY => u16,
                &attributes::GPS_TIME => f64,
                &mut storage
            );
            attributes_mut_view.for_each(|(intensity, gps_time)| {
                *intensity *= 2;
                *gps_time += 1.0;
            });
        }

        let modified_data = vec![(84_u16, 1.123), (86_u16, 1.456)];

        {
            let attributes_by_val_view = attributes!(
                &attributes::INTENSITY => u16,
                &attributes::GPS_TIME => f64,
                &storage
            );
            let attributes_by_val_collected = attributes_by_val_view.collect::<Vec<_>>();
            assert_eq!(modified_data, attributes_by_val_collected);
        }

        {
            let attributes_by_ref_view = attributes_ref!(
                &attributes::INTENSITY => u16,
                &attributes::GPS_TIME => f64,
                &storage
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

    #[test]
    #[should_panic(expected = "Type T does not match datatype of attribute")]
    fn test_attribute_with_wrong_type_fails() {
        let storage = InterleavedVecPointStorage::new(TestPointType::layout());
        attribute::<u32>(&storage, &attributes::INTENSITY);
    }

    #[test]
    #[should_panic(expected = "Type T does not match datatype of attribute")]
    fn test_attribute_ref_with_wrong_type_fails() {
        let storage = PerAttributeVecPointStorage::new(TestPointType::layout());
        attribute_ref::<u32>(&storage, &attributes::INTENSITY);
    }

    #[test]
    #[should_panic(expected = "Type T does not match datatype of attribute")]
    fn test_attribute_mut_with_wrong_type_fails() {
        let mut storage = PerAttributeVecPointStorage::new(TestPointType::layout());
        attribute_mut::<u32>(&mut storage, &attributes::INTENSITY);
    }

    #[test]
    #[should_panic(expected = "Type T does not match datatype of attribute")]
    fn test_attributes_with_wrong_type_fails() {
        let storage = InterleavedVecPointStorage::new(TestPointType::layout());
        attributes!(
            &attributes::INTENSITY => u32,
            &attributes::GPS_TIME => f32,
            &storage
        );
    }

    #[test]
    #[should_panic(expected = "Type T does not match datatype of attribute")]
    fn test_attributes_ref_with_wrong_type_fails() {
        let storage = PerAttributeVecPointStorage::new(TestPointType::layout());
        attributes_ref!(
            &attributes::INTENSITY => u32,
            &attributes::GPS_TIME => f32,
            &storage
        );
    }

    #[test]
    #[should_panic(expected = "Type T does not match datatype of attribute")]
    fn test_attributes_mut_with_wrong_type_fails() {
        let mut storage = PerAttributeVecPointStorage::new(TestPointType::layout());
        attributes_mut!(
            &attributes::INTENSITY => u32,
            &attributes::GPS_TIME => f32,
            &mut storage
        );
    }
}
