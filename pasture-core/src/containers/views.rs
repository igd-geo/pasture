use crate::containers::InterleavedPointBuffer;
use crate::containers::PerAttributePointBuffer;
use crate::containers::PointBuffer;
use crate::layout::{PointAttributeDefinition, PointType, PrimitiveType};

use super::{InterleavedPointBufferMut, PerAttributePointBufferMut};

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
        pub fn new<B: InterleavedPointBuffer + Sized>(buffer: &'a B) -> Self {
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
        pub fn new<B: InterleavedPointBufferMut>(buffer: &'a mut B) -> Self {
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

    /// Iterator over a `PointBuffer` that yields strongly typed data by value for a specific attribute for each point
    pub struct Point1AttributeIteratorByValue<'a, T: PrimitiveType> {
        buffer: &'a dyn PointBuffer,
        attribute: &'a PointAttributeDefinition,
        current_index: usize,
        _unused: PhantomData<T>,
    }

    impl<'a, T: PrimitiveType> Point1AttributeIteratorByValue<'a, T> {
        pub fn new(buffer: &'a dyn PointBuffer, attribute: &'a PointAttributeDefinition) -> Self {
            Self {
                buffer,
                attribute,
                current_index: 0,
                _unused: Default::default(),
            }
        }
    }

    impl<'a, T: PrimitiveType> Iterator for Point1AttributeIteratorByValue<'a, T> {
        type Item = T;

        fn next(&mut self) -> Option<Self::Item> {
            if self.current_index == self.buffer.len() {
                return None;
            }

            // Create an uninitialized T which is filled by the call to `buffer.get_point_by_copy`
            let mut attribute = MaybeUninit::<T>::uninit();
            unsafe {
                let attribute_byte_slice = std::slice::from_raw_parts_mut(
                    attribute.as_mut_ptr() as *mut u8,
                    std::mem::size_of::<T>(),
                );
                self.buffer.get_attribute_by_copy(
                    self.current_index,
                    self.attribute,
                    attribute_byte_slice,
                );
            }

            self.current_index += 1;

            unsafe { Some(attribute.assume_init()) }
        }
    }

    /// Iterator over a `PointBuffer` that yields strongly typed data by reference for a specific attribute for each point
    pub struct Point1AttributeIteratorByRef<'a, T: PrimitiveType> {
        attribute_data: &'a [T],
        current_index: usize,
    }

    impl<'a, T: PrimitiveType> Point1AttributeIteratorByRef<'a, T> {
        pub fn new<B: PerAttributePointBuffer>(
            buffer: &'a B,
            attribute: &'a PointAttributeDefinition,
        ) -> Self {
            let buffer_len = buffer.len();
            let attribute_data = unsafe {
                std::slice::from_raw_parts(
                    buffer
                        .get_attribute_range_ref(0..buffer_len, attribute)
                        .as_ptr() as *const T,
                    buffer_len,
                )
            };
            Self {
                attribute_data,
                current_index: 0,
            }
        }
    }

    impl<'a, T: PrimitiveType> Iterator for Point1AttributeIteratorByRef<'a, T> {
        type Item = &'a T;

        fn next(&mut self) -> Option<Self::Item> {
            if self.current_index == self.attribute_data.len() {
                return None;
            }

            let current_attribute = &self.attribute_data[self.current_index];
            self.current_index += 1;
            Some(current_attribute)
        }
    }

    pub struct Point1AttributeIteratorByMut<'a, T: PrimitiveType> {
        attribute_data: &'a mut [T],
        current_index: usize,
    }

    impl<'a, T: PrimitiveType> Point1AttributeIteratorByMut<'a, T> {
        pub fn new<B: PerAttributePointBufferMut>(
            buffer: &'a mut B,
            attribute: &'a PointAttributeDefinition,
        ) -> Self {
            let buffer_len = buffer.len();
            let attribute_data = unsafe {
                std::slice::from_raw_parts_mut(
                    buffer
                        .get_attribute_range_mut(0..buffer_len, attribute)
                        .as_mut_ptr() as *mut T,
                    buffer_len,
                )
            };
            Self {
                attribute_data,
                current_index: 0,
            }
        }
    }

    impl<'a, T: PrimitiveType> Iterator for Point1AttributeIteratorByMut<'a, T> {
        type Item = &'a mut T;

        fn next(&mut self) -> Option<Self::Item> {
            if self.current_index == self.attribute_data.len() {
                return None;
            }

            // Returning mutable references seems to require unsafe...

            let offset_in_buffer = self.current_index * std::mem::size_of::<T>();
            self.current_index += 1;
            unsafe {
                let ptr_to_current_attribute =
                    self.attribute_data.as_mut_ptr().add(offset_in_buffer);
                let item = &mut *(ptr_to_current_attribute as *mut T);
                Some(item)
            }
        }
    }

    /// Iterator over a `PointBuffer` that yields strongly typed data by value for two specific attributes for each point
    pub struct Point2AttributeIteratorByValue<'a, T1: PrimitiveType, T2: PrimitiveType> {
        buffer: &'a dyn PointBuffer,
        attributes: [&'a PointAttributeDefinition; 2],
        current_index: usize,
        _unused: PhantomData<(T1, T2)>,
    }

    impl<'a, T1: PrimitiveType, T2: PrimitiveType> Point2AttributeIteratorByValue<'a, T1, T2> {
        pub fn new(
            buffer: &'a dyn PointBuffer,
            attributes: [&'a PointAttributeDefinition; 2],
        ) -> Self {
            Self {
                buffer,
                attributes,
                current_index: 0,
                _unused: Default::default(),
            }
        }
    }

    impl<'a, T1: PrimitiveType, T2: PrimitiveType> Iterator
        for Point2AttributeIteratorByValue<'a, T1, T2>
    {
        type Item = (T1, T2);

        fn next(&mut self) -> Option<Self::Item> {
            if self.current_index == self.buffer.len() {
                return None;
            }

            // Create an uninitialized T which is filled by the call to `buffer.get_point_by_copy`
            let mut attribute1 = MaybeUninit::<T1>::uninit();
            let mut attribute2 = MaybeUninit::<T2>::uninit();
            unsafe {
                let attribute1_byte_slice = std::slice::from_raw_parts_mut(
                    attribute1.as_mut_ptr() as *mut u8,
                    std::mem::size_of::<T1>(),
                );
                let attribute2_byte_slice = std::slice::from_raw_parts_mut(
                    attribute2.as_mut_ptr() as *mut u8,
                    std::mem::size_of::<T2>(),
                );

                self.buffer.get_attribute_by_copy(
                    self.current_index,
                    self.attributes[0],
                    attribute1_byte_slice,
                );
                self.buffer.get_attribute_by_copy(
                    self.current_index,
                    self.attributes[1],
                    attribute2_byte_slice,
                );
            }

            self.current_index += 1;

            unsafe { Some((attribute1.assume_init(), attribute2.assume_init())) }
        }
    }

    pub struct Point2AttributeIteratorByRef<'a, T1: PrimitiveType, T2: PrimitiveType> {
        attribute_data: (&'a [T1], &'a [T2]),
        current_index: usize,
    }

    impl<'a, T1: PrimitiveType, T2: PrimitiveType> Point2AttributeIteratorByRef<'a, T1, T2> {
        pub fn new<B: PerAttributePointBuffer>(
            buffer: &'a B,
            attributes: [&'a PointAttributeDefinition; 2],
        ) -> Self {
            let buffer_len = buffer.len();
            let attribute1_data = unsafe {
                std::slice::from_raw_parts(
                    buffer
                        .get_attribute_range_ref(0..buffer_len, attributes[0])
                        .as_ptr() as *const T1,
                    buffer_len,
                )
            };
            let attribute2_data = unsafe {
                std::slice::from_raw_parts(
                    buffer
                        .get_attribute_range_ref(0..buffer_len, attributes[1])
                        .as_ptr() as *const T2,
                    buffer_len,
                )
            };
            Self {
                attribute_data: (attribute1_data, attribute2_data),
                current_index: 0,
            }
        }
    }

    impl<'a, T1: PrimitiveType, T2: PrimitiveType> Iterator
        for Point2AttributeIteratorByRef<'a, T1, T2>
    {
        type Item = (&'a T1, &'a T2);

        fn next(&mut self) -> Option<Self::Item> {
            if self.current_index == self.attribute_data.0.len() {
                return None;
            }

            let attr1 = &self.attribute_data.0[self.current_index];
            let attr2 = &self.attribute_data.1[self.current_index];
            self.current_index += 1;

            Some((attr1, attr2))
        }
    }

    pub struct Point2AttributeIteratorByMut<'a, T1: PrimitiveType, T2: PrimitiveType> {
        attribute_data: (&'a mut [T1], &'a mut [T2]),
        current_index: usize,
    }

    impl<'a, T1: PrimitiveType, T2: PrimitiveType> Point2AttributeIteratorByMut<'a, T1, T2> {
        pub fn new<B: PerAttributePointBufferMut>(
            buffer: &'a mut B,
            attributes: [&'a PointAttributeDefinition; 2],
        ) -> Self {
            let buffer_len = buffer.len();
            let attribute1_data = unsafe {
                std::slice::from_raw_parts_mut(
                    buffer
                        .get_attribute_range_mut(0..buffer_len, attributes[0])
                        .as_ptr() as *mut T1,
                    buffer_len,
                )
            };
            let attribute2_data = unsafe {
                std::slice::from_raw_parts_mut(
                    buffer
                        .get_attribute_range_mut(0..buffer_len, attributes[1])
                        .as_ptr() as *mut T2,
                    buffer_len,
                )
            };
            Self {
                attribute_data: (attribute1_data, attribute2_data),
                current_index: 0,
            }
        }
    }

    impl<'a, T1: PrimitiveType, T2: PrimitiveType> Iterator
        for Point2AttributeIteratorByMut<'a, T1, T2>
    {
        type Item = (&'a mut T1, &'a mut T2);

        fn next(&mut self) -> Option<Self::Item> {
            if self.current_index == self.attribute_data.0.len() {
                return None;
            }

            let offset1 = self.current_index * std::mem::size_of::<T1>();
            let offset2 = self.current_index * std::mem::size_of::<T2>();

            self.current_index += 1;

            unsafe {
                let ptr_attrib1 = self.attribute_data.0.as_mut_ptr().add(offset1);
                let attr1 = &mut *(ptr_attrib1 as *mut T1);

                let ptr_attrib2 = self.attribute_data.1.as_mut_ptr().add(offset2);
                let attr2 = &mut *(ptr_attrib2 as *mut T2);

                Some((attr1, attr2))
            }
        }
    }

    // TODO Implement iterators for more attributes (e.g. PerAttributeIterator<'a, T1, T2, ...> returning (&'a T1, &'a T2, ...))
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
            "points: PointLayouts do not match (type T has layout {:?}, buffer has layout {:?})",
            point_layout,
            buffer.point_layout()
        );
    }

    iterators::PointIteratorByValue::new(buffer)
}

/// Returns an iterator over references to all points within the given PointBuffer, strongly typed to the PointType T.
pub fn points_ref<'a, T: PointType + 'a, B: InterleavedPointBuffer>(
    buffer: &'a B,
) -> iterators::PointIteratorByRef<'a, T> {
    let point_layout = T::layout();
    if point_layout != *buffer.point_layout() {
        panic!(
            "points_ref: PointLayouts do not match (type T has layout {:?}, buffer has layout {:?})",
            point_layout,
            buffer.point_layout()
        );
    }

    iterators::PointIteratorByRef::new(buffer)
}

/// Returns an iterator over mutable references to all points within the given PointBuffer, strongly typed to the PointType T.
pub fn points_mut<'a, T: PointType + 'a, B: InterleavedPointBufferMut>(
    buffer: &'a mut B,
) -> iterators::PointIteratorByMut<'a, T> {
    let point_layout = T::layout();
    if point_layout != *buffer.point_layout() {
        panic!(
            "points_mut: PointLayouts do not match (type T has layout {:?}, buffer has layout {:?})",
            point_layout,
            buffer.point_layout()
        );
    }

    iterators::PointIteratorByMut::new(buffer)
}

/// Returns an iterator over the specific attribute for all points within the given `PointBuffer`, strongly typed over the `PrimitiveType` `T`
pub fn attributes<'a, T: PrimitiveType + 'a>(
    buffer: &'a dyn PointBuffer,
    attribute: &'a PointAttributeDefinition,
) -> impl Iterator<Item = T> + 'a {
    if !buffer.point_layout().has_attribute(attribute.name()) {
        panic!(
            "attributes: PointLayout of buffer does not contain attribute {}",
            attribute.name()
        );
    }

    iterators::Point1AttributeIteratorByValue::new(buffer, attribute)
}

/// Returns an iterator over references to the specific attribute for all points within the given `PointBuffer`, strongly typed over the `PrimitiveType` `T`
pub fn attributes_ref<'a, T: PrimitiveType + 'a, B: PerAttributePointBuffer>(
    buffer: &'a B,
    attribute: &'a PointAttributeDefinition,
) -> iterators::Point1AttributeIteratorByRef<'a, T> {
    if !buffer.point_layout().has_attribute(attribute.name()) {
        panic!(
            "attributes_ref: PointLayout of buffer does not contain attribute {}",
            attribute.name()
        );
    }

    iterators::Point1AttributeIteratorByRef::new(buffer, attribute)
}

/// Returns an iterator over mutable references to the specific attribute for all points within the given `PointBuffer`, strongly typed over the `PrimitiveType` `T`
pub fn attributes_mut<'a, T: PrimitiveType + 'a, B: PerAttributePointBufferMut>(
    buffer: &'a mut B,
    attribute: &'a PointAttributeDefinition,
) -> iterators::Point1AttributeIteratorByMut<'a, T> {
    if !buffer.point_layout().has_attribute(attribute.name()) {
        panic!(
            "attributes_mut: PointLayout of buffer does not contain attribute {}",
            attribute.name()
        );
    }

    iterators::Point1AttributeIteratorByMut::new(buffer, attribute)
}

/// Returns an iterator over the two specific attributes for all points within the given `PointBuffer`, strongly typed over the `PrimitiveType` `T`
pub fn attributes2<'a, T1: PrimitiveType + 'a, T2: PrimitiveType + 'a>(
    buffer: &'a dyn PointBuffer,
    attributes: [&'a PointAttributeDefinition; 2],
) -> iterators::Point2AttributeIteratorByValue<'a, T1, T2> {
    for attribute in attributes.iter() {
        if buffer.point_layout().has_attribute(attribute.name()) {
            panic!(
                "attributes2: PointLayout of buffer does not contain attribute {}",
                attribute.name()
            );
        }
    }

    iterators::Point2AttributeIteratorByValue::new(buffer, attributes)
}

/// Returns an iterator over references to the specific attribute for all points within the given `PointBuffer`, strongly typed over the `PrimitiveType` `T`
pub fn attributes2_ref<
    'a,
    T1: PrimitiveType + 'a,
    T2: PrimitiveType + 'a,
    B: PerAttributePointBuffer,
>(
    buffer: &'a B,
    attributes: [&'a PointAttributeDefinition; 2],
) -> iterators::Point2AttributeIteratorByRef<'a, T1, T2> {
    for attribute in attributes.iter() {
        if buffer.point_layout().has_attribute(attribute.name()) {
            panic!(
                "attributes2_ref: PointLayout of buffer does not contain attribute {}",
                attribute.name()
            );
        }
    }

    iterators::Point2AttributeIteratorByRef::new(buffer, attributes)
}

/// Returns an iterator over mutable references to the specific attribute for all points within the given `PointBuffer`, strongly typed over the `PrimitiveType` `T`
pub fn attributes2_mut<
    'a,
    T1: PrimitiveType + 'a,
    T2: PrimitiveType + 'a,
    B: PerAttributePointBufferMut,
>(
    buffer: &'a mut B,
    attributes: [&'a PointAttributeDefinition; 2],
) -> iterators::Point2AttributeIteratorByMut<'a, T1, T2> {
    for attribute in attributes.iter() {
        if buffer.point_layout().has_attribute(attribute.name()) {
            panic!(
                "attributes2_mut: PointLayout of buffer does not contain attribute {}",
                attribute.name()
            );
        }
    }

    iterators::Point2AttributeIteratorByMut::new(buffer, attributes)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::containers::{InterleavedVecPointStorage, PerAttributeVecPointStorage};
    use crate::layout::{attributes, PointLayout};
    use static_assertions::const_assert;

    #[repr(packed)]
    #[derive(Debug, Copy, Clone, PartialEq)]
    struct TestPointType(u16, f64);

    impl PointType for TestPointType {
        fn layout() -> PointLayout {
            PointLayout::from_attributes(&[attributes::INTENSITY, attributes::GPS_TIME])
        }
    }

    const_assert!(10 == std::mem::size_of::<TestPointType>());

    #[test]
    fn test_points_view_from_interleaved() {
        let reference_points = vec![TestPointType(42, 0.123), TestPointType(43, 0.456)];
        let mut storage = InterleavedVecPointStorage::new(TestPointType::layout());
        storage.push_point(reference_points[0]);
        storage.push_point(reference_points[1]);

        {
            let points_by_mut_view =
                points_mut::<TestPointType, InterleavedVecPointStorage>(&mut storage);
            points_by_mut_view.for_each(|point| {
                point.0 *= 2;
                point.1 += 1.0;
            });
        }

        let modified_points = vec![TestPointType(84, 1.123), TestPointType(86, 1.456)];

        {
            let points_by_val_view = points::<TestPointType>(&storage);
            let points_by_val_collected = points_by_val_view.collect::<Vec<_>>();
            assert_eq!(modified_points, points_by_val_collected);
        }

        {
            let points_by_ref_view =
                points_ref::<TestPointType, InterleavedVecPointStorage>(&storage);
            let points_by_ref_collected = points_by_ref_view.map(|r| *r).collect::<Vec<_>>();
            assert_eq!(modified_points, points_by_ref_collected);
        }
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
