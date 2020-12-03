use crate::containers::InterleavedPointBuffer;
use crate::containers::PerAttributePointBuffer;
use crate::containers::PointBuffer;
use crate::layout::{PointAttributeDefinition, PointType, PrimitiveType};

mod iterators {

    //! Contains `Iterator` implementations through which the untyped contents of `PointBuffer` structures
    //! can be accessed in a safe and strongly-typed manner.

    use super::*;
    use std::marker::PhantomData;
    use std::mem::MaybeUninit;

    /// Iterator over an arbitrary `PointBuffer` that yields strongly typed points
    pub struct DefaultPointIterator<'a, T: PointType> {
        buffer: &'a dyn PointBuffer,
        current_index: usize,
        unused: PhantomData<T>,
    }

    impl<'a, T: PointType> DefaultPointIterator<'a, T> {
        /// Creates a new `DefaultPointIterator` over all points in the given `PointBuffer`
        pub fn new(buffer: &'a dyn PointBuffer) -> Self {
            Self {
                buffer,
                current_index: 0,
                unused: Default::default(),
            }
        }
    }

    impl<'a, T: PointType> Iterator for DefaultPointIterator<'a, T> {
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

    /// Iterator over an interleaved `PointBuffer` that yields strongly typed points
    pub struct InterleavedPointIterator<'a, T: PointType + 'a, B: InterleavedPointBuffer + Sized> {
        buffer: &'a B,
        points: &'a [u8],
        current_index: usize,
        unused: PhantomData<T>,
    }

    impl<'a, T: PointType + 'a, B: InterleavedPointBuffer + Sized> InterleavedPointIterator<'a, T, B> {
        /// Creates a new `InterleavedPointIterator` over all points in the given `PointBuffer`
        pub fn new(buffer: &'a B) -> Self {
            Self {
                buffer,
                points: buffer.get_points_by_ref(0..buffer.len()),
                current_index: 0,
                unused: Default::default(),
            }
        }
    }

    impl<'a, T: PointType + 'a, B: InterleavedPointBuffer + Sized> Iterator
        for InterleavedPointIterator<'a, T, B>
    {
        type Item = &'a T;
        fn next(&mut self) -> Option<Self::Item> {
            if self.current_index == self.buffer.len() {
                return None;
            }

            let offset_in_buffer = self.current_index * std::mem::size_of::<T>();
            self.current_index += 1;
            unsafe {
                let ptr_to_current_point = self.points.as_ptr().add(offset_in_buffer);
                let item = &*(ptr_to_current_point as *const T);
                Some(item)
            }
        }
    }

    pub struct DefaultAttributeIterator<'a, T: PrimitiveType> {
        buffer: &'a dyn PointBuffer,
        attribute: &'a PointAttributeDefinition,
        current_index: usize,
        unused: PhantomData<T>,
    }

    impl<'a, T: PrimitiveType> DefaultAttributeIterator<'a, T> {
        pub fn new(buffer: &'a dyn PointBuffer, attribute: &'a PointAttributeDefinition) -> Self {
            Self {
                buffer,
                attribute,
                current_index: 0,
                unused: Default::default(),
            }
        }
    }

    impl<'a, T: PrimitiveType> Iterator for DefaultAttributeIterator<'a, T> {
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

    iterators::DefaultPointIterator::new(buffer)
}

/// Returns an iterator over all points within the given PointBuffer, strongly typed to the PointType T. Assumes an
/// interleaved memory layout for performance, so only works with an InterleavedPointBuffer. Panics if the PointBuffer
/// does not have the same PointLayout as the PointType T.
pub fn points_from_interleaved_buffer<
    'a,
    T: PointType + 'a,
    B: InterleavedPointBuffer + Sized + 'a,
>(
    buffer: &'a B,
) -> impl Iterator<Item = &'a T> + 'a {
    let point_layout = T::layout();
    if point_layout != *buffer.point_layout() {
        panic!("points_from_interleaved_buffer: PointLayouts do not match (type T has layout {:?}, buffer has layout {:?})", point_layout, buffer.point_layout());
    }

    iterators::InterleavedPointIterator::new(buffer)
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

    iterators::DefaultAttributeIterator::new(buffer, attribute)
}

/// Returns an iterator over the specific attribute for all points within the given `PointBuffer`, strongly typed over the `PrimitiveType` `T`
pub fn attributes_from_by_attribute_buffer<
    'a,
    T: PrimitiveType + 'a,
    B: PerAttributePointBuffer + Sized + 'a,
>(
    buffer: &'a B,
    attribute: &'a PointAttributeDefinition,
) -> impl Iterator<Item = T> + 'a {
    if !buffer.point_layout().has_attribute(attribute.name()) {
        panic!("attributes_from_by_attribute_buffer: PointLayout of buffer does not contain attribute {}", attribute.name());
    }
    iterators::DefaultAttributeIterator::new(buffer, attribute)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::containers::InterleavedVecPointStorage;
    use crate::layout::{attributes, PointLayout};
    use static_assertions::const_assert;

    #[repr(packed)]
    struct TestPointType(u16, f64);

    impl PointType for TestPointType {
        fn layout() -> PointLayout {
            PointLayout::from_attributes(&[attributes::INTENSITY, attributes::GPS_TIME])
        }
    }

    const_assert!(10 == std::mem::size_of::<TestPointType>());

    #[test]
    fn test_points_view_from_interleaved() {
        let mut storage = InterleavedVecPointStorage::new(TestPointType::layout());
        storage.push_point(TestPointType(42, 0.123));
        storage.push_point(TestPointType(43, 0.345));

        let points_view =
            points_from_interleaved_buffer::<TestPointType, InterleavedVecPointStorage>(&storage);

        let points_collected = points_view.collect::<Vec<_>>();

        assert_eq!(2, points_collected.len());
        assert_eq!(42, points_collected[0].0);
        assert_eq!(0.123, points_collected[0].1);
        assert_eq!(43, points_collected[1].0);
        assert_eq!(0.345, points_collected[1].1);
    }

    #[test]
    fn test_points_view() {
        let mut storage = InterleavedVecPointStorage::new(TestPointType::layout());
        storage.push_point(TestPointType(42, 0.123));
        storage.push_point(TestPointType(43, 0.345));

        let points_view = points::<TestPointType>(&storage);

        let points_collected = points_view.collect::<Vec<_>>();

        assert_eq!(2, points_collected.len());
        assert_eq!(42, points_collected[0].0);
        assert_eq!(0.123, points_collected[0].1);
        assert_eq!(43, points_collected[1].0);
        assert_eq!(0.345, points_collected[1].1);
    }

    #[test]
    fn test_attributes_view() {
        let mut storage = InterleavedVecPointStorage::new(TestPointType::layout());
        storage.push_point(TestPointType(42, 0.123));
        storage.push_point(TestPointType(43, 0.345));

        let attributes_view = attributes::<u16>(&storage, &attributes::INTENSITY);

        let attributes_collected = attributes_view.collect::<Vec<_>>();

        assert_eq!(2, attributes_collected.len());
        assert_eq!(42, attributes_collected[0]);
        assert_eq!(43, attributes_collected[1]);
    }
}
