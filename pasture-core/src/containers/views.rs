use crate::containers::InterleavedPointBuffer;
use crate::containers::PointBuffer;
use crate::layout::{PointAttributeDefinition, PointType, PrimitiveType};

mod iterators {

    use super::*;
    use std::marker::PhantomData;

    pub struct InterleavedPointIterator<'a, T: PointType + 'a, B: InterleavedPointBuffer + Sized> {
        buffer: &'a B,
        points: &'a [u8],
        current_index: usize,
        unused: PhantomData<T>,
    }

    impl<'a, T: PointType + 'a, B: InterleavedPointBuffer + Sized> InterleavedPointIterator<'a, T, B> {
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
}

// TODO points() and attributes() should be macro calls. Inside a macro, we can dispatch on the actual type,
// so in cases where the user has a specific PointBuffer type and calls points(&buf), we can return a specific
// iterator implementation instead of a boxed iterator. This will then be much faster in these cases because it
// alleviates the need for virtual dispatch on every iteration step

/// Returns an iterator over all points within the given PointBuffer, strongly typed to the PointType T. Assumes no
/// internal memory representation for the source buffer, so returns an opaque iterator type that works with arbitrary
/// PointBuffer implementations. If you know the type of your PointBuffer, prefer one of the points_from_... variants
/// as they will yield better performance. Or simply use the points! macro, which selects the best matching candidate.
pub fn points<'a, T: PointType>(_buffer: &'a dyn PointBuffer) -> Box<dyn Iterator<Item = T> + 'a> {
    todo!("implement")
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

/// Returns an iterator over the specific attribute for all points within the given PointBuffer, strongly typed over the PrimitiveType T
pub fn attributes<'a, T: PrimitiveType>(
    _buffer: &'a dyn PointBuffer,
    _attribute: &PointAttributeDefinition,
) -> Box<dyn Iterator<Item = T> + 'a> {
    todo!("implement")
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::containers::InterleavedVecPointStorage;
    use crate::layout::{attributes, PointLayout};

    struct TestPointType(u16);

    impl PointType for TestPointType {
        fn layout() -> PointLayout {
            PointLayout::from_attributes(&[attributes::INTENSITY])
        }
    }

    #[test]
    fn test_points_view_from_interleaved() {
        let mut storage = InterleavedVecPointStorage::new(TestPointType::layout());
        storage.push_point(TestPointType(42));

        let points_view =
            points_from_interleaved_buffer::<TestPointType, InterleavedVecPointStorage>(&storage);

        let points_collected = points_view.collect::<Vec<_>>();

        assert_eq!(1, points_collected.len());
        assert_eq!(42, points_collected[0].0);
    }
}
