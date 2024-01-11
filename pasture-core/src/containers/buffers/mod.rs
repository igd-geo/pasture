mod traits;
pub use self::traits::*;

mod vector_buffer;
pub use self::vector_buffer::*;

mod hashmap_buffer;
pub use self::hashmap_buffer::*;

mod borrowed_buffers;
pub use self::borrowed_buffers::*;

#[cfg(test)]
mod tests {
    use std::iter::FromIterator;

    use itertools::Itertools;
    use nalgebra::Vector3;
    use rand::{thread_rng, Rng};

    use crate::{
        containers::{SliceBuffer, SliceBufferMut},
        layout::{attributes::POSITION_3D, PointLayout, PointType},
        test_utils::{CustomPointTypeBig, CustomPointTypeSmall, DefaultPointDistribution},
    };

    use super::*;

    fn test_transform_attribute_generic<
        'a,
        B: BorrowedMutBuffer<'a> + FromIterator<CustomPointTypeBig> + 'a,
    >() {
        const COUNT: usize = 16;
        let test_data: Vec<CustomPointTypeBig> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();
        let overwrite_data: Vec<CustomPointTypeBig> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();

        let mut buffer = test_data.iter().copied().collect::<B>();
        // Overwrite the positions with the positions in `overwrite_data` using the `transform_attribute` function
        buffer.transform_attribute(&POSITION_3D, |index, _| -> Vector3<f64> {
            overwrite_data[index].position
        });

        let expected_positions = overwrite_data
            .iter()
            .map(|point| point.position)
            .collect::<Vec<_>>();
        let actual_positions = buffer
            .view_attribute(&POSITION_3D)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(expected_positions, actual_positions);
    }

    #[test]
    fn test_transform_attribute() {
        test_transform_attribute_generic::<VectorBuffer>();
        test_transform_attribute_generic::<HashMapBuffer>();
    }

    #[test]
    fn test_append() {
        const COUNT: usize = 16;
        let test_data: Vec<CustomPointTypeBig> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();

        let expected_buffer_interleaved = test_data.iter().copied().collect::<VectorBuffer>();
        let expected_buffer_columnar = test_data.iter().copied().collect::<HashMapBuffer>();

        {
            let mut vector_buffer = VectorBuffer::new_from_layout(CustomPointTypeBig::layout());
            vector_buffer.append(&expected_buffer_interleaved);
            assert_eq!(expected_buffer_interleaved, vector_buffer);
        }
        {
            let mut vector_buffer = VectorBuffer::new_from_layout(CustomPointTypeBig::layout());
            vector_buffer.append(&expected_buffer_columnar);
            assert_eq!(expected_buffer_interleaved, vector_buffer);
        }
        {
            let mut hashmap_buffer = HashMapBuffer::new_from_layout(CustomPointTypeBig::layout());
            hashmap_buffer.append(&expected_buffer_columnar);
            assert_eq!(expected_buffer_columnar, hashmap_buffer);
        }
        {
            let mut hashmap_buffer = HashMapBuffer::new_from_layout(CustomPointTypeBig::layout());
            hashmap_buffer.append(&expected_buffer_interleaved);
            assert_eq!(expected_buffer_columnar, hashmap_buffer);
        }
    }

    #[test]
    fn test_buffers_from_empty_layout() {
        let empty_layout = PointLayout::default();

        {
            let buffer = VectorBuffer::new_from_layout(empty_layout.clone());
            assert_eq!(0, buffer.len());
        }
        {
            let buffer = HashMapBuffer::new_from_layout(empty_layout.clone());
            assert_eq!(0, buffer.len());
        }
        {
            let empty_memory = Vec::default();
            let buffer = ExternalMemoryBuffer::new(&empty_memory, empty_layout.clone());
            assert_eq!(0, buffer.len());
        }
    }

    fn test_buffer_set_point_range_generic<
        B: for<'a> BorrowedMutBuffer<'a>
            + FromIterator<CustomPointTypeBig>
            + for<'a> SliceBufferMut<'a>,
    >() {
        const COUNT: usize = 16;
        let test_data: Vec<CustomPointTypeBig> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();
        let overwrite_data: Vec<CustomPointTypeBig> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();
        let raw_overwrite_data: &[u8] = bytemuck::cast_slice(&overwrite_data);

        let mut buffer = test_data.iter().copied().collect::<B>();
        // Safe because we know the point layout of buffer is equal to that of CustomPointTypeBig
        unsafe {
            buffer.set_point_range(0..COUNT, raw_overwrite_data);
        }

        let actual_data = buffer
            .view::<CustomPointTypeBig>()
            .into_iter()
            .collect_vec();
        assert_eq!(overwrite_data, actual_data);

        // Do the same thing, but with a slice
        buffer = test_data.iter().copied().collect::<B>();
        let mut buffer_slice = buffer.slice_mut(0..COUNT);
        unsafe {
            buffer_slice.set_point_range(0..COUNT, raw_overwrite_data);
        }
        drop(buffer_slice);

        let actual_data = buffer
            .view::<CustomPointTypeBig>()
            .into_iter()
            .collect_vec();
        assert_eq!(overwrite_data, actual_data);
    }

    #[test]
    fn test_buffers_set_point_range() {
        test_buffer_set_point_range_generic::<VectorBuffer>();
        test_buffer_set_point_range_generic::<HashMapBuffer>();
    }

    fn test_buffer_get_point_range_generic<
        B: for<'a> BorrowedMutBuffer<'a>
            + FromIterator<CustomPointTypeBig>
            + for<'a> SliceBufferMut<'a>,
    >() {
        const COUNT: usize = 16;
        let test_data: Vec<CustomPointTypeBig> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();
        let raw_test_data: &[u8] = bytemuck::cast_slice(test_data.as_slice());
        let size_of_single_point = std::mem::size_of::<CustomPointTypeBig>();

        let buffer = test_data.iter().copied().collect::<B>();

        let mut actual_point_data = vec![0; raw_test_data.len()];
        buffer.get_point_range(0..COUNT, &mut actual_point_data);

        assert_eq!(raw_test_data, actual_point_data);

        // Check that subset ranges work correctly as well
        let subset_slice = &mut actual_point_data[..(6 * size_of_single_point)];
        buffer.get_point_range(2..8, subset_slice);
        assert_eq!(
            &raw_test_data[(2 * size_of_single_point)..(8 * size_of_single_point)],
            subset_slice
        );
    }

    #[test]
    fn test_buffer_get_point_range() {
        test_buffer_get_point_range_generic::<VectorBuffer>();
        test_buffer_get_point_range_generic::<HashMapBuffer>();
    }

    fn test_buffer_set_attribute_range_generic<
        B: for<'a> BorrowedMutBuffer<'a>
            + FromIterator<CustomPointTypeBig>
            + for<'a> SliceBufferMut<'a>,
    >() {
        const COUNT: usize = 16;
        let test_data: Vec<CustomPointTypeBig> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();
        let overwrite_data: Vec<CustomPointTypeBig> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();
        let overwrite_positions = overwrite_data
            .iter()
            .map(|point| point.position)
            .collect_vec();
        let overwrite_positions_raw_data: &[u8] = bytemuck::cast_slice(&overwrite_positions);

        let mut buffer = test_data.iter().copied().collect::<B>();
        // Safe because we know the point layout of buffer
        unsafe {
            buffer.set_attribute_range(&POSITION_3D, 0..COUNT, overwrite_positions_raw_data);
        }

        let actual_positions = buffer
            .view_attribute(&POSITION_3D)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(overwrite_positions, actual_positions);

        // Do the same test, but for a slice
        buffer = test_data.iter().copied().collect::<B>();
        let mut buffer_slice = buffer.slice_mut(0..test_data.len());
        unsafe {
            buffer_slice.set_attribute_range(&POSITION_3D, 0..COUNT, overwrite_positions_raw_data);
        }
        drop(buffer_slice);

        let actual_positions = buffer
            .view_attribute(&POSITION_3D)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(overwrite_positions, actual_positions);
    }

    #[test]
    fn test_buffers_set_attribute_range() {
        test_buffer_set_attribute_range_generic::<VectorBuffer>();
        test_buffer_set_attribute_range_generic::<HashMapBuffer>();
    }

    #[test]
    fn test_buffers_as_memory_layout_accessors() {
        let mut vector_buffer = VectorBuffer::new_from_layout(CustomPointTypeSmall::layout());
        let mut hashmap_buffer = HashMapBuffer::new_from_layout(CustomPointTypeSmall::layout());

        let mut memory: Vec<u8> = Vec::default();
        let mut external_memory_buffer =
            ExternalMemoryBuffer::new(memory.as_mut_slice(), CustomPointTypeSmall::layout());

        assert!(vector_buffer.as_interleaved().is_some());
        assert!(vector_buffer.slice(0..0).as_interleaved().is_some());
        assert!(vector_buffer.slice_mut(0..0).as_interleaved().is_some());
        assert!(hashmap_buffer.as_interleaved().is_none());
        assert!(hashmap_buffer.slice(0..0).as_interleaved().is_none());
        assert!(hashmap_buffer.slice_mut(0..0).as_interleaved().is_none());
        assert!(external_memory_buffer.as_interleaved().is_some());
        assert!(external_memory_buffer
            .slice(0..0)
            .as_interleaved()
            .is_some());
        assert!(external_memory_buffer
            .slice_mut(0..0)
            .as_interleaved()
            .is_some());

        assert!(vector_buffer.as_interleaved_mut().is_some());
        assert!(vector_buffer.slice_mut(0..0).as_interleaved_mut().is_some());
        assert!(hashmap_buffer.as_interleaved_mut().is_none());
        assert!(hashmap_buffer
            .slice_mut(0..0)
            .as_interleaved_mut()
            .is_none());
        assert!(external_memory_buffer.as_interleaved_mut().is_some());
        assert!(external_memory_buffer
            .slice_mut(0..0)
            .as_interleaved_mut()
            .is_some());

        assert!(vector_buffer.as_columnar().is_none());
        assert!(vector_buffer.slice(0..0).as_columnar().is_none());
        assert!(vector_buffer.slice_mut(0..0).as_columnar().is_none());
        assert!(hashmap_buffer.as_columnar().is_some());
        assert!(hashmap_buffer.slice(0..0).as_columnar().is_some());
        assert!(hashmap_buffer.slice_mut(0..0).as_columnar().is_some());
        assert!(external_memory_buffer.as_columnar().is_none());
        assert!(external_memory_buffer.slice(0..0).as_columnar().is_none());
        assert!(external_memory_buffer
            .slice_mut(0..0)
            .as_columnar()
            .is_none());

        assert!(vector_buffer.as_columnar_mut().is_none());
        assert!(vector_buffer.slice_mut(0..0).as_columnar_mut().is_none());
        assert!(hashmap_buffer.as_columnar_mut().is_some());
        assert!(hashmap_buffer.slice_mut(0..0).as_columnar_mut().is_some());
        assert!(external_memory_buffer.as_columnar_mut().is_none());
        assert!(external_memory_buffer
            .slice_mut(0..0)
            .as_columnar_mut()
            .is_none());
    }
}
