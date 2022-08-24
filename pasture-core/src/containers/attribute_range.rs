use std::{
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
};

use rayon::{
    iter::plumbing::{bridge, Producer},
    prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
};

use crate::layout::{PointAttributeDefinition, PrimitiveType};

use super::PointBuffer;

/// Trait helper to allow using the different `Range` types in a unified way when indexing an `AttributeRange`
pub trait RangeIndex {
    fn to_clamped_bounds(self, min: usize, max_exclusive: usize) -> Range<usize>;
}

impl RangeIndex for Range<usize> {
    fn to_clamped_bounds(self, min: usize, max_exclusive: usize) -> Range<usize> {
        let start = min.max(self.start);
        let end = max_exclusive.min(self.end);
        start..end
    }
}

impl RangeIndex for RangeFrom<usize> {
    fn to_clamped_bounds(self, min: usize, max_exclusive: usize) -> Range<usize> {
        let start = min.max(self.start);
        start..max_exclusive
    }
}

impl RangeIndex for RangeTo<usize> {
    fn to_clamped_bounds(self, min: usize, max_exclusive: usize) -> Range<usize> {
        let end = max_exclusive.min(self.end);
        min..end
    }
}

impl RangeIndex for RangeToInclusive<usize> {
    fn to_clamped_bounds(self, min: usize, max_exclusive: usize) -> Range<usize> {
        if self.end == usize::MAX {
            min..max_exclusive
        } else {
            let end = max_exclusive.min(self.end + 1);
            min..end
        }
    }
}

impl RangeIndex for RangeInclusive<usize> {
    fn to_clamped_bounds(self, min: usize, max_exclusive: usize) -> Range<usize> {
        let start = min.max(*self.start());
        if *self.end() == usize::MAX {
            start..max_exclusive
        } else {
            let end = max_exclusive.min(self.end() + 1);
            start..end
        }
    }
}

impl RangeIndex for RangeFull {
    fn to_clamped_bounds(self, min: usize, max_exclusive: usize) -> Range<usize> {
        min..max_exclusive
    }
}

#[derive(Debug, Copy, Clone)]
pub struct AttributeRange<'a, T: PrimitiveType> {
    raw_data: &'a [u8],
    offset: usize,
    stride: usize,
    length: usize,
    _phantom: PhantomData<T>,
}

impl<'a, T: PrimitiveType> AttributeRange<'a, T> {
    pub fn new<B: PointBuffer + 'a>(
        point_buffer: &'a B,
        attribute: &PointAttributeDefinition,
    ) -> Self {
        if let Some(interleaved) = point_buffer.as_interleaved() {
            let stride = point_buffer.point_layout().size_of_point_entry() as usize;
            let offset = point_buffer
                .point_layout()
                .get_attribute(attribute)
                .map(|a| a.offset())
                .expect("PointAttribute not found in buffer") as usize;
            Self {
                raw_data: interleaved.get_raw_points_ref(0..interleaved.len()),
                offset,
                stride,
                length: interleaved.len(),
                _phantom: Default::default(),
            }
        } else if let Some(per_attribute) = point_buffer.as_per_attribute() {
            let stride = point_buffer
                .point_layout()
                .get_attribute(attribute)
                .map(|a| a.size())
                .expect("PointAttribute not found in buffer") as usize;
            let attribute_data =
                per_attribute.get_raw_attribute_range_ref(0..per_attribute.len(), attribute);
            Self {
                raw_data: attribute_data,
                offset: 0,
                stride,
                length: per_attribute.len(),
                _phantom: Default::default(),
            }
        } else {
            panic!("Invalid buffer type, must be either an interleaved buffer or an per-attribute buffer");
        }
    }

    /// Returns the attribute value at `index`
    pub fn at(&self, index: usize) -> T {
        let start = self.offset + (index * self.stride);
        let end = start + std::mem::size_of::<T>();
        let mut point = MaybeUninit::<T>::uninit();
        unsafe {
            let point_raw_data = std::slice::from_raw_parts_mut(
                point.as_mut_ptr() as *mut u8,
                std::mem::size_of::<T>(),
            );
            point_raw_data.copy_from_slice(&self.raw_data[start..end]);
            point.assume_init()
        }
    }

    /// Returns the length of this `AttributeRange1`
    pub fn len(&self) -> usize {
        self.length
    }

    /// Slice this `AttributeRange` with the given `range`. Behaves like a regular slice indexing operation with a range
    pub fn slice<R: RangeIndex>(&self, range: R) -> Self {
        let clamped_range = range.to_clamped_bounds(0, self.length);
        let data_start_index = clamped_range.start * self.stride;
        let data_end_index = clamped_range.end * self.stride;
        let sliced_data = &self.raw_data[data_start_index..data_end_index];
        Self {
            _phantom: Default::default(),
            length: clamped_range.len(),
            offset: self.offset,
            raw_data: sliced_data,
            stride: self.stride,
        }
    }

    /// Splits this `AttributeRange1` at the given index into two `AttributeRange1`s
    pub fn split_at(self, index: usize) -> (Self, Self) {
        let end_of_left = index * self.stride;
        let left_slice = &self.raw_data[..end_of_left];
        let right_slice = &self.raw_data[end_of_left..];
        (
            Self {
                raw_data: left_slice,
                length: index,
                offset: self.offset,
                stride: self.stride,
                _phantom: Default::default(),
            },
            Self {
                raw_data: right_slice,
                length: self.length - index,
                offset: self.offset,
                stride: self.stride,
                _phantom: Default::default(),
            },
        )
    }
}

pub struct AttributeIter<'a, T: PrimitiveType> {
    attribute_range: AttributeRange<'a, T>,
    head: usize,
    tail: usize,
}

impl<'a, T: PrimitiveType> Iterator for AttributeIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.head == self.tail {
            None
        } else {
            let ret = self.attribute_range.at(self.head);
            self.head += 1;
            Some(ret)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.tail - self.head;
        (remaining, Some(remaining))
    }
}

impl<'a, T: PrimitiveType> ExactSizeIterator for AttributeIter<'a, T> {}

impl<'a, T: PrimitiveType> DoubleEndedIterator for AttributeIter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.tail == self.head {
            None
        } else {
            self.tail -= 1;
            let ret = self.attribute_range.at(self.tail);
            Some(ret)
        }
    }
}

impl<'a, T: PrimitiveType> IntoIterator for AttributeRange<'a, T> {
    type Item = T;
    type IntoIter = AttributeIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        AttributeIter {
            attribute_range: self,
            head: 0,
            tail: self.length,
        }
    }
}

impl<'a, T: PrimitiveType + Send> IntoParallelIterator for AttributeRange<'a, T> {
    type Iter = AttributeParIter<'a, T>;
    type Item = T;

    fn into_par_iter(self) -> Self::Iter {
        AttributeParIter {
            attribute_range: self,
        }
    }
}

pub struct AttributeParIter<'a, T: PrimitiveType + Send> {
    attribute_range: AttributeRange<'a, T>,
}

impl<'a, T: PrimitiveType + Send> ParallelIterator for AttributeParIter<'a, T> {
    type Item = T;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.len())
    }
}

impl<'a, T: PrimitiveType + Send> IndexedParallelIterator for AttributeParIter<'a, T> {
    fn len(&self) -> usize {
        self.attribute_range.length
    }

    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(
        self,
        callback: CB,
    ) -> CB::Output {
        callback.callback(Attribute1Producer {
            attribute_range: self.attribute_range,
        })
    }
}

struct Attribute1Producer<'a, T: PrimitiveType> {
    attribute_range: AttributeRange<'a, T>,
}

impl<'a, T: PrimitiveType + Send> Producer for Attribute1Producer<'a, T> {
    type Item = T;
    type IntoIter = AttributeIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.attribute_range.into_iter()
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.attribute_range.split_at(index);
        (
            Attribute1Producer {
                attribute_range: left,
            },
            Attribute1Producer {
                attribute_range: right,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use nalgebra::Vector3;
    use pasture_derive::PointType;
    use rand::{thread_rng, Rng};

    use crate::{
        containers::{InterleavedVecPointStorage, PerAttributeVecPointStorage},
        layout::attributes::{COLOR_RGB, GPS_TIME, INTENSITY, POSITION_3D},
    };

    use super::*;

    #[derive(Debug, Copy, Clone, PartialEq, PointType)]
    #[repr(C)]
    struct TestPointType {
        #[pasture(BUILTIN_INTENSITY)]
        pub intensity: u16,
        #[pasture(BUILTIN_GPS_TIME)]
        pub gps_time: f64,
        #[pasture(BUILTIN_POSITION_3D)]
        pub position: Vector3<f64>,
    }

    fn get_test_data(count: usize) -> Vec<TestPointType> {
        let mut rng = thread_rng();
        (0..count)
            .map(|_| TestPointType {
                gps_time: rng.gen(),
                intensity: rng.gen(),
                position: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
            })
            .collect()
    }

    #[test]
    fn test_attribute_range_1_basic_functionality_interleaved() {
        let test_data = get_test_data(16);
        let buffer = InterleavedVecPointStorage::from(test_data.as_slice());

        let positions: AttributeRange<Vector3<f64>> = AttributeRange::new(&buffer, &POSITION_3D);
        assert_eq!(positions.len(), buffer.len());
        for idx in 0..buffer.len() {
            assert_eq!(
                test_data[idx].position,
                positions.at(idx),
                "Wrong position at index {}",
                idx
            );
        }
        // Should also work with iterators
        positions
            .into_iter()
            .eq(test_data.iter().copied().map(|p| p.position));

        let gps_times: AttributeRange<f64> = AttributeRange::new(&buffer, &GPS_TIME);
        assert_eq!(gps_times.len(), buffer.len());
        for idx in 0..buffer.len() {
            assert_eq!(
                test_data[idx].gps_time,
                gps_times.at(idx),
                "Wrong GPS time at index {}",
                idx
            );
        }
        gps_times
            .into_iter()
            .eq(test_data.iter().copied().map(|p| p.gps_time));

        let intensities: AttributeRange<u16> = AttributeRange::new(&buffer, &INTENSITY);
        assert_eq!(intensities.len(), buffer.len());
        for idx in 0..buffer.len() {
            assert_eq!(
                test_data[idx].intensity,
                intensities.at(idx),
                "Wrong intensity time at index {}",
                idx
            );
        }
        intensities
            .into_iter()
            .eq(test_data.iter().copied().map(|p| p.intensity));
    }

    #[test]
    fn test_attribute_range_1_basic_functionality_per_attribute() {
        let test_data = get_test_data(16);
        let buffer = PerAttributeVecPointStorage::from(test_data.as_slice());

        let positions: AttributeRange<Vector3<f64>> = AttributeRange::new(&buffer, &POSITION_3D);
        assert_eq!(positions.len(), buffer.len());
        for idx in 0..buffer.len() {
            assert_eq!(
                test_data[idx].position,
                positions.at(idx),
                "Wrong position at index {}",
                idx
            );
        }
        // Should also work with iterators
        positions
            .into_iter()
            .eq(test_data.iter().copied().map(|p| p.position));

        let gps_times: AttributeRange<f64> = AttributeRange::new(&buffer, &GPS_TIME);
        assert_eq!(gps_times.len(), buffer.len());
        for idx in 0..buffer.len() {
            assert_eq!(
                test_data[idx].gps_time,
                gps_times.at(idx),
                "Wrong GPS time at index {}",
                idx
            );
        }
        gps_times
            .into_iter()
            .eq(test_data.iter().copied().map(|p| p.gps_time));

        let intensities: AttributeRange<u16> = AttributeRange::new(&buffer, &INTENSITY);
        assert_eq!(intensities.len(), buffer.len());
        for idx in 0..buffer.len() {
            assert_eq!(
                test_data[idx].intensity,
                intensities.at(idx),
                "Wrong intensity time at index {}",
                idx
            );
        }
        intensities
            .into_iter()
            .eq(test_data.iter().copied().map(|p| p.intensity));
    }

    #[test]
    #[should_panic]
    fn test_attribute_range_1_invalid_attribute() {
        let test_data = get_test_data(16);
        let buffer = PerAttributeVecPointStorage::from(test_data.as_slice());
        let _: AttributeRange<Vector3<u8>> = AttributeRange::new(&buffer, &COLOR_RGB);
    }

    #[test]
    fn test_attribute_range_1_slice_interleaved() {
        let test_data = get_test_data(16);
        let buffer = InterleavedVecPointStorage::from(test_data.as_slice());

        let slice_range = 3..9;

        let positions: AttributeRange<Vector3<f64>> =
            AttributeRange::new(&buffer, &POSITION_3D).slice(slice_range.clone());
        assert_eq!(positions.len(), slice_range.len());
        for idx in slice_range.clone() {
            assert_eq!(
                test_data[idx].position,
                positions.at(idx - slice_range.start),
                "Wrong position at index {}",
                idx
            );
        }

        let gps_times: AttributeRange<f64> =
            AttributeRange::new(&buffer, &GPS_TIME).slice(slice_range.clone());
        assert_eq!(gps_times.len(), slice_range.len());
        for idx in slice_range.clone() {
            assert_eq!(
                test_data[idx].gps_time,
                gps_times.at(idx - slice_range.start),
                "Wrong GPS time at index {}",
                idx
            );
        }

        let intensities: AttributeRange<u16> =
            AttributeRange::new(&buffer, &INTENSITY).slice(slice_range.clone());
        assert_eq!(intensities.len(), slice_range.len());
        for idx in slice_range.clone() {
            assert_eq!(
                test_data[idx].intensity,
                intensities.at(idx - slice_range.start),
                "Wrong intensity time at index {}",
                idx
            );
        }
    }

    #[test]
    fn test_attribute_range_1_slice_per_attribute() {
        let test_data = get_test_data(16);
        let buffer = PerAttributeVecPointStorage::from(test_data.as_slice());

        let slice_range = 3..9;

        let positions: AttributeRange<Vector3<f64>> =
            AttributeRange::new(&buffer, &POSITION_3D).slice(slice_range.clone());
        assert_eq!(positions.len(), slice_range.len());
        for idx in slice_range.clone() {
            assert_eq!(
                test_data[idx].position,
                positions.at(idx - slice_range.start),
                "Wrong position at index {}",
                idx
            );
        }

        let gps_times: AttributeRange<f64> =
            AttributeRange::new(&buffer, &GPS_TIME).slice(slice_range.clone());
        assert_eq!(gps_times.len(), slice_range.len());
        for idx in slice_range.clone() {
            assert_eq!(
                test_data[idx].gps_time,
                gps_times.at(idx - slice_range.start),
                "Wrong GPS time at index {}",
                idx
            );
        }

        let intensities: AttributeRange<u16> =
            AttributeRange::new(&buffer, &INTENSITY).slice(slice_range.clone());
        assert_eq!(intensities.len(), slice_range.len());
        for idx in slice_range.clone() {
            assert_eq!(
                test_data[idx].intensity,
                intensities.at(idx - slice_range.start),
                "Wrong intensity time at index {}",
                idx
            );
        }
    }

    #[test]
    fn test_attribute_range_1_parallel_iterator_interleaved() {
        let test_data = get_test_data(1023);
        let buffer = InterleavedVecPointStorage::from(test_data.as_slice());

        let intensities: AttributeRange<u16> = AttributeRange::new(&buffer, &INTENSITY);
        let intensities_gathered = intensities.into_par_iter().collect::<HashSet<_>>();

        let expected_intensities = test_data
            .iter()
            .map(|p| p.intensity)
            .collect::<HashSet<_>>();
        assert_eq!(intensities_gathered, expected_intensities);
    }

    #[test]
    fn test_attribute_range_1_parallel_iterator_per_attribute() {
        let test_data = get_test_data(1023);
        let buffer = PerAttributeVecPointStorage::from(test_data.as_slice());

        let intensities: AttributeRange<u16> = AttributeRange::new(&buffer, &INTENSITY);
        let intensities_gathered = intensities.into_par_iter().collect::<HashSet<_>>();

        let expected_intensities = test_data
            .iter()
            .map(|p| p.intensity)
            .collect::<HashSet<_>>();
        assert_eq!(intensities_gathered, expected_intensities);
    }
}
