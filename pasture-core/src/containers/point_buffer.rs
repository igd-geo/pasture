use std::ops::Range;

use crate::layout::{PointAttributeDefinition, PointLayout, PointType};
use crate::util::view_raw_bytes;

/// Base trait for all containers that store point data. A PointBuffer stores any number of point entries
/// with a layout defined by the PointBuffers associated PointLayout structure.
///
/// Users will rarely have to work with this base trait directly as it exposes the underlying memory-unsafe
/// API for fast point and attribute access. Instead, prefer specific PointBuffer implementations or point views!
pub trait PointBuffer {
    /// Get the data for a single point from this PointBuffer and store it inside the given memory region.
    /// Panics if point_index is out of bounds. buf must be at least as big as a single point entry in the
    /// corresponding PointLayout of this PointBuffer
    fn get_point_by_copy(&self, point_index: usize, buf: &mut [u8]);
    /// Get the data for the given attribute of a single point from this PointBuffer and store it inside the
    /// given memory region. Panics if point_index is out of bounds or if the attribute is not part of the point_layout
    /// of this PointBuffer. buf must be at least as big as a single attribute entry of the given attribute.
    fn get_attribute_by_copy(
        &self,
        point_index: usize,
        attribute: &PointAttributeDefinition,
        buf: &mut [u8],
    );
    /// Get the data for a range of points from this PointBuffer and stores it inside the given memory region.
    /// Panics if any index in index_range is out of bounds. buf must be at least as big as the size of a single
    /// point entry in the corresponding PointLayout of this PointBuffer multiplied by the number of point indices
    /// in index_range
    fn get_points_by_copy(&self, index_range: Range<usize>, buf: &mut [u8]);
    // Get the data for the given attribute for a range of points from this PointBuffer and stores it inside the
    // given memory region. Panics if any index in index_range is out of bounds or if the attribute is not part of
    // the point_layout of this PointBuffer. buf must be at least as big as the size of a single entry of the given
    // attribute multiplied by the number of point indices in index_range.
    fn get_attribute_range_by_copy(
        &self,
        index_range: Range<usize>,
        attribute: &PointAttributeDefinition,
        buf: &mut [u8],
    );

    /// Returns the number of unique point entries stored inside this PointBuffer
    fn len(&self) -> usize;
    /// Returns true if the associated `PointBuffer` is empty, i.e. it stores zero points
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Returns a reference to the underlying PointLayout of this PointBuffer
    fn point_layout(&self) -> &PointLayout;
}

/// Trait for PointBuffer types that store interleaved point data. In an interleaved PointBuffer, all attributes
/// for a single point are stored together in memory. To illustrate this, suppose the PointLayout of some point
/// type defines the default attributes POSITION_3D (Vector3<f64>), INTENSITY (u16) and CLASSIFICATION (u8). In
/// an InterleavedPointBuffer, the data layout is like this:<br>
/// [Vector3<f64>, u16, u8, Vector3<f64>, u16, u8, ...]<br>
///  |------Point 1-------| |------Point 2-------| |--...<br>
pub trait InterleavedPointBuffer: PointBuffer {
    /// Returns a pointer to the raw memory of the point entry at the given index in this PointBuffer. In contrast
    /// to [get_point_by_copy](PointBuffer::get_point_by_copy), this function performs no copy operations and thus can
    /// yield better performance. Panics if point_index is out of bounds.
    fn get_point_by_ref(&self, point_index: usize) -> &[u8];
    /// Returns a pointer to the raw memory of a range of point entries in this PointBuffer. In contrast to
    /// [get_point_by_copy](PointBuffer::get_point_by_copy), this function performs no copy operations and thus can
    /// yield better performance. Panics if any index in index_range is out of bounds.
    fn get_points_by_ref(&self, index_range: Range<usize>) -> &[u8];
}

/// Trait for PointBuffer types that store point data per attribute. In buffers of this type, the data for a single
/// attribute of all points in stored together in memory. To illustrate this, suppose the PointLayout of some point
/// type defines the default attributes POSITION_3D (Vector3<f64>), INTENSITY (u16) and CLASSIFICATION (u8). In
/// a PerAttributePointBuffer, the data layout is like this:<br>
/// [Vector3<f64>, Vector3<f64>, Vector3<f64>, ...]<br>
/// [u16, u16, u16, ...]<br>
/// [u8, u8, u8, ...]<br>
pub trait PerAttributePointBuffer: PointBuffer {
    /// Returns a pointer to the raw memory for the attribute entry of the given point in this PointBuffer. In contrast
    /// to [get_attribute_by_copy](PointBuffer::get_attribute_by_copy), this function performs no copy operations and
    /// thus can yield better performance. Panics if point_index is out of bounds or if the attribute is not part of
    /// the point_layout of this PointBuffer.
    fn get_attribute_by_ref(
        &self,
        point_index: usize,
        attribute: &PointAttributeDefinition,
    ) -> &[u8];
    /// Returns a pointer to the raw memory for the given attribute of a range of points in this PointBuffer. In contrast
    /// to [get_attribute_range_by_copy](PointBuffer::get_attribute_range_by_copy), this function performs no copy operations
    /// and thus can yield better performance. Panics if any index in index_range is out of bounds or if the attribute is
    /// not part of the point_layout of this PointBuffer.
    fn get_attribute_range_by_ref(
        &self,
        index_range: Range<usize>,
        attribute: &PointAttributeDefinition,
    ) -> &[u8];
}

/// PointBuffer type that uses interleaved Vec-based storage for the points
pub struct InterleavedVecPointStorage {
    layout: PointLayout,
    points: Vec<u8>,
    size_of_point_entry: u64,
}

impl InterleavedVecPointStorage {
    /// Creates a new empty InterleavedVecPointStorage with the given PointLayout
    /// ```
    /// # use pasture_core::containers::*;
    /// # use pasture_core::layout::*;
    /// let layout = PointLayout::from_attributes(&[attributes::POSITION_3D]);
    /// let storage = InterleavedVecPointStorage::new(layout);
    /// # assert_eq!(0, storage.len());
    /// ```
    pub fn new(layout: PointLayout) -> Self {
        let size_of_point_entry = layout.size_of_point_entry();
        Self {
            layout,
            points: vec![],
            size_of_point_entry,
        }
    }

    /// Creates a new `InterleavedVecPointStorage` with enough capacity to store `capacity` points using
    /// the given `PointLayout`. Calling this method is similar to `Vec::with_capacity`: Internal memory
    /// is reserved but the `len()` is not affected.
    /// ```
    /// # use pasture_core::containers::*;
    /// # use pasture_core::layout::*;
    /// let layout = PointLayout::from_attributes(&[attributes::POSITION_3D]);
    /// let storage = InterleavedVecPointStorage::with_capacity(16, layout);
    /// # assert_eq!(0, storage.len());
    /// ```
    pub fn with_capacity(capacity: usize, layout: PointLayout) -> Self {
        let size_of_point_entry = layout.size_of_point_entry();
        let bytes_to_reserve = capacity * size_of_point_entry as usize;
        Self {
            layout,
            points: Vec::with_capacity(bytes_to_reserve),
            size_of_point_entry,
        }
    }

    /// Pushes a single point into the associated `InterleavedVecPointStorage`. Panics if the `PointLayout` of
    /// `T` does not match the `PointLayout` of the associated `InterleavedVecPointStorage`. *Note:* For safety
    /// reasons this function performs a `PointLayout` check. If you want to add many points quickly, either use
    /// the `push_points` variant which takes a range, or use the `push_point_unchecked` variant to circumvent checks.
    /// ```
    /// # use pasture_core::containers::*;
    /// # use pasture_core::layout::*;
    ///
    /// struct MyPointType(u16);
    ///
    /// impl PointType for MyPointType {
    ///   fn layout() -> PointLayout {
    ///     PointLayout::from_attributes(&[attributes::INTENSITY])
    ///   }
    /// }
    ///
    /// {
    ///   let mut storage = InterleavedVecPointStorage::new(MyPointType::layout());
    ///   storage.push_point(MyPointType(42));
    /// }
    /// ```
    pub fn push_point<T: PointType>(&mut self, point: T) {
        let point_layout = T::layout();
        if point_layout != self.layout {
            panic!(
                "push_point: PointLayouts don't match (T has layout {:?}, self has layout {:?})",
                point_layout, self.layout
            );
        }

        self.reserve(1);
        let point_bytes_and_size = unsafe { view_raw_bytes(&point) };

        self.points.extend_from_slice(point_bytes_and_size);
    }

    pub fn push_point_unchecked<T: PointType>(&mut self, _point: T) {
        todo!("implement")
    }

    pub fn push_points<T: PointType>(&mut self, _points: &[T]) {
        todo!("implement")
    }

    pub fn push_points_unchecked<T: PointType>(&mut self, _points: &[T]) {
        todo!("implement")
    }

    /// Reserve capacity for at least `additional_points` new points to be inserted into this `PointBuffer`
    fn reserve(&mut self, additional_points: usize) {
        let additional_bytes = additional_points * self.size_of_point_entry as usize;
        self.points.reserve(additional_bytes);
    }
}

impl PointBuffer for InterleavedVecPointStorage {
    fn get_point_by_copy(&self, point_index: usize, buf: &mut [u8]) {
        if point_index >= self.len() {
            panic!(
                "InterleavedVecPointStorage::get_point_by_copy: Point index {} out of bounds!",
                point_index
            );
        }

        let offset_to_point_bytes = point_index * self.size_of_point_entry as usize;
        buf.copy_from_slice(
            &self.points
                [offset_to_point_bytes..offset_to_point_bytes + self.size_of_point_entry as usize],
        );
    }

    fn get_attribute_by_copy(
        &self,
        point_index: usize,
        attribute: &PointAttributeDefinition,
        buf: &mut [u8],
    ) {
        if point_index >= self.len() {
            panic!(
                "InterleavedVecPointStorage::get_attribute_by_copy: Point index {} out of bounds!",
                point_index
            );
        }

        if let Some(attribute_offset) = self.layout.offset_of(attribute) {
            let offset_to_point_bytes = point_index * self.size_of_point_entry as usize;
            let offset_to_attribute = offset_to_point_bytes + attribute_offset as usize;
            let attribute_size = attribute.size() as usize;

            buf.copy_from_slice(
                &self.points[offset_to_attribute..offset_to_attribute + attribute_size],
            );
        } else {
            panic!("InterleavedVecPointStorage::get_attribute_by_copy: Attribute {:?} is not part of this PointBuffer's PointLayout!", attribute);
        }
    }

    fn get_points_by_copy(&self, point_indices: Range<usize>, buf: &mut [u8]) {
        let points_ref = self.get_points_by_ref(point_indices);
        buf.copy_from_slice(points_ref);
    }

    fn get_attribute_range_by_copy(
        &self,
        point_indices: Range<usize>,
        attribute: &PointAttributeDefinition,
        buf: &mut [u8],
    ) {
        if point_indices.end > self.len() {
            panic!(
                "InterleavedVecPointStorage::get_attribute_range_by_copy: Point indices {:?} out of bounds!",
                point_indices
            );
        }

        if let Some(attribute_offset) = self.layout.offset_of(attribute) {
            let attribute_size = attribute.size() as usize;
            let start_index = point_indices.start;

            for point_index in point_indices {
                let offset_to_point_bytes = point_index * self.size_of_point_entry as usize;
                let offset_to_attribute = offset_to_point_bytes + attribute_offset as usize;
                let offset_in_target_buf = (point_index - start_index) * attribute_size;
                let target_buf_slice =
                    &mut buf[offset_in_target_buf..offset_in_target_buf + attribute_size];

                target_buf_slice.copy_from_slice(
                    &self.points[offset_to_attribute..offset_to_attribute + attribute_size],
                );
            }
        } else {
            panic!("InterleavedVecPointStorage::get_attribute_by_copy: Attribute {:?} is not part of this PointBuffer's PointLayout!", attribute);
        }
    }

    fn len(&self) -> usize {
        self.points.len() / self.size_of_point_entry as usize
    }

    fn point_layout(&self) -> &PointLayout {
        &self.layout
    }
}

impl InterleavedPointBuffer for InterleavedVecPointStorage {
    fn get_point_by_ref(&self, point_index: usize) -> &[u8] {
        if point_index >= self.len() {
            panic!(
                "InterleavedVecPointStorage::get_point_by_ref: Point index {} out of bounds!",
                point_index
            );
        }

        let offset_to_point = point_index * self.size_of_point_entry as usize;
        &self.points[offset_to_point..offset_to_point + self.size_of_point_entry as usize]
    }

    fn get_points_by_ref(&self, point_indices: Range<usize>) -> &[u8] {
        if point_indices.end > self.len() {
            panic!(
                "InterleavedVecPointStorage::get_points_by_ref: Point indices {:?} out of bounds!",
                point_indices
            );
        }

        let offset_to_point = point_indices.start * self.size_of_point_entry as usize;
        let total_bytes_of_range =
            (point_indices.end - point_indices.start) * self.size_of_point_entry as usize;
        &self.points[offset_to_point..offset_to_point + total_bytes_of_range]
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::layout::{attributes, PointLayout};
    use crate::util::view_raw_bytes;

    #[repr(packed)]
    #[derive(Debug, Copy, Clone)]
    struct TestPointType(u16, f64);

    impl PointType for TestPointType {
        fn layout() -> PointLayout {
            PointLayout::from_attributes(&[attributes::INTENSITY, attributes::GPS_TIME])
        }
    }

    #[test]
    fn test_interleaved_vec_storage_len() {
        let mut storage = InterleavedVecPointStorage::new(TestPointType::layout());

        assert_eq!(0, storage.len());

        storage.push_point(TestPointType(42, 0.123));
        storage.push_point(TestPointType(43, 0.345));

        assert_eq!(2, storage.len());
    }

    // In the following two tests we test for byte equality when calling the raw API of `PointBuffer`
    // Mapping between bytes and strongly typed values is not tested here but instead in `views.rs`

    #[test]
    fn test_interleaved_vec_storage_get_point() {
        let mut storage = InterleavedVecPointStorage::new(TestPointType::layout());

        let reference_point_1 = TestPointType(42, 0.123);
        let reference_point_2 = TestPointType(42, 0.456);
        storage.push_point(reference_point_1);
        storage.push_point(reference_point_2);

        let reference_bytes_1 = unsafe { view_raw_bytes(&reference_point_1) };
        let reference_bytes_2 = unsafe { view_raw_bytes(&reference_point_2) };
        let mut reference_bytes_all = reference_bytes_1.iter().copied().collect::<Vec<_>>();
        reference_bytes_all.extend(reference_bytes_2);

        let mut buf: Vec<u8> = vec![0; reference_bytes_1.len()];

        storage.get_point_by_copy(0, &mut buf[..]);
        assert_eq!(
            reference_bytes_1, buf,
            "get_point_by_copy: Bytes are not equal!"
        );
        storage.get_point_by_copy(1, &mut buf[..]);
        assert_eq!(
            reference_bytes_2, buf,
            "get_point_by_copy: Bytes are not equal!"
        );

        let mut buf_for_both_points: Vec<u8> = vec![0; reference_bytes_all.len()];
        storage.get_points_by_copy(0..2, &mut buf_for_both_points[..]);
        assert_eq!(
            reference_bytes_all, buf_for_both_points,
            "get_points_by_copy: Bytes are not equal!"
        );

        let point_1_bytes_ref = storage.get_point_by_ref(0);
        assert_eq!(
            reference_bytes_1, point_1_bytes_ref,
            "get_point_by_ref: Bytes are not equal!"
        );

        let point_2_bytes_ref = storage.get_point_by_ref(1);
        assert_eq!(
            reference_bytes_2, point_2_bytes_ref,
            "get_point_by_ref: Bytes are not equal!"
        );

        let both_points_bytes_ref = storage.get_points_by_ref(0..2);
        assert_eq!(
            reference_bytes_all, both_points_bytes_ref,
            "get_points_by_ref: Bytes are not equal!"
        );
    }

    #[test]
    fn test_interleaved_vec_storage_get_attribute() {
        let mut storage = InterleavedVecPointStorage::new(TestPointType::layout());

        let reference_point_1 = TestPointType(42, 0.123);
        let reference_point_2 = TestPointType(42, 0.456);
        storage.push_point(reference_point_1);
        storage.push_point(reference_point_2);

        // Get the raw byte views for both attributes of both points
        let reference_bytes_1 = unsafe { view_raw_bytes(&reference_point_1) };
        let ref_bytes_p1_a1 = &reference_bytes_1[0..2];
        let ref_bytes_p1_a2 = &reference_bytes_1[2..10];

        let reference_bytes_2 = unsafe { view_raw_bytes(&reference_point_2) };
        let ref_bytes_p2_a1 = &reference_bytes_2[0..2];
        let ref_bytes_p2_a2 = &reference_bytes_2[2..10];

        let mut ref_bytes_all_a1 = ref_bytes_p1_a1.iter().copied().collect::<Vec<_>>();
        ref_bytes_all_a1.extend(ref_bytes_p2_a1);
        let mut ref_bytes_all_a2 = ref_bytes_p1_a2.iter().copied().collect::<Vec<_>>();
        ref_bytes_all_a2.extend(ref_bytes_p2_a2);

        // Get the attribute bytes through calls to the API of `PointBuffer`
        let mut attribute_1_buf: Vec<u8> = vec![0; 2];
        let mut attribute_2_buf: Vec<u8> = vec![0; 8];

        storage.get_attribute_by_copy(0, &attributes::INTENSITY, &mut attribute_1_buf[..]);
        assert_eq!(
            ref_bytes_p1_a1, attribute_1_buf,
            "get_attribute_by_copy: Bytes are not equal"
        );
        storage.get_attribute_by_copy(1, &attributes::INTENSITY, &mut attribute_1_buf[..]);
        assert_eq!(
            ref_bytes_p2_a1, attribute_1_buf,
            "get_attribute_by_copy: Bytes are not equal"
        );

        storage.get_attribute_by_copy(0, &attributes::GPS_TIME, &mut attribute_2_buf[..]);
        assert_eq!(
            ref_bytes_p1_a2, attribute_2_buf,
            "get_attribute_by_copy: Bytes are not equal"
        );
        storage.get_attribute_by_copy(1, &attributes::GPS_TIME, &mut attribute_2_buf[..]);
        assert_eq!(
            ref_bytes_p2_a2, attribute_2_buf,
            "get_attribute_by_copy: Bytes are not equal"
        );

        let mut all_attribute_1_buf: Vec<u8> = vec![0; 4];
        let mut all_attribute_2_buf: Vec<u8> = vec![0; 16];

        storage.get_attribute_range_by_copy(
            0..2,
            &attributes::INTENSITY,
            &mut all_attribute_1_buf[..],
        );
        assert_eq!(
            ref_bytes_all_a1, all_attribute_1_buf,
            "get_attribute_range_by_copy: Bytes are not equal"
        );

        storage.get_attribute_range_by_copy(
            0..2,
            &attributes::GPS_TIME,
            &mut all_attribute_2_buf[..],
        );
        assert_eq!(
            ref_bytes_all_a2, all_attribute_2_buf,
            "get_attribute_range_by_copy: Bytes are not equal"
        );
    }
}
