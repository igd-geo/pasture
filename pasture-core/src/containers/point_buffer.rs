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
    size_of_point_entry: usize,
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
        let bytes_to_reserve = capacity * size_of_point_entry;
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
        let additional_bytes = additional_points * self.size_of_point_entry;
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

        let offset_to_point_bytes = point_index * self.size_of_point_entry;
        buf.copy_from_slice(
            &self.points[offset_to_point_bytes..offset_to_point_bytes + self.size_of_point_entry],
        );
    }
    fn get_attribute_by_copy(&self, _: usize, _: &PointAttributeDefinition, _: &mut [u8]) {
        todo!()
    }

    fn get_points_by_copy(&self, _: Range<usize>, _: &mut [u8]) {
        todo!()
    }

    fn get_attribute_range_by_copy(
        &self,
        _: Range<usize>,
        _: &PointAttributeDefinition,
        _: &mut [u8],
    ) {
        todo!()
    }

    fn len(&self) -> usize {
        self.points.len() / self.layout.size_of_point_entry()
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

        let offset_to_point = point_index * self.size_of_point_entry;
        &self.points[offset_to_point..offset_to_point + self.size_of_point_entry]
    }

    fn get_points_by_ref(&self, point_indices: Range<usize>) -> &[u8] {
        if point_indices.end > self.len() {
            panic!(
                "InterleavedVecPointStorage::get_points_by_ref: Point indices {:?} out of bounds!",
                point_indices
            );
        }

        let offset_to_point = point_indices.start * self.size_of_point_entry;
        let total_bytes_of_range =
            (point_indices.end - point_indices.start) * self.size_of_point_entry;
        &self.points[offset_to_point..offset_to_point + total_bytes_of_range]
    }
}
