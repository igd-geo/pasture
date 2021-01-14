use std::collections::HashMap;
use std::ops::Index;
use std::ops::Range;

use crate::util::view_raw_bytes;
use crate::{
    layout::{PointAttributeDefinition, PointLayout, PointType, PrimitiveType},
    util::sort_untyped_slice_by_permutation,
};

use super::{InterleavedPointView, PerAttributePointView};

/**
 * TODOs
 *
 * Traits:
 *  - PointBuffer
 *      - Tests
 *  - PointBufferWriteable
 *  - InterleavedPointBuffer
 *  - InterleavedPointBufferMut
 *  - PerAttributePointBuffer
 *  - PerAttributePointBufferMut
 *
 * Structs:
 *  - InterleavedVecPointStorage
 *  - InterleavedPointBufferSlice
 *  - InterleavedPointBufferSliceMut
 *    - Implement
 *    - Tests
 *  - PerAttributeVecPointStorage
 *  - PerAttributePointBufferSlice
 *  - PerAttributePointBufferSliceMut
 *    - Implement
 *    - Tests
 */

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

/// Trait for all mutable `PointBuffer`s, that is all `PointBuffer`s where it is possible to push points into. Distinguishing between
/// read-only `PointBuffer` and mutable `PointBufferMut` traits enables read-only, non-owning views of a `PointBuffer` with the same interface
/// as a owning `PointBuffer`!
pub trait PointBufferWriteable: PointBuffer {
    /// Push points in interleaved format into the associated `PointBuffer`
    fn push_points_interleaved(&mut self, points: &InterleavedPointView<'_>);

    /// Push points in per-attribute format into the associated `PointBuffer`
    ///
    /// # Panics
    ///
    /// If an attribute from the `PointLayout` of the associated `PointBuffer` is missing in `attribute_buffers`.
    fn push_raw_points_per_attribute(&mut self, points: &PerAttributePointView<'_>);

    /// Appends all points from the given `InterleavedPointBuffer` to the end of the associated `PointBufferWriteable`
    fn extend_from_interleaved(&mut self, points: &dyn InterleavedPointBuffer);

    /// Appends all points from the given `PerAttributePointBuffer` to the end of the associated `PointBufferWriteable`
    fn extend_from_per_attribute(&mut self, points: &dyn PerAttributePointBuffer);

    /// Clears the contents of the associated `PointBufferMut`
    fn clear(&mut self);
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
    fn get_point_ref(&self, point_index: usize) -> &[u8];
    /// Returns a pointer to the raw memory of a range of point entries in this PointBuffer. In contrast to
    /// [get_point_by_copy](PointBuffer::get_point_by_copy), this function performs no copy operations and thus can
    /// yield better performance. Panics if any index in index_range is out of bounds.
    fn get_points_ref(&self, index_range: Range<usize>) -> &[u8];
}

/// Trait for PointBuffer types that store interleaved point data and are mutable
pub trait InterleavedPointBufferMut: InterleavedPointBuffer {
    /// Mutable version of [get_point_ref](InterleavedPointBuffer::get_point_ref)
    fn get_point_mut(&mut self, point_index: usize) -> &mut [u8];
    /// Mutable version of [get_points_ref](InterleavedPointBuffer::get_points_ref)
    fn get_points_mut(&mut self, index_range: Range<usize>) -> &mut [u8];
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
    fn get_attribute_ref(&self, point_index: usize, attribute: &PointAttributeDefinition) -> &[u8];
    /// Returns a pointer to the raw memory for the given attribute of a range of points in this PointBuffer. In contrast
    /// to [get_attribute_range_by_copy](PointBuffer::get_attribute_range_by_copy), this function performs no copy operations
    /// and thus can yield better performance. Panics if any index in index_range is out of bounds or if the attribute is
    /// not part of the point_layout of this PointBuffer.
    fn get_attribute_range_ref(
        &self,
        index_range: Range<usize>,
        attribute: &PointAttributeDefinition,
    ) -> &[u8];
}

pub trait PerAttributePointBufferMut: PerAttributePointBuffer {
    /// Mutable version of [get_attribute_ref](PerAttributePointBuffer::get_attribute_ref)
    fn get_attribute_mut(
        &mut self,
        point_index: usize,
        attribute: &PointAttributeDefinition,
    ) -> &mut [u8];
    /// Mutable version of [get_attribute_range_ref](PerAttributePointBuffer::get_attribute_range_ref)
    fn get_attribute_range_mut(
        &mut self,
        index_range: Range<usize>,
        attribute: &PointAttributeDefinition,
    ) -> &mut [u8];
}

/// PointBuffer type that uses interleaved Vec-based storage for the points
pub struct InterleavedVecPointStorage {
    layout: PointLayout,
    points: Vec<u8>,
    size_of_point_entry: u64,
}

impl InterleavedVecPointStorage {
    /// Creates a new empty `InterleavedVecPointStorage` with the given `PointLayout`
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

    /// Pushes a single point into the associated `InterleavedVecPointStorage`. *Note:* For safety
    /// reasons this function performs a `PointLayout` check. If you want to add many points quickly, either use
    /// the `push_points` variant which takes a range, or use the `push_point_unchecked` variant to circumvent checks.
    ///
    /// # Panics
    ///
    /// If the `PointLayout` of `T` does not match the `PointLayout` of the associated `InterleavedVecPointStorage`.
    ///
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

        self.push_point_unchecked(point);
    }

    /// Pushes a single point into the associated `InterleavedVecPointStorage`. *Note:* This method performs no checks
    /// regarding the point type `T`, so it is very unsafe! Only call this method if you know the point type matches
    /// the `PointLayout` of the associated `InterleavedVecPointStorage`!
    ///
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
    ///   storage.push_point_unchecked(MyPointType(42));
    /// }
    /// ```
    pub fn push_point_unchecked<T: PointType>(&mut self, point: T) {
        self.reserve(1);
        let point_bytes_and_size = unsafe { view_raw_bytes(&point) };

        self.points.extend_from_slice(point_bytes_and_size);
    }

    /// Pushes a range of points into the associated `InterleavedVecPointStorage`.
    ///
    /// # Panics
    ///
    /// If the `PointLayout` of type `T` does not match the layout of the associated `InterleavedVecPointStorage`.
    ///
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
    ///   let points = vec![MyPointType(42), MyPointType(43)];
    ///   storage.push_points(&points);
    /// }
    /// ```
    pub fn push_points<T: PointType>(&mut self, points: &[T]) {
        let point_layout = T::layout();
        if point_layout != self.layout {
            panic!(
                "push_points: PointLayouts don't match (T has layout {:?}, self has layout {:?})",
                point_layout, self.layout
            );
        }

        let points_as_bytes = unsafe {
            std::slice::from_raw_parts(
                points.as_ptr() as *const u8,
                points.len() * std::mem::size_of::<T>(),
            )
        };
        self.points.extend_from_slice(points_as_bytes);
    }

    /// Returns a slice of the associated `InterleavedVecPointStorage`
    pub fn slice(&self, range: Range<usize>) -> InterleavedPointBufferSlice<'_> {
        InterleavedPointBufferSlice::new(self, range)
    }

    /// Sorts all points in the associated `InterleavedVecPointStorage` using the order of the `PointType` `T`.
    ///
    /// # Panics
    ///
    /// If the `PointLayout` of `T` does not match the underlying layout
    pub fn sort<T: PointType + Ord>(&mut self) {
        if self.layout != T::layout() {
            panic!("InterleavedVecPointStorage::sort: Point type `T` does not match layout of this buffer!");
        }

        let typed_points = unsafe {
            std::slice::from_raw_parts_mut(self.points.as_mut_ptr() as *mut T, self.len())
        };
        typed_points.sort();
    }

    /// Sorts all points in the associated `InterleavedVecPointStorage` using the given comparator function
    ///
    /// # Panics
    ///
    /// If the `PointLayout` of `T` does not match the underlying layout
    pub fn sort_by<T: PointType, C: FnMut(&T, &T) -> std::cmp::Ordering>(&mut self, comparator: C) {
        if self.layout != T::layout() {
            panic!("InterleavedVecPointStorage::sort_by: Point type `T` does not match layout of this buffer!");
        }

        let typed_points = unsafe {
            std::slice::from_raw_parts_mut(self.points.as_mut_ptr() as *mut T, self.len())
        };
        typed_points.sort_by(comparator);
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
        let points_ref = self.get_points_ref(point_indices);
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

impl PointBufferWriteable for InterleavedVecPointStorage {
    fn push_points_interleaved(&mut self, points: &InterleavedPointView<'_>) {
        self.points.extend_from_slice(points.get_raw_data());
    }

    fn push_raw_points_per_attribute(&mut self, points: &PerAttributePointView<'_>) {
        if *points.get_point_layout() != self.layout {
            panic!("InterleavedVecPointStorage::push_raw_points_per_attribute: attribute_buffers do not match the PointLayout! Make sure that there is one entry for each PointAttributeDefinition in the PointLayout of this PointBuffer and that the order matches!");
        }

        // This function is essentially a data transpose!
        let attribute_buffers = self
            .layout
            .attributes()
            .map(|attribute| {
                (
                    attribute,
                    points.get_raw_data_for_attribute(attribute).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        for idx in 0..points.len() {
            for (attribute, attribute_buffer) in attribute_buffers.iter() {
                let slice_start = idx * attribute.size() as usize;
                let slice_end = (idx + 1) * attribute.size() as usize;
                self.points
                    .extend_from_slice(&attribute_buffer[slice_start..slice_end]);
            }
        }
    }

    fn extend_from_interleaved(&mut self, points: &dyn InterleavedPointBuffer) {
        if *points.point_layout() != self.layout {
            panic!("InterleavedVecPointStorage::extend_from_interleaved: Point layouts do not match, this buffer has layout {:?} but buffer to append from has layout {:?}", self.layout, points.point_layout());
        }

        self.points
            .extend_from_slice(points.get_points_ref(0..points.len()));
    }

    fn extend_from_per_attribute(&mut self, points: &dyn PerAttributePointBuffer) {
        if *points.point_layout() != self.layout {
            panic!("InterleavedVecPointStorage::extend_from_per_attribute: Point layouts do not match, this buffer has layout {:?} but buffer to append from has layout {:?}", self.layout, points.point_layout());
        }

        todo!()
    }

    fn clear(&mut self) {
        self.points.clear();
    }
}

impl InterleavedPointBuffer for InterleavedVecPointStorage {
    fn get_point_ref(&self, point_index: usize) -> &[u8] {
        if point_index >= self.len() {
            panic!(
                "InterleavedVecPointStorage::get_point_ref: Point index {} out of bounds!",
                point_index
            );
        }

        let offset_to_point = point_index * self.size_of_point_entry as usize;
        &self.points[offset_to_point..offset_to_point + self.size_of_point_entry as usize]
    }

    fn get_points_ref(&self, index_range: Range<usize>) -> &[u8] {
        if index_range.end > self.len() {
            panic!(
                "InterleavedVecPointStorage::get_points_ref: Point indices {:?} out of bounds!",
                index_range
            );
        }

        let offset_to_point = index_range.start * self.size_of_point_entry as usize;
        let total_bytes_of_range =
            (index_range.end - index_range.start) * self.size_of_point_entry as usize;
        &self.points[offset_to_point..offset_to_point + total_bytes_of_range]
    }
}

impl InterleavedPointBufferMut for InterleavedVecPointStorage {
    fn get_point_mut(&mut self, point_index: usize) -> &mut [u8] {
        if point_index >= self.len() {
            panic!(
                "InterleavedVecPointStorage::get_point_mut: Point index {} out of bounds!",
                point_index
            );
        }

        let offset_to_point = point_index * self.size_of_point_entry as usize;
        &mut self.points[offset_to_point..offset_to_point + self.size_of_point_entry as usize]
    }

    fn get_points_mut(&mut self, index_range: Range<usize>) -> &mut [u8] {
        if index_range.end > self.len() {
            panic!(
                "InterleavedVecPointStorage::get_points_mut: Point indices {:?} out of bounds!",
                index_range
            );
        }

        let offset_to_point = index_range.start * self.size_of_point_entry as usize;
        let total_bytes_of_range =
            (index_range.end - index_range.start) * self.size_of_point_entry as usize;
        &mut self.points[offset_to_point..offset_to_point + total_bytes_of_range]
    }
}

pub struct PerAttributeVecPointStorage {
    layout: PointLayout,
    attributes: HashMap<&'static str, Vec<u8>>,
}

impl PerAttributeVecPointStorage {
    /// Creates a new empty `PerAttributeVecPointStorage` with the given `PointLayout`
    /// ```
    /// # use pasture_core::containers::*;
    /// # use pasture_core::layout::*;
    /// let layout = PointLayout::from_attributes(&[attributes::POSITION_3D]);
    /// let storage = PerAttributeVecPointStorage::new(layout);
    /// # assert_eq!(0, storage.len());
    /// ```
    pub fn new(layout: PointLayout) -> Self {
        let attributes = layout
            .attributes()
            .map(|attribute| (attribute.name(), vec![]))
            .collect::<HashMap<_, _>>();
        Self { layout, attributes }
    }

    /// Creates a new `PerAttributeVecPointStorage` with enough capacity to store `capacity` points using
    /// the given `PointLayout`. Calling this method is similar to `Vec::with_capacity`: Internal memory
    /// is reserved but the `len()` is not affected.
    /// ```
    /// # use pasture_core::containers::*;
    /// # use pasture_core::layout::*;
    /// let layout = PointLayout::from_attributes(&[attributes::POSITION_3D]);
    /// let storage = PerAttributeVecPointStorage::with_capacity(16, layout);
    /// # assert_eq!(0, storage.len());
    /// ```
    pub fn with_capacity(capacity: usize, layout: PointLayout) -> Self {
        let attributes = layout
            .attributes()
            .map(|attribute| {
                let attribute_bytes = capacity * attribute.size() as usize;
                (attribute.name(), Vec::with_capacity(attribute_bytes))
            })
            .collect::<HashMap<_, _>>();
        Self { layout, attributes }
    }

    /// Pushes a single point into the associated `PerAttributeVecPointStorage`.
    ///
    /// # Panics
    ///
    /// If the `PointLayout` of type `T` does not match the layout of the associated `PerAttributeVecPointStorage`.
    ///
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
    ///   let mut storage = PerAttributeVecPointStorage::new(MyPointType::layout());
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

        let point_bytes = unsafe { view_raw_bytes(&point) };
        for attribute in self.layout.attributes() {
            let offset_to_attribute_in_point = self.layout.offset_of(attribute).unwrap() as usize;
            let point_slice = &point_bytes[offset_to_attribute_in_point
                ..offset_to_attribute_in_point + attribute.size() as usize];

            let attribute_buffer = self.attributes.get_mut(attribute.name()).unwrap();
            attribute_buffer.extend_from_slice(point_slice);
        }
    }

    /// Pushes a range of points into the associated `PerAttributeVecPointStorage`.
    ///
    /// # Panics
    ///
    /// If the `PointLayout` of type `T` does not match the layout of the associated `PerAttributeVecPointStorage`.
    ///
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
    ///   let mut storage = PerAttributeVecPointStorage::new(MyPointType::layout());
    ///   storage.push_points(&[MyPointType(42), MyPointType(43)]);
    /// }
    /// ```
    pub fn push_points<T: PointType>(&mut self, points: &[T]) {
        let point_layout = T::layout();
        if point_layout != self.layout {
            panic!(
                "push_point: PointLayouts don't match (T has layout {:?}, self has layout {:?})",
                point_layout, self.layout
            );
        }

        self.reserve(points.len());

        for attribute in self.layout.attributes() {
            let offset_to_attribute_in_point = self.layout.offset_of(attribute).unwrap() as usize;
            let attribute_buffer = self.attributes.get_mut(attribute.name()).unwrap();

            for point in points {
                let point_bytes = unsafe { view_raw_bytes(point) };
                let point_slice = &point_bytes[offset_to_attribute_in_point
                    ..offset_to_attribute_in_point + attribute.size() as usize];

                attribute_buffer.extend_from_slice(point_slice);
            }
        }
    }

    /// Pushes a single attribute into the associated `PerAttributeVecPointStorage`. *Note:* This function adds only part
    /// of a point to the storage, thus leaving the interal data structure in a state where one attribute buffer might contain
    /// more entries than the buffer for another attribute. It is the users responsibility to ensure that all attributes for
    /// all points are fully added before iterating over the contents of the associated `PerAttributeVecPointStorage`!
    ///
    /// TODO its probably better to create a separate type that takes ownership of the `PerAttributeVecPointStorage`, allows only
    /// these edit operations and then has a method that checks correctness of the internal datastructure before yielding the
    /// `PerAttributeVecPointStorage` again
    ///
    /// # Panics
    ///
    /// If the given `PointAttributeDefinition` is not part of the internal `PointLayout` of the associated `PerAttributeVecPointStorage`
    ///
    /// ```
    /// # use pasture_core::containers::*;
    /// # use pasture_core::layout::*;
    ///
    /// #[repr(packed)]
    /// struct MyPointType(u16, f64);
    ///
    /// impl PointType for MyPointType {
    ///   fn layout() -> PointLayout {
    ///     PointLayout::from_attributes(&[attributes::INTENSITY, attributes::GPS_TIME])
    ///   }
    /// }
    ///
    /// {
    ///   let mut storage = PerAttributeVecPointStorage::new(MyPointType::layout());
    ///   storage.push_attribute(&attributes::INTENSITY, 42);
    ///   storage.push_attribute(&attributes::GPS_TIME, 0.123);
    /// }
    /// ```
    pub fn push_attribute<T: PrimitiveType>(
        &mut self,
        attribute: &PointAttributeDefinition,
        value: T,
    ) {
        match self.attributes.get_mut(attribute.name()) {
            None => panic!("PerAttributeVecPointStorage::push_attribute: Attribute {:?} is not part of this PointBuffer's PointLayout!", attribute),
            Some(attribute_buffer) => {
                let value_bytes = unsafe { view_raw_bytes(&value) };
                attribute_buffer.extend_from_slice(value_bytes);
            },
        }
    }

    /// Pushes a range of values for a single attribute into the associated `PerAttributeVecPointStorage`. *Note:* This function adds only
    /// part of a point to the storage, thus leaving the interal data structure in a state where one attribute buffer might contain
    /// more entries than the buffer for another attribute. It is the users responsibility to ensure that all attributes for
    /// all points are fully added before iterating over the contents of the associated `PerAttributeVecPointStorage`!
    ///
    /// TODO its probably better to create a separate type that takes ownership of the `PerAttributeVecPointStorage`, allows only
    /// these edit operations and then has a method that checks correctness of the internal datastructure before yielding the
    /// `PerAttributeVecPointStorage` again
    ///
    /// # Panics
    ///
    /// If the given `PointAttributeDefinition` is not part of the internal `PointLayout` of the associated `PerAttributeVecPointStorage`
    ///
    /// ```
    /// # use pasture_core::containers::*;
    /// # use pasture_core::layout::*;
    ///
    /// #[repr(packed)]
    /// struct MyPointType(u16, f64);
    ///
    /// impl PointType for MyPointType {
    ///   fn layout() -> PointLayout {
    ///     PointLayout::from_attributes(&[attributes::INTENSITY, attributes::GPS_TIME])
    ///   }
    /// }
    ///
    /// {
    ///   let mut storage = PerAttributeVecPointStorage::new(MyPointType::layout());
    ///   storage.push_attribute_range(&attributes::INTENSITY, &[42, 43, 44]);
    ///   storage.push_attribute_range(&attributes::GPS_TIME, &[0.123, 0.456, 0.789]);
    /// }
    /// ```
    pub fn push_attribute_range<T: PrimitiveType>(
        &mut self,
        attribute: &PointAttributeDefinition,
        values: &[T],
    ) {
        match self.attributes.get_mut(attribute.name()) {
            None => panic!("PerAttributeVecPointStorage::push_attributes: Attribute {:?} is not part of this PointBuffer's PointLayout!", attribute),
            Some(attribute_buffer) => {
                let value_bytes = unsafe { &*(values as *const [T] as *const [u8]) };
                attribute_buffer.extend_from_slice(value_bytes);
            },
        }
    }

    /// Returns a read-only slice of the associated `PerAttributePointBuffer`
    pub fn slice(&self, range: Range<usize>) -> PerAttributePointBufferSlice<'_> {
        PerAttributePointBufferSlice::new(self, range)
    }

    /// Sorts all points in the associated `PerAttributePointBuffer` using the order of the `PointType` `T`.
    ///
    /// # Panics
    ///
    /// If the `PointLayout` of `T` does not match the underlying layout
    pub fn sort_by_attribute<T: PrimitiveType + Ord>(
        &mut self,
        attribute: &PointAttributeDefinition,
    ) {
        // TODO What is the fastest way to sort multiple vectors based on one of the vectors as the sort key?
        // Here we simply create an index permutation and then shuffle all vectors using that permutation

        let typed_attribute = self.attributes.get(attribute.name()).map(|untyped_attribute| {
            return unsafe {
                std::slice::from_raw_parts(untyped_attribute.as_ptr() as *const T, self.len())
            };
        }).expect(&format!("PerAttributePointBuffer:sort_by_attribute: Attribute {:?} not contained in this buffers PointLayout!", attribute));

        let mut indices = (0..self.len()).collect::<Vec<_>>();
        indices.sort_by(|&idx_a, &idx_b| typed_attribute[idx_a].cmp(&typed_attribute[idx_b]));

        let attribute_sizes = self
            .attributes
            .keys()
            .map(|&key| self.layout.get_attribute_by_name(key).unwrap().size())
            .collect::<Vec<_>>();

        self.attributes
            .values_mut()
            .enumerate()
            .for_each(|(idx, untyped_attribute)| {
                let attribute_size = attribute_sizes[idx];
                sort_untyped_slice_by_permutation(
                    untyped_attribute.as_mut_slice(),
                    indices.as_slice(),
                    attribute_size as usize,
                );
            });
    }

    /// Reserves space for at least `additional_points` additional points in the associated `PerAttributeVecPointStorage`
    fn reserve(&mut self, additional_points: usize) {
        for attribute in self.layout.attributes() {
            let attribute_buffer = self.attributes.get_mut(attribute.name()).unwrap();
            let additional_bytes = additional_points * attribute.size() as usize;
            attribute_buffer.reserve(additional_bytes);
        }
    }
}

impl PointBuffer for PerAttributeVecPointStorage {
    fn get_point_by_copy(&self, point_index: usize, buf: &mut [u8]) {
        if point_index >= self.len() {
            panic!(
                "PerAttributeVecPointStorage::get_point_by_copy: Point index {} out of bounds!",
                point_index
            );
        }

        for attribute in self.layout.attributes() {
            let attribute_buffer = self.attributes.get(attribute.name()).unwrap();
            let attribute_size = attribute.size() as usize;
            let offset_in_buffer = point_index * attribute_size;
            let offset_in_point = self.layout.offset_of(attribute).unwrap() as usize;

            let buf_slice = &mut buf[offset_in_point..offset_in_point + attribute_size];
            let attribute_slice =
                &attribute_buffer[offset_in_buffer..offset_in_buffer + attribute_size];
            buf_slice.copy_from_slice(attribute_slice);
        }
    }

    fn get_attribute_by_copy(
        &self,
        point_index: usize,
        attribute: &PointAttributeDefinition,
        buf: &mut [u8],
    ) {
        let attribute_slice = self.get_attribute_ref(point_index, attribute);
        buf.copy_from_slice(attribute_slice);
    }

    fn get_points_by_copy(&self, point_indices: Range<usize>, buf: &mut [u8]) {
        if point_indices.end > self.len() {
            panic!(
                "PerAttributeVecPointStorage::get_points_by_copy: Point indices {:?} out of bounds!",
                point_indices
            );
        }

        let point_size = self.layout.size_of_point_entry() as usize;

        for attribute in self.layout.attributes() {
            let attribute_buffer = self.attributes.get(attribute.name()).unwrap();
            let attribute_size = attribute.size() as usize;
            for point_index in point_indices.clone() {
                // Get the appropriate subsections of the attribute buffer and the point buffer that is passed in
                let offset_in_attribute_buffer = point_index * attribute_size;
                let attribute_slice = &attribute_buffer
                    [offset_in_attribute_buffer..offset_in_attribute_buffer + attribute_size];

                let offset_in_point = self.layout.offset_of(attribute).unwrap() as usize;
                let offset_in_points_buffer = point_index * point_size + offset_in_point;
                let buf_slice =
                    &mut buf[offset_in_points_buffer..offset_in_points_buffer + attribute_size];

                buf_slice.copy_from_slice(attribute_slice);
            }
        }
    }

    fn get_attribute_range_by_copy(
        &self,
        point_indices: Range<usize>,
        attribute: &PointAttributeDefinition,
        buf: &mut [u8],
    ) {
        let attribute_buffer_slice = self.get_attribute_range_ref(point_indices, attribute);
        buf.copy_from_slice(attribute_buffer_slice);
    }

    fn len(&self) -> usize {
        // TODO This assumes that the buffer is always complete, i.e. all attribute buffers store the same number of points
        // This relates to the comment of the `push_attribute` method
        let attribute = self.layout.attributes().next().unwrap();
        let attribute_buf = self.attributes.get(attribute.name()).unwrap();
        attribute_buf.len() / attribute.size() as usize
    }

    fn point_layout(&self) -> &PointLayout {
        &self.layout
    }
}

impl PointBufferWriteable for PerAttributeVecPointStorage {
    fn push_points_interleaved(&mut self, points: &InterleavedPointView<'_>) {
        if points.get_point_layout() != self.point_layout() {
            panic!("PerAttributeVecPointStorage::push_points_interleaved: Layout of 'points' does not match layout of this buffer!");
        }

        let raw_point_data = points.get_raw_data();
        let stride = self.layout.size_of_point_entry() as usize;

        for (attribute_name, attribute_data) in self.attributes.iter_mut() {
            let current_attribute = self.layout.get_attribute_by_name(attribute_name).unwrap();
            let base_offset = self.layout.offset_of(current_attribute).unwrap() as usize;
            let attribute_size = current_attribute.size() as usize;

            for idx in 0..points.len() {
                let attribute_start = base_offset + (idx * stride);
                let attribute_end = attribute_start + attribute_size;
                let attribute_slice = &raw_point_data[attribute_start..attribute_end];
                attribute_data.extend_from_slice(attribute_slice);
            }
        }
    }

    fn push_raw_points_per_attribute(&mut self, points: &PerAttributePointView<'_>) {
        if *points.get_point_layout() != self.layout {
            panic!("PerAttributeVecPointStorage::push_raw_points_per_attribute: points layout does not match this PointLayout!");
        }
        for attribute in self.layout.attributes() {
            self.attributes
                .get_mut(attribute.name())
                .unwrap()
                .extend_from_slice(points.get_raw_data_for_attribute(attribute).unwrap());
        }
    }

    fn extend_from_interleaved(&mut self, points: &dyn InterleavedPointBuffer) {
        todo!()
    }

    fn extend_from_per_attribute(&mut self, points: &dyn PerAttributePointBuffer) {
        if *points.point_layout() != self.layout {
            panic!("PerAttributeVecPointStorage::extend_from_per_attribute: Point layouts do not match, this buffer has layout {:?} but buffer to append from has layout {:?}", self.layout, points.point_layout());
        }

        for (&key, attribute_data) in self.attributes.iter_mut() {
            let attribute_def = self
                .layout
                .get_attribute_by_name(key)
                .expect("Attribute not present in point layout");
            let new_attribute_data = points.get_attribute_range_ref(0..points.len(), attribute_def);
            attribute_data.extend_from_slice(new_attribute_data);
        }
    }

    fn clear(&mut self) {
        self.attributes.iter_mut().for_each(|(_, vec)| vec.clear());
    }
}

impl PerAttributePointBuffer for PerAttributeVecPointStorage {
    fn get_attribute_ref(&self, point_index: usize, attribute: &PointAttributeDefinition) -> &[u8] {
        if point_index >= self.len() {
            panic!(
                "PerAttributeVecPointStorage::get_attribute_ref: Point index {} out of bounds!",
                point_index
            );
        }

        if !self.layout.has_attribute(attribute.name()) {
            panic!("PerAttributeVecPointStorage::get_attribute_ref: Attribute {:?} is not part of this PointBuffer's PointLayout!", attribute);
        }

        let attribute_buffer = self.attributes.get(attribute.name()).unwrap();
        let attribute_size = attribute.size() as usize;
        let offset_in_attribute_buffer = point_index * attribute_size;
        &attribute_buffer[offset_in_attribute_buffer..offset_in_attribute_buffer + attribute_size]
    }

    fn get_attribute_range_ref(
        &self,
        index_range: Range<usize>,
        attribute: &PointAttributeDefinition,
    ) -> &[u8] {
        if index_range.end > self.len() {
            panic!(
                "PerAttributeVecPointStorage::get_attribute_range_ref: Point indices {:?} out of bounds!",
                index_range
            );
        }

        if !self.layout.has_attribute(attribute.name()) {
            panic!("PerAttributeVecPointStorage::get_attribute_ref: Attribute {:?} is not part of this PointBuffer's PointLayout!", attribute);
        }

        let attribute_buffer = self.attributes.get(attribute.name()).unwrap();
        let start_offset_in_attribute_buffer = index_range.start * attribute.size() as usize;
        let end_offset_in_attribute_buffer = start_offset_in_attribute_buffer
            + (index_range.end - index_range.start) * attribute.size() as usize;
        &attribute_buffer[start_offset_in_attribute_buffer..end_offset_in_attribute_buffer]
    }
}

impl PerAttributePointBufferMut for PerAttributeVecPointStorage {
    fn get_attribute_mut(
        &mut self,
        point_index: usize,
        attribute: &PointAttributeDefinition,
    ) -> &mut [u8] {
        if point_index >= self.len() {
            panic!(
                "PerAttributeVecPointStorage::get_attribute_mut: Point index {} out of bounds!",
                point_index
            );
        }

        if !self.layout.has_attribute(attribute.name()) {
            panic!("PerAttributeVecPointStorage::get_attribute_mut: Attribute {:?} is not part of this PointBuffer's PointLayout!", attribute);
        }

        let attribute_buffer = self.attributes.get_mut(attribute.name()).unwrap();
        let attribute_size = attribute.size() as usize;
        let offset_in_attribute_buffer = point_index * attribute_size;
        &mut attribute_buffer
            [offset_in_attribute_buffer..offset_in_attribute_buffer + attribute_size]
    }

    fn get_attribute_range_mut(
        &mut self,
        index_range: Range<usize>,
        attribute: &PointAttributeDefinition,
    ) -> &mut [u8] {
        if index_range.end > self.len() {
            panic!(
                "PerAttributeVecPointStorage::get_attribute_range_mut: Point indices {:?} out of bounds!",
                index_range
            );
        }

        if !self.layout.has_attribute(attribute.name()) {
            panic!("PerAttributeVecPointStorage::get_attribute_range_mut: Attribute {:?} is not part of this PointBuffer's PointLayout!", attribute);
        }

        let attribute_buffer = self.attributes.get_mut(attribute.name()).unwrap();
        let start_offset_in_attribute_buffer = index_range.start * attribute.size() as usize;
        let end_offset_in_attribute_buffer = start_offset_in_attribute_buffer
            + (index_range.end - index_range.start) * attribute.size() as usize;
        &mut attribute_buffer[start_offset_in_attribute_buffer..end_offset_in_attribute_buffer]
    }
}

/// Non-owning, read-only slice of the data of an `InterleavedPointBuffer`
pub struct InterleavedPointBufferSlice<'p> {
    buffer: &'p dyn InterleavedPointBuffer,
    range_in_buffer: Range<usize>,
}

// impl <'p> Index<Range<usize>> for InterleavedPointBuffer {
//     type Output = InterleavedPointBufferSlice<'p>;

//     fn index(&self, index: Idx) -> &Self::Output {
//         // Cannot return Self::Output instead of &Self::Output :(
//     }
// }

impl<'p> InterleavedPointBufferSlice<'p> {
    /// Creates a new `InterleavedPointBufferSlice` pointing to the given range within the given buffer
    pub fn new(buffer: &'p dyn InterleavedPointBuffer, range_in_buffer: Range<usize>) -> Self {
        if range_in_buffer.end > buffer.len() {
            panic!(
                "InterleavedPointBufferSlice::new: Range {:?} is out of bounds!",
                range_in_buffer
            );
        }
        Self {
            buffer,
            range_in_buffer,
        }
    }
}

impl<'p> PointBuffer for InterleavedPointBufferSlice<'p> {
    fn get_point_by_copy(&self, point_index: usize, buf: &mut [u8]) {
        let point_index_in_buffer = point_index + self.range_in_buffer.start;
        self.buffer.get_point_by_copy(point_index_in_buffer, buf);
    }

    fn get_attribute_by_copy(
        &self,
        point_index: usize,
        attribute: &PointAttributeDefinition,
        buf: &mut [u8],
    ) {
        let point_index_in_buffer = point_index + self.range_in_buffer.start;
        self.buffer
            .get_attribute_by_copy(point_index_in_buffer, attribute, buf);
    }

    fn get_points_by_copy(&self, index_range: Range<usize>, buf: &mut [u8]) {
        let range_in_buffer = index_range.start + self.range_in_buffer.start
            ..index_range.end + self.range_in_buffer.start;
        self.buffer.get_points_by_copy(range_in_buffer, buf);
    }

    fn get_attribute_range_by_copy(
        &self,
        index_range: Range<usize>,
        attribute: &PointAttributeDefinition,
        buf: &mut [u8],
    ) {
        let range_in_buffer = index_range.start + self.range_in_buffer.start
            ..index_range.end + self.range_in_buffer.start;
        self.buffer
            .get_attribute_range_by_copy(range_in_buffer, attribute, buf);
    }

    fn len(&self) -> usize {
        self.range_in_buffer.end - self.range_in_buffer.start
    }

    fn point_layout(&self) -> &PointLayout {
        self.buffer.point_layout()
    }
}

impl<'p> InterleavedPointBuffer for InterleavedPointBufferSlice<'p> {
    fn get_point_ref(&self, point_index: usize) -> &[u8] {
        let point_index_in_buffer = point_index + self.range_in_buffer.start;
        self.buffer.get_point_ref(point_index_in_buffer)
    }

    fn get_points_ref(&self, index_range: Range<usize>) -> &[u8] {
        let range_in_buffer = index_range.start + self.range_in_buffer.start
            ..index_range.end + self.range_in_buffer.start;
        self.buffer.get_points_ref(range_in_buffer)
    }
}

pub struct PerAttributePointBufferSlice<'p> {
    buffer: &'p dyn PerAttributePointBuffer,
    range_in_buffer: Range<usize>,
}

impl<'p> PerAttributePointBufferSlice<'p> {
    /// Creates a new `PerAttributePointBufferSlice` pointing to the given range within the given buffer
    pub fn new(buffer: &'p dyn PerAttributePointBuffer, range_in_buffer: Range<usize>) -> Self {
        if range_in_buffer.end > buffer.len() {
            panic!(
                "PerAttributePointBufferSlice::new: Range {:?} is out of bounds!",
                range_in_buffer
            );
        }
        Self {
            buffer,
            range_in_buffer,
        }
    }
}

impl<'p> PointBuffer for PerAttributePointBufferSlice<'p> {
    fn get_point_by_copy(&self, point_index: usize, buf: &mut [u8]) {
        let point_index_in_buffer = point_index + self.range_in_buffer.start;
        self.buffer.get_point_by_copy(point_index_in_buffer, buf);
    }

    fn get_attribute_by_copy(
        &self,
        point_index: usize,
        attribute: &PointAttributeDefinition,
        buf: &mut [u8],
    ) {
        let point_index_in_buffer = point_index + self.range_in_buffer.start;
        self.buffer
            .get_attribute_by_copy(point_index_in_buffer, attribute, buf);
    }

    fn get_points_by_copy(&self, index_range: Range<usize>, buf: &mut [u8]) {
        let range_in_buffer = index_range.start + self.range_in_buffer.start
            ..index_range.end + self.range_in_buffer.start;
        self.buffer.get_points_by_copy(range_in_buffer, buf);
    }

    fn get_attribute_range_by_copy(
        &self,
        index_range: Range<usize>,
        attribute: &PointAttributeDefinition,
        buf: &mut [u8],
    ) {
        let range_in_buffer = index_range.start + self.range_in_buffer.start
            ..index_range.end + self.range_in_buffer.start;
        self.buffer
            .get_attribute_range_by_copy(range_in_buffer, attribute, buf);
    }

    fn len(&self) -> usize {
        self.range_in_buffer.end - self.range_in_buffer.start
    }

    fn point_layout(&self) -> &PointLayout {
        self.buffer.point_layout()
    }
}

impl<'p> PerAttributePointBuffer for PerAttributePointBufferSlice<'p> {
    fn get_attribute_ref(&self, point_index: usize, attribute: &PointAttributeDefinition) -> &[u8] {
        let point_index_in_buffer = point_index + self.range_in_buffer.start;
        self.buffer
            .get_attribute_ref(point_index_in_buffer, attribute)
    }

    fn get_attribute_range_ref(
        &self,
        index_range: Range<usize>,
        attribute: &PointAttributeDefinition,
    ) -> &[u8] {
        let range_in_buffer = index_range.start + self.range_in_buffer.start
            ..index_range.end + self.range_in_buffer.start;
        self.buffer
            .get_attribute_range_ref(range_in_buffer, attribute)
    }
}

#[cfg(test)]
mod tests {

    use nalgebra::Vector3;

    use super::*;
    use crate::util::view_raw_bytes;
    use crate::{
        layout::{attributes, PointLayout},
        util::view_raw_bytes_mut,
    };

    #[repr(packed)]
    #[derive(Debug, Copy, Clone, PartialEq)]
    struct TestPointType(u16, f64);

    impl PointType for TestPointType {
        fn layout() -> PointLayout {
            PointLayout::from_attributes(&[attributes::INTENSITY, attributes::GPS_TIME])
        }
    }

    #[repr(packed)]
    #[derive(Debug, Copy, Clone, PartialEq)]
    struct OtherPointType(Vector3<f64>, u8);

    impl PointType for OtherPointType {
        fn layout() -> PointLayout {
            PointLayout::from_attributes(&[attributes::POSITION_3D, attributes::RETURN_NUMBER])
        }
    }

    trait OpqaueInterleavedBuffer: InterleavedPointBufferMut + PointBufferWriteable {}
    impl OpqaueInterleavedBuffer for InterleavedVecPointStorage {}

    trait OpqauePerAttributeBuffer: PerAttributePointBufferMut + PointBufferWriteable {}
    impl OpqauePerAttributeBuffer for PerAttributeVecPointStorage {}

    fn get_empty_interleaved_point_buffer(layout: PointLayout) -> Box<dyn OpqaueInterleavedBuffer> {
        Box::new(InterleavedVecPointStorage::new(layout))
    }

    fn get_interleaved_point_buffer_from_points<T: PointType>(
        points: &[T],
    ) -> Box<dyn OpqaueInterleavedBuffer> {
        let mut buffer = InterleavedVecPointStorage::new(T::layout());
        buffer.push_points(points);
        Box::new(buffer)
    }

    fn get_empty_per_attribute_point_buffer(
        layout: PointLayout,
    ) -> Box<dyn OpqauePerAttributeBuffer> {
        Box::new(PerAttributeVecPointStorage::new(layout))
    }

    fn get_per_attribute_point_buffer_from_points<T: PointType>(
        points: &[T],
    ) -> Box<dyn OpqauePerAttributeBuffer> {
        let mut buffer = PerAttributeVecPointStorage::new(T::layout());
        buffer.push_points(points);
        Box::new(buffer)
    }

    #[test]
    fn test_point_buffer_len() {
        let interleaved_buffer = get_empty_interleaved_point_buffer(TestPointType::layout());
        assert_eq!(0, interleaved_buffer.len());

        let per_attribute_buffer = get_empty_per_attribute_point_buffer(TestPointType::layout());
        assert_eq!(0, per_attribute_buffer.len());
    }

    #[test]
    fn test_point_buffer_is_empty() {
        let interleaved_buffer = get_empty_interleaved_point_buffer(TestPointType::layout());
        assert!(interleaved_buffer.is_empty());

        let per_attribute_buffer = get_empty_per_attribute_point_buffer(TestPointType::layout());
        assert!(per_attribute_buffer.is_empty());
    }

    #[test]
    fn test_point_buffer_get_layout() {
        let interleaved_buffer = get_empty_interleaved_point_buffer(TestPointType::layout());
        assert_eq!(&TestPointType::layout(), interleaved_buffer.point_layout());

        let per_attribute_buffer = get_empty_per_attribute_point_buffer(TestPointType::layout());
        assert_eq!(
            &TestPointType::layout(),
            per_attribute_buffer.point_layout()
        );
    }

    #[test]
    fn test_point_buffer_get_point_by_copy() {
        let interleaved_buffer =
            get_interleaved_point_buffer_from_points(&[TestPointType(42, 0.123)]);

        let mut ref_point = TestPointType(0, 0.0);
        unsafe {
            interleaved_buffer.get_point_by_copy(0, view_raw_bytes_mut(&mut ref_point));
        }

        assert_eq!(TestPointType(42, 0.123), ref_point);

        let per_attribute_buffer =
            get_per_attribute_point_buffer_from_points(&[TestPointType(43, 0.456)]);
        unsafe {
            per_attribute_buffer.get_point_by_copy(0, view_raw_bytes_mut(&mut ref_point));
        }

        assert_eq!(TestPointType(43, 0.456), ref_point);
    }

    #[test]
    #[should_panic]
    fn test_point_buffer_get_point_by_copy_on_empty_interleaved_buffer() {
        let buffer = get_empty_interleaved_point_buffer(TestPointType::layout());

        let mut ref_point = TestPointType(0, 0.0);
        unsafe {
            buffer.get_point_by_copy(0, view_raw_bytes_mut(&mut ref_point));
        }
    }

    #[test]
    #[should_panic]
    fn test_point_buffer_get_point_by_copy_on_empty_per_attribute_buffer() {
        let buffer = get_empty_per_attribute_point_buffer(TestPointType::layout());

        let mut ref_point = TestPointType(0, 0.0);
        unsafe {
            buffer.get_point_by_copy(0, view_raw_bytes_mut(&mut ref_point));
        }
    }

    #[test]
    fn test_point_buffer_get_attribute_by_copy() {
        let interleaved_buffer =
            get_interleaved_point_buffer_from_points(&[TestPointType(42, 0.123)]);

        let mut ref_attribute: u16 = 0;
        unsafe {
            interleaved_buffer.get_attribute_by_copy(
                0,
                &attributes::INTENSITY,
                view_raw_bytes_mut(&mut ref_attribute),
            );
        }

        assert_eq!(42, ref_attribute);

        let per_attribute_buffer =
            get_per_attribute_point_buffer_from_points(&[TestPointType(43, 0.456)]);
        unsafe {
            per_attribute_buffer.get_attribute_by_copy(
                0,
                &attributes::INTENSITY,
                view_raw_bytes_mut(&mut ref_attribute),
            );
        }

        assert_eq!(43, ref_attribute);
    }

    #[test]
    #[should_panic]
    fn test_point_buffer_get_attribute_by_copy_on_empty_interleaved_buffer() {
        let buffer = get_empty_interleaved_point_buffer(TestPointType::layout());

        let mut ref_attribute: u16 = 0;
        unsafe {
            buffer.get_attribute_by_copy(
                0,
                &attributes::INTENSITY,
                view_raw_bytes_mut(&mut ref_attribute),
            );
        }
    }

    #[test]
    #[should_panic]
    fn test_point_buffer_get_attribute_by_copy_on_empty_per_attribute_buffer() {
        let buffer = get_empty_per_attribute_point_buffer(TestPointType::layout());

        let mut ref_attribute: u16 = 0;
        unsafe {
            buffer.get_attribute_by_copy(
                0,
                &attributes::INTENSITY,
                view_raw_bytes_mut(&mut ref_attribute),
            );
        }
    }

    #[test]
    #[should_panic]
    fn test_point_buffer_get_attribute_by_copy_for_invalid_attribute_interleaved() {
        let buffer = get_interleaved_point_buffer_from_points(&[TestPointType(42, 0.123)]);

        let mut ref_attribute: u16 = 0;
        unsafe {
            buffer.get_attribute_by_copy(
                0,
                &attributes::POSITION_3D,
                view_raw_bytes_mut(&mut ref_attribute),
            );
        }
    }

    #[test]
    #[should_panic]
    fn test_point_buffer_get_attribute_by_copy_for_invalid_attribute_per_attribute() {
        let buffer = get_per_attribute_point_buffer_from_points(&[TestPointType(43, 0.456)]);

        let mut ref_attribute: u16 = 0;
        unsafe {
            buffer.get_attribute_by_copy(
                0,
                &attributes::POSITION_3D,
                view_raw_bytes_mut(&mut ref_attribute),
            );
        }
    }

    #[test]
    fn test_point_buffer_get_points_by_copy() {
        let interleaved_buffer = get_interleaved_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);

        let mut ref_points = [TestPointType(0, 0.0), TestPointType(0, 0.0)];
        unsafe { interleaved_buffer.get_points_by_copy(0..2, view_raw_bytes_mut(&mut ref_points)) }

        assert_eq!(TestPointType(42, 0.123), ref_points[0]);
        assert_eq!(TestPointType(43, 0.456), ref_points[1]);

        let per_attribute_buffer = get_per_attribute_point_buffer_from_points(&[
            TestPointType(44, 0.321),
            TestPointType(45, 0.654),
        ]);
        unsafe {
            per_attribute_buffer.get_points_by_copy(0..2, view_raw_bytes_mut(&mut ref_points));
        }

        assert_eq!(TestPointType(44, 0.321), ref_points[0]);
        assert_eq!(TestPointType(45, 0.654), ref_points[1]);
    }

    #[test]
    #[should_panic]
    fn test_point_buffer_get_points_by_copy_out_of_bounds_interleaved() {
        let interleaved_buffer = get_empty_interleaved_point_buffer(TestPointType::layout());
        interleaved_buffer.get_points_by_copy(0..2, &mut [0]);
    }

    #[test]
    #[should_panic]
    fn test_point_buffer_get_points_by_copy_out_of_bounds_per_attribute() {
        let interleaved_buffer = get_empty_per_attribute_point_buffer(TestPointType::layout());
        interleaved_buffer.get_points_by_copy(0..2, &mut [0]);
    }

    #[test]
    fn test_point_buffer_get_attribute_range_by_copy() {
        let interleaved_buffer = get_interleaved_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);

        let mut ref_attributes: [u16; 2] = [0, 0];
        unsafe {
            interleaved_buffer.get_attribute_range_by_copy(
                0..2,
                &attributes::INTENSITY,
                view_raw_bytes_mut(&mut ref_attributes),
            )
        }

        assert_eq!([42, 43], ref_attributes);

        let per_attribute_buffer = get_per_attribute_point_buffer_from_points(&[
            TestPointType(44, 0.321),
            TestPointType(45, 0.654),
        ]);
        unsafe {
            per_attribute_buffer.get_attribute_range_by_copy(
                0..2,
                &attributes::INTENSITY,
                view_raw_bytes_mut(&mut ref_attributes),
            )
        }

        assert_eq!([44, 45], ref_attributes);
    }

    #[test]
    #[should_panic]
    fn test_point_buffer_get_attribute_range_by_copy_invalid_attribute_interleaved() {
        let buffer = get_interleaved_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);
        buffer.get_attribute_range_by_copy(0..2, &attributes::POINT_ID, &mut [0]);
    }

    #[test]
    #[should_panic]
    fn test_point_buffer_get_attribute_range_by_copy_invalid_attribute_per_attribute() {
        let buffer = get_per_attribute_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);
        buffer.get_attribute_range_by_copy(0..2, &attributes::POINT_ID, &mut [0]);
    }

    #[test]
    fn test_point_buffer_writeable_clear() {
        let mut interleaved_buffer = get_interleaved_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);

        interleaved_buffer.clear();
        assert_eq!(0, interleaved_buffer.len());

        let mut per_attribute_buffer = get_per_attribute_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);
        per_attribute_buffer.clear();
        assert_eq!(0, per_attribute_buffer.len());
    }

    #[test]
    fn test_point_buffer_writeable_push_points_interleaved() {
        let mut interleaved_buffer = get_empty_interleaved_point_buffer(TestPointType::layout());
        interleaved_buffer.push_points_interleaved(&InterleavedPointView::from_slice(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]));

        assert_eq!(2, interleaved_buffer.len());

        let mut ref_points = [TestPointType(0, 0.0), TestPointType(0, 0.0)];
        unsafe { interleaved_buffer.get_points_by_copy(0..2, view_raw_bytes_mut(&mut ref_points)) }

        assert_eq!(TestPointType(42, 0.123), ref_points[0]);
        assert_eq!(TestPointType(43, 0.456), ref_points[1]);

        let mut per_attribute_buffer =
            get_empty_per_attribute_point_buffer(TestPointType::layout());
        per_attribute_buffer.push_points_interleaved(&InterleavedPointView::from_slice(&[
            TestPointType(44, 0.321),
            TestPointType(45, 0.654),
        ]));

        assert_eq!(2, per_attribute_buffer.len());

        unsafe {
            per_attribute_buffer.get_points_by_copy(0..2, view_raw_bytes_mut(&mut ref_points))
        }

        assert_eq!(TestPointType(44, 0.321), ref_points[0]);
        assert_eq!(TestPointType(45, 0.654), ref_points[1]);
    }

    #[test]
    fn test_point_buffer_writeable_push_raw_points_per_attribute() {
        let mut data_view = PerAttributePointView::new();
        let intensities: [u16; 2] = [42, 43];
        let gps_times: [f64; 2] = [0.123, 0.456];
        data_view.push_attribute(&intensities, &attributes::INTENSITY);
        data_view.push_attribute(&gps_times, &attributes::GPS_TIME);

        let mut interleaved_buffer = get_empty_interleaved_point_buffer(TestPointType::layout());
        interleaved_buffer.push_raw_points_per_attribute(&data_view);

        assert_eq!(2, interleaved_buffer.len());

        let mut ref_points = [TestPointType(0, 0.0), TestPointType(0, 0.0)];
        unsafe { interleaved_buffer.get_points_by_copy(0..2, view_raw_bytes_mut(&mut ref_points)) }

        assert_eq!(TestPointType(42, 0.123), ref_points[0]);
        assert_eq!(TestPointType(43, 0.456), ref_points[1]);

        let mut per_attribute_buffer =
            get_empty_per_attribute_point_buffer(TestPointType::layout());
        per_attribute_buffer.push_raw_points_per_attribute(&data_view);

        assert_eq!(2, per_attribute_buffer.len());

        ref_points[0] = TestPointType(0, 0.0);
        ref_points[1] = TestPointType(0, 0.0);

        unsafe {
            per_attribute_buffer.get_points_by_copy(0..2, view_raw_bytes_mut(&mut ref_points))
        }

        assert_eq!(TestPointType(42, 0.123), ref_points[0]);
        assert_eq!(TestPointType(43, 0.456), ref_points[1]);
    }

    #[test]
    #[should_panic]
    fn test_point_buffer_writeable_interleaved_push_raw_points_per_attribute_invalid_layout() {
        let data_view = PerAttributePointView::new();

        let mut buffer = get_empty_interleaved_point_buffer(TestPointType::layout());
        buffer.push_raw_points_per_attribute(&data_view);
    }

    #[test]
    #[should_panic]
    fn test_point_buffer_writeable_per_attribute_push_raw_points_per_attribute_invalid_layout() {
        let data_view = PerAttributePointView::new();

        let mut buffer = get_empty_per_attribute_point_buffer(TestPointType::layout());
        buffer.push_raw_points_per_attribute(&data_view);
    }

    #[test]
    fn test_interleaved_point_buffer_get_point_ref() {
        let interleaved_buffer = get_interleaved_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);

        let first_point_ref = interleaved_buffer.get_point_ref(0);
        let first_point_ref_typed = unsafe { *(first_point_ref.as_ptr() as *const TestPointType) };

        assert_eq!(TestPointType(42, 0.123), first_point_ref_typed);

        let second_point_ref = interleaved_buffer.get_point_ref(1);
        let second_point_ref_typed =
            unsafe { *(second_point_ref.as_ptr() as *const TestPointType) };

        assert_eq!(TestPointType(43, 0.456), second_point_ref_typed);
    }

    #[test]
    #[should_panic]
    fn test_interleaved_point_buffer_get_point_ref_on_empty_buffer() {
        let buffer = get_empty_interleaved_point_buffer(TestPointType::layout());

        buffer.get_point_ref(0);
    }

    #[test]
    fn test_interleaved_point_buffer_get_points_ref() {
        let interleaved_buffer = get_interleaved_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);

        let points_ref = interleaved_buffer.get_points_ref(0..2);
        let typed_points_ref =
            unsafe { std::slice::from_raw_parts(points_ref.as_ptr() as *const TestPointType, 2) };

        let reference_points = &[TestPointType(42, 0.123), TestPointType(43, 0.456)];

        assert_eq!(reference_points, typed_points_ref);
    }

    #[test]
    #[should_panic]
    fn test_interleaved_point_buffer_get_points_ref_on_empty_buffer() {
        let buffer = get_empty_interleaved_point_buffer(TestPointType::layout());

        buffer.get_points_ref(0..2);
    }

    #[test]
    fn test_interleaved_point_buffer_mut_get_point_mut() {
        let mut interleaved_buffer = get_interleaved_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);

        {
            let mut_point = interleaved_buffer.get_point_mut(0);
            let mut_point_typed = unsafe { &mut *(mut_point.as_mut_ptr() as *mut TestPointType) };

            mut_point_typed.0 = 128;
            mut_point_typed.1 = 3.14159;
        }

        let point_ref = interleaved_buffer.get_point_ref(0);
        let point_ref_typed = unsafe { *(point_ref.as_ptr() as *const TestPointType) };

        assert_eq!(TestPointType(128, 3.14159), point_ref_typed);
    }

    #[test]
    #[should_panic]
    fn test_interleaved_point_buffer_mut_get_point_mut_on_empty_buffer() {
        let mut buffer = get_empty_interleaved_point_buffer(TestPointType::layout());
        buffer.get_point_mut(0);
    }

    #[test]
    fn test_interleaved_point_buffer_mut_get_points_mut() {
        let mut interleaved_buffer = get_interleaved_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);

        let reference_points = &[TestPointType(128, 3.14159), TestPointType(129, 2.71828)];

        {
            let mut_points = interleaved_buffer.get_points_mut(0..2);
            let mut_points_typed = unsafe {
                std::slice::from_raw_parts_mut(mut_points.as_mut_ptr() as *mut TestPointType, 2)
            };

            mut_points_typed[0] = reference_points[0];
            mut_points_typed[1] = reference_points[1];
        }

        let points_ref = interleaved_buffer.get_points_ref(0..2);
        let typed_points_ref =
            unsafe { std::slice::from_raw_parts(points_ref.as_ptr() as *const TestPointType, 2) };

        assert_eq!(reference_points, typed_points_ref);
    }

    #[test]
    #[should_panic]
    fn test_interleaved_point_buffer_mut_get_points_mut_on_empty_buffer() {
        let mut buffer = get_empty_interleaved_point_buffer(TestPointType::layout());
        buffer.get_points_mut(0..2);
    }

    #[test]
    fn test_per_attribute_point_buffer_get_attribute_ref() {
        let buffer = get_per_attribute_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);

        let first_point_attribute_ref = buffer.get_attribute_ref(0, &attributes::INTENSITY);
        let first_point_attribute_ref_typed =
            unsafe { *(first_point_attribute_ref.as_ptr() as *const u16) };

        assert_eq!(42, first_point_attribute_ref_typed);

        let second_point_attribute_ref = buffer.get_attribute_ref(1, &attributes::GPS_TIME);
        let second_point_attribute_ref_typed =
            unsafe { *(second_point_attribute_ref.as_ptr() as *const f64) };

        assert_eq!(0.456, second_point_attribute_ref_typed);
    }

    #[test]
    #[should_panic]
    fn test_per_attribute_point_buffer_get_attribute_ref_out_of_bounds() {
        let buffer = get_empty_per_attribute_point_buffer(TestPointType::layout());
        buffer.get_attribute_ref(0, &attributes::INTENSITY);
    }

    #[test]
    #[should_panic]
    fn test_per_attribute_point_buffer_get_attribute_ref_invalid_attribute() {
        let buffer = get_per_attribute_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);
        buffer.get_attribute_ref(0, &attributes::SCAN_ANGLE_RANK);
    }

    #[test]
    fn test_per_attribute_point_buffer_get_attribute_range_ref() {
        let buffer = get_per_attribute_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);

        let intensity_attribute_range =
            buffer.get_attribute_range_ref(0..2, &attributes::INTENSITY);
        let intensity_attribute_range_typed = unsafe {
            std::slice::from_raw_parts(intensity_attribute_range.as_ptr() as *const u16, 2)
        };

        assert_eq!(&[42, 43], intensity_attribute_range_typed);

        let gps_time_attribute_range = buffer.get_attribute_range_ref(0..2, &attributes::GPS_TIME);
        let gps_time_attribute_range_typed = unsafe {
            std::slice::from_raw_parts(gps_time_attribute_range.as_ptr() as *const f64, 2)
        };

        assert_eq!(&[0.123, 0.456], gps_time_attribute_range_typed);
    }

    #[test]
    #[should_panic]
    fn test_per_attribute_point_buffer_get_attribute_range_ref_out_of_bounds() {
        let buffer = get_empty_per_attribute_point_buffer(TestPointType::layout());
        buffer.get_attribute_range_ref(0..2, &attributes::INTENSITY);
    }

    #[test]
    #[should_panic]
    fn test_per_attribute_point_buffer_get_attribute_range_ref_invalid_attribute() {
        let buffer = get_per_attribute_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);
        buffer.get_attribute_range_ref(0..2, &attributes::SCAN_ANGLE_RANK);
    }

    #[test]
    fn test_per_attribute_point_buffer_mut_get_attribute_mut() {
        let mut buffer = get_per_attribute_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);

        {
            let first_point_attribute_mut = buffer.get_attribute_mut(0, &attributes::INTENSITY);
            let first_point_attribute_mut_typed =
                unsafe { &mut *(first_point_attribute_mut.as_mut_ptr() as *mut u16) };
            *first_point_attribute_mut_typed = 128;
        }

        let first_point_attribute_ref = buffer.get_attribute_ref(0, &attributes::INTENSITY);
        let first_point_attribute_ref_typed =
            unsafe { *(first_point_attribute_ref.as_ptr() as *const u16) };

        assert_eq!(128, first_point_attribute_ref_typed);

        {
            let second_point_attribute_mut = buffer.get_attribute_mut(1, &attributes::GPS_TIME);
            let second_point_attribute_mut_typed =
                unsafe { &mut *(second_point_attribute_mut.as_mut_ptr() as *mut f64) };
            *second_point_attribute_mut_typed = 3.14159;
        }

        let second_point_attribute_ref = buffer.get_attribute_ref(1, &attributes::GPS_TIME);
        let second_point_attribute_ref_typed =
            unsafe { *(second_point_attribute_ref.as_ptr() as *const f64) };

        assert_eq!(3.14159, second_point_attribute_ref_typed);
    }

    #[test]
    #[should_panic]
    fn test_per_attribute_point_buffer_mut_get_attribute_mut_out_of_bounds() {
        let mut buffer = get_empty_per_attribute_point_buffer(TestPointType::layout());
        buffer.get_attribute_mut(0, &attributes::INTENSITY);
    }

    #[test]
    #[should_panic]
    fn test_per_attribute_point_buffer_mut_get_attribute_mut_invalid_attribute() {
        let mut buffer = get_per_attribute_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);
        buffer.get_attribute_mut(0, &attributes::SCAN_ANGLE_RANK);
    }

    #[test]
    fn test_per_attribute_point_buffer_mut_get_attribute_range_mut() {
        let mut buffer = get_per_attribute_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);

        {
            let intensities_mut = buffer.get_attribute_range_mut(0..2, &attributes::INTENSITY);
            let intensities_mut_typed = unsafe {
                std::slice::from_raw_parts_mut(intensities_mut.as_mut_ptr() as *mut u16, 2)
            };

            intensities_mut_typed[0] = 128;
            intensities_mut_typed[1] = 129;
        }

        let intensity_attribute_range =
            buffer.get_attribute_range_ref(0..2, &attributes::INTENSITY);
        let intensity_attribute_range_typed = unsafe {
            std::slice::from_raw_parts(intensity_attribute_range.as_ptr() as *const u16, 2)
        };

        assert_eq!(&[128, 129], intensity_attribute_range_typed);

        {
            let gps_times_mut = buffer.get_attribute_range_mut(0..2, &attributes::GPS_TIME);
            let gps_times_mut_typed = unsafe {
                std::slice::from_raw_parts_mut(gps_times_mut.as_mut_ptr() as *mut f64, 2)
            };

            gps_times_mut_typed[0] = 3.14159;
            gps_times_mut_typed[1] = 2.71828;
        }

        let gps_time_attribute_range = buffer.get_attribute_range_ref(0..2, &attributes::GPS_TIME);
        let gps_time_attribute_range_typed = unsafe {
            std::slice::from_raw_parts(gps_time_attribute_range.as_ptr() as *const f64, 2)
        };

        assert_eq!(&[3.14159, 2.71828], gps_time_attribute_range_typed);
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

        let point_1_bytes_ref = storage.get_point_ref(0);
        assert_eq!(
            reference_bytes_1, point_1_bytes_ref,
            "get_point_by_ref: Bytes are not equal!"
        );

        let point_2_bytes_ref = storage.get_point_ref(1);
        assert_eq!(
            reference_bytes_2, point_2_bytes_ref,
            "get_point_by_ref: Bytes are not equal!"
        );

        let both_points_bytes_ref = storage.get_points_ref(0..2);
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

    #[test]
    #[should_panic]
    fn test_interleaved_vec_storage_push_point_invalid_format() {
        let mut buffer = InterleavedVecPointStorage::new(TestPointType::layout());
        buffer.push_point(OtherPointType(Vector3::new(0.0, 1.0, 2.0), 23));
    }

    #[test]
    #[should_panic]
    fn test_interleaved_vec_storage_push_points_invalid_format() {
        let mut buffer = InterleavedVecPointStorage::new(TestPointType::layout());
        buffer.push_points(&[OtherPointType(Vector3::new(0.0, 1.0, 2.0), 23)]);
    }

    #[test]
    fn test_interleaved_vec_storage_slice() {
        let mut buffer = InterleavedVecPointStorage::new(TestPointType::layout());

        let reference_point_1 = TestPointType(42, 0.123);
        let reference_point_2 = TestPointType(42, 0.456);
        buffer.push_point(reference_point_1);
        buffer.push_point(reference_point_2);

        let slice = buffer.slice(0..1);
        assert_eq!(1, slice.len());

        let first_point_ref = slice.get_point_ref(0);
        let first_point_ref_typed = unsafe { &*(first_point_ref.as_ptr() as *const TestPointType) };

        assert_eq!(&reference_point_1, first_point_ref_typed);
    }

    #[test]
    fn test_per_attribute_vec_storage_len() {
        let storage1 = PerAttributeVecPointStorage::new(TestPointType::layout());

        assert_eq!(0, storage1.len());

        let mut storage2 = PerAttributeVecPointStorage::with_capacity(2, TestPointType::layout());

        storage2.push_point(TestPointType(42, 0.123));
        storage2.push_point(TestPointType(43, 0.456));

        assert_eq!(2, storage2.len());
    }

    #[test]
    fn test_per_attribute_vec_storage_get_point() {
        let mut storage = PerAttributeVecPointStorage::new(TestPointType::layout());

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
    }

    #[test]
    fn test_per_attribute_vec_storage_get_attribute() {
        let mut storage = PerAttributeVecPointStorage::new(TestPointType::layout());

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

        assert_eq!(
            ref_bytes_p1_a1,
            storage.get_attribute_ref(0, &attributes::INTENSITY),
            "get_attribute_by_ref: Bytes are not equal"
        );
        assert_eq!(
            ref_bytes_p2_a1,
            storage.get_attribute_ref(1, &attributes::INTENSITY),
            "get_attribute_by_ref: Bytes are not equal"
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

        assert_eq!(
            ref_bytes_p1_a2,
            storage.get_attribute_ref(0, &attributes::GPS_TIME),
            "get_attribute_by_ref: Bytes are not equal"
        );
        assert_eq!(
            ref_bytes_p2_a2,
            storage.get_attribute_ref(1, &attributes::GPS_TIME),
            "get_attribute_by_ref: Bytes are not equal"
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
        assert_eq!(
            ref_bytes_all_a1,
            storage.get_attribute_range_ref(0..2, &attributes::INTENSITY),
            "get_attribute_range_by_ref: Bytes are not equal"
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
        assert_eq!(
            ref_bytes_all_a2,
            storage.get_attribute_range_ref(0..2, &attributes::GPS_TIME),
            "get_attribute_range_by_ref: Bytes are not equal"
        );
    }

    #[test]
    #[should_panic]
    fn test_per_attribute_vec_storage_push_point_invalid_format() {
        let mut buffer = PerAttributeVecPointStorage::new(TestPointType::layout());
        buffer.push_point(OtherPointType(Vector3::new(0.0, 1.0, 2.0), 23));
    }

    #[test]
    #[should_panic]
    fn test_per_attribute_vec_storage_push_points_invalid_format() {
        let mut buffer = PerAttributeVecPointStorage::new(TestPointType::layout());
        buffer.push_points(&[OtherPointType(Vector3::new(0.0, 1.0, 2.0), 23)]);
    }
}
