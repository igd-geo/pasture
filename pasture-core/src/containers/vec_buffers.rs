use std::{collections::HashMap, iter::FromIterator, ops::Range};

use crate::{
    layout::{PointAttributeDefinition, PointLayout, PointType, PrimitiveType},
    util::{sort_untyped_slice_by_permutation, view_raw_bytes},
};

use super::{
    InterleavedPointBuffer, InterleavedPointBufferMut, InterleavedPointBufferSlice,
    PerAttributePointBuffer, PerAttributePointBufferMut, PerAttributePointBufferSlice,
    PerAttributePointBufferSliceMut, PointBuffer, PointBufferWriteable,
};
use rayon::prelude::*;

/// `PointBuffer` type that uses Interleaved memory layout and `Vec`-based owning storage for point data
pub struct InterleavedVecPointStorage {
    layout: PointLayout,
    points: Vec<u8>,
    size_of_point_entry: u64,
}

impl InterleavedVecPointStorage {
    /// Creates a new empty `InterleavedVecPointStorage` with the given `PointLayout`
    ///
    /// # Examples
    ///
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
    ///
    /// # Examples
    ///
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
    /// # Examples
    ///
    /// ```
    /// # use pasture_core::containers::*;
    /// # use pasture_core::layout::*;
    /// # use pasture_derive::PointType;
    ///
    /// #[repr(C)]
    /// #[derive(PointType)]
    /// struct MyPointType(#[pasture(BUILTIN_INTENSITY)] u16);
    ///
    /// {
    ///   let mut storage = InterleavedVecPointStorage::new(MyPointType::layout());
    ///   storage.push_point(MyPointType(42));
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// If the `PointLayout` of `T` does not match the `PointLayout` of the associated `InterleavedVecPointStorage`.
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
    /// # Examples
    ///
    /// ```
    /// # use pasture_core::containers::*;
    /// # use pasture_core::layout::*;
    /// # use pasture_derive::PointType;
    ///
    /// #[repr(C)]
    /// #[derive(PointType)]
    /// struct MyPointType(#[pasture(BUILTIN_INTENSITY)] u16);
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
    /// # Examples
    ///
    /// ```
    /// # use pasture_core::containers::*;
    /// # use pasture_core::layout::*;
    /// # use pasture_derive::PointType;
    ///
    /// #[repr(C)]
    /// #[derive(PointType)]
    /// struct MyPointType(#[pasture(BUILTIN_INTENSITY)] u16);
    ///
    /// {
    ///   let mut storage = InterleavedVecPointStorage::new(MyPointType::layout());
    ///   let points = vec![MyPointType(42), MyPointType(43)];
    ///   storage.push_points(&points);
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// If the `PointLayout` of type `T` does not match the layout of the associated `InterleavedVecPointStorage`.
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

    /// Returns a slice of the associated `InterleavedVecPointStorage`.
    ///
    /// # Note on `std::ops::Index`
    ///
    /// This method semantically does what a hypothetical `buf[a..b]` call would do, i.e. what `std::ops::Index` is designed
    /// for. Sadly, `std::ops::Index` requires that the resulting output type is returned by reference, which prevents any
    /// view types from being used with `std::ops::Index`. Since slicing a `PointBuffer` has to return a view object, Pasture
    /// can't use `std::ops::Index` together with `PointBuffer`.
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

    fn push_interleaved(&mut self, points: &dyn InterleavedPointBuffer) {
        self.points
            .extend_from_slice(points.get_raw_points_ref(0..points.len()));
    }

    fn push_per_attribute(&mut self, points: &dyn PerAttributePointBuffer) {
        // This function is essentially a data transpose!
        let attribute_buffers = self
            .layout
            .attributes()
            .map(|attribute| {
                (
                    attribute,
                    points.get_raw_attribute_range_ref(0..points.len(), &attribute.into()),
                )
            })
            .collect::<Vec<_>>();
        let mut single_point_blob = vec![0; self.layout.size_of_point_entry() as usize];
        for idx in 0..points.len() {
            for (attribute, attribute_buffer) in attribute_buffers.iter() {
                let slice_start = idx * attribute.size() as usize;
                let slice_end = (idx + 1) * attribute.size() as usize;
                let offset_in_point = attribute.offset() as usize;

                let point_slice = &mut single_point_blob
                    [offset_in_point..offset_in_point + attribute.size() as usize];
                let attribute_slice = &attribute_buffer[slice_start..slice_end];
                point_slice.copy_from_slice(attribute_slice);
            }
            self.points.extend_from_slice(single_point_blob.as_slice());
        }
    }

    fn splice_interleaved(&mut self, range: Range<usize>, points: &dyn InterleavedPointBuffer) {
        if points.point_layout() != self.point_layout() {
            panic!(
                "InterleavedVecPointStorage::splice_interleaved: points layout does not match this PointLayout!"
            );
        }
        if range.start > range.end {
            panic!("Range start is greater than range end");
        }
        if range.end > self.len() {
            panic!("Range is out of bounds");
        }
        let this_offset = range.start * self.size_of_point_entry as usize;
        let this_range_len = range.len() * self.size_of_point_entry as usize;
        let this_slice = &mut self.points[this_offset..(this_offset + this_range_len)];
        let other_slice = points.get_raw_points_ref(0..range.len());
        this_slice.copy_from_slice(other_slice);
    }

    fn splice_per_attribute(&mut self, range: Range<usize>, points: &dyn PerAttributePointBuffer) {
        if !points
            .point_layout()
            .compare_without_offsets(self.point_layout())
        {
            panic!(
                "InterleavedVecPointStorage::splice_per_attribute: points layout does not match this PointLayout!"
            );
        }
        if range.start > range.end {
            panic!("Range start is greater than range end");
        }
        if range.end > self.len() {
            panic!("Range is out of bounds");
        }

        let attribute_buffers = self
            .layout
            .attributes()
            .map(|attribute| {
                (
                    attribute,
                    points.get_raw_attribute_range_ref(0..range.len(), &attribute.into()),
                )
            })
            .collect::<Vec<_>>();
        for idx in 0..range.len() {
            let current_point_offset = (range.start + idx) * self.size_of_point_entry as usize;
            for (attribute, attribute_buffer) in attribute_buffers.iter() {
                let slice_start = idx * attribute.size() as usize;
                let slice_end = (idx + 1) * attribute.size() as usize;
                let offset_in_point = attribute.offset() as usize;
                let attribute_slice = &attribute_buffer[slice_start..slice_end];

                let offset_in_this_buffer = current_point_offset + offset_in_point;
                let point_slice = &mut self.points
                    [offset_in_this_buffer..offset_in_this_buffer + attribute.size() as usize];
                point_slice.copy_from_slice(attribute_slice);
            }
        }
    }
}

impl PointBuffer for InterleavedVecPointStorage {
    fn get_raw_point(&self, point_index: usize, buf: &mut [u8]) {
        if point_index >= self.len() {
            panic!(
                "InterleavedVecPointStorage::get_raw_point: Point index {} out of bounds!",
                point_index
            );
        }

        let offset_to_point_bytes = point_index * self.size_of_point_entry as usize;
        buf.copy_from_slice(
            &self.points
                [offset_to_point_bytes..offset_to_point_bytes + self.size_of_point_entry as usize],
        );
    }

    fn get_raw_attribute(
        &self,
        point_index: usize,
        attribute: &PointAttributeDefinition,
        buf: &mut [u8],
    ) {
        if point_index >= self.len() {
            panic!(
                "InterleavedVecPointStorage::get_raw_attribute: Point index {} out of bounds!",
                point_index
            );
        }

        if let Some(attribute_in_buffer) = self.layout.get_attribute(attribute) {
            let offset_to_point_bytes = point_index * self.size_of_point_entry as usize;
            let offset_to_attribute = offset_to_point_bytes + attribute_in_buffer.offset() as usize;
            let attribute_size = attribute.size() as usize;

            buf.copy_from_slice(
                &self.points[offset_to_attribute..offset_to_attribute + attribute_size],
            );
        } else {
            panic!("InterleavedVecPointStorage::get_raw_attribute: Attribute {:?} is not part of this PointBuffer's PointLayout!", attribute);
        }
    }

    fn get_raw_points(&self, point_indices: Range<usize>, buf: &mut [u8]) {
        let points_ref = self.get_raw_points_ref(point_indices);
        buf[0..points_ref.len()].copy_from_slice(points_ref);
    }

    fn get_raw_attribute_range(
        &self,
        point_indices: Range<usize>,
        attribute: &PointAttributeDefinition,
        buf: &mut [u8],
    ) {
        if point_indices.end > self.len() {
            panic!(
                "InterleavedVecPointStorage::get_raw_attribute_range: Point indices {:?} out of bounds!",
                point_indices
            );
        }

        if let Some(attribute_in_buffer) = self.layout.get_attribute(attribute) {
            let attribute_size = attribute.size() as usize;
            let start_index = point_indices.start;

            for point_index in point_indices {
                let offset_to_point_bytes = point_index * self.size_of_point_entry as usize;
                let offset_to_attribute =
                    offset_to_point_bytes + attribute_in_buffer.offset() as usize;
                let offset_in_target_buf = (point_index - start_index) * attribute_size;
                let target_buf_slice =
                    &mut buf[offset_in_target_buf..offset_in_target_buf + attribute_size];

                target_buf_slice.copy_from_slice(
                    &self.points[offset_to_attribute..offset_to_attribute + attribute_size],
                );
            }
        } else {
            panic!("InterleavedVecPointStorage::get_raw_attribute: Attribute {:?} is not part of this PointBuffer's PointLayout!", attribute);
        }
    }

    fn len(&self) -> usize {
        self.points.len() / self.size_of_point_entry as usize
    }

    fn point_layout(&self) -> &PointLayout {
        &self.layout
    }

    fn as_interleaved(&self) -> Option<&dyn InterleavedPointBuffer> {
        Some(self)
    }
}

impl PointBufferWriteable for InterleavedVecPointStorage {
    fn push(&mut self, points: &dyn PointBuffer) {
        if let Some(interleaved) = points.as_interleaved() {
            self.push_interleaved(interleaved);
        } else if let Some(per_attribute) = points.as_per_attribute() {
            self.push_per_attribute(per_attribute);
        } else {
            panic!("InterleavedVecPointStorage::push: points buffer implements neither the InterleavedPointBuffer nor the PerAttributePointBuffer traits");
        }
    }

    fn splice(&mut self, range: Range<usize>, replace_with: &dyn PointBuffer) {
        if let Some(interleaved) = replace_with.as_interleaved() {
            self.splice_interleaved(range, interleaved);
        } else if let Some(per_attribute) = replace_with.as_per_attribute() {
            self.splice_per_attribute(range, per_attribute);
        } else {
            panic!("InterleavedVecPointStorage::splice: replace_with buffer implements neither the InterleavedPointBuffer nor the PerAttributePointBuffer traits");
        }
    }

    fn clear(&mut self) {
        self.points.clear();
    }
}

impl InterleavedPointBuffer for InterleavedVecPointStorage {
    fn get_raw_point_ref(&self, point_index: usize) -> &[u8] {
        if point_index >= self.len() {
            panic!(
                "InterleavedVecPointStorage::get_raw_point_ref: Point index {} out of bounds!",
                point_index
            );
        }

        let offset_to_point = point_index * self.size_of_point_entry as usize;
        &self.points[offset_to_point..offset_to_point + self.size_of_point_entry as usize]
    }

    fn get_raw_points_ref(&self, index_range: Range<usize>) -> &[u8] {
        if index_range.end > self.len() {
            panic!(
                "InterleavedVecPointStorage::get_raw_points_ref: Point indices {:?} out of bounds!",
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
    fn get_raw_point_mut(&mut self, point_index: usize) -> &mut [u8] {
        if point_index >= self.len() {
            panic!(
                "InterleavedVecPointStorage::get_raw_point_mut: Point index {} out of bounds!",
                point_index
            );
        }

        let offset_to_point = point_index * self.size_of_point_entry as usize;
        &mut self.points[offset_to_point..offset_to_point + self.size_of_point_entry as usize]
    }

    fn get_raw_points_mut(&mut self, index_range: Range<usize>) -> &mut [u8] {
        if index_range.end > self.len() {
            panic!(
                "InterleavedVecPointStorage::get_raw_points_mut: Point indices {:?} out of bounds!",
                index_range
            );
        }

        let offset_to_point = index_range.start * self.size_of_point_entry as usize;
        let total_bytes_of_range =
            (index_range.end - index_range.start) * self.size_of_point_entry as usize;
        &mut self.points[offset_to_point..offset_to_point + total_bytes_of_range]
    }
}

impl<T: PointType> FromIterator<T> for InterleavedVecPointStorage {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut buffer = Self::new(T::layout());
        for point in iter {
            buffer.push_point_unchecked(point);
        }
        buffer
    }
}

impl<T: PointType> From<&'_ [T]> for InterleavedVecPointStorage {
    fn from(slice: &'_ [T]) -> Self {
        let mut buffer = Self::with_capacity(slice.len(), T::layout());
        buffer.push_points(slice);
        buffer
    }
}

impl<T: PointType> From<&'_ mut [T]> for InterleavedVecPointStorage {
    fn from(slice: &'_ mut [T]) -> Self {
        let mut buffer = Self::with_capacity(slice.len(), T::layout());
        buffer.push_points(slice);
        buffer
    }
}

impl<T: PointType> From<Vec<T>> for InterleavedVecPointStorage {
    fn from(vec: Vec<T>) -> Self {
        //TODO We could optimize this by transmogrifying the Vec<T> into a Vec<u8> and moving this vec
        // into Self. But it requires unsafe code and we have to make sure that this always works for any
        // possible PointType
        Self::from(vec.as_slice())
    }
}

/// `PointBuffer` type that uses PerAttribute memory layout and `Vec`-based owning storage for point data
pub struct PerAttributeVecPointStorage {
    layout: PointLayout,
    attributes: HashMap<&'static str, Vec<u8>>,
}

impl PerAttributeVecPointStorage {
    /// Creates a new empty `PerAttributeVecPointStorage` with the given `PointLayout`
    ///
    /// # Examples
    ///
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
    ///
    /// # Examples
    ///
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
    /// # Examples
    ///
    /// ```
    /// # use pasture_core::containers::*;
    /// # use pasture_core::layout::*;
    /// # use pasture_derive::PointType;
    ///
    /// #[repr(C)]
    /// #[derive(PointType)]
    /// struct MyPointType(#[pasture(BUILTIN_INTENSITY)] u16);
    ///
    /// {
    ///   let mut storage = PerAttributeVecPointStorage::new(MyPointType::layout());
    ///   storage.push_point(MyPointType(42));
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// If the `PointLayout` of type `T` does not match the layout of the associated `PerAttributeVecPointStorage`.
    pub fn push_point<T: PointType>(&mut self, point: T) {
        // We don't care for the attribute offsets in T::layout(), because PerAttributeVecPointStorage stores each
        // attribute in a separate Vec. So we only compare that all attributes of T::layout() exist and that their
        // types are equal. This is done implicitly with the 'self.attributes.get_mut' call below

        self.reserve(1);

        let point_bytes = unsafe { view_raw_bytes(&point) };
        let point_layout = T::layout();
        for attribute in point_layout.attributes() {
            let offset_to_attribute_in_point = attribute.offset() as usize;
            let point_slice = &point_bytes[offset_to_attribute_in_point
                ..offset_to_attribute_in_point + attribute.size() as usize];

            let attribute_buffer = self.attributes.get_mut(attribute.name()).unwrap_or_else(|| panic!("PerAttributeVecPointStorage::push_point: Attribute {} of point does not exist in the PointLayout of this buffer!", attribute));
            attribute_buffer.extend_from_slice(point_slice);
        }
    }

    /// Pushes a range of points into the associated `PerAttributeVecPointStorage`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use pasture_core::containers::*;
    /// # use pasture_core::layout::*;
    /// # use pasture_derive::PointType;
    ///
    /// #[repr(C)]
    /// #[derive(PointType)]
    /// struct MyPointType(#[pasture(BUILTIN_INTENSITY)] u16);
    ///
    /// {
    ///   let mut storage = PerAttributeVecPointStorage::new(MyPointType::layout());
    ///   storage.push_points(&[MyPointType(42), MyPointType(43)]);
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// If the `PointLayout` of type `T` does not match the layout of the associated `PerAttributeVecPointStorage`.
    pub fn push_points<T: PointType>(&mut self, points: &[T]) {
        self.reserve(points.len());

        let point_layout = T::layout();
        for attribute in point_layout.attributes() {
            let offset_to_attribute_in_point = attribute.offset() as usize;
            let attribute_buffer = self.attributes.get_mut(attribute.name()).unwrap_or_else(|| panic!("PerAttributeVecPointStorage::push_points: Attribute {} of points does not exist in the PointLayout of this buffer!", attribute));

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
    /// # Examples
    ///
    /// ```
    /// # use pasture_core::containers::*;
    /// # use pasture_core::layout::*;
    /// # use pasture_derive::PointType;
    ///
    /// #[repr(C)]
    /// #[derive(PointType)]
    /// struct MyPointType(#[pasture(BUILTIN_INTENSITY)] u16, #[pasture(BUILTIN_GPS_TIME)] f64);
    ///
    /// {
    ///   let mut storage = PerAttributeVecPointStorage::new(MyPointType::layout());
    ///   storage.push_attribute(&attributes::INTENSITY, 42_u16);
    ///   storage.push_attribute(&attributes::GPS_TIME, 0.123);
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// If the given `PointAttributeDefinition` is not part of the internal `PointLayout` of the associated `PerAttributeVecPointStorage`
    pub fn push_attribute<T: PrimitiveType>(
        &mut self,
        attribute: &PointAttributeDefinition,
        value: T,
    ) {
        if T::data_type() != attribute.datatype() {
            panic!("PerAttributeVecPointStorage::push_attribute: Was called with generic argument {} but the datatype in the PointAttributeDefinition is {}!", T::data_type(), attribute.datatype());
        }
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
    /// # Examples
    ///
    /// ```
    /// # use pasture_core::containers::*;
    /// # use pasture_core::layout::*;
    /// # use pasture_derive::PointType;
    ///
    /// #[repr(C)]
    /// #[derive(PointType)]
    /// struct MyPointType(#[pasture(BUILTIN_INTENSITY)] u16, #[pasture(BUILTIN_GPS_TIME)] f64);
    ///
    /// {
    ///   let mut storage = PerAttributeVecPointStorage::new(MyPointType::layout());
    ///   storage.push_attribute_range(&attributes::INTENSITY, &[42_u16, 43_u16, 44_u16]);
    ///   storage.push_attribute_range(&attributes::GPS_TIME, &[0.123, 0.456, 0.789]);
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// If the given `PointAttributeDefinition` is not part of the internal `PointLayout` of the associated `PerAttributeVecPointStorage`
    pub fn push_attribute_range<T: PrimitiveType>(
        &mut self,
        attribute: &PointAttributeDefinition,
        values: &[T],
    ) {
        if T::data_type() != attribute.datatype() {
            panic!("PerAttributeVecPointStorage::push_attribute: Was called with generic argument {} but the datatype in the PointAttributeDefinition is {}!", T::data_type(), attribute.datatype());
        }
        match self.attributes.get_mut(attribute.name()) {
            None => panic!("PerAttributeVecPointStorage::push_attributes: Attribute {:?} is not part of this PointBuffer's PointLayout!", attribute),
            Some(attribute_buffer) => {
                let value_bytes = unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * std::mem::size_of::<T>()) };
                attribute_buffer.extend_from_slice(value_bytes);
            },
        }
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
        if attribute.datatype() != T::data_type() {
            panic!("PerAttributePointBuffer:sort_by_attribute: Type T does not match the datatype of the attribute {}", attribute);
        }
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

    /// Like `sort_by_attribute`, but sorts each attribute in parallel. Uses the [`rayon`]() crate for parallelization
    pub fn par_sort_by_attribute<T: PrimitiveType + Ord>(
        &mut self,
        attribute: &PointAttributeDefinition,
    ) {
        if attribute.datatype() != T::data_type() {
            panic!("PerAttributePointBuffer:sort_by_attribute: Type T does not match the datatype of the attribute {}", attribute);
        }

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
            .map(|&key| {
                (
                    key.to_owned(),
                    self.layout.get_attribute_by_name(key).unwrap().size(),
                )
            })
            .collect::<HashMap<_, _>>();

        self.attributes
            .par_iter_mut()
            .for_each(|(&key, untyped_attribute)| {
                let size = *attribute_sizes.get(key).unwrap();
                sort_untyped_slice_by_permutation(
                    untyped_attribute.as_mut_slice(),
                    indices.as_slice(),
                    size as usize,
                );
            });
    }

    /// Reserves space for at least `additional_points` additional points in the associated `PerAttributeVecPointStorage`
    pub fn reserve(&mut self, additional_points: usize) {
        for attribute in self.layout.attributes() {
            let attribute_buffer = self.attributes.get_mut(attribute.name()).unwrap();
            let additional_bytes = additional_points * attribute.size() as usize;
            attribute_buffer.reserve(additional_bytes);
        }
    }

    fn push_interleaved(&mut self, points: &dyn InterleavedPointBuffer) {
        if !points
            .point_layout()
            .compare_without_offsets(self.point_layout())
        {
            panic!("PerAttributeVecPointStorage::push_points_interleaved: Layout of 'points' does not match layout of this buffer!");
        }

        let raw_point_data = points.get_raw_points_ref(0..points.len());
        let stride = self.layout.size_of_point_entry() as usize;

        for (attribute_name, attribute_data) in self.attributes.iter_mut() {
            let current_attribute = self.layout.get_attribute_by_name(attribute_name).unwrap();
            let base_offset = current_attribute.offset() as usize;
            let attribute_size = current_attribute.size() as usize;

            for idx in 0..points.len() {
                let attribute_start = base_offset + (idx * stride);
                let attribute_end = attribute_start + attribute_size;
                let attribute_slice = &raw_point_data[attribute_start..attribute_end];
                attribute_data.extend_from_slice(attribute_slice);
            }
        }
    }

    fn push_per_attribute(&mut self, points: &dyn PerAttributePointBuffer) {
        if !points
            .point_layout()
            .compare_without_offsets(self.point_layout())
        {
            panic!("PerAttributeVecPointStorage::push_raw_points_per_attribute: points layout does not match this PointLayout!");
        }
        for attribute in self.layout.attributes() {
            self.attributes
                .get_mut(attribute.name())
                .unwrap()
                .extend_from_slice(
                    points.get_raw_attribute_range_ref(0..points.len(), &attribute.into()),
                );
        }
    }

    fn splice_interleaved(&mut self, range: Range<usize>, points: &dyn InterleavedPointBuffer) {
        if range.start > range.end {
            panic!("Range start is greater than range end");
        }
        if range.end > self.len() {
            panic!("Range is out of bounds");
        }
        if !points
            .point_layout()
            .compare_without_offsets(self.point_layout())
        {
            panic!("PerAttributeVecPointStorage::splice_interleaved: points layout does not match this PointLayout!");
        }

        let raw_point_data = points.get_raw_points_ref(0..range.len());
        let stride = self.layout.size_of_point_entry() as usize;

        for (attribute_name, attribute_data) in self.attributes.iter_mut() {
            let current_attribute = self.layout.get_attribute_by_name(attribute_name).unwrap();
            let base_offset = current_attribute.offset() as usize;
            let attribute_size = current_attribute.size() as usize;

            for idx in 0..range.len() {
                let this_attribute_start = (range.start + idx) * attribute_size;
                let this_attribute_end = this_attribute_start + attribute_size;
                let this_attribute_slice =
                    &mut attribute_data[this_attribute_start..this_attribute_end];

                let new_attribute_start = base_offset + (idx * stride);
                let new_attribute_end = new_attribute_start + attribute_size;
                let new_attribute_slice = &raw_point_data[new_attribute_start..new_attribute_end];

                this_attribute_slice.copy_from_slice(new_attribute_slice);
            }
        }
    }

    fn splice_per_attribute(&mut self, range: Range<usize>, points: &dyn PerAttributePointBuffer) {
        if range.start > range.end {
            panic!("Range start is greater than range end");
        }
        if range.end > self.len() {
            panic!("Range is out of bounds");
        }
        if !points
            .point_layout()
            .compare_without_offsets(self.point_layout())
        {
            panic!("PerAttributeVecPointStorage::splice_per_attribute: points layout does not match this PointLayout!");
        }
        for attribute in self.layout.attributes() {
            let this_attribute_offset = range.start * attribute.size() as usize;
            let this_attribute_slice = &mut self.attributes.get_mut(attribute.name()).unwrap()
                [this_attribute_offset
                    ..this_attribute_offset + range.len() * attribute.size() as usize];

            let new_attribute_slice =
                points.get_raw_attribute_range_ref(0..range.len(), &attribute.into());

            this_attribute_slice.copy_from_slice(new_attribute_slice);
        }
    }
}

impl PointBuffer for PerAttributeVecPointStorage {
    fn get_raw_point(&self, point_index: usize, buf: &mut [u8]) {
        if point_index >= self.len() {
            panic!(
                "PerAttributeVecPointStorage::get_raw_point: Point index {} out of bounds!",
                point_index
            );
        }

        for attribute in self.layout.attributes() {
            let attribute_buffer = self.attributes.get(attribute.name()).unwrap();
            let attribute_size = attribute.size() as usize;
            let offset_in_buffer = point_index * attribute_size;
            let offset_in_point = attribute.offset() as usize;

            let buf_slice = &mut buf[offset_in_point..offset_in_point + attribute_size];
            let attribute_slice =
                &attribute_buffer[offset_in_buffer..offset_in_buffer + attribute_size];
            buf_slice.copy_from_slice(attribute_slice);
        }
    }

    fn get_raw_attribute(
        &self,
        point_index: usize,
        attribute: &PointAttributeDefinition,
        buf: &mut [u8],
    ) {
        let attribute_slice = self.get_raw_attribute_ref(point_index, attribute);
        buf.copy_from_slice(attribute_slice);
    }

    fn get_raw_points(&self, point_indices: Range<usize>, buf: &mut [u8]) {
        if point_indices.end > self.len() {
            panic!(
                "PerAttributeVecPointStorage::get_raw_points: Point indices {:?} out of bounds!",
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

                let offset_in_point = attribute.offset() as usize;
                let offset_in_points_buffer = point_index * point_size + offset_in_point;
                let buf_slice =
                    &mut buf[offset_in_points_buffer..offset_in_points_buffer + attribute_size];

                buf_slice.copy_from_slice(attribute_slice);
            }
        }
    }

    fn get_raw_attribute_range(
        &self,
        point_indices: Range<usize>,
        attribute: &PointAttributeDefinition,
        buf: &mut [u8],
    ) {
        let attribute_buffer_slice = self.get_raw_attribute_range_ref(point_indices, attribute);
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

    fn as_per_attribute(&self) -> Option<&dyn PerAttributePointBuffer> {
        Some(self)
    }
}

impl PointBufferWriteable for PerAttributeVecPointStorage {
    fn push(&mut self, points: &dyn PointBuffer) {
        if let Some(interleaved) = points.as_interleaved() {
            self.push_interleaved(interleaved);
        } else if let Some(per_attribute) = points.as_per_attribute() {
            self.push_per_attribute(per_attribute);
        } else {
            panic!("PerAttributeVecPointStorage::push: points buffer does not implement the InterleavedPointBuffer or the PerAttributePointBuffer trait");
        }
    }

    fn splice(&mut self, range: Range<usize>, replace_with: &dyn PointBuffer) {
        if let Some(interleaved) = replace_with.as_interleaved() {
            self.splice_interleaved(range, interleaved);
        } else if let Some(per_attribute) = replace_with.as_per_attribute() {
            self.splice_per_attribute(range, per_attribute);
        } else {
            panic!("PerAttributeVecPointStorage::splice: replace_with buffer does not implement the InterleavedPointBuffer or the PerAttributePointBuffer trait");
        }
    }

    fn clear(&mut self) {
        self.attributes.iter_mut().for_each(|(_, vec)| vec.clear());
    }
}

impl PerAttributePointBuffer for PerAttributeVecPointStorage {
    fn get_raw_attribute_ref(
        &self,
        point_index: usize,
        attribute: &PointAttributeDefinition,
    ) -> &[u8] {
        if point_index >= self.len() {
            panic!(
                "PerAttributeVecPointStorage::get_raw_attribute_ref: Point index {} out of bounds!",
                point_index
            );
        }

        if !self.layout.has_attribute(attribute) {
            panic!("PerAttributeVecPointStorage::get_raw_attribute_ref: Attribute {:?} is not part of this PointBuffer's PointLayout!", attribute);
        }

        let attribute_buffer = self.attributes.get(attribute.name()).unwrap();
        let attribute_size = attribute.size() as usize;
        let offset_in_attribute_buffer = point_index * attribute_size;
        &attribute_buffer[offset_in_attribute_buffer..offset_in_attribute_buffer + attribute_size]
    }

    fn get_raw_attribute_range_ref(
        &self,
        index_range: Range<usize>,
        attribute: &PointAttributeDefinition,
    ) -> &[u8] {
        if index_range.end > self.len() {
            panic!(
                "PerAttributeVecPointStorage::get_raw_attribute_range_ref: Point indices {:?} out of bounds!",
                index_range
            );
        }

        if !self.layout.has_attribute(attribute) {
            panic!("PerAttributeVecPointStorage::get_raw_attribute_ref: Attribute {:?} is not part of this PointBuffer's PointLayout!", attribute);
        }

        let attribute_buffer = self.attributes.get(attribute.name()).unwrap();
        let start_offset_in_attribute_buffer = index_range.start * attribute.size() as usize;
        let end_offset_in_attribute_buffer = start_offset_in_attribute_buffer
            + (index_range.end - index_range.start) * attribute.size() as usize;
        &attribute_buffer[start_offset_in_attribute_buffer..end_offset_in_attribute_buffer]
    }

    fn slice(&self, range: Range<usize>) -> PerAttributePointBufferSlice<'_> {
        PerAttributePointBufferSlice::new(self, range)
    }
}

impl<'p> PerAttributePointBufferMut<'p> for PerAttributeVecPointStorage {
    fn get_raw_attribute_mut(
        &mut self,
        point_index: usize,
        attribute: &PointAttributeDefinition,
    ) -> &mut [u8] {
        if point_index >= self.len() {
            panic!(
                "PerAttributeVecPointStorage::get_raw_attribute_mut: Point index {} out of bounds!",
                point_index
            );
        }

        if !self.layout.has_attribute(attribute) {
            panic!("PerAttributeVecPointStorage::get_raw_attribute_mut: Attribute {:?} is not part of this PointBuffer's PointLayout!", attribute);
        }

        let attribute_buffer = self.attributes.get_mut(attribute.name()).unwrap();
        let attribute_size = attribute.size() as usize;
        let offset_in_attribute_buffer = point_index * attribute_size;
        &mut attribute_buffer
            [offset_in_attribute_buffer..offset_in_attribute_buffer + attribute_size]
    }

    fn get_raw_attribute_range_mut(
        &mut self,
        index_range: Range<usize>,
        attribute: &PointAttributeDefinition,
    ) -> &mut [u8] {
        if index_range.end > self.len() {
            panic!(
                "PerAttributeVecPointStorage::get_raw_attribute_range_mut: Point indices {:?} out of bounds!",
                index_range
            );
        }

        if !self.layout.has_attribute(attribute) {
            panic!("PerAttributeVecPointStorage::get_raw_attribute_range_mut: Attribute {:?} is not part of this PointBuffer's PointLayout!", attribute);
        }

        let attribute_buffer = self.attributes.get_mut(attribute.name()).unwrap();
        let start_offset_in_attribute_buffer = index_range.start * attribute.size() as usize;
        let end_offset_in_attribute_buffer = start_offset_in_attribute_buffer
            + (index_range.end - index_range.start) * attribute.size() as usize;
        &mut attribute_buffer[start_offset_in_attribute_buffer..end_offset_in_attribute_buffer]
    }

    fn slice_mut(&'p mut self, range: Range<usize>) -> PerAttributePointBufferSliceMut<'p> {
        PerAttributePointBufferSliceMut::new(self, range)
    }

    fn disjunct_slices_mut<'b>(
        &'b mut self,
        ranges: &[Range<usize>],
    ) -> Vec<PerAttributePointBufferSliceMut<'p>>
    where
        'p: 'b,
    {
        let self_ptr = self as *mut dyn PerAttributePointBufferMut<'p>;

        ranges
            .iter()
            .map(|range| PerAttributePointBufferSliceMut::from_raw_ptr(self_ptr, range.clone()))
            .collect()
    }

    fn as_per_attribute_point_buffer(&self) -> &dyn PerAttributePointBuffer {
        self
    }
}

impl<T: PointType> FromIterator<T> for PerAttributeVecPointStorage {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut buffer = Self::new(T::layout());
        for point in iter {
            buffer.push_point(point);
        }
        buffer
    }
}

impl<T: PointType> From<&'_ [T]> for PerAttributeVecPointStorage {
    fn from(slice: &'_ [T]) -> Self {
        let mut buffer = Self::with_capacity(slice.len(), T::layout());
        buffer.push_points(slice);
        buffer
    }
}

impl<T: PointType> From<&'_ mut [T]> for PerAttributeVecPointStorage {
    fn from(slice: &'_ mut [T]) -> Self {
        let mut buffer = Self::with_capacity(slice.len(), T::layout());
        buffer.push_points(slice);
        buffer
    }
}

impl<T: PointType> From<Vec<T>> for PerAttributeVecPointStorage {
    fn from(vec: Vec<T>) -> Self {
        Self::from(vec.as_slice())
    }
}

#[cfg(test)]
mod tests {

    use nalgebra::Vector3;

    use super::*;
    use crate::containers::{
        attribute_slice, InterleavedPointView, PerAttributePointView, PointBufferExt,
    };
    use crate::util::view_raw_bytes;
    use crate::{
        layout::{attributes, PointLayout},
        util::view_raw_bytes_mut,
    };
    use pasture_derive::PointType;

    // We need this, otherwise we can't use the derive(PointType) macro from within pasture_core because the macro
    // doesn't recognize the name 'pasture_core' :/
    use crate as pasture_core;

    #[repr(packed)]
    #[derive(Debug, Copy, Clone, PartialEq, PointType)]
    struct TestPointType(
        #[pasture(BUILTIN_INTENSITY)] u16,
        #[pasture(BUILTIN_GPS_TIME)] f64,
    );

    #[repr(packed)]
    #[derive(Debug, Copy, Clone, PartialEq, PointType)]
    struct OtherPointType(
        #[pasture(BUILTIN_POSITION_3D)] Vector3<f64>,
        #[pasture(BUILTIN_RETURN_NUMBER)] u8,
    );

    trait OpqaueInterleavedBuffer: InterleavedPointBufferMut + PointBufferWriteable {}
    impl OpqaueInterleavedBuffer for InterleavedVecPointStorage {}

    trait OpqauePerAttributeBuffer<'b>: PerAttributePointBufferMut<'b> + PointBufferWriteable {}
    impl<'b> OpqauePerAttributeBuffer<'b> for PerAttributeVecPointStorage {}

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
    ) -> Box<dyn OpqauePerAttributeBuffer<'static>> {
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
    fn test_point_buffer_get_raw_point() {
        let interleaved_buffer =
            get_interleaved_point_buffer_from_points(&[TestPointType(42, 0.123)]);

        let mut ref_point = TestPointType(0, 0.0);
        unsafe {
            interleaved_buffer.get_raw_point(0, view_raw_bytes_mut(&mut ref_point));
        }

        assert_eq!(TestPointType(42, 0.123), ref_point);

        let per_attribute_buffer =
            get_per_attribute_point_buffer_from_points(&[TestPointType(43, 0.456)]);
        unsafe {
            per_attribute_buffer.get_raw_point(0, view_raw_bytes_mut(&mut ref_point));
        }

        assert_eq!(TestPointType(43, 0.456), ref_point);
    }

    #[test]
    #[should_panic]
    fn test_point_buffer_get_raw_point_on_empty_interleaved_buffer() {
        let buffer = get_empty_interleaved_point_buffer(TestPointType::layout());

        let mut ref_point = TestPointType(0, 0.0);
        unsafe {
            buffer.get_raw_point(0, view_raw_bytes_mut(&mut ref_point));
        }
    }

    #[test]
    #[should_panic]
    fn test_point_buffer_get_raw_point_on_empty_per_attribute_buffer() {
        let buffer = get_empty_per_attribute_point_buffer(TestPointType::layout());

        let mut ref_point = TestPointType(0, 0.0);
        unsafe {
            buffer.get_raw_point(0, view_raw_bytes_mut(&mut ref_point));
        }
    }

    #[test]
    fn test_point_buffer_get_raw_attribute() {
        let interleaved_buffer =
            get_interleaved_point_buffer_from_points(&[TestPointType(42, 0.123)]);

        let mut ref_attribute: u16 = 0;
        unsafe {
            interleaved_buffer.get_raw_attribute(
                0,
                &attributes::INTENSITY,
                view_raw_bytes_mut(&mut ref_attribute),
            );
        }

        assert_eq!(42, ref_attribute);

        let per_attribute_buffer =
            get_per_attribute_point_buffer_from_points(&[TestPointType(43, 0.456)]);
        unsafe {
            per_attribute_buffer.get_raw_attribute(
                0,
                &attributes::INTENSITY,
                view_raw_bytes_mut(&mut ref_attribute),
            );
        }

        assert_eq!(43, ref_attribute);
    }

    #[test]
    #[should_panic]
    fn test_point_buffer_get_raw_attribute_on_empty_interleaved_buffer() {
        let buffer = get_empty_interleaved_point_buffer(TestPointType::layout());

        let mut ref_attribute: u16 = 0;
        unsafe {
            buffer.get_raw_attribute(
                0,
                &attributes::INTENSITY,
                view_raw_bytes_mut(&mut ref_attribute),
            );
        }
    }

    #[test]
    #[should_panic]
    fn test_point_buffer_get_raw_attribute_on_empty_per_attribute_buffer() {
        let buffer = get_empty_per_attribute_point_buffer(TestPointType::layout());

        let mut ref_attribute: u16 = 0;
        unsafe {
            buffer.get_raw_attribute(
                0,
                &attributes::INTENSITY,
                view_raw_bytes_mut(&mut ref_attribute),
            );
        }
    }

    #[test]
    #[should_panic]
    fn test_point_buffer_get_raw_attribute_for_invalid_attribute_interleaved() {
        let buffer = get_interleaved_point_buffer_from_points(&[TestPointType(42, 0.123)]);

        let mut ref_attribute: u16 = 0;
        unsafe {
            buffer.get_raw_attribute(
                0,
                &attributes::POSITION_3D,
                view_raw_bytes_mut(&mut ref_attribute),
            );
        }
    }

    #[test]
    #[should_panic]
    fn test_point_buffer_get_raw_attribute_for_invalid_attribute_per_attribute() {
        let buffer = get_per_attribute_point_buffer_from_points(&[TestPointType(43, 0.456)]);

        let mut ref_attribute: u16 = 0;
        unsafe {
            buffer.get_raw_attribute(
                0,
                &attributes::POSITION_3D,
                view_raw_bytes_mut(&mut ref_attribute),
            );
        }
    }

    #[test]
    fn test_point_buffer_get_raw_points() {
        let interleaved_buffer = get_interleaved_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);

        let mut ref_points = [TestPointType(0, 0.0), TestPointType(0, 0.0)];
        unsafe { interleaved_buffer.get_raw_points(0..2, view_raw_bytes_mut(&mut ref_points)) }

        assert_eq!(TestPointType(42, 0.123), ref_points[0]);
        assert_eq!(TestPointType(43, 0.456), ref_points[1]);

        let per_attribute_buffer = get_per_attribute_point_buffer_from_points(&[
            TestPointType(44, 0.321),
            TestPointType(45, 0.654),
        ]);
        unsafe {
            per_attribute_buffer.get_raw_points(0..2, view_raw_bytes_mut(&mut ref_points));
        }

        assert_eq!(TestPointType(44, 0.321), ref_points[0]);
        assert_eq!(TestPointType(45, 0.654), ref_points[1]);
    }

    #[test]
    #[should_panic]
    fn test_point_buffer_get_raw_points_out_of_bounds_interleaved() {
        let interleaved_buffer = get_empty_interleaved_point_buffer(TestPointType::layout());
        interleaved_buffer.get_raw_points(0..2, &mut [0]);
    }

    #[test]
    #[should_panic]
    fn test_point_buffer_get_raw_points_out_of_bounds_per_attribute() {
        let interleaved_buffer = get_empty_per_attribute_point_buffer(TestPointType::layout());
        interleaved_buffer.get_raw_points(0..2, &mut [0]);
    }

    #[test]
    fn test_point_buffer_get_raw_attribute_range() {
        let interleaved_buffer = get_interleaved_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);

        let mut ref_attributes: [u16; 2] = [0, 0];
        unsafe {
            interleaved_buffer.get_raw_attribute_range(
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
            per_attribute_buffer.get_raw_attribute_range(
                0..2,
                &attributes::INTENSITY,
                view_raw_bytes_mut(&mut ref_attributes),
            )
        }

        assert_eq!([44, 45], ref_attributes);
    }

    #[test]
    #[should_panic]
    fn test_point_buffer_get_raw_attribute_range_invalid_attribute_interleaved() {
        let buffer = get_interleaved_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);
        buffer.get_raw_attribute_range(0..2, &attributes::POINT_ID, &mut [0]);
    }

    #[test]
    #[should_panic]
    fn test_point_buffer_get_raw_attribute_range_invalid_attribute_per_attribute() {
        let buffer = get_per_attribute_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);
        buffer.get_raw_attribute_range(0..2, &attributes::POINT_ID, &mut [0]);
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
        interleaved_buffer.push(&InterleavedPointView::from_slice(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]));

        assert_eq!(2, interleaved_buffer.len());

        let mut ref_points = [TestPointType(0, 0.0), TestPointType(0, 0.0)];
        unsafe { interleaved_buffer.get_raw_points(0..2, view_raw_bytes_mut(&mut ref_points)) }

        assert_eq!(TestPointType(42, 0.123), ref_points[0]);
        assert_eq!(TestPointType(43, 0.456), ref_points[1]);

        let mut per_attribute_buffer =
            get_empty_per_attribute_point_buffer(TestPointType::layout());
        per_attribute_buffer.push(&InterleavedPointView::from_slice(&[
            TestPointType(44, 0.321),
            TestPointType(45, 0.654),
        ]));

        assert_eq!(2, per_attribute_buffer.len());

        unsafe { per_attribute_buffer.get_raw_points(0..2, view_raw_bytes_mut(&mut ref_points)) }

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
        interleaved_buffer.push(&data_view);

        assert_eq!(2, interleaved_buffer.len());

        let mut ref_points = [TestPointType(0, 0.0), TestPointType(0, 0.0)];
        unsafe { interleaved_buffer.get_raw_points(0..2, view_raw_bytes_mut(&mut ref_points)) }

        assert_eq!(TestPointType(42, 0.123), ref_points[0]);
        assert_eq!(TestPointType(43, 0.456), ref_points[1]);

        let mut per_attribute_buffer =
            get_empty_per_attribute_point_buffer(TestPointType::layout());
        per_attribute_buffer.push(&data_view);

        assert_eq!(2, per_attribute_buffer.len());

        ref_points[0] = TestPointType(0, 0.0);
        ref_points[1] = TestPointType(0, 0.0);

        unsafe { per_attribute_buffer.get_raw_points(0..2, view_raw_bytes_mut(&mut ref_points)) }

        assert_eq!(TestPointType(42, 0.123), ref_points[0]);
        assert_eq!(TestPointType(43, 0.456), ref_points[1]);
    }

    #[test]
    #[should_panic]
    fn test_point_buffer_writeable_interleaved_push_raw_points_per_attribute_invalid_layout() {
        let data_view = PerAttributePointView::new();

        let mut buffer = get_empty_interleaved_point_buffer(TestPointType::layout());
        buffer.push(&data_view);
    }

    #[test]
    #[should_panic]
    fn test_point_buffer_writeable_per_attribute_push_raw_points_per_attribute_invalid_layout() {
        let data_view = PerAttributePointView::new();

        let mut buffer = get_empty_per_attribute_point_buffer(TestPointType::layout());
        buffer.push(&data_view);
    }

    #[test]
    fn test_interleaved_point_buffer_get_raw_point_ref() {
        let interleaved_buffer = get_interleaved_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);

        let first_point_ref = interleaved_buffer.get_raw_point_ref(0);
        let first_point_ref_typed = unsafe { *(first_point_ref.as_ptr() as *const TestPointType) };

        assert_eq!(TestPointType(42, 0.123), first_point_ref_typed);

        let second_point_ref = interleaved_buffer.get_raw_point_ref(1);
        let second_point_ref_typed =
            unsafe { *(second_point_ref.as_ptr() as *const TestPointType) };

        assert_eq!(TestPointType(43, 0.456), second_point_ref_typed);
    }

    #[test]
    #[should_panic]
    fn test_interleaved_point_buffer_get_raw_point_ref_on_empty_buffer() {
        let buffer = get_empty_interleaved_point_buffer(TestPointType::layout());

        buffer.get_raw_point_ref(0);
    }

    #[test]
    fn test_interleaved_point_buffer_get_raw_points_ref() {
        let interleaved_buffer = get_interleaved_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);

        let points_ref = interleaved_buffer.get_raw_points_ref(0..2);
        let typed_points_ref =
            unsafe { std::slice::from_raw_parts(points_ref.as_ptr() as *const TestPointType, 2) };

        let reference_points = &[TestPointType(42, 0.123), TestPointType(43, 0.456)];

        assert_eq!(reference_points, typed_points_ref);
    }

    #[test]
    #[should_panic]
    fn test_interleaved_point_buffer_get_raw_points_ref_on_empty_buffer() {
        let buffer = get_empty_interleaved_point_buffer(TestPointType::layout());

        buffer.get_raw_points_ref(0..2);
    }

    #[test]
    fn test_interleaved_point_buffer_mut_get_raw_point_mut() {
        let mut interleaved_buffer = get_interleaved_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);

        {
            let mut_point = interleaved_buffer.get_raw_point_mut(0);
            let mut_point_typed = unsafe { &mut *(mut_point.as_mut_ptr() as *mut TestPointType) };

            mut_point_typed.0 = 128;
            mut_point_typed.1 = 3.14159;
        }

        let point_ref = interleaved_buffer.get_raw_point_ref(0);
        let point_ref_typed = unsafe { *(point_ref.as_ptr() as *const TestPointType) };

        assert_eq!(TestPointType(128, 3.14159), point_ref_typed);
    }

    #[test]
    #[should_panic]
    fn test_interleaved_point_buffer_mut_get_raw_point_mut_on_empty_buffer() {
        let mut buffer = get_empty_interleaved_point_buffer(TestPointType::layout());
        buffer.get_raw_point_mut(0);
    }

    #[test]
    fn test_interleaved_point_buffer_mut_get_raw_points_mut() {
        let mut interleaved_buffer = get_interleaved_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);

        let reference_points = &[TestPointType(128, 3.14159), TestPointType(129, 2.71828)];

        {
            let mut_points = interleaved_buffer.get_raw_points_mut(0..2);
            let mut_points_typed = unsafe {
                std::slice::from_raw_parts_mut(mut_points.as_mut_ptr() as *mut TestPointType, 2)
            };

            mut_points_typed[0] = reference_points[0];
            mut_points_typed[1] = reference_points[1];
        }

        let points_ref = interleaved_buffer.get_raw_points_ref(0..2);
        let typed_points_ref =
            unsafe { std::slice::from_raw_parts(points_ref.as_ptr() as *const TestPointType, 2) };

        assert_eq!(reference_points, typed_points_ref);
    }

    #[test]
    #[should_panic]
    fn test_interleaved_point_buffer_mut_get_raw_points_mut_on_empty_buffer() {
        let mut buffer = get_empty_interleaved_point_buffer(TestPointType::layout());
        buffer.get_raw_points_mut(0..2);
    }

    #[test]
    fn test_per_attribute_point_buffer_get_raw_attribute_ref() {
        let buffer = get_per_attribute_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);

        let first_point_attribute_ref = buffer.get_raw_attribute_ref(0, &attributes::INTENSITY);
        let first_point_attribute_ref_typed =
            unsafe { *(first_point_attribute_ref.as_ptr() as *const u16) };

        assert_eq!(42, first_point_attribute_ref_typed);

        let second_point_attribute_ref = buffer.get_raw_attribute_ref(1, &attributes::GPS_TIME);
        let second_point_attribute_ref_typed =
            unsafe { *(second_point_attribute_ref.as_ptr() as *const f64) };

        assert_eq!(0.456, second_point_attribute_ref_typed);
    }

    #[test]
    #[should_panic]
    fn test_per_attribute_point_buffer_get_raw_attribute_ref_out_of_bounds() {
        let buffer = get_empty_per_attribute_point_buffer(TestPointType::layout());
        buffer.get_raw_attribute_ref(0, &attributes::INTENSITY);
    }

    #[test]
    #[should_panic]
    fn test_per_attribute_point_buffer_get_raw_attribute_ref_invalid_attribute() {
        let buffer = get_per_attribute_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);
        buffer.get_raw_attribute_ref(0, &attributes::SCAN_ANGLE_RANK);
    }

    #[test]
    fn test_per_attribute_point_buffer_get_raw_attribute_range_ref() {
        let buffer = get_per_attribute_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);

        let intensity_attribute_range =
            buffer.get_raw_attribute_range_ref(0..2, &attributes::INTENSITY);
        let intensity_attribute_range_typed = unsafe {
            std::slice::from_raw_parts(intensity_attribute_range.as_ptr() as *const u16, 2)
        };

        assert_eq!(&[42, 43], intensity_attribute_range_typed);

        let gps_time_attribute_range =
            buffer.get_raw_attribute_range_ref(0..2, &attributes::GPS_TIME);
        let gps_time_attribute_range_typed = unsafe {
            std::slice::from_raw_parts(gps_time_attribute_range.as_ptr() as *const f64, 2)
        };

        assert_eq!(&[0.123, 0.456], gps_time_attribute_range_typed);
    }

    #[test]
    #[should_panic]
    fn test_per_attribute_point_buffer_get_raw_attribute_range_ref_out_of_bounds() {
        let buffer = get_empty_per_attribute_point_buffer(TestPointType::layout());
        buffer.get_raw_attribute_range_ref(0..2, &attributes::INTENSITY);
    }

    #[test]
    #[should_panic]
    fn test_per_attribute_point_buffer_get_raw_attribute_range_ref_invalid_attribute() {
        let buffer = get_per_attribute_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);
        buffer.get_raw_attribute_range_ref(0..2, &attributes::SCAN_ANGLE_RANK);
    }

    #[test]
    fn test_per_attribute_point_buffer_mut_get_raw_attribute_mut() {
        let mut buffer = get_per_attribute_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);

        {
            let first_point_attribute_mut = buffer.get_raw_attribute_mut(0, &attributes::INTENSITY);
            let first_point_attribute_mut_typed =
                unsafe { &mut *(first_point_attribute_mut.as_mut_ptr() as *mut u16) };
            *first_point_attribute_mut_typed = 128;
        }

        let first_point_attribute_ref = buffer.get_raw_attribute_ref(0, &attributes::INTENSITY);
        let first_point_attribute_ref_typed =
            unsafe { *(first_point_attribute_ref.as_ptr() as *const u16) };

        assert_eq!(128, first_point_attribute_ref_typed);

        {
            let second_point_attribute_mut = buffer.get_raw_attribute_mut(1, &attributes::GPS_TIME);
            let second_point_attribute_mut_typed =
                unsafe { &mut *(second_point_attribute_mut.as_mut_ptr() as *mut f64) };
            *second_point_attribute_mut_typed = 3.14159;
        }

        let second_point_attribute_ref = buffer.get_raw_attribute_ref(1, &attributes::GPS_TIME);
        let second_point_attribute_ref_typed =
            unsafe { *(second_point_attribute_ref.as_ptr() as *const f64) };

        assert_eq!(3.14159, second_point_attribute_ref_typed);
    }

    #[test]
    #[should_panic]
    fn test_per_attribute_point_buffer_mut_get_raw_attribute_mut_out_of_bounds() {
        let mut buffer = get_empty_per_attribute_point_buffer(TestPointType::layout());
        buffer.get_raw_attribute_mut(0, &attributes::INTENSITY);
    }

    #[test]
    #[should_panic]
    fn test_per_attribute_point_buffer_mut_get_raw_attribute_mut_invalid_attribute() {
        let mut buffer = get_per_attribute_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);
        buffer.get_raw_attribute_mut(0, &attributes::SCAN_ANGLE_RANK);
    }

    #[test]
    fn test_per_attribute_point_buffer_mut_get_raw_attribute_range_mut() {
        let mut buffer = get_per_attribute_point_buffer_from_points(&[
            TestPointType(42, 0.123),
            TestPointType(43, 0.456),
        ]);

        {
            let intensities_mut = buffer.get_raw_attribute_range_mut(0..2, &attributes::INTENSITY);
            let intensities_mut_typed = unsafe {
                std::slice::from_raw_parts_mut(intensities_mut.as_mut_ptr() as *mut u16, 2)
            };

            intensities_mut_typed[0] = 128;
            intensities_mut_typed[1] = 129;
        }

        let intensity_attribute_range =
            buffer.get_raw_attribute_range_ref(0..2, &attributes::INTENSITY);
        let intensity_attribute_range_typed = unsafe {
            std::slice::from_raw_parts(intensity_attribute_range.as_ptr() as *const u16, 2)
        };

        assert_eq!(&[128, 129], intensity_attribute_range_typed);

        {
            let gps_times_mut = buffer.get_raw_attribute_range_mut(0..2, &attributes::GPS_TIME);
            let gps_times_mut_typed = unsafe {
                std::slice::from_raw_parts_mut(gps_times_mut.as_mut_ptr() as *mut f64, 2)
            };

            gps_times_mut_typed[0] = 3.14159;
            gps_times_mut_typed[1] = 2.71828;
        }

        let gps_time_attribute_range =
            buffer.get_raw_attribute_range_ref(0..2, &attributes::GPS_TIME);
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

        storage.get_raw_point(0, &mut buf[..]);
        assert_eq!(
            reference_bytes_1, buf,
            "get_raw_point: Bytes are not equal!"
        );
        storage.get_raw_point(1, &mut buf[..]);
        assert_eq!(
            reference_bytes_2, buf,
            "get_raw_point: Bytes are not equal!"
        );

        let mut buf_for_both_points: Vec<u8> = vec![0; reference_bytes_all.len()];
        storage.get_raw_points(0..2, &mut buf_for_both_points[..]);
        assert_eq!(
            reference_bytes_all, buf_for_both_points,
            "get_raw_points: Bytes are not equal!"
        );

        let point_1_bytes_ref = storage.get_raw_point_ref(0);
        assert_eq!(
            reference_bytes_1, point_1_bytes_ref,
            "get_point_by_ref: Bytes are not equal!"
        );

        let point_2_bytes_ref = storage.get_raw_point_ref(1);
        assert_eq!(
            reference_bytes_2, point_2_bytes_ref,
            "get_point_by_ref: Bytes are not equal!"
        );

        let both_points_bytes_ref = storage.get_raw_points_ref(0..2);
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

        storage.get_raw_attribute(0, &attributes::INTENSITY, &mut attribute_1_buf[..]);
        assert_eq!(
            ref_bytes_p1_a1, attribute_1_buf,
            "get_raw_attribute: Bytes are not equal"
        );
        storage.get_raw_attribute(1, &attributes::INTENSITY, &mut attribute_1_buf[..]);
        assert_eq!(
            ref_bytes_p2_a1, attribute_1_buf,
            "get_raw_attribute: Bytes are not equal"
        );

        storage.get_raw_attribute(0, &attributes::GPS_TIME, &mut attribute_2_buf[..]);
        assert_eq!(
            ref_bytes_p1_a2, attribute_2_buf,
            "get_raw_attribute: Bytes are not equal"
        );
        storage.get_raw_attribute(1, &attributes::GPS_TIME, &mut attribute_2_buf[..]);
        assert_eq!(
            ref_bytes_p2_a2, attribute_2_buf,
            "get_raw_attribute: Bytes are not equal"
        );

        let mut all_attribute_1_buf: Vec<u8> = vec![0; 4];
        let mut all_attribute_2_buf: Vec<u8> = vec![0; 16];

        storage.get_raw_attribute_range(0..2, &attributes::INTENSITY, &mut all_attribute_1_buf[..]);
        assert_eq!(
            ref_bytes_all_a1, all_attribute_1_buf,
            "get_raw_attribute_range: Bytes are not equal"
        );

        storage.get_raw_attribute_range(0..2, &attributes::GPS_TIME, &mut all_attribute_2_buf[..]);
        assert_eq!(
            ref_bytes_all_a2, all_attribute_2_buf,
            "get_raw_attribute_range: Bytes are not equal"
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

        let first_point_ref = slice.get_raw_point_ref(0);
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

        storage.get_raw_point(0, &mut buf[..]);
        assert_eq!(
            reference_bytes_1, buf,
            "get_raw_point: Bytes are not equal!"
        );
        storage.get_raw_point(1, &mut buf[..]);
        assert_eq!(
            reference_bytes_2, buf,
            "get_raw_point: Bytes are not equal!"
        );

        let mut buf_for_both_points: Vec<u8> = vec![0; reference_bytes_all.len()];
        storage.get_raw_points(0..2, &mut buf_for_both_points[..]);
        assert_eq!(
            reference_bytes_all, buf_for_both_points,
            "get_raw_points: Bytes are not equal!"
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

        storage.get_raw_attribute(0, &attributes::INTENSITY, &mut attribute_1_buf[..]);
        assert_eq!(
            ref_bytes_p1_a1, attribute_1_buf,
            "get_raw_attribute: Bytes are not equal"
        );
        storage.get_raw_attribute(1, &attributes::INTENSITY, &mut attribute_1_buf[..]);
        assert_eq!(
            ref_bytes_p2_a1, attribute_1_buf,
            "get_raw_attribute: Bytes are not equal"
        );

        assert_eq!(
            ref_bytes_p1_a1,
            storage.get_raw_attribute_ref(0, &attributes::INTENSITY),
            "get_attribute_by_ref: Bytes are not equal"
        );
        assert_eq!(
            ref_bytes_p2_a1,
            storage.get_raw_attribute_ref(1, &attributes::INTENSITY),
            "get_attribute_by_ref: Bytes are not equal"
        );

        storage.get_raw_attribute(0, &attributes::GPS_TIME, &mut attribute_2_buf[..]);
        assert_eq!(
            ref_bytes_p1_a2, attribute_2_buf,
            "get_raw_attribute: Bytes are not equal"
        );
        storage.get_raw_attribute(1, &attributes::GPS_TIME, &mut attribute_2_buf[..]);
        assert_eq!(
            ref_bytes_p2_a2, attribute_2_buf,
            "get_raw_attribute: Bytes are not equal"
        );

        assert_eq!(
            ref_bytes_p1_a2,
            storage.get_raw_attribute_ref(0, &attributes::GPS_TIME),
            "get_attribute_by_ref: Bytes are not equal"
        );
        assert_eq!(
            ref_bytes_p2_a2,
            storage.get_raw_attribute_ref(1, &attributes::GPS_TIME),
            "get_attribute_by_ref: Bytes are not equal"
        );

        let mut all_attribute_1_buf: Vec<u8> = vec![0; 4];
        let mut all_attribute_2_buf: Vec<u8> = vec![0; 16];

        storage.get_raw_attribute_range(0..2, &attributes::INTENSITY, &mut all_attribute_1_buf[..]);
        assert_eq!(
            ref_bytes_all_a1, all_attribute_1_buf,
            "get_raw_attribute_range: Bytes are not equal"
        );
        assert_eq!(
            ref_bytes_all_a1,
            storage.get_raw_attribute_range_ref(0..2, &attributes::INTENSITY),
            "get_attribute_range_by_ref: Bytes are not equal"
        );

        storage.get_raw_attribute_range(0..2, &attributes::GPS_TIME, &mut all_attribute_2_buf[..]);
        assert_eq!(
            ref_bytes_all_a2, all_attribute_2_buf,
            "get_raw_attribute_range: Bytes are not equal"
        );
        assert_eq!(
            ref_bytes_all_a2,
            storage.get_raw_attribute_range_ref(0..2, &attributes::GPS_TIME),
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

    #[test]
    fn test_per_attribute_vec_storage_extend_from_interleaved() {
        let mut per_attribute_buffer = PerAttributeVecPointStorage::new(TestPointType::layout());

        let mut interleaved_buffer = InterleavedVecPointStorage::new(TestPointType::layout());
        interleaved_buffer.push_points(&[TestPointType(42, 0.123), TestPointType(43, 0.456)]);

        per_attribute_buffer.push(&interleaved_buffer);

        assert_eq!(2, per_attribute_buffer.len());
        let attrib1 = attribute_slice::<u16>(&per_attribute_buffer, 0..2, &attributes::INTENSITY);
        assert_eq!(attrib1, &[42, 43]);

        let attrib2 = attribute_slice::<f64>(&per_attribute_buffer, 0..2, &attributes::GPS_TIME);
        assert_eq!(attrib2, &[0.123, 0.456]);
    }

    #[test]
    fn test_per_attribute_vec_storage_push_attribute() {
        let mut per_attribute_buffer = PerAttributeVecPointStorage::new(TestPointType::layout());

        per_attribute_buffer.push_attribute(&attributes::INTENSITY, 42_u16);
        per_attribute_buffer.push_attribute(&attributes::GPS_TIME, 0.123);

        assert_eq!(1, per_attribute_buffer.len());

        let attrib1 = attribute_slice::<u16>(&per_attribute_buffer, 0..1, &attributes::INTENSITY);
        assert_eq!(attrib1, &[42]);

        let attrib2 = attribute_slice::<f64>(&per_attribute_buffer, 0..1, &attributes::GPS_TIME);
        assert_eq!(attrib2, &[0.123]);
    }

    #[test]
    #[should_panic]
    fn test_per_attribute_vec_storage_push_attribute_wrong_type() {
        let mut per_attribute_buffer = PerAttributeVecPointStorage::new(TestPointType::layout());

        // This is a subtle bug that absolutely has to be caught by pasture: The attribute INTENSITY has default datatype U16,
        // however integer literals are i32 by default. Since we don't specify the generic argument of 'push_attribute' it is
        // deduced as 'i32', which doesn't match the datatype of the INTENSITY attribute!
        per_attribute_buffer.push_attribute(&attributes::INTENSITY, 42);
    }

    #[test]
    fn test_per_attribute_vec_storage_push_attribute_range() {
        let mut per_attribute_buffer = PerAttributeVecPointStorage::new(TestPointType::layout());

        per_attribute_buffer.push_attribute_range(&attributes::INTENSITY, &[42_u16, 43_u16]);
        per_attribute_buffer.push_attribute_range(&attributes::GPS_TIME, &[0.123, 0.456]);

        assert_eq!(2, per_attribute_buffer.len());

        let attrib1 = attribute_slice::<u16>(&per_attribute_buffer, 0..2, &attributes::INTENSITY);
        assert_eq!(attrib1, &[42, 43]);

        let attrib2 = attribute_slice::<f64>(&per_attribute_buffer, 0..2, &attributes::GPS_TIME);
        assert_eq!(attrib2, &[0.123, 0.456]);
    }

    #[test]
    #[should_panic]
    fn test_per_attribute_vec_storage_push_attribute_range_wrong_type() {
        let mut per_attribute_buffer = PerAttributeVecPointStorage::new(TestPointType::layout());

        // See comment in test_per_attribute_vec_storage_push_attribute_wrong_type()
        per_attribute_buffer.push_attribute_range(&attributes::INTENSITY, &[42, 43]);
    }

    #[test]
    fn test_interleaved_point_buffer_extend_from_per_attribute() {
        let mut interleaved_buffer = InterleavedVecPointStorage::new(TestPointType::layout());

        let mut per_attribute_buffer = PerAttributeVecPointStorage::new(TestPointType::layout());
        per_attribute_buffer.push_points(&[TestPointType(42, 0.123), TestPointType(43, 0.456)]);

        interleaved_buffer.push(&per_attribute_buffer);

        assert_eq!(2, interleaved_buffer.len());

        let points: Vec<TestPointType> = interleaved_buffer.iter_point().collect();
        assert_eq!(
            points,
            vec![TestPointType(42, 0.123), TestPointType(43, 0.456)]
        );
    }

    #[test]
    fn test_per_attribute_point_buffer_extend_from_interleaved() {
        let mut per_attribute_buffer = PerAttributeVecPointStorage::new(TestPointType::layout());

        let mut interleaved_buffer = InterleavedVecPointStorage::new(TestPointType::layout());
        interleaved_buffer.push_points(&[TestPointType(42, 0.123), TestPointType(43, 0.456)]);

        per_attribute_buffer.push(&interleaved_buffer);

        assert_eq!(2, per_attribute_buffer.len());

        let points: Vec<TestPointType> = per_attribute_buffer.iter_point().collect();
        assert_eq!(
            points,
            vec![TestPointType(42, 0.123), TestPointType(43, 0.456)]
        );
    }

    #[test]
    fn test_interleaved_point_buffer_splice_from_interleaved() {
        let mut source_points = InterleavedVecPointStorage::new(TestPointType::layout());
        source_points.push_points(&[
            TestPointType(1, 0.1),
            TestPointType(2, 0.2),
            TestPointType(3, 0.3),
            TestPointType(4, 0.4),
        ]);

        let mut new_points = InterleavedVecPointStorage::new(TestPointType::layout());
        new_points.push_points(&[TestPointType(42, 0.42), TestPointType(43, 0.43)]);

        source_points.splice(1..3, &new_points);

        assert_eq!(4, source_points.len());

        let points: Vec<TestPointType> = source_points.iter_point().collect();
        assert_eq!(
            points,
            vec![
                TestPointType(1, 0.1),
                TestPointType(42, 0.42),
                TestPointType(43, 0.43),
                TestPointType(4, 0.4)
            ]
        );
    }

    #[test]
    fn test_interleaved_point_buffer_splice_from_per_attribute() {
        let mut source_points = InterleavedVecPointStorage::new(TestPointType::layout());
        source_points.push_points(&[
            TestPointType(1, 0.1),
            TestPointType(2, 0.2),
            TestPointType(3, 0.3),
            TestPointType(4, 0.4),
        ]);

        let mut new_points = PerAttributeVecPointStorage::new(TestPointType::layout());
        new_points.push_points(&[TestPointType(42, 0.42), TestPointType(43, 0.43)]);

        source_points.splice(1..3, &new_points);

        assert_eq!(4, source_points.len());

        let points: Vec<TestPointType> = source_points.iter_point().collect();
        assert_eq!(
            points,
            vec![
                TestPointType(1, 0.1),
                TestPointType(42, 0.42),
                TestPointType(43, 0.43),
                TestPointType(4, 0.4)
            ]
        );
    }

    #[test]
    fn test_per_attribute_point_buffer_splice_from_interleaved() {
        let mut source_points = PerAttributeVecPointStorage::new(TestPointType::layout());
        source_points.push_points(&[
            TestPointType(1, 0.1),
            TestPointType(2, 0.2),
            TestPointType(3, 0.3),
            TestPointType(4, 0.4),
        ]);

        let mut new_points = InterleavedVecPointStorage::new(TestPointType::layout());
        new_points.push_points(&[TestPointType(42, 0.42), TestPointType(43, 0.43)]);

        source_points.splice(1..3, &new_points);

        assert_eq!(4, source_points.len());

        let points: Vec<TestPointType> = source_points.iter_point().collect();
        assert_eq!(
            points,
            vec![
                TestPointType(1, 0.1),
                TestPointType(42, 0.42),
                TestPointType(43, 0.43),
                TestPointType(4, 0.4)
            ]
        );
    }

    #[test]
    fn test_per_attribute_point_buffer_splice_from_per_attribute() {
        let mut source_points = PerAttributeVecPointStorage::new(TestPointType::layout());
        source_points.push_points(&[
            TestPointType(1, 0.1),
            TestPointType(2, 0.2),
            TestPointType(3, 0.3),
            TestPointType(4, 0.4),
        ]);

        let mut new_points = PerAttributeVecPointStorage::new(TestPointType::layout());
        new_points.push_points(&[TestPointType(42, 0.42), TestPointType(43, 0.43)]);

        source_points.splice(1..3, &new_points);

        assert_eq!(4, source_points.len());

        let points: Vec<TestPointType> = source_points.iter_point().collect();
        assert_eq!(
            points,
            vec![
                TestPointType(1, 0.1),
                TestPointType(42, 0.42),
                TestPointType(43, 0.43),
                TestPointType(4, 0.4)
            ]
        );
    }

    #[test]
    #[should_panic(expected = "does not match this PointLayout")]
    fn test_interleaved_point_buffer_splice_from_interleaved_wrong_layout() {
        let mut source_points = InterleavedVecPointStorage::new(TestPointType::layout());
        source_points.push_points(&[
            TestPointType(1, 0.1),
            TestPointType(2, 0.2),
            TestPointType(3, 0.3),
            TestPointType(4, 0.4),
        ]);

        let mut other_points = InterleavedVecPointStorage::new(OtherPointType::layout());
        other_points.push_points(&[OtherPointType(Vector3::new(0.0, 1.0, 2.0), 23)]);

        source_points.splice(0..1, &other_points);
    }

    #[test]
    #[should_panic(expected = "does not match this PointLayout")]
    fn test_interleaved_point_buffer_splice_from_per_attribute_wrong_layout() {
        let mut source_points = InterleavedVecPointStorage::new(TestPointType::layout());
        source_points.push_points(&[
            TestPointType(1, 0.1),
            TestPointType(2, 0.2),
            TestPointType(3, 0.3),
            TestPointType(4, 0.4),
        ]);

        let mut other_points = PerAttributeVecPointStorage::new(OtherPointType::layout());
        other_points.push_points(&[OtherPointType(Vector3::new(0.0, 1.0, 2.0), 23)]);

        source_points.splice(0..1, &other_points);
    }

    #[test]
    #[should_panic(expected = "does not match this PointLayout")]
    fn test_per_attribute_point_buffer_splice_from_interleaved_wrong_layout() {
        let mut source_points = PerAttributeVecPointStorage::new(TestPointType::layout());
        source_points.push_points(&[
            TestPointType(1, 0.1),
            TestPointType(2, 0.2),
            TestPointType(3, 0.3),
            TestPointType(4, 0.4),
        ]);

        let mut other_points = InterleavedVecPointStorage::new(OtherPointType::layout());
        other_points.push_points(&[OtherPointType(Vector3::new(0.0, 1.0, 2.0), 23)]);

        source_points.splice(0..1, &other_points);
    }

    #[test]
    #[should_panic(expected = "does not match this PointLayout")]
    fn test_per_attribute_point_buffer_splice_from_per_attribute_wrong_layout() {
        let mut source_points = PerAttributeVecPointStorage::new(TestPointType::layout());
        source_points.push_points(&[
            TestPointType(1, 0.1),
            TestPointType(2, 0.2),
            TestPointType(3, 0.3),
            TestPointType(4, 0.4),
        ]);

        let mut other_points = PerAttributeVecPointStorage::new(OtherPointType::layout());
        other_points.push_points(&[OtherPointType(Vector3::new(0.0, 1.0, 2.0), 23)]);

        source_points.splice(0..1, &other_points);
    }

    #[test]
    fn test_interleaved_point_buffer_from_iterator() {
        let no_points: Vec<TestPointType> = vec![];
        let some_points: Vec<TestPointType> =
            vec![TestPointType(42, 0.123), TestPointType(43, 0.456)];

        {
            let empty_buffer: InterleavedVecPointStorage = no_points.into_iter().collect();
            assert_eq!(0, empty_buffer.len());
            assert_eq!(TestPointType::layout(), *empty_buffer.point_layout());
        }

        {
            let non_empty_buffer: InterleavedVecPointStorage =
                some_points.clone().into_iter().collect();
            assert_eq!(some_points.len(), non_empty_buffer.len());
            assert_eq!(TestPointType::layout(), *non_empty_buffer.point_layout());

            let points: Vec<TestPointType> = non_empty_buffer.iter_point().collect();
            assert_eq!(some_points, points);
        }
    }

    #[test]
    fn test_interleaved_point_buffer_from_slice() {
        let no_points: Vec<TestPointType> = vec![];
        let some_points: Vec<TestPointType> =
            vec![TestPointType(42, 0.123), TestPointType(43, 0.456)];

        {
            let empty_buffer = InterleavedVecPointStorage::from(no_points.as_slice());
            assert_eq!(0, empty_buffer.len());
            assert_eq!(TestPointType::layout(), *empty_buffer.point_layout());
        }

        {
            let non_empty_buffer = InterleavedVecPointStorage::from(some_points.as_slice());
            assert_eq!(2, non_empty_buffer.len());
            assert_eq!(TestPointType::layout(), *non_empty_buffer.point_layout());

            let points: Vec<TestPointType> = non_empty_buffer.iter_point().collect();
            assert_eq!(some_points, points);
        }
    }

    #[test]
    fn test_per_attribute_point_buffer_from_iterator() {
        let no_points: Vec<TestPointType> = vec![];
        let some_points: Vec<TestPointType> =
            vec![TestPointType(42, 0.123), TestPointType(43, 0.456)];

        {
            let empty_buffer: PerAttributeVecPointStorage = no_points.into_iter().collect();
            assert_eq!(0, empty_buffer.len());
            assert_eq!(TestPointType::layout(), *empty_buffer.point_layout());
        }

        {
            let non_empty_buffer: PerAttributeVecPointStorage =
                some_points.clone().into_iter().collect();
            assert_eq!(some_points.len(), non_empty_buffer.len());
            assert_eq!(TestPointType::layout(), *non_empty_buffer.point_layout());

            let points: Vec<TestPointType> = non_empty_buffer.iter_point().collect();
            assert_eq!(some_points, points);
        }
    }

    #[test]
    fn test_per_attribute_point_buffer_from_slice() {
        let no_points: Vec<TestPointType> = vec![];
        let some_points: Vec<TestPointType> =
            vec![TestPointType(42, 0.123), TestPointType(43, 0.456)];

        {
            let empty_buffer = PerAttributeVecPointStorage::from(no_points.as_slice());
            assert_eq!(0, empty_buffer.len());
            assert_eq!(TestPointType::layout(), *empty_buffer.point_layout());
        }

        {
            let non_empty_buffer = PerAttributeVecPointStorage::from(some_points.as_slice());
            assert_eq!(2, non_empty_buffer.len());
            assert_eq!(TestPointType::layout(), *non_empty_buffer.point_layout());

            let points: Vec<TestPointType> = non_empty_buffer.iter_point().collect();
            assert_eq!(some_points, points);
        }
    }

    #[test]
    fn test_point_buffer_extension_trait() {
        use crate::containers::PointBufferExt;

        let reference_points = vec![TestPointType(42, 0.123), TestPointType(43, 0.456)];

        // Interleaved
        {
            let buf = get_interleaved_point_buffer_from_points(reference_points.as_slice());

            assert_eq!(TestPointType(42, 0.123), buf.get_point(0));
            assert_eq!(TestPointType(43, 0.456), buf.get_point(1));

            assert_eq!(42, buf.get_attribute::<u16>(&attributes::INTENSITY, 0));
            assert_eq!(43, buf.get_attribute::<u16>(&attributes::INTENSITY, 1));
            assert_eq!(0.123, buf.get_attribute::<f64>(&attributes::GPS_TIME, 0));
            assert_eq!(0.456, buf.get_attribute::<f64>(&attributes::GPS_TIME, 1));

            let all_points: Vec<TestPointType> = buf.iter_point().collect();
            assert_eq!(all_points, reference_points);
        }

        // PerAttribute
        {
            let buf = get_per_attribute_point_buffer_from_points(reference_points.as_slice());

            assert_eq!(TestPointType(42, 0.123), buf.get_point(0));
            assert_eq!(TestPointType(43, 0.456), buf.get_point(1));

            assert_eq!(42, buf.get_attribute::<u16>(&attributes::INTENSITY, 0));
            assert_eq!(43, buf.get_attribute::<u16>(&attributes::INTENSITY, 1));
            assert_eq!(0.123, buf.get_attribute::<f64>(&attributes::GPS_TIME, 0));
            assert_eq!(0.456, buf.get_attribute::<f64>(&attributes::GPS_TIME, 1));

            let all_points: Vec<TestPointType> = buf.iter_point().collect();
            assert_eq!(all_points, reference_points);
        }
    }
}
