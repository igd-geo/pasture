use std::ops::Range;

use crate::layout::{
    FieldAlignment, PointAttributeDefinition, PointLayout, PointType, PrimitiveType,
};

use super::{
    InterleavedPointBuffer, PerAttributePointBuffer, PerAttributePointBufferSlice, PointBuffer,
};

/// A non-owning view for a contiguous slice of interleaved point data. This is like `InterleavedVecPointBuffer`, but it
/// does not own the point data. It is useful for passing around point data in an untyped but safe manner, for example in
/// I/O heavy code.
pub struct InterleavedPointView<'d> {
    point_data: &'d [u8],
    point_layout: PointLayout,
    point_count: usize,
    size_of_point_entry: usize,
}

impl<'d> InterleavedPointView<'d> {
    /// Creates a new `InterleavedPointView` referencing the given slice of points
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
    /// let points = vec![MyPointType(42), MyPointType(43)];
    /// let view = InterleavedPointView::from_slice(points.as_slice());
    /// ```
    pub fn from_slice<T: PointType>(points: &'d [T]) -> Self {
        let raw_points_data = unsafe {
            std::slice::from_raw_parts(
                points.as_ptr() as *const u8,
                points.len() * std::mem::size_of::<T>(),
            )
        };
        let point_layout = T::layout();
        let size_of_point_entry = point_layout.size_of_point_entry() as usize;
        Self {
            point_data: raw_points_data,
            point_layout,
            point_count: points.len(),
            size_of_point_entry,
        }
    }

    /// Creates a new `InterleavedPointView` referencing the given slice of untyped point data
    pub fn from_raw_slice(points: &'d [u8], layout: PointLayout) -> Self {
        let size_of_single_point = layout.size_of_point_entry() as usize;
        if points.len() % size_of_single_point != 0 {
            panic!("InterleavedPointView::from_raw_slice: points.len() is no multiple of point entry size in PointLayout!");
        }
        let num_points = points.len() / size_of_single_point;
        let size_of_point_entry = layout.size_of_point_entry() as usize;
        Self {
            point_data: points,
            point_layout: layout,
            point_count: num_points,
            size_of_point_entry,
        }
    }

    /// Returns the data for the points of the associated `InterleavedPointView` as a typed slice.
    ///
    /// # Panics
    ///
    /// If the `PointLayout` of the view does not match the layout of type `T`
    ///
    /// ```
    /// # use pasture_core::containers::*;
    /// # use pasture_core::layout::*;
    /// # use pasture_derive::PointType;
    ///
    /// #[repr(C)]
    /// #[derive(Debug, Copy, Clone, PartialEq, Eq, PointType)]
    /// struct MyPointType(#[pasture(BUILTIN_INTENSITY)] u16);
    ///
    /// let points = vec![MyPointType(42), MyPointType(43)];
    /// let view = InterleavedPointView::from_slice(points.as_slice());
    /// let points_ref = view.get_typed_data::<MyPointType>();
    /// assert_eq!(points_ref, points.as_slice());
    /// ```
    pub fn get_typed_data<T: PointType>(&self) -> &'d [T] {
        if self.point_layout != T::layout() {
            panic!("InterleavedPointView::get_typed_data: Point layout does not match type T!");
        }
        unsafe {
            std::slice::from_raw_parts(self.point_data.as_ptr() as *const T, self.point_count)
        }
    }
}

impl<'d> PointBuffer for InterleavedPointView<'d> {
    // TODO Refactor the code here and in InterleavedVecPointStorage (they are basically identical) by extracting them into a standalone function
    fn get_point_by_copy(&self, point_index: usize, buf: &mut [u8]) {
        if point_index >= self.len() {
            panic!(
                "InterleavedPointView::get_point_by_copy: Point index {} out of bounds",
                point_index
            );
        }

        let offset_to_point = point_index * self.size_of_point_entry;
        let point_slice =
            &self.point_data[offset_to_point..(offset_to_point + self.size_of_point_entry)];
        buf.copy_from_slice(point_slice);
    }

    fn get_attribute_by_copy(
        &self,
        point_index: usize,
        attribute: &PointAttributeDefinition,
        buf: &mut [u8],
    ) {
        if point_index >= self.len() {
            panic!(
                "InterleavedPointView::get_attribute_by_copy: Point index {} out of bounds!",
                point_index
            );
        }

        if let Some(attribute_in_buffer) = self.point_layout.get_attribute(attribute) {
            let offset_to_point_bytes = point_index * self.size_of_point_entry as usize;
            let offset_to_attribute = offset_to_point_bytes + attribute_in_buffer.offset() as usize;
            let attribute_size = attribute.size() as usize;

            buf.copy_from_slice(
                &self.point_data[offset_to_attribute..offset_to_attribute + attribute_size],
            );
        } else {
            panic!("InterleavedPointView::get_attribute_by_copy: Attribute {:?} is not part of this PointBuffer's PointLayout!", attribute);
        }
    }

    fn get_points_by_copy(&self, index_range: std::ops::Range<usize>, buf: &mut [u8]) {
        let points_ref = self.get_points_ref(index_range);
        buf[0..points_ref.len()].copy_from_slice(points_ref);
    }

    fn get_attribute_range_by_copy(
        &self,
        index_range: std::ops::Range<usize>,
        attribute: &PointAttributeDefinition,
        buf: &mut [u8],
    ) {
        if index_range.end > self.len() {
            panic!(
                "InterleavedPointView::get_attribute_range_by_copy: Point indices {:?} out of bounds!",
                index_range
            );
        }

        if let Some(attribute_in_buffer) = self.point_layout.get_attribute(attribute) {
            let attribute_size = attribute.size() as usize;
            let start_index = index_range.start;

            for point_index in index_range {
                let offset_to_point_bytes = point_index * self.size_of_point_entry as usize;
                let offset_to_attribute =
                    offset_to_point_bytes + attribute_in_buffer.offset() as usize;
                let offset_in_target_buf = (point_index - start_index) * attribute_size;
                let target_buf_slice =
                    &mut buf[offset_in_target_buf..offset_in_target_buf + attribute_size];

                target_buf_slice.copy_from_slice(
                    &self.point_data[offset_to_attribute..offset_to_attribute + attribute_size],
                );
            }
        } else {
            panic!("InterleavedPointView::get_attribute_by_copy: Attribute {:?} is not part of this PointBuffer's PointLayout!", attribute);
        }
    }

    fn len(&self) -> usize {
        self.point_count
    }

    fn point_layout(&self) -> &PointLayout {
        &self.point_layout
    }

    fn as_interleaved(&self) -> Option<&dyn InterleavedPointBuffer> {
        Some(self)
    }
}

impl<'d> InterleavedPointBuffer for InterleavedPointView<'d> {
    fn get_point_ref(&self, point_index: usize) -> &[u8] {
        if point_index >= self.len() {
            panic!(
                "InterleavedPointView::get_point_ref: Point index {} out of bounds!",
                point_index
            );
        }

        let offset_to_point = point_index * self.size_of_point_entry as usize;
        &self.point_data[offset_to_point..offset_to_point + self.size_of_point_entry as usize]
    }

    fn get_points_ref(&self, index_range: std::ops::Range<usize>) -> &[u8] {
        if index_range.end > self.len() {
            panic!(
                "InterleavedPointView::get_points_ref: Point indices {:?} out of bounds!",
                index_range
            );
        }

        let offset_to_point = index_range.start * self.size_of_point_entry as usize;
        let total_bytes_of_range =
            (index_range.end - index_range.start) * self.size_of_point_entry as usize;
        &self.point_data[offset_to_point..offset_to_point + total_bytes_of_range]
    }
}

/// A non-owning view for per-attribute point data. This is like `PerAttributeVecPointBuffer`, but it does not own the
/// point data. It is useful for passing around point data in an untyped but safe manner, for example in I/O heavy code.
pub struct PerAttributePointView<'d> {
    point_data: Vec<&'d [u8]>,
    point_layout: PointLayout,
    point_count: usize,
}

impl<'d> PerAttributePointView<'d> {
    /// Creates a new empty `PerAttributePointView` that stores no data and has an empty `PointLayout`
    /// ```
    /// # use pasture_core::containers::*;
    /// # use pasture_core::layout::*;
    ///
    /// let point_view = PerAttributePointView::new();
    /// assert_eq!(point_view.point_layout(), &PointLayout::default());
    /// ```
    pub fn new() -> Self {
        Self {
            point_data: vec![],
            point_layout: PointLayout::default(),
            point_count: 0,
        }
    }

    /// Creates a new `PerAttributePointView` from the given attribute buffers and `PointLayout`. The
    /// `attributes` parameter must contain one slice for each `PointAttributeDefinition` in the given
    /// `PointLayout`, in the exact order in which they are defined in the `PointLayout`.
    /// *Note*: You will rarely have to use this method directly, instead prefer to use the `per_attribute_point_view!`
    /// macro or use `PerAttributePointView::new()` together with `PerAttributePointView::push_attribute`
    ///
    /// # Panics
    ///
    /// If the slices in `attributes` don't match the expected data layout of `point_layout`. Reasons for this
    /// can be that `attributes.len()` does not match the number of attributes in the `PointLayout`, or that
    /// the length of one of the slices in `attributes` is no multiple of the size of a single entry of the
    /// corresponding point attribute in the `PointLayout`.
    pub fn from_slices(attributes: Vec<&'d [u8]>, point_layout: PointLayout) -> Self {
        if !Self::attribute_buffers_match_layout(&attributes, &point_layout) {
            panic!("PerAttributePointView::from_slices: attributes don't match the PointLayout!");
        }

        let point_count = if attributes.len() != 0 {
            attributes[0].len() / point_layout.attributes().next().unwrap().size() as usize
        } else {
            0
        };

        Self {
            point_data: attributes,
            point_layout,
            point_count,
        }
    }

    /// Push data for a new point attribute into the associated `PerAttributePointView`. This extends the internal
    /// `PointLayout` of the view. The number of entries in `attribute` must match the number of entries in all other
    /// attributes that this view stores, if there are any.
    ///
    /// # Panics
    ///
    /// If there are attributes already stored in this view, but `attribute.len()` does not match the length of the
    /// other attributes
    ///
    /// ```
    /// # use pasture_core::containers::*;
    /// # use pasture_core::layout::*;
    /// # use nalgebra::Vector3;
    ///
    /// let positions = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 1.0)];
    ///
    /// let mut point_view = PerAttributePointView::new();
    /// point_view.push_attribute(&positions, &attributes::POSITION_3D);
    /// ```
    pub fn push_attribute<T: PrimitiveType>(
        &mut self,
        attribute: &'d [T],
        attribute_definition: &PointAttributeDefinition,
    ) {
        if attribute_definition.datatype() != T::data_type() {
            panic!("PerAttributePointView::push_attribute: data type of T and PointAttributeDefinition do not match!");
        }
        if self.point_data.is_empty() {
            self.push_first_attribute(attribute, attribute_definition);
        } else {
            self.push_additional_attribute(attribute, attribute_definition);
        }
    }

    /// Returns the raw data for the given point attribute. Returns `None` if no matching attribute
    /// is stored in the associated `PerAttributePointView`.
    ///
    /// ```
    /// # use pasture_core::containers::*;
    /// # use pasture_core::layout::*;
    /// # use nalgebra::Vector3;
    ///
    /// let positions = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 1.0)];
    ///
    /// let mut point_view = PerAttributePointView::new();
    /// point_view.push_attribute(&positions, &attributes::POSITION_3D);
    ///
    /// let raw_positions = point_view.get_raw_data_for_attribute(&attributes::POSITION_3D);
    /// ```
    pub fn get_raw_data_for_attribute(
        &self,
        attribute: &PointAttributeDefinition,
    ) -> Option<&'d [u8]> {
        self.point_layout
            .index_of(attribute)
            .map(|attribute_index| self.point_data[attribute_index])
    }

    /// Returns the data for the given point attribute as a typed slice. Returns `None` if no matching
    /// attribute is stored in the associated `PerAttributePointView`.
    ///
    /// # Panics
    ///
    /// If the associated `PointAttributeDefinition` does not store values of data type `T`
    ///
    /// ```
    /// # use pasture_core::containers::*;
    /// # use pasture_core::layout::*;
    /// # use nalgebra::Vector3;
    ///
    /// let positions = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 1.0)];
    ///
    /// let mut point_view = PerAttributePointView::new();
    /// point_view.push_attribute(&positions, &attributes::POSITION_3D);
    ///
    /// let retrieved_positions = point_view.get_typed_data_for_attribute::<Vector3<f64>>(&attributes::POSITION_3D).unwrap();
    /// assert_eq!(retrieved_positions, positions.as_slice());
    /// ```
    pub fn get_typed_data_for_attribute<T: PrimitiveType>(
        &self,
        attribute: &PointAttributeDefinition,
    ) -> Option<&'d [T]> {
        if attribute.datatype() != T::data_type() {
            panic!("PerAttributePointView::get_typed_data_for_attribute: PointAttributeDefinition does not have datatype T!");
        }
        self.point_layout
            .index_of(attribute)
            .map(|attribute_index| {
                let raw_slice = self.point_data[attribute_index];
                unsafe {
                    std::slice::from_raw_parts(raw_slice.as_ptr() as *const T, self.point_count)
                }
            })
    }

    /// Returns the `PointLayout` of the associated `PerAttributePointView`
    ///
    /// ```
    /// # use pasture_core::containers::*;
    /// # use pasture_core::layout::*;
    /// # use nalgebra::Vector3;
    ///
    /// let positions = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 1.0)];
    ///
    /// let mut point_view = PerAttributePointView::new();
    /// point_view.push_attribute(&positions, &attributes::POSITION_3D);
    ///
    /// assert_eq!(point_view.point_layout(), &PointLayout::from_attributes(&[attributes::POSITION_3D]));
    /// ```
    pub fn point_layout(&self) -> &PointLayout {
        &self.point_layout
    }

    /// Returns the number of unique point entries in the associated `PerAttributePointView`
    ///
    /// ```
    /// # use pasture_core::containers::*;
    /// # use pasture_core::layout::*;
    /// # use nalgebra::Vector3;
    ///
    /// let positions = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 1.0)];
    ///
    /// let mut point_view = PerAttributePointView::new();
    /// point_view.push_attribute(&positions, &attributes::POSITION_3D);
    ///
    /// assert_eq!(2, point_view.len());
    /// ```
    pub fn len(&self) -> usize {
        self.point_count
    }

    fn push_first_attribute<T: PrimitiveType>(
        &mut self,
        attribute: &'d [T],
        attribute_definition: &PointAttributeDefinition,
    ) {
        self.point_count = attribute.len();

        let attribute_bytes = unsafe {
            std::slice::from_raw_parts(
                attribute.as_ptr() as *const u8,
                attribute.len() * std::mem::size_of::<T>(),
            )
        };
        self.point_data.push(attribute_bytes);
        self.point_layout
            .add_attribute(attribute_definition.clone(), FieldAlignment::Default);
    }

    fn push_additional_attribute<T: PrimitiveType>(
        &mut self,
        attribute: &'d [T],
        attribute_definition: &PointAttributeDefinition,
    ) {
        if attribute.len() != self.point_count {
            panic!("PerAttributePointView::push_attribute: attribute data length does not match point count of this view!");
        }

        let attribute_bytes = unsafe {
            std::slice::from_raw_parts(
                attribute.as_ptr() as *const u8,
                attribute.len() * std::mem::size_of::<T>(),
            )
        };
        self.point_data.push(attribute_bytes);
        self.point_layout
            .add_attribute(attribute_definition.clone(), FieldAlignment::Default);
    }

    fn attribute_buffers_match_layout(attributes: &[&[u8]], point_layout: &PointLayout) -> bool {
        if attributes.len() != point_layout.attributes().count() {
            return false;
        }

        // Check that all attribute slices store the same number of entries
        let mut num_points = None;
        for (attribute_definition, &data) in point_layout.attributes().zip(attributes.iter()) {
            if data.len() as u64 % attribute_definition.size() != 0 {
                return false;
            }
            let point_count_in_data = data.len() as u64 / attribute_definition.size();
            if let Some(point_count) = num_points {
                if point_count != point_count_in_data {
                    return false;
                }
            } else {
                num_points = Some(point_count_in_data);
            }
        }

        true
    }
}

impl<'d> PointBuffer for PerAttributePointView<'d> {
    fn get_point_by_copy(&self, point_index: usize, buf: &mut [u8]) {
        if point_index >= self.len() {
            panic!(
                "PerAttributePointView::get_point_by_copy: Point index {} out of bounds!",
                point_index
            );
        }

        for (idx, attribute) in self.point_layout.attributes().enumerate() {
            let attribute_buffer = self.point_data[idx];
            let attribute_size = attribute.size() as usize;
            let offset_in_buffer = point_index * attribute_size;
            let offset_in_point = attribute.offset() as usize;

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

    fn get_points_by_copy(&self, index_range: std::ops::Range<usize>, buf: &mut [u8]) {
        if index_range.end > self.len() {
            panic!(
                "PerAttributePointView::get_points_by_copy: Point indices {:?} out of bounds!",
                index_range
            );
        }

        let point_size = self.point_layout.size_of_point_entry() as usize;

        for (idx, attribute) in self.point_layout.attributes().enumerate() {
            let attribute_buffer = self.point_data[idx];
            let attribute_size = attribute.size() as usize;
            for point_index in index_range.clone() {
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

    fn get_attribute_range_by_copy(
        &self,
        index_range: std::ops::Range<usize>,
        attribute: &PointAttributeDefinition,
        buf: &mut [u8],
    ) {
        let attribute_buffer_slice = self.get_attribute_range_ref(index_range, attribute);
        buf.copy_from_slice(attribute_buffer_slice);
    }

    fn len(&self) -> usize {
        self.point_count
    }

    fn point_layout(&self) -> &PointLayout {
        &self.point_layout
    }

    fn as_per_attribute(&self) -> Option<&dyn PerAttributePointBuffer> {
        Some(self)
    }
}

impl<'d> PerAttributePointBuffer for PerAttributePointView<'d> {
    fn get_attribute_ref(&self, point_index: usize, attribute: &PointAttributeDefinition) -> &[u8] {
        if point_index >= self.len() {
            panic!(
                "PerAttributePointView::get_attribute_ref: Point index {} out of bounds!",
                point_index
            );
        }

        let attribute_index = match self.point_layout.index_of(attribute) {
            Some(idx) => idx,
            None => panic!("PerAttributePointView::get_attribute_ref: Attribute {:?} is not part of this PointBuffer's PointLayout!", attribute),
        };

        let attribute_buffer = self.point_data[attribute_index];
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
                "PerAttributePointView::get_attribute_range_ref: Point indices {:?} out of bounds!",
                index_range
            );
        }

        let attribute_index = match self.point_layout.index_of(attribute) {
            Some(idx) => idx,
            None => panic!("PerAttributePointView::get_attribute_range_ref: Attribute {:?} is not part of this PointBuffer's PointLayout!", attribute),
        };

        let attribute_buffer = self.point_data[attribute_index];
        let start_offset_in_attribute_buffer = index_range.start * attribute.size() as usize;
        let end_offset_in_attribute_buffer = start_offset_in_attribute_buffer
            + (index_range.end - index_range.start) * attribute.size() as usize;
        &attribute_buffer[start_offset_in_attribute_buffer..end_offset_in_attribute_buffer]
    }

    fn slice(&self, range: Range<usize>) -> PerAttributePointBufferSlice<'_> {
        PerAttributePointBufferSlice::new(self, range)
    }
}
