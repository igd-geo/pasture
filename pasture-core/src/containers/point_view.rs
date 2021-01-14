use crate::layout::{PointAttributeDefinition, PointLayout, PointType, PrimitiveType};

/// A non-owning view for a contiguous slice of interleaved point data. This is a low-cost
/// abstraction that enables passing around point data in an untyped but safe way! Its primary
/// use is for pushing point data into a `PointBuffer` through its trait interface
pub struct InterleavedPointView<'data> {
    point_data: &'data [u8],
    point_layout: PointLayout,
    point_count: usize,
}

impl<'data> InterleavedPointView<'data> {
    /// Creates a new `InterleavedPointView` referencing the given slice of points
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
    /// let points = vec![MyPointType(42), MyPointType(43)];
    /// let view = InterleavedPointView::from_slice(points.as_slice());
    /// ```
    pub fn from_slice<T: PointType>(points: &'data [T]) -> Self {
        let raw_points_data = unsafe {
            std::slice::from_raw_parts(
                points.as_ptr() as *const u8,
                points.len() * std::mem::size_of::<T>(),
            )
        };
        Self {
            point_data: raw_points_data,
            point_layout: T::layout(),
            point_count: points.len(),
        }
    }

    /// Creates a new `InterleavedPointView` referencing the given slice of untyped point data
    pub fn from_raw_slice(points: &'data [u8], layout : PointLayout) -> Self {
        let size_of_single_point = layout.size_of_point_entry() as usize;
        if points.len() % size_of_single_point != 0 {
            panic!("InterleavedPointView::from_raw_slice: points.len() is not multiple of point entry size in PointLayout!");
        }
        let num_points = points.len() / size_of_single_point;
        Self {
            point_data: points,
            point_layout: layout,
            point_count: num_points,
        }
    }

    /// Returns the raw data for the points that the associated `InterleavedPointView` is pointing to
    pub fn get_raw_data(&self) -> &'data [u8] {
        self.point_data
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
    ///
    /// #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    /// struct MyPointType(u16);
    ///
    /// impl PointType for MyPointType {
    ///   fn layout() -> PointLayout {
    ///     PointLayout::from_attributes(&[attributes::INTENSITY])
    ///   }
    /// }
    ///
    /// let points = vec![MyPointType(42), MyPointType(43)];
    /// let view = InterleavedPointView::from_slice(points.as_slice());
    /// let points_ref = view.get_typed_data::<MyPointType>();
    /// assert_eq!(points_ref, points.as_slice());
    /// ```
    pub fn get_typed_data<T: PointType>(&self) -> &'data [T] {
        if self.point_layout != T::layout() {
            panic!("InterleavedPointView::get_typed_data: Point layout does not match type T!");
        }
        unsafe {
            std::slice::from_raw_parts(self.point_data.as_ptr() as *const T, self.point_count)
        }
    }

    /// Returns the `PointLayout` of the associated `InterleavedPointView`
    ///
    /// ```
    /// # use pasture_core::containers::*;
    /// # use pasture_core::layout::*;
    ///
    /// #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    /// struct MyPointType(u16);
    ///
    /// impl PointType for MyPointType {
    ///   fn layout() -> PointLayout {
    ///     PointLayout::from_attributes(&[attributes::INTENSITY])
    ///   }
    /// }
    ///
    /// let points = vec![MyPointType(42), MyPointType(43)];
    /// let view = InterleavedPointView::from_slice(points.as_slice());
    /// assert_eq!(view.get_point_layout(), &MyPointType::layout());
    /// ```
    pub fn get_point_layout(&self) -> &PointLayout {
        &self.point_layout
    }

    /// Returns the number of point entries in the associated `InterleavedPointView`
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
    /// let points = vec![MyPointType(42), MyPointType(43)];
    /// let view = InterleavedPointView::from_slice(points.as_slice());
    /// assert_eq!(view.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.point_count
    }
}

/// A non-owning view for per-attribute point data
pub struct PerAttributePointView<'data> {
    point_data: Vec<&'data [u8]>,
    point_layout: PointLayout,
    point_count: usize,
}

impl<'data> PerAttributePointView<'data> {
    /// Creates a new empty `PerAttributePointView` that stores no data and has an empty `PointLayout`
    /// ```
    /// # use pasture_core::containers::*;
    /// # use pasture_core::layout::*;
    ///
    /// let point_view = PerAttributePointView::new();
    /// assert_eq!(point_view.get_point_layout(), &PointLayout::new());
    /// ```
    pub fn new() -> Self {
        Self {
            point_data: vec![],
            point_layout: PointLayout::new(),
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
    pub fn from_slices(attributes: Vec<&'data [u8]>, point_layout: PointLayout) -> Self {
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
        attribute: &'data [T],
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
    ) -> Option<&'data [u8]> {
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
    ) -> Option<&'data [T]> {
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
    /// assert_eq!(point_view.get_point_layout(), &PointLayout::from_attributes(&[attributes::POSITION_3D]));
    /// ```
    pub fn get_point_layout(&self) -> &PointLayout {
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
        attribute: &'data [T],
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
            .add_attribute(attribute_definition.clone());
    }

    fn push_additional_attribute<T: PrimitiveType>(
        &mut self,
        attribute: &'data [T],
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
            .add_attribute(attribute_definition.clone());
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
