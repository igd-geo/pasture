use nalgebra::Vector3;
use static_assertions::const_assert;

/// Possible data types for individual point attributes
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PointAttributeDataType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F32,
    F64,
    Bool,
    Vec3u8,
    Vec3u16,
    Vec3f32,
    Vec3f64,
}

/// Marker trait for all types that can be used as primitive types within a PointAttributeDefinition
pub trait PrimitiveType {}

impl PrimitiveType for u8 {}
impl PrimitiveType for u16 {}
impl PrimitiveType for u32 {}
impl PrimitiveType for u64 {}
impl PrimitiveType for i8 {}
impl PrimitiveType for i16 {}
impl PrimitiveType for i32 {}
impl PrimitiveType for i64 {}
impl PrimitiveType for f32 {}
impl PrimitiveType for f64 {}
impl PrimitiveType for bool {}
impl PrimitiveType for Vector3<u8> {}
impl PrimitiveType for Vector3<u16> {}
impl PrimitiveType for Vector3<f32> {}
impl PrimitiveType for Vector3<f64> {}

// Assert sizes of vector types are as we expect. Primitive types always are the same size, but we don't know
// what nalgebra does with the Vector3 types on the target machine...
const_assert!(std::mem::size_of::<Vector3<u8>>() == 3);
const_assert!(std::mem::size_of::<Vector3<u16>>() == 6);
const_assert!(std::mem::size_of::<Vector3<f32>>() == 12);
const_assert!(std::mem::size_of::<Vector3<f64>>() == 24);

/// A definition for a single point attribute of a point cloud. Point attributes are things like the position,
/// GPS time, intensity etc. In Pasture, attributes are identified by a unique name together with the data type
/// that a single record of the attribute is stored in. Attributes can be grouped into two categories: Built-in
/// attributes (e.g. POSITION_3D, INTENSITY, GPS_TIME etc.) and custom attributes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointAttributeDefinition {
    name: &'static str,
    datatype: PointAttributeDataType,
}

impl PointAttributeDefinition {
    /// Creates a new custom PointAttributeDefinition with the given name and data type
    /// ```
    /// # use pasture_core::layout::*;
    /// let custom_attribute = PointAttributeDefinition::custom("Custom", PointAttributeDataType::F32);
    /// # assert_eq!(custom_attribute.name(), "Custom");
    /// # assert_eq!(*custom_attribute.datatype(), PointAttributeDataType::F32);
    /// ```
    pub fn custom(name: &'static str, datatype: PointAttributeDataType) -> Self {
        Self { name, datatype }
    }

    /// Returns the name of this PointAttributeDefinition
    /// ```
    /// # use pasture_core::layout::*;
    /// let custom_attribute = PointAttributeDefinition::custom("Custom", PointAttributeDataType::F32);
    /// let name = custom_attribute.name();
    /// # assert_eq!(name, "Custom");
    /// ```
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Returns the datatype of this PointAttributeDefinition
    /// ```
    /// # use pasture_core::layout::*;
    /// let custom_attribute = PointAttributeDefinition::custom("Custom", PointAttributeDataType::F32);
    /// let datatype = custom_attribute.datatype();
    /// # assert_eq!(*datatype, PointAttributeDataType::F32);
    /// ```
    pub fn datatype(&self) -> &PointAttributeDataType {
        &self.datatype
    }

    /// Returns the size in bytes of this attribute
    pub fn size(&self) -> u64 {
        match self.datatype {
            PointAttributeDataType::Bool => 1,
            PointAttributeDataType::F32 => 4,
            PointAttributeDataType::F64 => 8,
            PointAttributeDataType::I8 => 1,
            PointAttributeDataType::I16 => 2,
            PointAttributeDataType::I32 => 4,
            PointAttributeDataType::I64 => 8,
            PointAttributeDataType::U8 => 1,
            PointAttributeDataType::U16 => 2,
            PointAttributeDataType::U32 => 4,
            PointAttributeDataType::U64 => 8,
            PointAttributeDataType::Vec3f32 => 3 * 4,
            PointAttributeDataType::Vec3f64 => 3 * 8,
            PointAttributeDataType::Vec3u16 => 3 * 2,
            PointAttributeDataType::Vec3u8 => 3,
        }
    }

    /// Returns a new PointAttributeDefinition based on this PointAttributeDefinition, but with a different datatype
    /// ```
    /// # use pasture_core::layout::*;
    /// let custom_position_attribute = attributes::POSITION_3D.with_custom_datatype(PointAttributeDataType::Vec3f32);
    /// # assert_eq!(custom_position_attribute.name(), attributes::POSITION_3D.name());
    /// # assert_eq!(*custom_position_attribute.datatype(), PointAttributeDataType::Vec3f32);
    /// ```
    pub fn with_custom_datatype(&self, new_datatype: PointAttributeDataType) -> Self {
        Self {
            name: self.name,
            datatype: new_datatype,
        }
    }
}

/// Module containing default attribute definitions
pub mod attributes {
    use super::{PointAttributeDataType, PointAttributeDefinition};

    /// Attribute definition for a 3D position. Default datatype is Vec3f64
    pub const POSITION_3D: PointAttributeDefinition = PointAttributeDefinition {
        name: "Position3D",
        datatype: PointAttributeDataType::Vec3f64,
    };

    /// Attribute definition for an intensity value. Default datatype is U16
    pub const INTENSITY: PointAttributeDefinition = PointAttributeDefinition {
        name: "Intensity",
        datatype: PointAttributeDataType::U16,
    };

    /// Attribute definition for a return number. Default datatype is U8
    pub const RETURN_NUMBER: PointAttributeDefinition = PointAttributeDefinition {
        name: "ReturnNumber",
        datatype: PointAttributeDataType::U8,
    };

    /// Attribute definition for the number of returns. Default datatype is U8
    pub const NUMBER_OF_RETURNS: PointAttributeDefinition = PointAttributeDefinition {
        name: "NumberOfReturns",
        datatype: PointAttributeDataType::U8,
    };

    /// Attribute definition for a scan direction flag. Default datatype is Bool
    pub const SCAN_DIRECTION_FLAG: PointAttributeDefinition = PointAttributeDefinition {
        name: "ScanDirectionFlag",
        datatype: PointAttributeDataType::Bool,
    };

    /// Attribute definition for an edge of flight line flag. Default datatype is Bool
    pub const EDGE_OF_FLIGHT_LINE: PointAttributeDefinition = PointAttributeDefinition {
        name: "EdgeOfFlightLine",
        datatype: PointAttributeDataType::Bool,
    };

    /// Attribute definition for a classification. Default datatype is U8
    pub const CLASSIFICATION: PointAttributeDefinition = PointAttributeDefinition {
        name: "Classification",
        datatype: PointAttributeDataType::U8,
    };

    /// Attribute definition for a scan angle rank. Default datatype is I8
    pub const SCAN_ANGLE_RANK: PointAttributeDefinition = PointAttributeDefinition {
        name: "ScanAngleRank",
        datatype: PointAttributeDataType::I8,
    };

    /// Attribute definition for a user data field. Default datatype is U8
    pub const USER_DATA: PointAttributeDefinition = PointAttributeDefinition {
        name: "UserData",
        datatype: PointAttributeDataType::U8,
    };

    /// Attribute definition for a point source ID. Default datatype is U16
    pub const POINT_SOURCE_ID: PointAttributeDefinition = PointAttributeDefinition {
        name: "PointSourceID",
        datatype: PointAttributeDataType::U16,
    };

    /// Attribute definition for an RGB color. Default datatype is Vec3u16
    pub const COLOR_RGB: PointAttributeDefinition = PointAttributeDefinition {
        name: "ColorRGB",
        datatype: PointAttributeDataType::Vec3u16,
    };

    /// Attribute definition for a GPS timestamp. Default datatype is F64
    pub const GPS_TIME: PointAttributeDefinition = PointAttributeDefinition {
        name: "GpsTime",
        datatype: PointAttributeDataType::F64,
    };

    /// Attribute definition for near-infrared records (NIR). Default datatype is U16
    pub const NIR: PointAttributeDefinition = PointAttributeDefinition {
        name: "NIR",
        datatype: PointAttributeDataType::U16,
    };

    /// Attribute definition for a point ID. Default datatype is U64
    pub const POINT_ID: PointAttributeDefinition = PointAttributeDefinition {
        name: "PointID",
        datatype: PointAttributeDataType::U64,
    };

    /// Attribute definition for a 3D point normal. Default datatype is Vec3f32
    pub const NORMAL: PointAttributeDefinition = PointAttributeDefinition {
        name: "Normal",
        datatype: PointAttributeDataType::Vec3f32,
    };
}

/// Describes the layout of a single point in a point cloud
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PointLayout {
    attributes: Vec<PointAttributeDefinition>,
    attribute_offsets: Vec<u64>,
}

impl PointLayout {
    /// Creates a new empty PointLayout
    /// ```
    /// # use pasture_core::layout::*;
    /// let layout = PointLayout::new();
    /// # assert_eq!(0, layout.attributes().count());
    /// ```
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    /// Creates a new PointLayout from the given sequence of attributes. Panics if any two attributes within
    /// the sequence share the same attribute name.
    /// ```
    /// # use pasture_core::layout::*;
    /// let layout = PointLayout::from_attributes(&[attributes::POSITION_3D, attributes::INTENSITY]);
    /// # assert_eq!(2, layout.attributes().count());
    /// ```
    pub fn from_attributes(attributes: &[PointAttributeDefinition]) -> Self {
        let mut layout = Self::new();
        for attribute in attributes {
            layout.add_attribute(attribute.clone());
        }
        layout
    }

    /// Adds the given PointAttributeDefinition to this PointLayout. Panics if an attribute with the same
    /// name is already part of this PointLayout.
    /// ```
    /// # use pasture_core::layout::*;
    /// let mut layout = PointLayout::new();
    /// layout.add_attribute(attributes::POSITION_3D);
    /// # assert_eq!(1, layout.attributes().count());
    /// ```
    pub fn add_attribute(&mut self, point_attribute: PointAttributeDefinition) {
        if let Some(old_attribute) = self.get_attribute_by_name(point_attribute.name()) {
            panic!(
                "Point attribute {} is already present in this PointLayout!",
                old_attribute.name()
            );
        }

        // Store the offset to the start of the new attribute within this layout
        if self.attributes.is_empty() {
            self.attribute_offsets.push(0);
        } else {
            // If there are previous attributes, the offset to this attribute is equal to the offset
            // to the previous attribute plus the previous attribute's size
            self.attribute_offsets.push(
                self.attribute_offsets.last().unwrap() + self.attributes.last().unwrap().size(),
            );
        }

        self.attributes.push(point_attribute);
    }

    /// Returns true if an attribute with the given name is part of this PointLayout.
    /// ```
    /// # use pasture_core::layout::*;
    /// let mut layout = PointLayout::new();
    /// layout.add_attribute(attributes::POSITION_3D);
    /// assert!(layout.has_attribute(attributes::POSITION_3D.name()));
    /// ```
    pub fn has_attribute(&self, attribute_name: &str) -> bool {
        self.attributes
            .iter()
            .any(|attribute| attribute.name() == attribute_name)
    }

    /// Returns the attribute with the given name from this PointLayout. Returns None if no such attribute exists.
    /// ```
    /// # use pasture_core::layout::*;
    /// let mut layout = PointLayout::new();
    /// layout.add_attribute(attributes::POSITION_3D);
    /// let attribute = layout.get_attribute_by_name(attributes::POSITION_3D.name());
    /// # assert!(attribute.is_some());
    /// assert_eq!(attributes::POSITION_3D, *attribute.unwrap());
    /// ```
    pub fn get_attribute_by_name(&self, attribute_name: &str) -> Option<&PointAttributeDefinition> {
        self.attributes
            .iter()
            .find(|attribute| attribute.name() == attribute_name)
    }

    /// Returns an iterator over all attributes in this PointLayout.
    /// ```
    /// # use pasture_core::layout::*;
    /// let mut layout = PointLayout::new();
    /// layout.add_attribute(attributes::POSITION_3D);
    /// layout.add_attribute(attributes::INTENSITY);
    /// # let attributes = layout.attributes().collect::<Vec<_>>();
    /// # assert_eq!(2, attributes.len());
    /// # assert_eq!(attributes::POSITION_3D, *attributes[0]);
    /// # assert_eq!(attributes::INTENSITY, *attributes[1]);
    /// for attribute in layout.attributes() {
    ///    println!("{:?}", attribute);
    /// }
    /// ```
    pub fn attributes<'a>(&'a self) -> impl Iterator<Item = &'a PointAttributeDefinition> + 'a {
        self.attributes.iter()
    }

    /// Returns the size in bytes of a single point entry with the associated `PointLayout`.
    /// ```
    /// # use pasture_core::layout::*;
    /// let layout = PointLayout::from_attributes(&[attributes::POSITION_3D, attributes::INTENSITY]);
    /// let size_of_point = layout.size_of_point_entry();
    /// assert_eq!(26, size_of_point);
    /// ```
    pub fn size_of_point_entry(&self) -> u64 {
        self.attributes
            .iter()
            .fold(0, |acc, attribute| acc + attribute.size())
    }

    /// Returns the offset in bytes to the start of the given attribute within this layout. This assumes tight packing of
    /// the attributes. Returns `None` if the given attribute is not part of the associated `PointLayout`.
    ///
    /// ```
    /// # use pasture_core::layout::*;
    /// let layout = PointLayout::from_attributes(&[attributes::POSITION_3D, attributes::INTENSITY]);
    /// let intensity_offset = layout.offset_of(&attributes::INTENSITY).unwrap();
    /// assert_eq!(attributes::POSITION_3D.size(), intensity_offset);
    /// ```
    pub fn offset_of(&self, attribute: &PointAttributeDefinition) -> Option<u64> {
        self.index_of(attribute)
            .map(|idx| self.attribute_offsets[idx])
    }

    /// Returns the index of the given attribute within the associated `PointLayout`, or `None` if the attribute is not
    /// part of the `PointLayout`. The index depends on the order in which the attributes have been added to the associated
    /// `PointLayout`:
    /// ```
    /// # use pasture_core::layout::*;
    /// let layout = PointLayout::from_attributes(&[attributes::POSITION_3D, attributes::INTENSITY]);
    /// assert_eq!(Some(0), layout.index_of(&attributes::POSITION_3D));
    /// assert_eq!(Some(1), layout.index_of(&attributes::INTENSITY));
    /// assert_eq!(None, layout.index_of(&attributes::CLASSIFICATION));
    /// ```
    pub fn index_of(&self, attribute: &PointAttributeDefinition) -> Option<usize> {
        self.attributes
            .iter()
            .position(|this_attribute| this_attribute == attribute)
    }
}
