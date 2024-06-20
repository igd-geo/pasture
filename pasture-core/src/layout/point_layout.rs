use std::{alloc::Layout, borrow::Cow, fmt::Display, iter::FromIterator, ops::Range};

use itertools::Itertools;
use nalgebra::{Vector3, Vector4};
use static_assertions::const_assert;
use uuid::Uuid;

use crate::math::Alignable;

mod private {
    use super::*;

    pub trait Sealed {}

    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for bool {}
    impl Sealed for Vector3<u8> {}
    impl Sealed for Vector3<u16> {}
    impl Sealed for Vector3<f32> {}
    impl Sealed for Vector3<f64> {}
    impl Sealed for Vector4<u8> {}
}

/// Possible data types for individual point attributes
///
/// # Why no `bool` anymore?
///
/// Previous versions of pasture allowed `bool` as a `PointAttributeDataType`. This introduced undefined behavior (UB)
/// into pasture since (contrary to C) there are only two valid bit patterns for `bool` in Rust (`0x0` and `0x1`). Hence,
/// casting from a byte slice to a type containing `bool` values will result in UB unless we guarantee that the bytes
/// are a valid bit pattern. In theory, these checks could be implemented, but they would have to be implemented everywhere
/// where we deal with raw point bytes. Since pasture switched to the `bytemuck` crate for all byte casting, and `bytemuck`
/// simply disallows slice-to-T casts for types that are not valid for any bit pattern, is was deemed that this is not worth
/// the effort. If you need `bool`-like behavior, you can always use an `u8` type and check for `value != 0`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PointAttributeDataType {
    /// An unsigned 8-bit integer value, corresponding to Rusts `u8` type
    U8,
    /// A signed 8-bit integer value, corresponding to Rusts `i8` type
    I8,
    /// An unsigned 16-bit integer value, corresponding to Rusts `u16` type
    U16,
    /// A signed 16-bit integer value, corresponding to Rusts `i16` type
    I16,
    /// An unsigned 32-bit integer value, corresponding to Rusts `u32` type
    U32,
    /// A signed 32-bit integer value, corresponding to Rusts `i32` type
    I32,
    /// An unsigned 64-bit integer value, corresponding to Rusts `u64` type
    U64,
    /// A signed 64-bit integer value, corresponding to Rusts `i64` type
    I64,
    /// A single-precision floating point value, corresponding to Rusts `f32` type
    F32,
    /// A double-precision floating point value, corresponding to Rusts `f64` type
    F64,
    /// A 3-component vector storing unsigned 8-bit integer values. Corresponding to the `Vector3<u8>` type of the [nalgebra crate](https://crates.io/crates/nalgebra)
    Vec3u8,
    /// A 3-component vector storing unsigned 16-bit integer values. Corresponding to the `Vector3<u16>` type of the [nalgebra crate](https://crates.io/crates/nalgebra)
    Vec3u16,
    /// A 3-component vector storing single-precision floating point values. Corresponding to the `Vector3<f32>` type of the [nalgebra crate](https://crates.io/crates/nalgebra)
    Vec3f32,
    /// A 3-component vector storing singed 32-bit integer values. Corresponding to the `Vector3<i32>` type of the [nalgebra crate](https://crates.io/crates/nalgebra)
    Vec3i32,
    /// A 3-component vector storing double-precision floating point values. Corresponding to the `Vector3<f32>` type of the [nalgebra crate](https://crates.io/crates/nalgebra)
    Vec3f64,
    /// A 4-component vector storing unsigned 8-bit integer values. Corresponding to the `Vector4<u8>` type of the [nalgebra crate](https://crates.io/crates/nalgebra)
    Vec4u8,
    /// A raw byte array of a given size determined at runtime. This corresponds to the Rust type `[u8; N]`
    ByteArray(u64),
    /// A custom data type. This makes pasture extensible to types that it does not know. To use a custom type `T` with
    /// pasture, implement the `PrimitiveType` trait for this type and have it return `PointAttributeDataType::Custom`
    /// with the correct size and alignment
    Custom {
        size: u64,
        min_alignment: u64,
        name: Uuid,
    }, //TODO REFACTOR Vector types should probably be Point3 instead, or at least use nalgebra::Point3 as their underlying type!
       //TODO Instead of representing each VecN<T> type as a separate literal, might it be possible to do: Vec3(PointAttributeDataType)?
       //Not in that way of course, because of recursive datastructures, but something like that?
}

impl PointAttributeDataType {
    /// Size of the associated `PointAttributeDataType`
    pub const fn size(&self) -> u64 {
        match self {
            PointAttributeDataType::U8 => 1,
            PointAttributeDataType::I8 => 1,
            PointAttributeDataType::U16 => 2,
            PointAttributeDataType::I16 => 2,
            PointAttributeDataType::U32 => 4,
            PointAttributeDataType::I32 => 4,
            PointAttributeDataType::U64 => 8,
            PointAttributeDataType::I64 => 8,
            PointAttributeDataType::F32 => 4,
            PointAttributeDataType::F64 => 8,
            PointAttributeDataType::Vec3u8 => 3,
            PointAttributeDataType::Vec3u16 => 6,
            PointAttributeDataType::Vec3i32 => 12,
            PointAttributeDataType::Vec3f32 => 12,
            PointAttributeDataType::Vec3f64 => 24,
            PointAttributeDataType::Vec4u8 => 4,
            PointAttributeDataType::ByteArray(length) => *length,
            PointAttributeDataType::Custom {
                size,
                min_alignment: _,
                name: _,
            } => *size,
        }
    }

    /// Minimum required alignment of the associated `PointAttributeDataType`
    pub fn min_alignment(&self) -> u64 {
        let align = match self {
            PointAttributeDataType::U8 => std::mem::align_of::<u8>(),
            PointAttributeDataType::I8 => std::mem::align_of::<i8>(),
            PointAttributeDataType::U16 => std::mem::align_of::<u16>(),
            PointAttributeDataType::I16 => std::mem::align_of::<i16>(),
            PointAttributeDataType::U32 => std::mem::align_of::<u32>(),
            PointAttributeDataType::I32 => std::mem::align_of::<i32>(),
            PointAttributeDataType::U64 => std::mem::align_of::<u64>(),
            PointAttributeDataType::I64 => std::mem::align_of::<i64>(),
            PointAttributeDataType::F32 => std::mem::align_of::<f32>(),
            PointAttributeDataType::F64 => std::mem::align_of::<f64>(),
            PointAttributeDataType::Vec3u8 => std::mem::align_of::<Vector3<u8>>(),
            PointAttributeDataType::Vec3u16 => std::mem::align_of::<Vector3<u16>>(),
            PointAttributeDataType::Vec3i32 => std::mem::align_of::<Vector3<i32>>(),
            PointAttributeDataType::Vec3f32 => std::mem::align_of::<Vector3<f32>>(),
            PointAttributeDataType::Vec3f64 => std::mem::align_of::<Vector3<f64>>(),
            PointAttributeDataType::Vec4u8 => std::mem::align_of::<Vector4<u8>>(),
            PointAttributeDataType::ByteArray(_) => 1,
            PointAttributeDataType::Custom {
                size: _,
                min_alignment,
                name: _,
            } => *min_alignment as usize,
        };
        align as u64
    }
}

impl Display for PointAttributeDataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PointAttributeDataType::U8 => write!(f, "U8"),
            PointAttributeDataType::I8 => write!(f, "I8"),
            PointAttributeDataType::U16 => write!(f, "U16"),
            PointAttributeDataType::I16 => write!(f, "I16"),
            PointAttributeDataType::U32 => write!(f, "U32"),
            PointAttributeDataType::I32 => write!(f, "I32"),
            PointAttributeDataType::U64 => write!(f, "U64"),
            PointAttributeDataType::I64 => write!(f, "I64"),
            PointAttributeDataType::F32 => write!(f, "F32"),
            PointAttributeDataType::F64 => write!(f, "F64"),
            PointAttributeDataType::Vec3u8 => write!(f, "Vec3<u8>"),
            PointAttributeDataType::Vec3u16 => write!(f, "Vec3<u16>"),
            PointAttributeDataType::Vec3i32 => write!(f, "Vec3<i32>"),
            PointAttributeDataType::Vec3f32 => write!(f, "Vec3<f32>"),
            PointAttributeDataType::Vec3f64 => write!(f, "Vec3<f64>"),
            PointAttributeDataType::Vec4u8 => write!(f, "Vec4<u8>"),
            PointAttributeDataType::ByteArray(length) => write!(f, "ByteArray[{length}]"),
            PointAttributeDataType::Custom {
                size: _,
                min_alignment: _,
                name,
            } => write!(f, "{name}"),
        }
    }
}

/// Marker trait for all types that can be used as primitive types within a `PointAttributeDefinition`. It provides a mapping
/// between Rust types and the `PointAttributeDataType` enum.
pub trait PrimitiveType: Copy + bytemuck::Pod {
    /// Returns the corresponding `PointAttributeDataType` for the implementing type
    fn data_type() -> PointAttributeDataType;
}

impl PrimitiveType for u8 {
    fn data_type() -> PointAttributeDataType {
        PointAttributeDataType::U8
    }
}
impl PrimitiveType for u16 {
    fn data_type() -> PointAttributeDataType {
        PointAttributeDataType::U16
    }
}
impl PrimitiveType for u32 {
    fn data_type() -> PointAttributeDataType {
        PointAttributeDataType::U32
    }
}
impl PrimitiveType for u64 {
    fn data_type() -> PointAttributeDataType {
        PointAttributeDataType::U64
    }
}
impl PrimitiveType for i8 {
    fn data_type() -> PointAttributeDataType {
        PointAttributeDataType::I8
    }
}
impl PrimitiveType for i16 {
    fn data_type() -> PointAttributeDataType {
        PointAttributeDataType::I16
    }
}
impl PrimitiveType for i32 {
    fn data_type() -> PointAttributeDataType {
        PointAttributeDataType::I32
    }
}
impl PrimitiveType for i64 {
    fn data_type() -> PointAttributeDataType {
        PointAttributeDataType::I64
    }
}
impl PrimitiveType for f32 {
    fn data_type() -> PointAttributeDataType {
        PointAttributeDataType::F32
    }
}
impl PrimitiveType for f64 {
    fn data_type() -> PointAttributeDataType {
        PointAttributeDataType::F64
    }
}
impl PrimitiveType for Vector3<u8> {
    fn data_type() -> PointAttributeDataType {
        PointAttributeDataType::Vec3u8
    }
}
impl PrimitiveType for Vector3<u16> {
    fn data_type() -> PointAttributeDataType {
        PointAttributeDataType::Vec3u16
    }
}
impl PrimitiveType for Vector3<i32> {
    fn data_type() -> PointAttributeDataType {
        PointAttributeDataType::Vec3i32
    }
}
impl PrimitiveType for Vector3<f32> {
    fn data_type() -> PointAttributeDataType {
        PointAttributeDataType::Vec3f32
    }
}
impl PrimitiveType for Vector3<f64> {
    fn data_type() -> PointAttributeDataType {
        PointAttributeDataType::Vec3f64
    }
}

impl PrimitiveType for Vector4<u8> {
    fn data_type() -> PointAttributeDataType {
        PointAttributeDataType::Vec4u8
    }
}

// Assert sizes of vector types are as we expect. Primitive types always are the same size, but we don't know
// what nalgebra does with the Vector3 types on the target machine...
const_assert!(std::mem::size_of::<Vector3<u8>>() == 3);
const_assert!(std::mem::size_of::<Vector3<u16>>() == 6);
const_assert!(std::mem::size_of::<Vector3<f32>>() == 12);
const_assert!(std::mem::size_of::<Vector3<f64>>() == 24);
const_assert!(std::mem::size_of::<Vector4<u8>>() == 4);

/// A definition for a single point attribute of a point cloud. Point attributes are things like the position,
/// GPS time, intensity etc. In Pasture, attributes are identified by a unique name together with the data type
/// that a single record of the attribute is stored in. Attributes can be grouped into two categories: Built-in
/// attributes (e.g. POSITION_3D, INTENSITY, GPS_TIME etc.) and custom attributes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PointAttributeDefinition {
    name: Cow<'static, str>,
    datatype: PointAttributeDataType,
}

impl PointAttributeDefinition {
    /// Creates a new custom PointAttributeDefinition with the given name and data type
    /// ```
    /// # use pasture_core::layout::*;
    /// # use std::borrow::Cow;
    /// let custom_attribute = PointAttributeDefinition::custom(Cow::Borrowed("Custom"), PointAttributeDataType::F32);
    /// # assert_eq!(custom_attribute.name(), "Custom");
    /// # assert_eq!(custom_attribute.datatype(), PointAttributeDataType::F32);
    /// ```
    pub const fn custom(name: Cow<'static, str>, datatype: PointAttributeDataType) -> Self {
        Self { name, datatype }
    }

    /// Returns the name of this PointAttributeDefinition
    /// ```
    /// # use pasture_core::layout::*;
    /// # use std::borrow::Cow;
    /// let custom_attribute = PointAttributeDefinition::custom(Cow::Borrowed("Custom"), PointAttributeDataType::F32);
    /// let name = custom_attribute.name();
    /// # assert_eq!(name, "Custom");
    /// ```
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the datatype of this PointAttributeDefinition
    /// ```
    /// # use pasture_core::layout::*;
    /// # use std::borrow::Cow;
    /// let custom_attribute = PointAttributeDefinition::custom(Cow::Borrowed("Custom"), PointAttributeDataType::F32);
    /// let datatype = custom_attribute.datatype();
    /// # assert_eq!(datatype, PointAttributeDataType::F32);
    /// ```
    #[inline]
    pub const fn datatype(&self) -> PointAttributeDataType {
        self.datatype
    }

    /// Returns the size in bytes of this attribute
    #[inline]
    pub const fn size(&self) -> u64 {
        self.datatype.size()
    }

    /// Returns a new PointAttributeDefinition based on this PointAttributeDefinition, but with a different datatype
    /// ```
    /// # use pasture_core::layout::*;
    /// let custom_position_attribute = attributes::POSITION_3D.with_custom_datatype(PointAttributeDataType::Vec3f32);
    /// # assert_eq!(custom_position_attribute.name(), attributes::POSITION_3D.name());
    /// # assert_eq!(custom_position_attribute.datatype(), PointAttributeDataType::Vec3f32);
    /// ```
    pub fn with_custom_datatype(&self, new_datatype: PointAttributeDataType) -> Self {
        Self {
            name: self.name.clone(),
            datatype: new_datatype,
        }
    }

    /// Creates a `PointAttributeMember` from the associated `PointAttributeDefinition` by specifying an offset
    /// of the attribute within a `PointType`. This turns an abstract `PointAttributeDefinition` into a concrete
    /// `PointAttributeMember`
    /// ```
    /// # use pasture_core::layout::*;
    /// let custom_position_attribute = attributes::POSITION_3D.at_offset_in_type(8);
    /// # assert_eq!(custom_position_attribute.name(), attributes::POSITION_3D.name());
    /// # assert_eq!(custom_position_attribute.datatype(), attributes::POSITION_3D.datatype());
    /// # assert_eq!(custom_position_attribute.offset(), 8);
    /// ```
    pub fn at_offset_in_type(&self, offset: u64) -> PointAttributeMember {
        PointAttributeMember {
            attribute_definition: self.clone(),
            offset,
            size: self.size(),
        }
    }
}

impl Display for PointAttributeDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{};{}]", self.name, self.datatype)
    }
}

/// A point attribute within a `PointType` structure. This is similar to a `PointAttributeDefinition`, but includes the
/// offset of the member within the structure
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PointAttributeMember {
    attribute_definition: PointAttributeDefinition,
    offset: u64,
    size: u64,
}

impl PointAttributeMember {
    /// Creates a new custom `PointAttributeMember` with the given name, datatype and byte offset
    /// ```
    /// # use pasture_core::layout::*;
    /// let custom_attribute = PointAttributeMember::custom("Custom", PointAttributeDataType::F32, 8);
    /// # assert_eq!(custom_attribute.name(), "Custom");
    /// # assert_eq!(custom_attribute.datatype(), PointAttributeDataType::F32);
    /// # assert_eq!(custom_attribute.offset(), 8);
    /// ```
    pub fn custom(name: &'static str, datatype: PointAttributeDataType, offset: u64) -> Self {
        Self {
            attribute_definition: PointAttributeDefinition {
                name: Cow::Borrowed(name),
                datatype,
            },
            offset,
            size: datatype.size(),
        }
    }

    /// Returns the name of the associated `PointAttributeMember`
    /// ```
    /// # use pasture_core::layout::*;
    /// let custom_attribute = PointAttributeMember::custom("Custom", PointAttributeDataType::F32, 8);
    /// let name = custom_attribute.name();
    /// # assert_eq!(name, "Custom");
    /// ```
    pub fn name(&self) -> &str {
        self.attribute_definition.name()
    }

    /// Returns the datatype of the associated `PointAttributeMember`
    /// ```
    /// # use pasture_core::layout::*;
    /// let custom_attribute = PointAttributeMember::custom("Custom", PointAttributeDataType::F32, 0);
    /// let datatype = custom_attribute.datatype();
    /// # assert_eq!(datatype, PointAttributeDataType::F32);
    /// ```
    #[inline]
    pub const fn datatype(&self) -> PointAttributeDataType {
        self.attribute_definition.datatype()
    }

    /// Returns the byte offset of the associated `PointAttributeMember`
    /// ```
    /// # use pasture_core::layout::*;
    /// let custom_attribute = PointAttributeMember::custom("Custom", PointAttributeDataType::F32, 8);
    /// let offset = custom_attribute.offset();
    /// # assert_eq!(offset, 8);
    /// ```
    #[inline]
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    /// Returns the underlying `PointAttributeDefinition` for the associated `PointAttributeMember`
    pub fn attribute_definition(&self) -> &PointAttributeDefinition {
        &self.attribute_definition
    }

    /// Returns the size in bytes of the associated `PointAttributeMember`
    #[inline]
    pub const fn size(&self) -> u64 {
        self.size
    }

    /// Returns the byte range within the `PointType` for this attribute
    pub fn byte_range_within_point(&self) -> Range<usize> {
        let start = self.offset as usize;
        let end = start + self.size() as usize;
        start..end
    }
}

impl Display for PointAttributeMember {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{};{} @ offset {}]",
            self.name(),
            self.datatype(),
            self.offset
        )
    }
}

// impl PartialEq for PointAttributeMember {
//     fn eq(&self, other: &Self) -> bool {
//         self.name == other.name && self.datatype == other.datatype
//     }
// }

// impl Eq for PointAttributeMember {}

/// Module containing default attribute definitions
pub mod attributes {
    use std::borrow::Cow;

    use super::{PointAttributeDataType, PointAttributeDefinition};

    /// Attribute definition for a 3D position. Default datatype is Vec3f64
    pub const POSITION_3D: PointAttributeDefinition = PointAttributeDefinition {
        name: Cow::Borrowed("Position3D"),
        datatype: PointAttributeDataType::Vec3f64,
    };

    /// Attribute definition for an intensity value. Default datatype is U16
    pub const INTENSITY: PointAttributeDefinition = PointAttributeDefinition {
        name: Cow::Borrowed("Intensity"),
        datatype: PointAttributeDataType::U16,
    };

    /// Attribute definition for a return number. Default datatype is U8
    pub const RETURN_NUMBER: PointAttributeDefinition = PointAttributeDefinition {
        name: Cow::Borrowed("ReturnNumber"),
        datatype: PointAttributeDataType::U8,
    };

    /// Attribute definition for the number of returns. Default datatype is U8
    pub const NUMBER_OF_RETURNS: PointAttributeDefinition = PointAttributeDefinition {
        name: Cow::Borrowed("NumberOfReturns"),
        datatype: PointAttributeDataType::U8,
    };

    /// Attribute definition for the classification flags. Default datatype is U8
    pub const CLASSIFICATION_FLAGS: PointAttributeDefinition = PointAttributeDefinition {
        name: Cow::Borrowed("ClassificationFlags"),
        datatype: PointAttributeDataType::U8,
    };

    /// Attribute definition for the scanner channel. Default datatype is U8
    pub const SCANNER_CHANNEL: PointAttributeDefinition = PointAttributeDefinition {
        name: Cow::Borrowed("ScannerChannel"),
        datatype: PointAttributeDataType::U8,
    };

    /// Attribute definition for a scan direction flag. Default datatype is U8
    pub const SCAN_DIRECTION_FLAG: PointAttributeDefinition = PointAttributeDefinition {
        name: Cow::Borrowed("ScanDirectionFlag"),
        datatype: PointAttributeDataType::U8,
    };

    /// Attribute definition for an edge of flight line flag. Default datatype is U8
    pub const EDGE_OF_FLIGHT_LINE: PointAttributeDefinition = PointAttributeDefinition {
        name: Cow::Borrowed("EdgeOfFlightLine"),
        datatype: PointAttributeDataType::U8,
    };

    /// Attribute definition for a classification. Default datatype is U8
    pub const CLASSIFICATION: PointAttributeDefinition = PointAttributeDefinition {
        name: Cow::Borrowed("Classification"),
        datatype: PointAttributeDataType::U8,
    };

    /// Attribute definition for a scan angle rank. Default datatype is I8
    pub const SCAN_ANGLE_RANK: PointAttributeDefinition = PointAttributeDefinition {
        name: Cow::Borrowed("ScanAngleRank"),
        datatype: PointAttributeDataType::I8,
    };

    /// Attribute definition for a scan angle with extended precision (like in LAS format 1.4). Default datatype is I16
    pub const SCAN_ANGLE: PointAttributeDefinition = PointAttributeDefinition {
        name: Cow::Borrowed("ScanAngle"),
        datatype: PointAttributeDataType::I16,
    };

    /// Attribute definition for a user data field. Default datatype is U8
    pub const USER_DATA: PointAttributeDefinition = PointAttributeDefinition {
        name: Cow::Borrowed("UserData"),
        datatype: PointAttributeDataType::U8,
    };

    /// Attribute definition for a point source ID. Default datatype is U16
    pub const POINT_SOURCE_ID: PointAttributeDefinition = PointAttributeDefinition {
        name: Cow::Borrowed("PointSourceID"),
        datatype: PointAttributeDataType::U16,
    };

    /// Attribute definition for an RGB color. Default datatype is Vec3u16
    pub const COLOR_RGB: PointAttributeDefinition = PointAttributeDefinition {
        name: Cow::Borrowed("ColorRGB"),
        datatype: PointAttributeDataType::Vec3u16,
    };

    /// Attribute definition for a GPS timestamp. Default datatype is F64
    pub const GPS_TIME: PointAttributeDefinition = PointAttributeDefinition {
        name: Cow::Borrowed("GpsTime"),
        datatype: PointAttributeDataType::F64,
    };

    /// Attribute definition for near-infrared records (NIR). Default datatype is U16
    /// TODO NIR semantically belongs to the color attributes, so there should be a separate
    /// attribute for 4-channel color that includes NIR!
    pub const NIR: PointAttributeDefinition = PointAttributeDefinition {
        name: Cow::Borrowed("NIR"),
        datatype: PointAttributeDataType::U16,
    };

    /// Attribute definition for the wave packet descriptor index in the LAS format. Default datatype is U8
    pub const WAVE_PACKET_DESCRIPTOR_INDEX: PointAttributeDefinition = PointAttributeDefinition {
        name: Cow::Borrowed("WavePacketDescriptorIndex"),
        datatype: PointAttributeDataType::U8,
    };

    /// Attribute definition for the offset to the waveform data in the LAS format. Default datatype is U64
    pub const WAVEFORM_DATA_OFFSET: PointAttributeDefinition = PointAttributeDefinition {
        name: Cow::Borrowed("WaveformDataOffset"),
        datatype: PointAttributeDataType::U64,
    };

    /// Attribute definition for the size of a waveform data packet in the LAS format. Default datatype is U32
    pub const WAVEFORM_PACKET_SIZE: PointAttributeDefinition = PointAttributeDefinition {
        name: Cow::Borrowed("WaveformPacketSize"),
        datatype: PointAttributeDataType::U32,
    };

    /// Attribute definition for the return point waveform location in the LAS format. Default datatype is F32
    pub const RETURN_POINT_WAVEFORM_LOCATION: PointAttributeDefinition = PointAttributeDefinition {
        name: Cow::Borrowed("ReturnPointWaveformLocation"),
        datatype: PointAttributeDataType::F32,
    };

    /// Attribute definition for the waveform parameters in the LAS format. Default datatype is Vector3<f32>
    pub const WAVEFORM_PARAMETERS: PointAttributeDefinition = PointAttributeDefinition {
        name: Cow::Borrowed("WaveformParameters"),
        datatype: PointAttributeDataType::Vec3f32,
    };

    /// Attribute definition for a point ID. Default datatype is U64
    pub const POINT_ID: PointAttributeDefinition = PointAttributeDefinition {
        name: Cow::Borrowed("PointID"),
        datatype: PointAttributeDataType::U64,
    };

    /// Attribute definition for a 3D point normal. Default datatype is Vec3f32
    pub const NORMAL: PointAttributeDefinition = PointAttributeDefinition {
        name: Cow::Borrowed("Normal"),
        datatype: PointAttributeDataType::Vec3f32,
    };
}

/// How is a field within the associated in-memory type of a `PointLayout` aligned?
pub enum FieldAlignment {
    /// Use alignment as if the type is [`#[repr(C)]`](https://doc.rust-lang.org/reference/type-layout.html#reprc-structs)
    Default,
    /// Use alignment as if the type is [`#[repr(packed(N))]`](https://doc.rust-lang.org/reference/type-layout.html#the-alignment-modifiers)
    Packed(u64),
}

/// Describes the data layout of a single point in a point cloud
///
/// # Detailed explanation
///
/// To understand `PointLayout`, it is necessary to understand the memory model of Pasture. Pasture is a library
/// for handling point cloud data, so the first thing worth understanding is what 'point cloud data' means in the context
/// of Pasture:
///
/// A point cloud in Pasture is modeled as a collection of tuples of attributes (a_1, a_2, ..., a_n). An
/// *attribute* can be any datum associated with a point, such as its position in 3D space, an intensity value, an object
/// classification etc. The set of all unique attributes in a point cloud make up the point clouds *point layout*, which
/// is represented in Pasture by the `PointLayout` type. The Pasture memory model describes how the attributes for each
/// point in a point cloud are layed out in memory. There are two major memory layouts supported by Pasture: *Interleaved*
/// and *PerAttribute*. In an *Interleaved* layout, all attributes for a single point are stored together in memory:
///
/// \[a_1(p_1), a_2(p_1), ..., a_n(p_1), a_1(p_2), a_2(p_2), ..., a_n(p_2), ...\]
///
/// This layout is equivalent to storing a type `T` inside a `Vec`, where `T` has members a_1, a_2, ..., a_n and is packed
/// tightly.
///
/// In a *PerAttribute* layout, all attributes of a single type are stored together in memory, often in separate memory regions:
///
/// \[a_1(p_1), a_1(p_2), ..., a_1(p_n)\]
/// \[a_2(p_1), a_2(p_2), ..., a_2(p_n)\]
/// ...
/// \[a_n(p_1), a_n(p_2), ..., a_n(p_n)\]
///
/// These layouts are sometimes called 'Array of Structs' (Interleaved) and 'Struct of Arrays' (PerAttribute).
///
/// Most code in Pasture supports point clouds with either of these memory layouts. To correctly handle memory layout and access
/// in both Interleaved and PerAttribute layout, each buffer in Pasture that stores point cloud data requires a piece of metadata
/// that describes the attributes of the point cloud with their [respective Rust types](PointAttributeDataType), their order, their memory alignment
/// and their potential offset within a single point entry in Interleaved format. All this information is stored inside the `PointLayout`
/// structure.
///
/// To support the different memory layouts, Pasture buffers store point data as raw binary buffers internally. To work with the data,
/// you will want to use strongly typed Rust structures. Any type `T` that you want to use for accessing point data in a strongly typed manner
/// must implement the `PointType` trait and thus provide Pasture with a way of figuring out the attributes and memory layout of this type `T`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PointLayout {
    attributes: Vec<PointAttributeMember>,
    #[cfg_attr(feature = "serde", serde(with = "serde_layout"))]
    memory_layout: Layout,
}

impl PointLayout {
    /// Creates a new PointLayout from the given sequence of attributes. The attributes will be aligned using the
    /// default alignments for their respective datatypes, in accordance with the [Rust alignment rules for `repr(C)` structs](https://doc.rust-lang.org/reference/type-layout.html#reprc-structs)
    ///
    /// #Panics
    ///
    /// If any two attributes within the sequence share the same attribute name.
    ///
    /// ```
    /// # use pasture_core::layout::*;
    /// let layout = PointLayout::from_attributes(&[attributes::POSITION_3D, attributes::INTENSITY]);
    /// # assert_eq!(2, layout.attributes().count());
    /// # assert_eq!(0, layout.at(0).offset());
    /// # assert_eq!(attributes::POSITION_3D.size(), layout.at(1).offset());
    /// ```
    pub fn from_attributes(attributes: &[PointAttributeDefinition]) -> Self {
        attributes.iter().cloned().collect()
    }

    /// Creates a new PointLayout from the given sequence of attributes. The attributes will be aligned to a 1 byte boundary
    /// in accordance with the [Rust alignment rules for `repr(packed)` structs](https://doc.rust-lang.org/reference/type-layout.html#the-alignment-modifiers)
    ///
    /// #Panics
    ///
    /// If any two attributes within the sequence share the same attribute name.
    ///
    /// ```
    /// # use pasture_core::layout::*;
    /// // Default INTENSITY attribute uses u16 datatype. In a packed(1) struct, the next field will have offset 2
    /// // even though the POSITION_3D attribute has an alignment requirement of 8
    /// let layout_packed_1 = PointLayout::from_attributes_packed(&[attributes::INTENSITY, attributes::POSITION_3D], 1);
    /// # assert_eq!(2, layout_packed_1.attributes().count());
    /// assert_eq!(0, layout_packed_1.at(0).offset());
    /// assert_eq!(2, layout_packed_1.at(1).offset());
    ///
    /// // If we use packed(4), POSITION_3D will start at byte 4:
    /// let layout_packed_4 = PointLayout::from_attributes_packed(&[attributes::INTENSITY, attributes::POSITION_3D], 4);
    /// assert_eq!(4, layout_packed_4.at(1).offset());
    /// ```
    pub fn from_attributes_packed(
        attributes: &[PointAttributeDefinition],
        max_alignment: u64,
    ) -> Self {
        let mut layout = Self::default();
        for attribute in attributes {
            layout.add_attribute(attribute.clone(), FieldAlignment::Packed(max_alignment));
        }
        layout
    }

    /// Creates a new PointLayout from the given `PointAttributeMember` sequence as well as the given `type_alignment`.
    ///
    /// #Panics
    ///
    /// If any two attributes within the sequence share the same attribute name, or if there is overlap between any two
    /// attributes based on their sizes and offsets.
    ///
    /// ```
    /// # use pasture_core::layout::*;
    /// let layout = PointLayout::from_members_and_alignment(&[attributes::INTENSITY.at_offset_in_type(2), attributes::POSITION_3D.at_offset_in_type(8)], 8);
    /// # assert_eq!(2, layout.attributes().count());
    /// assert_eq!(2, layout.at(0).offset());
    /// assert_eq!(8, layout.at(1).offset());
    /// assert_eq!(32, layout.size_of_point_entry());
    /// ```
    pub fn from_members_and_alignment(
        attributes: &[PointAttributeMember],
        type_alignment: u64,
    ) -> Self {
        // Conduct extensive checks for uniqueness and non-overlap. The checks are a bit expensive, however
        // they are absolutely necessary because this method is dangerous!
        let unique_names = attributes.iter().map(|a| a.name()).unique();
        if unique_names.count() != attributes.len() {
            panic!(
                "PointLayout::from_attributes_and_offsets: All attributes must have unique names!"
            );
        }

        let mut unaligned_ranges = attributes
            .iter()
            .map(|a| (a.offset()..(a.offset() + a.size())))
            .collect::<Vec<_>>();
        unaligned_ranges.sort_by(|a, b| a.start.cmp(&b.start));
        for next_idx in 1..unaligned_ranges.len() {
            let this_range = &unaligned_ranges[next_idx - 1];
            let next_range = &unaligned_ranges[next_idx];
            if this_range.end > next_range.start {
                panic!("PointLayout::from_attributes_and_offsets: All attributes must span non-overlapping memory regions!")
            }
        }

        let unaligned_size = attributes
            .iter()
            .max_by(|a, b| a.offset().cmp(&b.offset()))
            .map(|last_attribute| last_attribute.offset() + last_attribute.size())
            .unwrap_or(0);

        Self {
            attributes: attributes.to_vec(),
            memory_layout: Layout::from_size_align(
                unaligned_size.align_to(type_alignment) as usize,
                type_alignment as usize,
            )
            .expect("Could not create memory layout for PointLayout"),
        }
    }

    /// Adds the given PointAttributeDefinition to this PointLayout. Sets the offset of the new attribute
    /// within the `PointLayout` based on the given `FieldAlignment`
    ///
    /// #Panics
    ///
    /// If an attribute with the same name is already part of this PointLayout.
    /// ```
    /// # use pasture_core::layout::*;
    /// let mut layout = PointLayout::default();
    /// layout.add_attribute(attributes::INTENSITY, FieldAlignment::Default);
    /// layout.add_attribute(attributes::POSITION_3D, FieldAlignment::Default);
    /// # assert_eq!(2, layout.attributes().count());
    /// # assert_eq!(&attributes::INTENSITY.at_offset_in_type(0), layout.at(0));
    /// # assert_eq!(&attributes::POSITION_3D.at_offset_in_type(8), layout.at(1));
    /// // Default field alignment respects the 8-byte alignment requirement of default POSITION_3D (Vector3<f64>), even though default INTENSITY takes only 2 bytes
    /// assert_eq!(8, layout.at(1).offset());
    /// ```
    pub fn add_attribute(
        &mut self,
        point_attribute: PointAttributeDefinition,
        field_alignment: FieldAlignment,
    ) {
        if let Some(old_attribute) = self.get_attribute_by_name(point_attribute.name()) {
            panic!(
                "Point attribute {} is already present in this PointLayout!",
                old_attribute.name()
            );
        }

        let alignment_requirement_of_field = match field_alignment {
            FieldAlignment::Default => point_attribute.datatype().min_alignment(),
            FieldAlignment::Packed(max_alignment) => {
                std::cmp::min(max_alignment, point_attribute.datatype().min_alignment())
            }
        };
        let offset = self
            .packed_offset_of_next_field()
            .align_to(alignment_requirement_of_field);

        let current_max_alignment = self.memory_layout.align() as u64;
        let new_max_alignment = match field_alignment {
            FieldAlignment::Default => std::cmp::max(
                current_max_alignment,
                point_attribute.datatype().min_alignment(),
            ),
            FieldAlignment::Packed(max_alignment) => {
                std::cmp::min(max_alignment, current_max_alignment)
            }
        };

        self.attributes
            .push(point_attribute.at_offset_in_type(offset));

        let old_size = self.memory_layout.size() as u64;
        let attribute_end = offset + point_attribute.size();
        let new_size_unaligned = std::cmp::max(old_size, attribute_end);
        self.memory_layout = Layout::from_size_align(
            new_size_unaligned.align_to(new_max_alignment) as usize,
            new_max_alignment as usize,
        )
        .expect("Could not create memory layout for PointLayout");
    }

    /// Returns true if an attribute with the given name is part of this PointLayout.
    /// ```
    /// # use pasture_core::layout::*;
    /// let mut layout = PointLayout::default();
    /// layout.add_attribute(attributes::POSITION_3D, FieldAlignment::Default);
    /// assert!(layout.has_attribute_with_name(attributes::POSITION_3D.name()));
    /// ```
    pub fn has_attribute_with_name(&self, attribute_name: &str) -> bool {
        self.attributes
            .iter()
            .any(|attribute| attribute.name() == attribute_name)
    }

    /// Returns `true` if the associated `PointLayout` contains the given `attribute`. Both the name of `attribute` as well as
    /// its datatype must match for this method to return `true`. This is a more strict form of [`has_attribute_with_name`](Self::has_attribute_with_name)
    ///
    /// # Example
    /// ```
    /// # use pasture_core::layout::*;
    /// let mut layout = PointLayout::default();
    /// layout.add_attribute(attributes::POSITION_3D, FieldAlignment::Default);
    /// assert!(layout.has_attribute(&attributes::POSITION_3D));
    ///
    /// layout.add_attribute(attributes::INTENSITY.with_custom_datatype(PointAttributeDataType::U32), FieldAlignment::Default);
    /// assert!(!layout.has_attribute(&attributes::INTENSITY));
    /// ```
    pub fn has_attribute(&self, attribute: &PointAttributeDefinition) -> bool {
        self.attributes.iter().any(|this_attribute| {
            this_attribute.name() == attribute.name()
                && this_attribute.datatype() == attribute.datatype()
        })
    }

    /// Returns the attribute that matches the given `attribute` in name and datatype from the associated `PointLayout`. Returns `None` if
    /// no attribute with the same name and datatype exists
    /// ```
    /// # use pasture_core::layout::*;
    /// let mut layout = PointLayout::default();
    /// layout.add_attribute(attributes::POSITION_3D, FieldAlignment::Default);
    /// let attribute = layout.get_attribute(&attributes::POSITION_3D);
    /// assert!(attribute.is_some());
    /// let invalid_attribute = layout.get_attribute(&attributes::POSITION_3D.with_custom_datatype(PointAttributeDataType::U32));
    /// assert!(invalid_attribute.is_none());
    /// ```
    pub fn get_attribute(
        &self,
        attribute: &PointAttributeDefinition,
    ) -> Option<&PointAttributeMember> {
        self.attributes.iter().find(|self_attribute| {
            self_attribute.name() == attribute.name()
                && self_attribute.datatype() == attribute.datatype()
        })
    }

    /// Returns the attribute with the given name from this PointLayout. Returns None if no such attribute exists.
    /// ```
    /// # use pasture_core::layout::*;
    /// let mut layout = PointLayout::default();
    /// layout.add_attribute(attributes::POSITION_3D, FieldAlignment::Default);
    /// let attribute = layout.get_attribute_by_name(attributes::POSITION_3D.name());
    /// # assert!(attribute.is_some());
    /// assert_eq!(attributes::POSITION_3D.at_offset_in_type(0), *attribute.unwrap());
    /// ```
    pub fn get_attribute_by_name(&self, attribute_name: &str) -> Option<&PointAttributeMember> {
        self.attributes
            .iter()
            .find(|attribute| attribute.name() == attribute_name)
    }

    /// Returns the attribute at the given index from the associated `PointLayout`
    ///
    /// # Panics
    ///
    /// If `index` is out of bounds
    pub fn at(&self, index: usize) -> &PointAttributeMember {
        &self.attributes[index]
    }

    /// Returns an iterator over all attributes in this `PointLayout`. The attributes are returned in the order
    /// in which they were added to this `PointLayout`:
    /// ```
    /// # use pasture_core::layout::*;
    /// let mut layout = PointLayout::default();
    /// layout.add_attribute(attributes::POSITION_3D, FieldAlignment::Default);
    /// layout.add_attribute(attributes::INTENSITY, FieldAlignment::Default);
    /// let attributes = layout.attributes().collect::<Vec<_>>();
    /// # assert_eq!(2, attributes.len());
    /// assert_eq!(attributes::POSITION_3D.at_offset_in_type(0), *attributes[0]);
    /// assert_eq!(attributes::INTENSITY.at_offset_in_type(24), *attributes[1]);
    /// ```
    pub fn attributes(&self) -> impl Iterator<Item = &PointAttributeMember> {
        self.attributes.iter()
    }

    /// Returns the size in bytes of a single point entry with the associated `PointLayout`. Note that the size can be
    /// larger than the sum of the sizes of all attributes because of alignment requirements!
    ///
    /// # Example
    /// ```
    /// # use pasture_core::layout::*;
    /// let layout = PointLayout::from_attributes(&[attributes::POSITION_3D, attributes::INTENSITY]);
    /// // from_attributes respects the alignment requirements of each attribute. Default POSITION_3D uses Vector3<f64> and as such
    /// // has an 8-byte minimum alignment, so the whole PointLayout is aligned to an 8-byte boundary. This is reflected in its size:
    /// assert_eq!(32, layout.size_of_point_entry());
    /// ```
    #[inline]
    pub const fn size_of_point_entry(&self) -> u64 {
        self.memory_layout.size() as u64
    }

    /// Returns the index of the given attribute within the associated `PointLayout`, or `None` if the attribute is not
    /// part of the `PointLayout`. The index depends on the order in which the attributes have been added to the associated
    /// `PointLayout`, but does not necessarily reflect the order of the attributes in memory.
    ///
    /// # Example
    /// ```
    /// # use pasture_core::layout::*;
    /// let layout = PointLayout::from_attributes(&[attributes::POSITION_3D, attributes::INTENSITY]);
    /// assert_eq!(Some(0), layout.index_of(&attributes::POSITION_3D));
    /// assert_eq!(Some(1), layout.index_of(&attributes::INTENSITY));
    /// # assert_eq!(None, layout.index_of(&attributes::CLASSIFICATION));
    ///
    /// // Create a layout where we add INTENSITY as first attribute, however in memory, INTENSITY comes after POSITION_3D
    /// let reordered_layout = PointLayout::from_members_and_alignment(&[attributes::INTENSITY.at_offset_in_type(24), attributes::POSITION_3D.at_offset_in_type(0)], 8);
    /// assert_eq!(Some(0), reordered_layout.index_of(&attributes::INTENSITY));
    /// assert_eq!(Some(1), reordered_layout.index_of(&attributes::POSITION_3D));
    /// ```
    pub fn index_of(&self, attribute: &PointAttributeDefinition) -> Option<usize> {
        self.attributes.iter().position(|this_attribute| {
            this_attribute.name() == attribute.name()
                && this_attribute.datatype() == attribute.datatype()
        })
    }

    /// Compares the associated `PointLayout` with the `other` layout, ignoring the attribute offsets. This way, only the names and datatypes
    /// of the attributes are compared. This method is useful when dealing with data in a non-interleaved format, where offsets are irrelevant
    pub fn compare_without_offsets(&self, other: &PointLayout) -> bool {
        if self.attributes.len() != other.attributes.len() {
            return false;
        }

        self.attributes.iter().all(|self_attribute| {
            other
                .get_attribute_by_name(self_attribute.name())
                .map(|other_attribute| other_attribute.datatype() == self_attribute.datatype())
                .unwrap_or(false)
        })
    }

    /// Returns the offset from an attribute.
    /// If the attribute don't exist in the layout this function returns None.
    pub fn offset_of(&self, attribute: &PointAttributeDefinition) -> Option<u64> {
        self.attributes
            .iter()
            .find(|this_attribute| {
                this_attribute.name() == attribute.name()
                    && this_attribute.datatype() == attribute.datatype()
            })
            .map(|member| member.offset())
    }

    /// Returns the offset of the next field that could be added to this `PointLayout`, without any alignment
    /// requirements
    fn packed_offset_of_next_field(&self) -> u64 {
        if self.attributes.is_empty() {
            0
        } else {
            // If there are previous attributes, the offset to this attribute is equal to the offset
            // to the previous attribute plus the previous attribute's size
            let last_attribute = self.attributes.last().unwrap();
            last_attribute.offset() + last_attribute.size()
        }
    }
}

impl Display for PointLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "PointLayout {{")?;

        for attribute in self.attributes() {
            writeln!(f, "\t{}", attribute)?;
        }

        writeln!(f, "}}")
    }
}

impl Default for PointLayout {
    /// Creates a new empty PointLayout
    /// ```
    /// # use pasture_core::layout::*;
    /// let layout = PointLayout::default();
    /// # assert_eq!(0, layout.attributes().count());
    /// ```
    fn default() -> Self {
        Self {
            attributes: vec![],
            memory_layout: Layout::from_size_align(0, 1).unwrap(),
        }
    }
}

impl FromIterator<PointAttributeDefinition> for PointLayout {
    fn from_iter<T: IntoIterator<Item = PointAttributeDefinition>>(iter: T) -> Self {
        let mut layout = Self::default();
        for attribute in iter.into_iter() {
            layout.add_attribute(attribute.clone(), FieldAlignment::Default);
        }
        layout
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{
        attributes::{COLOR_RGB, INTENSITY, POSITION_3D},
        PointType,
    };
    use pasture_derive::PointType;

    #[derive(
        Debug, PointType, Copy, Clone, PartialEq, bytemuck::NoUninit, bytemuck::AnyBitPattern,
    )]
    #[repr(C, packed)]
    struct TestPoint1 {
        #[pasture(BUILTIN_POSITION_3D)]
        position: Vector3<f64>,
        #[pasture(BUILTIN_COLOR_RGB)]
        color: Vector3<u16>,
        #[pasture(BUILTIN_INTENSITY)]
        intensity: u16,
    }

    #[test]
    fn test_derive_point_type() {
        let expected_layout_1 = PointLayout::from_attributes_packed(
            &[
                POSITION_3D.with_custom_datatype(PointAttributeDataType::Vec3f64),
                COLOR_RGB.with_custom_datatype(PointAttributeDataType::Vec3u16),
                INTENSITY.with_custom_datatype(PointAttributeDataType::U16),
            ],
            1,
        );

        assert_eq!(expected_layout_1, TestPoint1::layout());
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_point_layout_serde() {
        let original_value = PointLayout {
            attributes: vec![],
            memory_layout: Layout::from_size_align(20, 4).unwrap(),
        };
        let serialized = serde_json::to_value(original_value.clone()).unwrap();
        let expected = json!({
            "attributes": [],
            "memory_layout": {
                "align": 4,
                "size": 20
            },
        });
        assert_eq!(serialized, expected);
        let deserialized: PointLayout = serde_json::from_value(serialized).unwrap();
        assert_eq!(deserialized, original_value);
    }
}

#[cfg(feature = "serde")]
mod serde_layout {
    use std::alloc::Layout;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    #[serde(rename = "Layout")]
    struct SizeAndAlignment {
        size: usize,
        align: usize,
    }

    pub fn serialize<S>(layout: &Layout, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SizeAndAlignment {
            size: layout.size(),
            align: layout.align(),
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, D>(de: D) -> Result<Layout, D::Error>
    where
        D: Deserializer<'de>,
    {
        let fields = SizeAndAlignment::deserialize(de)?;
        Layout::from_size_align(fields.size, fields.align).map_err(serde::de::Error::custom)
    }
}
