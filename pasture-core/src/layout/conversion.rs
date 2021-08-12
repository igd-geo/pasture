//! Contains helper function and structures for raw binary point format conversions. This module contains a lot of unsafe
//! code because it has to support conversions between various point formats at runtime. The conversions operate on binary
//! buffers (`&[u8]` and `&mut [u8]`) that represent the binary layout of strongly typed `PointTypes`. Given two point
//! types `A: PointType` and `B: PointType`, a runtime conversion from `A` to `B` works by first obtaining the binary
//! representations of `A` and `B` through `view_raw_bytes`/`view_raw_bytes_mut`:
//! ```ignore
//! let point_a : A = Default::default();
//! let point_b : B = Default::default();
//!
//! let buf_a = unsafe { view_raw_bytes(&a) };
//! let buf_b = unsafe { view_raw_bytes_mut(&mut b) };
//! ```
//! The conversion then operates on these two buffers. As this is a *highly* unsafe operation where all sorts of things
//! could go wrong, any conversion is only valid together with the *exact* `PointLayout` of both `A` and `B`!

use lazy_static::lazy_static;
use nalgebra::{Scalar, Vector3};
use std::{collections::HashMap, ops::Range};

use crate::layout::{PointAttributeDataType, PointAttributeDefinition, PointLayout};

/// Helper structure that contains the relevant data to convert a single attribute from a source binary
/// buffer to a target binary buffer.
struct RawAttributeConverter {
    conversion_fn: AttributeConversionFn,
    source_range: Range<usize>,
    target_range: Range<usize>,
}

impl RawAttributeConverter {
    pub fn new(
        conversion_fn: AttributeConversionFn,
        source_offset: u64,
        source_size: u64,
        target_offset: u64,
        target_size: u64,
    ) -> Self {
        Self {
            conversion_fn,
            source_range: Range {
                start: source_offset as usize,
                end: (source_offset + source_size) as usize,
            },
            target_range: Range {
                start: target_offset as usize,
                end: (target_offset + target_size) as usize,
            },
        }
    }

    /// Performs the conversion
    unsafe fn convert(&self, source_point: &[u8], target_point: &mut [u8]) {
        let source_slice = &source_point[self.source_range.start..self.source_range.end];
        let target_slice = &mut target_point[self.target_range.start..self.source_range.end];

        (self.conversion_fn)(source_slice, target_slice);
    }
}

/// Helper struct that encapsulates all `RawAttributeConverter`s necessary for converting a point in a specific layout
pub struct RawPointConverter {
    attribute_converters: Vec<RawAttributeConverter>,
}

impl RawPointConverter {
    /// Creates a new `RawPointConverter` that converts points `from_layout` to `to_layout`. The converter converts
    /// all attributes that are present in both `from_layout` and `to_layout` and which can be converted.
    pub fn from_to(from_layout: &PointLayout, to_layout: &PointLayout) -> RawPointConverter {
        let converters = from_layout
            .attributes()
            .filter(|&from_attribute| to_layout.has_attribute_with_name(from_attribute.name()))
            .map(|from_attribute| {
                let to_attribute = to_layout
                    .get_attribute_by_name(from_attribute.name())
                    .unwrap();
                let conversion_fn =
                    get_converter_for_attributes(&from_attribute.into(), &to_attribute.into());
                conversion_fn.map(|conversion_fn| {
                    RawAttributeConverter::new(
                        conversion_fn,
                        from_attribute.offset(),
                        from_attribute.size(),
                        to_attribute.offset(),
                        to_attribute.size(),
                    )
                })
            })
            .filter(|converter| converter.is_some())
            .map(|converter| converter.unwrap())
            .collect::<Vec<_>>();

        Self {
            attribute_converters: converters,
        }
    }

    /// Converts the `source_point` into the `target_point`
    pub unsafe fn convert(&self, source_point: &[u8], target_point: &mut [u8]) {
        for converter in self.attribute_converters.iter() {
            converter.convert(source_point, target_point);
        }
    }
}

/// Function pointer type for functions that convert between attributes with different datatypes
pub type AttributeConversionFn = unsafe fn(&[u8], &mut [u8]) -> ();

/// Returns a conversion function for converting from `from_attribute` into `to_attribute`. Both attributes must have the
/// same name but can have different datatypes. Conversion functions operate on raw byte buffers, where the first argument
/// is a buffer that represents a single value of `from_attribute` and the second buffer is a single mutable value of
/// `to_attribute`. If both attributes are equal, `None` is returned.
///
/// # Panics
///
/// If no conversion from `from_attribute` into `to_attribute` is possible
pub fn get_converter_for_attributes(
    from_attribute: &PointAttributeDefinition,
    to_attribute: &PointAttributeDefinition,
) -> Option<AttributeConversionFn> {
    if from_attribute.name() != to_attribute.name() {
        panic!("get_converter_for_attributes: from and to attributes must have the same name!");
    }
    if from_attribute.datatype() == to_attribute.datatype() {
        return None;
    }

    match from_attribute.name() {
        "Position3D" => get_position_converter(from_attribute.datatype(), to_attribute.datatype()),
        "ColorRGB" => get_color_rgb_converter(from_attribute.datatype(), to_attribute.datatype()),
        _ => get_generic_converter(from_attribute.datatype(), to_attribute.datatype()),
    }
}

fn get_position_converter(
    from_type: PointAttributeDataType,
    to_type: PointAttributeDataType,
) -> Option<AttributeConversionFn> {
    lazy_static! {
        static ref POSITION_CONVERTERS: HashMap<(PointAttributeDataType, PointAttributeDataType), AttributeConversionFn> = {
            let mut converters = HashMap::<
                (PointAttributeDataType, PointAttributeDataType),
                AttributeConversionFn,
            >::new();
            converters.insert(
                (
                    PointAttributeDataType::Vec3f64,
                    PointAttributeDataType::Vec3f32,
                ),
                convert_position_from_vec3f64_to_vec3f32,
            );
            converters.insert(
                (
                    PointAttributeDataType::Vec3f32,
                    PointAttributeDataType::Vec3f64,
                ),
                convert_position_from_vec3f32_to_vec3f64,
            );
            converters
        };
    }

    let key = (from_type, to_type);
    POSITION_CONVERTERS.get(&key).map(|&fptr| fptr)
}

fn get_color_rgb_converter(
    from_type: PointAttributeDataType,
    to_type: PointAttributeDataType,
) -> Option<AttributeConversionFn> {
    lazy_static! {
        static ref COLOR_RGB_CONVERTERS: HashMap<(PointAttributeDataType, PointAttributeDataType), AttributeConversionFn> = {
            let mut converters = HashMap::<
                (PointAttributeDataType, PointAttributeDataType),
                AttributeConversionFn,
            >::new();
            converters.insert(
                (
                    PointAttributeDataType::Vec3u16,
                    PointAttributeDataType::Vec3u8,
                ),
                convert_color_rgb_from_vec3u16_to_vec3u8,
            );
            converters.insert(
                (
                    PointAttributeDataType::Vec3u8,
                    PointAttributeDataType::Vec3u16,
                ),
                convert_color_rgb_from_vec3u8_to_vec3u16,
            );
            converters
        };
    }

    let key = (from_type, to_type);
    COLOR_RGB_CONVERTERS.get(&key).map(|&fptr| fptr)
}

macro_rules! insert_converter_using_into {
    ($prim_from:ident, $prim_to:ident, $type_from:ident, $type_to:ident, $map:expr) => {
        ($map).insert(
            (
                PointAttributeDataType::$type_from,
                PointAttributeDataType::$type_to,
            ),
            convert_using_into::<$prim_from, $prim_to>,
        )
    };
}

macro_rules! insert_converter_using_as {
    ($type_from:ident, $type_to:ident, $convert_fn:ident, $map:expr) => {
        ($map).insert(
            (
                PointAttributeDataType::$type_from,
                PointAttributeDataType::$type_to,
            ),
            $convert_fn,
        )
    };
}

/// Returns a generic converter that can convert between primitive types. Going from smaller to larger types is realized
/// through `.into()` calls, while going from larger to smaller types is done through coercions (using `as`) where possible
fn get_generic_converter(
    from_type: PointAttributeDataType,
    to_type: PointAttributeDataType,
) -> Option<AttributeConversionFn> {
    lazy_static! {
        static ref GENERIC_CONVERTERS: HashMap<(PointAttributeDataType, PointAttributeDataType), AttributeConversionFn> = {
            let mut converters = HashMap::<
                (PointAttributeDataType, PointAttributeDataType),
                AttributeConversionFn,
            >::new();
            insert_converter_using_into!(u8, u16, U8, U16, converters);
            insert_converter_using_into!(u8, u32, U8, U32, converters);
            insert_converter_using_into!(u8, u64, U8, U64, converters);
            insert_converter_using_into!(u16, u32, U16, U32, converters);
            insert_converter_using_into!(u16, u64, U16, U64, converters);
            insert_converter_using_into!(u32, u64, U32, U64, converters);

            insert_converter_using_into!(i8, i16, I8, I16, converters);
            insert_converter_using_into!(i8, i32, I8, I32, converters);
            insert_converter_using_into!(i8, i64, I8, I64, converters);
            insert_converter_using_into!(i16, i32, I16, I32, converters);
            insert_converter_using_into!(i16, i64, I16, I64, converters);
            insert_converter_using_into!(i32, i64, I32, I64, converters);

            insert_converter_using_as!(U16, U8, convert_u16_to_u8, converters);
            insert_converter_using_as!(U32, U8, convert_u32_to_u8, converters);
            insert_converter_using_as!(U64, U8, convert_u64_to_u8, converters);
            insert_converter_using_as!(U32, U16, convert_u32_to_u16, converters);
            insert_converter_using_as!(U64, U16, convert_u64_to_u16, converters);
            insert_converter_using_as!(U64, U32, convert_u64_to_u32, converters);

            insert_converter_using_as!(I16, I8, convert_i16_to_i8, converters);
            insert_converter_using_as!(I32, I8, convert_i32_to_i8, converters);
            insert_converter_using_as!(I64, I8, convert_i64_to_i8, converters);
            insert_converter_using_as!(I32, I16, convert_i32_to_i16, converters);
            insert_converter_using_as!(I64, I16, convert_i64_to_i16, converters);
            insert_converter_using_as!(I64, I32, convert_i64_to_i32, converters);

            insert_converter_using_as!(F64, F32, convert_f64_to_f32, converters);

            converters
        };
    }

    let key = (from_type, to_type);
    let f = GENERIC_CONVERTERS
        .get(&key)
        .unwrap_or_else(|| panic!("Invalid conversion {} -> {}", from_type, to_type));
    Some(*f)
}

/// Unit conversion function (when from and to represent the same datatype)
/// ```unsafe
/// # use nalgebra::Vector3;
/// # use pasture_core::layout::*;
/// # use pasture_core::util::*;
///
/// let source : Vector3<f64> = Vector3::new(1.0, 2.0, 3.0);
/// let mut dest : Vector3<f64> = Default::default();
///
/// let source_bytes = view_raw_bytes(&source);
/// let dest_bytes = view_raw_bytes_mut(&mut dest);
/// convert_unit(source_bytes, dest_bytes);
///
/// assert_eq!(1.0, dest.x);
/// assert_eq!(2.0, dest.y);
/// assert_eq!(3.0, dest.z);
/// ```
unsafe fn _convert_unit(from: &[u8], to: &mut [u8]) {
    to.copy_from_slice(from)
}

/// Unsafe conversion of a `Vector3<f64>` to a `Vector3<f32>` using their binary representations
/// ```unsafe
/// # use nalgebra::Vector3;
/// # use pasture_core::layout::*;
/// # use pasture_core::util::*;
///
/// let source : Vector3<f64> = Vector3::new(1.0, 2.0, 3.0);
/// let mut dest : Vector3<f32> = Default::default();
///
/// let source_bytes = view_raw_bytes(&source);
/// let dest_bytes = view_raw_bytes_mut(&mut dest);
/// convert_position_from_vec3f64_to_vec3f32(source_bytes, dest_bytes);
///
/// assert_eq!(1.0 as f32, dest.x);
/// assert_eq!(2.0 as f32, dest.y);
/// assert_eq!(3.0 as f32, dest.z);
/// ```
unsafe fn convert_position_from_vec3f64_to_vec3f32(from: &[u8], to: &mut [u8]) {
    let from_vec = &*(from.as_ptr() as *const Vector3<f64>);
    let to_vec = &mut *(to.as_mut_ptr() as *mut Vector3<f32>);

    to_vec.x = from_vec.x as f32;
    to_vec.y = from_vec.y as f32;
    to_vec.z = from_vec.z as f32;
}

/// Unsafe conversion of a `Vector3<f32>` to a `Vector3<f64>` using their binary representations
/// ```unsafe
/// # use nalgebra::Vector3;
/// # use pasture_core::layout::*;
/// # use pasture_core::util::*;
///
/// let source : Vector3<f32> = Vector3::new(1.0 as f32, 2.0 as f32, 3.0 as f32);
/// let mut dest : Vector3<f64> = Default::default();
///
/// let source_bytes = view_raw_bytes(&source);
/// let dest_bytes = view_raw_bytes_mut(&mut dest);
/// convert_position_from_vec3f32_to_vec3f64(source_bytes, dest_bytes);
///
/// assert_eq!(1.0, dest.x);
/// assert_eq!(2.0, dest.y);
/// assert_eq!(3.0, dest.z);
/// ```
unsafe fn convert_position_from_vec3f32_to_vec3f64(from: &[u8], to: &mut [u8]) {
    let from_vec = &*(from.as_ptr() as *const Vector3<f32>);
    let to_vec = &mut *(to.as_mut_ptr() as *mut Vector3<f64>);

    to_vec.x = from_vec.x as f64;
    to_vec.y = from_vec.y as f64;
    to_vec.z = from_vec.z as f64;
}

/// Unsafe conversion of a `Vector3<u16>` RGB color to a `Vector3<u8>` RGB color using their binary representations.
/// This conversion performs a bit shift instead of a truncation to reduce the dynamic range of the color.
/// ```unsafe
/// # use nalgebra::Vector3;
/// # use pasture_core::layout::*;
/// # use pasture_core::util::*;
///
/// let source : Vector3<u16> = Vector3::new(0xFFFF as u16, 0xF00F as u16, 0x8008 as u16);
/// let mut dest : Vector3<u8> = Default::default();
///
/// let source_bytes = view_raw_bytes(&source);
/// let dest_bytes = view_raw_bytes_mut(&mut dest);
/// convert_color_rgb_from_vec3u16_to_vec3u8(source_bytes, dest_bytes);
///
/// assert_eq!(0xFF as u8, dest.x);
/// assert_eq!(0xF0 as u8, dest.y);
/// assert_eq!(0x80 as u8, dest.z);
/// ```
unsafe fn convert_color_rgb_from_vec3u16_to_vec3u8(from: &[u8], to: &mut [u8]) {
    let from_vec = &*(from.as_ptr() as *const Vector3<u16>);
    let to_vec = &mut *(to.as_mut_ptr() as *mut Vector3<u8>);

    to_vec.x = (from_vec.x >> 8) as u8;
    to_vec.y = (from_vec.y >> 8) as u8;
    to_vec.z = (from_vec.z >> 8) as u8;
}

/// Unsafe conversion of a `Vector3<u8>` RGB color to a `Vector3<u16>` RGB color using their binary representations.
/// This conversion performs a bit shift instead of a truncation to increase the dynamic range of the color.
/// ```unsafe
/// # use nalgebra::Vector3;
/// # use pasture_core::layout::*;
/// # use pasture_core::util::*;
///
/// let source : Vector3<u8> = Vector3::new(0xFF as u8, 0xF0 as u8, 0x80 as u8);
/// let mut dest : Vector3<u16> = Default::default();
///
/// let source_bytes = view_raw_bytes(&source);
/// let dest_bytes = view_raw_bytes_mut(&mut dest);
/// convert_color_rgb_from_vec3u16_to_vec3u8(source_bytes, dest_bytes);
///
/// assert_eq!(0xFF00 as u16, dest.x);
/// assert_eq!(0xF000 as u16, dest.y);
/// assert_eq!(0x8000 as u16, dest.z);
/// ```
unsafe fn convert_color_rgb_from_vec3u8_to_vec3u16(from: &[u8], to: &mut [u8]) {
    let from_vec = &*(from.as_ptr() as *const Vector3<u8>);
    let to_vec = &mut *(to.as_mut_ptr() as *mut Vector3<u16>);

    to_vec.x = ((from_vec.x as u16) << 8) as u16;
    to_vec.y = ((from_vec.y as u16) << 8) as u16;
    to_vec.z = ((from_vec.z as u16) << 8) as u16;
}

unsafe fn _convert_generic_vec3<F, T>(from: &[u8], to: &mut [u8])
where
    F: Into<T> + Copy + Scalar,
    T: Copy + Scalar,
{
    let from_typed = &*(from.as_ptr() as *const Vector3<F>);
    let to_typed = &mut *(to.as_mut_ptr() as *mut Vector3<T>);

    to_typed.x = from_typed.x.into();
    to_typed.y = from_typed.y.into();
    to_typed.z = from_typed.z.into();
}

unsafe fn convert_using_into<F, T>(from: &[u8], to: &mut [u8])
where
    F: Into<T> + Copy,
    T: Copy,
{
    let from_typed = (from.as_ptr() as *const F).read_unaligned();
    (to.as_mut_ptr() as *mut T).write_unaligned(from_typed.into());
}

macro_rules! convert_using_as {
    ($type_from:ident, $type_to:ident, $name:ident) => {
        unsafe fn $name(from: &[u8], to: &mut [u8]) {
            let from_typed = (from.as_ptr() as *const $type_from).read_unaligned();
            (to.as_mut_ptr() as *mut $type_to).write_unaligned(from_typed as $type_to);
        }
    };
}

convert_using_as!(u16, u8, convert_u16_to_u8);
convert_using_as!(u32, u8, convert_u32_to_u8);
convert_using_as!(u64, u8, convert_u64_to_u8);
convert_using_as!(u32, u16, convert_u32_to_u16);
convert_using_as!(u64, u16, convert_u64_to_u16);
convert_using_as!(u64, u32, convert_u64_to_u32);

convert_using_as!(i16, i8, convert_i16_to_i8);
convert_using_as!(i32, i8, convert_i32_to_i8);
convert_using_as!(i64, i8, convert_i64_to_i8);
convert_using_as!(i32, i16, convert_i32_to_i16);
convert_using_as!(i64, i16, convert_i64_to_i16);
convert_using_as!(i64, i32, convert_i64_to_i32);

convert_using_as!(f64, f32, convert_f64_to_f32);
