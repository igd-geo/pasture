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
use nalgebra::Vector3;
use num_traits::AsPrimitive;
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
        let source_slice = &source_point[self.source_range.clone()];
        let target_slice = &mut target_point[self.target_range.clone()];

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
                let conversion_fn = get_converter_for_attributes(
                    from_attribute.attribute_definition(),
                    to_attribute.attribute_definition(),
                );
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
    assert_eq!(from_attribute.name(), to_attribute.name());
    if from_attribute.datatype() == to_attribute.datatype() {
        return None;
    }

    get_generic_converter(from_attribute.datatype(), to_attribute.datatype())
}

macro_rules! insert_scalar_converter_using_as {
    ($prim_from:ident, $prim_to:ident, $type_from:ident, $type_to:ident, $map:expr) => {
        // Insert symmetric conversion function from<->to and assert that they are unique
        assert!(($map)
            .insert(
                (
                    PointAttributeDataType::$type_from,
                    PointAttributeDataType::$type_to,
                ),
                convert_scalar_using_as::<$prim_from, $prim_to>,
            )
            .is_none());
        assert!(($map)
            .insert(
                (
                    PointAttributeDataType::$type_to,
                    PointAttributeDataType::$type_from,
                ),
                convert_scalar_using_as::<$prim_to, $prim_from>,
            )
            .is_none());
    };
}

macro_rules! insert_vec3_converter_using_as {
    ($prim_from:ident, $prim_to:ident, $type_from:ident, $type_to:ident, $map:expr) => {
        // Insert symmetric conversion function from<->to and assert that they are unique
        assert!(($map)
            .insert(
                (
                    PointAttributeDataType::$type_from,
                    PointAttributeDataType::$type_to,
                ),
                convert_vec3_using_as::<$prim_from, $prim_to>,
            )
            .is_none());
        assert!(($map)
            .insert(
                (
                    PointAttributeDataType::$type_to,
                    PointAttributeDataType::$type_from,
                ),
                convert_vec3_using_as::<$prim_to, $prim_from>,
            )
            .is_none());
    };
}

/// Returns a generic converter that can convert between primitive types. Going from smaller to larger types is realized
/// through `.into()` calls, while going from larger to smaller types is done through coercions (using `as`) where possible
pub fn get_generic_converter(
    from_type: PointAttributeDataType,
    to_type: PointAttributeDataType,
) -> Option<AttributeConversionFn> {
    lazy_static! {
        static ref GENERIC_CONVERTERS: HashMap<(PointAttributeDataType, PointAttributeDataType), AttributeConversionFn> = {
            let mut converters = HashMap::<
                (PointAttributeDataType, PointAttributeDataType),
                AttributeConversionFn,
            >::new();
            insert_scalar_converter_using_as!(u8, u16, U8, U16, converters);
            insert_scalar_converter_using_as!(u8, u32, U8, U32, converters);
            insert_scalar_converter_using_as!(u8, u64, U8, U64, converters);
            insert_scalar_converter_using_as!(u8, i8, U8, I8, converters);
            insert_scalar_converter_using_as!(u8, i16, U8, I16, converters);
            insert_scalar_converter_using_as!(u8, i32, U8, I32, converters);
            insert_scalar_converter_using_as!(u8, i64, U8, I64, converters);
            insert_scalar_converter_using_as!(u8, f32, U8, F32, converters);
            insert_scalar_converter_using_as!(u8, f64, U8, F64, converters);

            insert_scalar_converter_using_as!(u16, u32, U16, U32, converters);
            insert_scalar_converter_using_as!(u16, u64, U16, U64, converters);
            insert_scalar_converter_using_as!(u16, i8, U16, I8, converters);
            insert_scalar_converter_using_as!(u16, i16, U16, I16, converters);
            insert_scalar_converter_using_as!(u16, i32, U16, I32, converters);
            insert_scalar_converter_using_as!(u16, i64, U16, I64, converters);
            insert_scalar_converter_using_as!(u16, f32, U16, F32, converters);
            insert_scalar_converter_using_as!(u16, f64, U16, F64, converters);

            insert_scalar_converter_using_as!(u32, u64, U32, U64, converters);
            insert_scalar_converter_using_as!(u32, i8, U32, I8, converters);
            insert_scalar_converter_using_as!(u32, i16, U32, I16, converters);
            insert_scalar_converter_using_as!(u32, i32, U32, I32, converters);
            insert_scalar_converter_using_as!(u32, i64, U32, I64, converters);
            insert_scalar_converter_using_as!(u32, f32, U32, F32, converters);
            insert_scalar_converter_using_as!(u32, f64, U32, F64, converters);

            insert_scalar_converter_using_as!(u64, i8, U64, I8, converters);
            insert_scalar_converter_using_as!(u64, i16, U64, I16, converters);
            insert_scalar_converter_using_as!(u64, i32, U64, I32, converters);
            insert_scalar_converter_using_as!(u64, i64, U64, I64, converters);
            insert_scalar_converter_using_as!(u64, f32, U64, F32, converters);
            insert_scalar_converter_using_as!(u64, f64, U64, F64, converters);

            insert_scalar_converter_using_as!(i8, i16, I8, I16, converters);
            insert_scalar_converter_using_as!(i8, i32, I8, I32, converters);
            insert_scalar_converter_using_as!(i8, i64, I8, I64, converters);
            insert_scalar_converter_using_as!(i8, f32, I8, F32, converters);
            insert_scalar_converter_using_as!(i8, f64, I8, F64, converters);

            insert_scalar_converter_using_as!(i16, i32, I16, I32, converters);
            insert_scalar_converter_using_as!(i16, i64, I16, I64, converters);
            insert_scalar_converter_using_as!(i16, f32, I16, F32, converters);
            insert_scalar_converter_using_as!(i16, f64, I16, F64, converters);

            insert_scalar_converter_using_as!(i32, i64, I32, I64, converters);
            insert_scalar_converter_using_as!(i32, f32, I32, F32, converters);
            insert_scalar_converter_using_as!(i32, f64, I32, F64, converters);

            insert_scalar_converter_using_as!(i64, f32, I64, F32, converters);
            insert_scalar_converter_using_as!(i64, f64, I64, F64, converters);

            insert_scalar_converter_using_as!(f32, f64, F32, F64, converters);

            insert_vec3_converter_using_as!(f32, f64, Vec3f32, Vec3f64, converters);

            insert_vec3_converter_using_as!(u8, u16, Vec3u8, Vec3u16, converters);
            insert_vec3_converter_using_as!(u8, i32, Vec3u8, Vec3i32, converters);
            insert_vec3_converter_using_as!(u8, f32, Vec3u8, Vec3f32, converters);
            insert_vec3_converter_using_as!(u8, f64, Vec3u8, Vec3f64, converters);

            insert_vec3_converter_using_as!(u16, i32, Vec3u16, Vec3i32, converters);
            insert_vec3_converter_using_as!(u16, f32, Vec3u16, Vec3f32, converters);
            insert_vec3_converter_using_as!(u16, f64, Vec3u16, Vec3f64, converters);

            insert_vec3_converter_using_as!(i32, f32, Vec3i32, Vec3f32, converters);
            insert_vec3_converter_using_as!(i32, f64, Vec3i32, Vec3f64, converters);

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

/// Generic conversion function from scalar values of type `From` to type `To`. Assumes that `From` and
/// `To` are primitive types so that the conversion can happen by using `as`. Boils down to `*to_value = from_value as To;`
/// where `from_value` comes from the bytes `from` interpreted as `From`, and `to_value` comes from the bytes
/// `to` interpreted as `To`.
///
/// # Safety
///
/// `from` and `to` can be unaligned, but must point to valid initialized memory of the types `From` and
/// `To`, respectively
unsafe fn convert_scalar_using_as<From, To>(from: &[u8], to: &mut [u8])
where
    From: AsPrimitive<To> + Copy,
    To: Copy + 'static,
{
    let from_ptr = from.as_ptr() as *const From;
    let to_ptr = to.as_mut_ptr() as *mut To;

    let from_value = from_ptr.read_unaligned();
    let to_value = from_value.as_();
    to_ptr.write_unaligned(to_value);
}

/// Generic conversion function from a `Vector3<From>` into a `Vector3<To>`. Assumes that `From` and `To`
/// are primitive types so that the conversion can happen by using `as` for the components of the vector.
/// Boils down to `to_vector.x = from_vector.x as To;` etc. where `from_vector` comes from the bytes `from`
/// interpreted as `Vector3<From>` and `to_vector` comes from the bytes `to` interpreted as `Vector3<To>`.
///
/// # Safety
///
/// `from` and `to` can be unaligned, but must point to valid initialized memory of the type `Vector3<From>`
/// and `Vector3<To>`, respectively.
unsafe fn convert_vec3_using_as<From, To>(from: &[u8], to: &mut [u8])
where
    From: AsPrimitive<To> + Copy,
    To: Copy + 'static,
{
    let from_ptr = from.as_ptr() as *const Vector3<From>;
    let to_ptr = to.as_mut_ptr() as *mut Vector3<To>;

    let from_vec = from_ptr.read_unaligned();
    let to_vec = Vector3::<To>::new(from_vec[0].as_(), from_vec[1].as_(), from_vec[2].as_());
    to_ptr.write_unaligned(to_vec);
}
