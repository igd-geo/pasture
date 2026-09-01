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

use num_traits::AsPrimitive;
use std::{any::type_name, ops::Range};

use crate::layout::{
    PointAttributeDataType, PointAttributeDefinition, PointLayout, ScalarDataType,
};

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
        source_offset: usize,
        source_size: usize,
        target_offset: usize,
        target_size: usize,
    ) -> Self {
        Self {
            conversion_fn,
            source_range: Range {
                start: source_offset,
                end: source_offset + source_size,
            },
            target_range: Range {
                start: target_offset,
                end: target_offset + target_size,
            },
        }
    }

    /// Performs the conversion
    unsafe fn convert(&self, source_point: &[u8], target_point: &mut [u8]) {
        unsafe {
            let source_slice = &source_point[self.source_range.clone()];
            let target_slice = &mut target_point[self.target_range.clone()];

            (self.conversion_fn)(source_slice, target_slice);
        }
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
            .filter_map(|from_attribute| {
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
            .collect::<Vec<_>>();

        Self {
            attribute_converters: converters,
        }
    }

    /// Converts the `source_point` into the `target_point`
    ///
    /// # Safety
    ///
    /// `source_point` must contain memory for an initialized `PointType` `T` that has the exact same
    /// `PointLayout` as the one passed to [`Self::from_to`] as its first argument!
    pub unsafe fn convert(&self, source_point: &[u8], target_point: &mut [u8]) {
        unsafe {
            for converter in self.attribute_converters.iter() {
                converter.convert(source_point, target_point);
            }
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

/// Returns a generic converter that can convert between primitive types. These functions implement primitive type conversions
/// as if using the `as` operator, using the [`num_traits::AsPrimitive`] trait
pub fn get_generic_converter(
    from_type: PointAttributeDataType,
    to_type: PointAttributeDataType,
) -> Option<AttributeConversionFn> {
    if from_type.components != to_type.components {
        return None;
    }
    let components = from_type.components;

    if from_type == to_type {
        return Some(convert_unit);
    }

    macro_rules! conversion_match_components {
        ($t1:ty, $t2:ty) => {
            match components {
                1 => Some(convert_using_as::<$t1, $t2, 1>),
                2 => Some(convert_using_as::<$t1, $t2, 2>),
                3 => Some(convert_using_as::<$t1, $t2, 3>),
                4 => Some(convert_using_as::<$t1, $t2, 4>),
                _ => Some(convert_array_using_as::<$t1, $t2>),
            }
        };
    }
    macro_rules! conversion_match_to_type {
        ($t1:ty) => {
            match to_type.scalar {
                ScalarDataType::U8 => conversion_match_components!($t1, u8),
                ScalarDataType::I8 => conversion_match_components!($t1, i8),
                ScalarDataType::U16 => conversion_match_components!($t1, u16),
                ScalarDataType::I16 => conversion_match_components!($t1, i16),
                ScalarDataType::U32 => conversion_match_components!($t1, u32),
                ScalarDataType::I32 => conversion_match_components!($t1, i32),
                ScalarDataType::U64 => conversion_match_components!($t1, u64),
                ScalarDataType::I64 => conversion_match_components!($t1, i64),
                ScalarDataType::F32 => conversion_match_components!($t1, f32),
                ScalarDataType::F64 => conversion_match_components!($t1, f64),
                ScalarDataType::Custom { .. } => None,
            }
        };
    }
    match from_type.scalar {
        ScalarDataType::U8 => conversion_match_to_type!(u8),
        ScalarDataType::I8 => conversion_match_to_type!(i8),
        ScalarDataType::U16 => conversion_match_to_type!(u16),
        ScalarDataType::I16 => conversion_match_to_type!(i16),
        ScalarDataType::U32 => conversion_match_to_type!(u32),
        ScalarDataType::I32 => conversion_match_to_type!(i32),
        ScalarDataType::U64 => conversion_match_to_type!(u64),
        ScalarDataType::I64 => conversion_match_to_type!(i64),
        ScalarDataType::F32 => conversion_match_to_type!(f32),
        ScalarDataType::F64 => conversion_match_to_type!(f64),
        ScalarDataType::Custom { .. } => None,
    }
}

/// Unit conversion function (when from and to represent the same datatype)
///
/// # Safety
///
/// Even though this function only performs a `memcpy`, it is only valid to call it if both
/// `from` and `to` point to the same `PrimitiveType` (so `from` represents a `&T` and `to`
/// a `&mut T`)
///
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
pub unsafe fn convert_unit(from: &[u8], to: &mut [u8]) {
    to.copy_from_slice(from)
}

/// Generic conversion function from scalar or vector values of type `From` to type `To`. Assumes that `From` and
/// `To` are primitive types so that the conversion can happen by using `as`. Boils down to `*to_value = from_value as To;`
/// where `from_value` comes from the bytes `from` interpreted as `From`, and `to_value` comes from the bytes
/// `to` interpreted as `To`.
///
/// # Safety
///
/// `from` and `to` can be unaligned, but must point to valid initialized memory of the types `From` and
/// `To`, respectively
unsafe fn convert_using_as<From, To, const C: usize>(from: &[u8], to: &mut [u8])
where
    From: AsPrimitive<To> + Copy,
    To: Copy + 'static,
{
    // Relying on compiler optimizations here to do loop unrolling etc for optimal versions for
    // scalars (C==1) and small vectors (mostly Vec3, C==3).
    // todo: measure performance to validate this claim.
    dbg!(type_name::<From>(), type_name::<To>(), C);
    for i in 0..C {
        unsafe {
            let from_ptr = (from.as_ptr() as *const From).add(i);
            let to_ptr = (to.as_mut_ptr() as *mut To).add(i);
            let from_value = from_ptr.read_unaligned();
            let to_value = from_value.as_();
            to_ptr.write_unaligned(to_value);
        }
    }
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
unsafe fn convert_array_using_as<From, To>(from: &[u8], to: &mut [u8])
where
    From: AsPrimitive<To> + Copy,
    To: Copy + 'static,
{
    unsafe {
        let mut from_ptr = from.as_ptr() as *const From;
        let mut to_ptr = to.as_mut_ptr() as *mut To;
        let from_end = from.as_ptr().add(from.len()) as *const From;
        let to_end = to.as_ptr().add(to.len()) as *mut To;
        while from_ptr < from_end && to_ptr < to_end {
            let from_value = from_ptr.read_unaligned();
            let to_value = from_value.as_();
            to_ptr.write_unaligned(to_value);
            from_ptr = from_ptr.add(1);
            to_ptr = to_ptr.add(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::slice;

    use bytemuck::{Pod, Zeroable};
    use nalgebra::Vector3;
    use rand::RngExt;
    use uuid::Uuid;

    use crate::layout::{
        PointAttributeDataType, PrimitiveType, ScalarDataType, conversion::get_generic_converter,
    };

    #[test]
    fn test_same_type() {
        fn test_case<T>(test_value: T)
        where
            T: PrimitiveType + Default + PartialEq + std::fmt::Debug,
        {
            let converter_function = get_generic_converter(T::DATA_TYPE, T::DATA_TYPE).unwrap();
            let mut result_value: T = T::default();
            let src = bytemuck::cast_slice::<T, u8>(slice::from_ref(&test_value));
            let dst = bytemuck::cast_slice_mut(slice::from_mut(&mut result_value));
            unsafe {
                converter_function(src, dst);
            }
            assert_eq!(
                test_value,
                result_value,
                "Test failed for type {}",
                T::DATA_TYPE
            );
        }

        #[repr(transparent)]
        #[derive(Debug, Default, Eq, PartialEq, Copy, Clone, Pod, Zeroable)]
        struct CustomType([u8; 16]);

        impl PrimitiveType for CustomType {
            const DATA_TYPE: PointAttributeDataType =
                PointAttributeDataType::scalar(ScalarDataType::Custom {
                    size: 16,
                    min_alignment: 1,
                    name: Uuid::from_bytes_le(*b"My custom type. "),
                });
        }

        let mut rng = rand::rng();
        test_case(rng.random::<u8>());
        test_case(rng.random::<u16>());
        test_case(rng.random::<u32>());
        test_case(rng.random::<u64>());
        test_case(rng.random::<i8>());
        test_case(rng.random::<i16>());
        test_case(rng.random::<i32>());
        test_case(rng.random::<i64>());
        test_case(rng.random::<f32>());
        test_case(rng.random::<f64>());
        test_case(Vector3::new(
            rng.random::<f64>(),
            rng.random::<f64>(),
            rng.random::<f64>(),
        ));
        test_case(rng.random::<[u8; 20]>());
        test_case(CustomType(rng.random()));
    }

    #[test]
    fn test_as_conversion() {
        let mut rng = rand::rng();
        macro_rules! test_case {
            ($t1:ty, $t2:ty, $c:tt) => {{
                let converter_function =
                    get_generic_converter(<[$t1; $c]>::DATA_TYPE, <[$t2; $c]>::DATA_TYPE).unwrap();

                let test_value: [$t1; $c] = rng.random();
                let mut result_value: [$t2; $c] = Default::default();
                let expected_result_value: [$t2; $c] = test_value.map(|x| x as $t2);
                let src = bytemuck::cast_slice::<$t1, u8>(&test_value);
                let dst = bytemuck::cast_slice_mut::<$t2, u8>(&mut result_value);
                unsafe {
                    converter_function(src, dst);
                }
                assert_eq!(
                    result_value,
                    expected_result_value,
                    "Test failed for conversion {} as {}. Input value was {:?}, converted value was {:?}, but expected value is {:?}.",
                    <[$t1; $c]>::DATA_TYPE,
                    <[$t2; $c]>::DATA_TYPE,
                    test_value,
                    result_value,
                    expected_result_value
                );
            }};
        }
        macro_rules! test_cases_target {
            ($t1:ty, $c:tt) => {
                test_case!($t1, u8, $c);
                test_case!($t1, u16, $c);
                test_case!($t1, u32, $c);
                test_case!($t1, u64, $c);
                test_case!($t1, i8, $c);
                test_case!($t1, i16, $c);
                test_case!($t1, i32, $c);
                test_case!($t1, i64, $c);
                test_case!($t1, f32, $c);
                test_case!($t1, f64, $c);
            };
        }
        macro_rules! test_cases_source {
            ($c:tt) => {
                test_cases_target!(u8, $c);
                test_cases_target!(u16, $c);
                test_cases_target!(u32, $c);
                test_cases_target!(u64, $c);
                test_cases_target!(i8, $c);
                test_cases_target!(i16, $c);
                test_cases_target!(i32, $c);
                test_cases_target!(i64, $c);
                test_cases_target!(f32, $c);
                test_cases_target!(f64, $c);
            };
        }

        test_cases_source!(1); // scalars
        test_cases_source!(2); // vec2
        test_cases_source!(3); // vec3
        test_cases_source!(4); // vec4
        test_cases_source!(10); // arbitrarily sized arrays (here: [T; 10])
    }
}
