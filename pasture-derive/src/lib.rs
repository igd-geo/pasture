extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::spanned::Spanned;
use syn::{
    Attribute, Data, Error, Expr, Field, Fields, Ident, Index, Lit, Member, Path, Result, Type,
    parse_macro_input,
};
use syn::{DeriveInput, Meta};

fn get_pasture_attr(field: &Field) -> Result<Option<&Attribute>> {
    let pasture_attributes: Vec<&Attribute> = field
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("pasture"))
        .collect();
    if pasture_attributes.is_empty() {
        Ok(None)
    } else if pasture_attributes.len() == 1 {
        Ok(Some(pasture_attributes[0]))
    } else {
        Err(Error::new_spanned(
            field,
            "Not more than one #[pasture] attribute per member is allowed!",
        ))
    }
}

/// Checks, if the given path is a known builtin (BUILTIN_POSITION_3D, etc...)
/// and if so, returns the builtin attribute name.
fn builtin_attribute(path: &Path) -> Option<&'static str> {
    let ident = path.get_ident()?;
    let name = ident.to_string();

    match name.as_str() {
        "BUILTIN_POSITION_3D" => Some("Position3D"),
        "BUILTIN_INTENSITY" => Some("Intensity"),
        "BUILTIN_RETURN_NUMBER" => Some("ReturnNumber"),
        "BUILTIN_NUMBER_OF_RETURNS" => Some("NumberOfReturns"),
        "BUILTIN_CLASSIFICATION_FLAGS" => Some("ClassificationFlags"),
        "BUILTIN_SCANNER_CHANNEL" => Some("ScannerChannel"),
        "BUILTIN_SCAN_DIRECTION_FLAG" => Some("ScanDirectionFlag"),
        "BUILTIN_EDGE_OF_FLIGHT_LINE" => Some("EdgeOfFlightLine"),
        "BUILTIN_CLASSIFICATION" => Some("Classification"),
        "BUILTIN_SCAN_ANGLE_RANK" => Some("ScanAngleRank"),
        "BUILTIN_SCAN_ANGLE" => Some("ScanAngle"),
        "BUILTIN_USER_DATA" => Some("UserData"),
        "BUILTIN_POINT_SOURCE_ID" => Some("PointSourceID"),
        "BUILTIN_COLOR_RGB" => Some("ColorRGB"),
        "BUILTIN_GPS_TIME" => Some("GpsTime"),
        "BUILTIN_NIR" => Some("NIR"),
        "BUILTIN_WAVE_PACKET_DESCRIPTOR_INDEX" => Some("WavePacketDescriptorIndex"),
        "BUILTIN_WAVEFORM_DATA_OFFSET" => Some("WaveformDataOffset"),
        "BUILTIN_WAVEFORM_PACKET_SIZE" => Some("WaveformPacketSize"),
        "BUILTIN_RETURN_POINT_WAVEFORM_LOCATION" => Some("ReturnPointWaveformLocation"),
        "BUILTIN_WAVEFORM_PARAMETERS" => Some("WaveformParameters"),
        "BUILTIN_POINT_ID" => Some("PointID"),
        "BUILTIN_NORMAL" => Some("Normal"),
        _ => None,
    }
}

enum PointAttributeInfo {
    Ignore,
    FromField { name: String, ty: Type },
    FromConst { path: Path, ty: Type },
}

/// Describes a single field within a `PointType` struct. Contains the name of the field, the point attribute
/// that the field maps to, as well as the primitive type of the field
struct FieldLayoutDescription {
    pub ident: Member,
    pub point_attribute: PointAttributeInfo,
}

fn get_field_layout_descriptions(fields: &Fields) -> Result<Vec<FieldLayoutDescription>> {
    fields
        .iter()
        .enumerate()
        .map(|(idx, field)| {
            let pasture_attribute = get_pasture_attr(field)?;
            let ident = field.ident.clone().map_or_else(
                || {
                    Member::Unnamed(Index {
                        index: idx as u32,
                        span: field.span(),
                    })
                },
                Member::Named,
            );

            let malformed_field_error_msg = "#[pasture] attribute is malformed.";
            let point_attribute = match pasture_attribute {
                None => PointAttributeInfo::FromField {
                    name: field
                        .ident
                        .as_ref()
                        .map_or_else(|| format!("Attribute{idx}"), |i| i.to_string()),
                    ty: field.ty.clone(),
                },
                Some(attr) => match &attr.meta {
                    Meta::List(list) => {
                        let nested_meta = list.parse_args::<Meta>()?;
                        match nested_meta {
                            Meta::Path(path) => {
                                if path.is_ident("ignore") {
                                    PointAttributeInfo::Ignore
                                } else if let Some(builtin) = builtin_attribute(&path) {
                                    PointAttributeInfo::FromField {
                                        name: builtin.to_string(),
                                        ty: field.ty.clone(),
                                    }
                                } else {
                                    PointAttributeInfo::FromConst {
                                        path,
                                        ty: field.ty.clone(),
                                    }
                                }
                            }
                            Meta::NameValue(meta_name_value) => {
                                if meta_name_value.path.is_ident("rename")
                                    || meta_name_value.path.is_ident("attribute")
                                {
                                    PointAttributeInfo::FromField {
                                        name: match meta_name_value.value {
                                            Expr::Lit(lit) => match lit.lit {
                                                Lit::Str(string_literal) => string_literal.value(),
                                                bad => {
                                                    return Err(Error::new_spanned(
                                                        bad,
                                                        malformed_field_error_msg,
                                                    ));
                                                }
                                            },
                                            bad => {
                                                return Err(Error::new_spanned(
                                                    bad,
                                                    malformed_field_error_msg,
                                                ));
                                            }
                                        },
                                        ty: field.ty.clone(),
                                    }
                                } else {
                                    return Err(Error::new_spanned(
                                        meta_name_value,
                                        malformed_field_error_msg,
                                    ));
                                }
                            }
                            bad => return Err(Error::new_spanned(bad, malformed_field_error_msg)),
                        }
                    }
                    bad => return Err(Error::new_spanned(bad, malformed_field_error_msg)),
                },
            };

            Ok(FieldLayoutDescription {
                ident,
                point_attribute,
            })
        })
        .collect::<Result<Vec<FieldLayoutDescription>>>()
}

fn field_parameters(data: &Data, ident: &Ident) -> Result<Vec<FieldLayoutDescription>> {
    // TODO Make sure that structrs are #[repr(C)] - OR figure out the exact layout of the members in the struct. But #[repr(rust)] is allowed
    // to re-order the fields in the struct, which would (maybe?) break the Layout. Then again, if we correctly determine offsets and sizes of
    // fields, the order might not be important anymore?! It's really quite tricky to get this right and will need a lot of tests
    // We can use this maybe: https://doc.rust-lang.org/std/alloc/struct.Layout.html
    //
    //let member_layout = get_struct_member_layout(type_attributes, struct_data)?;

    match data {
        Data::Struct(struct_data) => get_field_layout_descriptions(&struct_data.fields),
        _ => Err(Error::new_spanned(
            ident,
            "#[derive(PointType)] is only valid for structs",
        )),
    }
}

/// Custom `derive` macro that implements the [`PointType`](pasture_core::layout::PointType) trait for the type that it is applied to.
///
/// Any that that wants to implement `PointType` using this `derive` macro must fulfill the following requirements:
/// - It must be `#[repr(C)]`
/// - It must implement [bytemuck::NoUninit] and [bytemuck::AnyBitPattern]
/// - All its members may only be [Pasture primitive types](pasture_core::layout::PointAttributeDataType)
///
/// The derived point layout will contain a point attribute for each struct member.
///
/// Members can be annotated with the `#[pasture]` attribute to control the name and type of the generated point attributes.
///
/// # Default behaviour
///
/// By default, each generated point attribute will have the same name and datatype as the struct member.
///
/// # Rename fields
///
/// Struct members can be annotated with: `#[pasture(rename = "newname")]`
///
/// The provided name will be used for the generated point attribute instead of the name of the struct member.
///
/// # Ignore fields
///
/// Struct members can be annotated with: `#[pasture(ignore)]`
///
/// No point attributes will be created for ignored members. This is usefull for inserting padding between point attributes.
///
/// # External point attributes
///
/// Struct members can be annotated with: `#[pasture(path::to::some::CUSTOM_POINT_ATTRIBUTE)]`
///
/// The path has to point to a constant [`PointAttributeDefinition`](pasture_core::layout::PointAttributeDefinition). (`const CUSTOM_POINT_ATTRIBUTE: PointAttributeDefinition = PointAttributeDefinition::custom(...)`)
///
/// The attribute will use the provided `PointAttributeDefinition` (both name and type).
///
/// The type of the struct member will be checked to match the [`datatype`](pasture_core::layout::PointAttributeDefinition::datatype) of the point attribute definition.
///
/// # Builtin attributes
///
/// To associate a member of a custom `PointType` with one of the builtin point attributes in Pasture, use the `#[pasture(X)]` attribute, where `X` is one of:
///
/// - `BUILTIN_POSITION_3D` corresponding to the [POSITION_3D](pasture_core::layout::attributes::POSITION_3D) attribute
/// - `BUILTIN_INTENSITY` corresponding to the [INTENSITY](pasture_core::layout::attributes::INTENSITY) attribute
/// - `BUILTIN_RETURN_NUMBER` corresponding to the [RETURN_NUMBER](pasture_core::layout::attributes::RETURN_NUMBER) attribute
/// - `BUILTIN_NUMBER_OF_RETURNS` corresponding to the [NUMBER_OF_RETURNS](pasture_core::layout::attributes::NUMBER_OF_RETURNS) attribute
/// - `BUILTIN_CLASSIFICATION_FLAGS` corresponding to the [CLASSIFICATION_FLAGS](pasture_core::layout::attributes::CLASSIFICATION_FLAGS) attribute
/// - `BUILTIN_SCANNER_CHANNEL` corresponding to the [SCANNER_CHANNEL](pasture_core::layout::attributes::SCANNER_CHANNEL) attribute
/// - `BUILTIN_SCAN_DIRECTION_FLAG` corresponding to the [SCAN_DIRECTION_FLAG](pasture_core::layout::attributes::SCAN_DIRECTION_FLAG) attribute
/// - `BUILTIN_EDGE_OF_FLIGHT_LINE` corresponding to the [EDGE_OF_FLIGHT_LINE](pasture_core::layout::attributes::EDGE_OF_FLIGHT_LINE) attribute
/// - `BUILTIN_CLASSIFICATION` corresponding to the [CLASSIFICATION](pasture_core::layout::attributes::CLASSIFICATION) attribute
/// - `BUILTIN_SCAN_ANGLE_RANK` corresponding to the [SCAN_ANGLE_RANK](pasture_core::layout::attributes::SCAN_ANGLE_RANK) attribute
/// - `BUILTIN_SCAN_ANGLE` corresponding to the [SCAN_ANGLE](pasture_core::layout::attributes::SCAN_ANGLE) attribute
/// - `BUILTIN_USER_DATA` corresponding to the [USER_DATA](pasture_core::layout::attributes::USER_DATA) attribute
/// - `BUILTIN_POINT_SOURCE_ID` corresponding to the [POINT_SOURCE_ID](pasture_core::layout::attributes::POINT_SOURCE_ID) attribute
/// - `BUILTIN_COLOR_RGB` corresponding to the [COLOR_RGB](pasture_core::layout::attributes::COLOR_RGB) attribute
/// - `BUILTIN_GPS_TIME` corresponding to the [GPS_TIME](pasture_core::layout::attributes::GPS_TIME) attribute
/// - `BUILTIN_NIR` corresponding to the [NIR](pasture_core::layout::attributes::NIR) attribute
/// - `BUILTIN_WAVE_PACKET_DESCRIPTOR_INDEX` corresponding to the [WAVE_PACKET_DESCRIPTOR_INDEX](pasture_core::layout::attributes::WAVE_PACKET_DESCRIPTOR_INDEX) attribute
/// - `BUILTIN_WAVEFORM_DATA_OFFSET` corresponding to the [WAVEFORM_DATA_OFFSET](pasture_core::layout::attributes::WAVEFORM_DATA_OFFSET) attribute
/// - `BUILTIN_WAVEFORM_PACKET_SIZE` corresponding to the [WAVEFORM_PACKET_SIZE](pasture_core::layout::attributes::WAVEFORM_PACKET_SIZE) attribute
/// - `BUILTIN_RETURN_POINT_WAVEFORM_LOCATION` corresponding to the [RETURN_POINT_WAVEFORM_LOCATION](pasture_core::layout::attributes::RETURN_POINT_WAVEFORM_LOCATION) attribute
/// - `BUILTIN_WAVEFORM_PARAMETERS` corresponding to the [WAVEFORM_PARAMETERS](pasture_core::layout::attributes::WAVEFORM_PARAMETERS) attribute
/// - `BUILTIN_POINT_ID` corresponding to the [POINT_ID](pasture_core::layout::attributes::POINT_ID) attribute
/// - `BUILTIN_NORMAL` corresponding to the [NORMAL](pasture_core::layout::attributes::NORMAL) attribute
///
/// Warning: Builtins only set the name of the point attribute. The datatype is always inferred from the
/// type of the field itself. To get type checked attributes, use the path to the attributes in pasture_core.
///
/// For example:
///
///  - `#[pasture(BUILTIN_POSITION_3D)]` will only set the name of the point attribute and infer its type from the attribute type. (Basically equivalent to `#[pasture(rename = "xxx")]` but with predefined attribute names.)
///  - `#[pasture(pasture_core::layout::attributes::POSITION_3D)]` will set the name of the point attribute and also check that the field type matches the point attribute datatype (VEC3F64).
///
/// # Example
///
/// ```
/// use pasture_derive::PointType;
/// use pasture_core::layout::{PointAttributeDefinition, PointAttributeDataType};
/// use std::borrow::Cow;
/// use nalgebra::Vector3;
///
/// const EXTRA_BYTES_8: PointAttributeDefinition = PointAttributeDefinition::custom(
///     Cow::Borrowed("ExtraBytes"),
///     PointAttributeDataType::byte_array(8),
/// );
///
/// #[repr(C)]
/// #[derive(PointType, Copy, Clone, bytemuck::NoUninit, bytemuck::AnyBitPattern)]
/// struct ExamplePoint {
///
///     #[pasture(BUILTIN_POSITION_3D)]
///     pub position: Vector3<f64>,
///
///     pub classification: u8,
///
///     #[pasture(ignore)]
///     pub _padding: [u8; 3],
///
///     #[pasture(rename = "ClassificationConfidence")]
///     pub confidence: f32,
///
///     #[pasture(EXTRA_BYTES_8)]
///     pub custom: [u8; 8],
/// }
/// ```
///
/// The point layout for the `ExamplePoint` struct will then have the following point attributes:
///
/// | Bytes  | Attribute name           | Attribute Type                        |
/// | ------ | ------------------------ | ------------------------------------- |
/// | 0..24  | Position3D               | PointAttributeDataType::VEC3F64       |
/// | 24..25 | classification           | PointAttributeDataType::U8            |
/// | 25..28 | -                        | -                                     |
/// | 28..32 | ClassificationConfidence | PointAttributeDataType::F32           |
/// | 32..40 | ExtraBytes               | PointAttributeDataType::byte_array(8) |
///
#[proc_macro_derive(PointType, attributes(pasture))]
pub fn derive_point_type(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);

    // What we want:
    //   - Ensure that the current type is a struct and not an enum
    //   - Get all members of the struct
    //   - Ensure that all members are one of the accepted primitive types that pasture-core defines
    //   - Ensure that each member has an appropriate attribute on it for the name of the attribute
    //   - Get the name, size and offset of each member, in order
    //   - Generate an impl PointType for the struct, where we build the layout using the types, names, sizes and offsets

    if !input.generics.params.is_empty() {
        return Error::new_spanned(input, "derive(PointType) is not valid for generic types")
            .to_compile_error()
            .into();
    }

    let struct_name = &input.ident;

    let fields = match field_parameters(&input.data, struct_name) {
        Ok(inner) => inner,
        Err(why) => {
            return why.to_compile_error().into();
        }
    };

    let attribute_descriptions = fields.iter().filter_map(|field| {
        let FieldLayoutDescription {
            ident,
            point_attribute,
        } = field;
        match point_attribute {
            PointAttributeInfo::Ignore => None,
            PointAttributeInfo::FromField { name, ty } => Some(quote! {
                pasture_core::layout::PointAttributeDefinition::custom(
                    std::borrow::Cow::Borrowed(#name),
                    <#ty as pasture_core::layout::PrimitiveType>::DATA_TYPE
                ).at_offset_in_type(offset_of!(#struct_name, #ident))
            }),
            PointAttributeInfo::FromConst { path, ty } => {
                let field = match ident {
                    Member::Named(ident) => ident.to_string(),
                    Member::Unnamed(index) => index.index.to_string(),
                };
                let error_message = format!(
                    "Field {struct_name}::{field} has wrong type for point attribute {}.",
                    quote! {#path}.to_string()
                ).replace('{', "{{").replace('}', "}}");

                Some(quote! {
                const {
                    let point_attr: pasture_core::layout::PointAttributeDefinition = #path;
                    let point_attr_datatype = pasture_core::layout::PointAttributeDefinition::datatype(&point_attr);
                    let struct_member_datatype = <#ty as pasture_core::layout::PrimitiveType>::DATA_TYPE;
                    if !point_attr_datatype.const_eq(&struct_member_datatype) {
                        ::core::panic!(#error_message); // panics in const contexts become compile errors.
                    }
                    point_attr
                }.at_offset_in_type(offset_of!(#struct_name, #ident))
            })},
        }
    });

    // let r#gen = quote! {
    //     impl pasture_core::layout::PointType for #name {
    //         fn layout() -> pasture_core::layout::PointLayout {
    //             pasture_core::layout::PointLayout::from_members_and_alignment(&[
    //                 #(#attribute_descriptions ,)*
    //             ], #type_alignment)
    //         }
    //     }
    // };

    let r#gen = quote! {
        impl pasture_core::layout::PointType for #struct_name {
            fn layout() -> pasture_core::layout::PointLayout {
                use core::mem::{align_of, offset_of};

                pasture_core::layout::PointLayout::from_members_and_alignment(&[
                    #(#attribute_descriptions ,)*
                ], align_of::<#struct_name>())
            }
        }
    };

    r#gen.into()
}
