extern crate proc_macro;
use std::collections::HashSet;

//use anyhow::{anyhow, bail, Result};
use layout::{get_struct_member_layout, StructMemberLayout};
use proc_macro::TokenStream;
use quote::quote;
use syn::DeriveInput;
use syn::{
    parse_macro_input, Attribute, Data, Error, Field, Fields, GenericArgument, Ident, Lit,
    NestedMeta, PathArguments, Result, Type, TypePath,
};

mod layout;

enum PasturePrimitiveType {
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
    Vec4u8,
}

impl PasturePrimitiveType {
    fn min_alignment(&self) -> u64 {
        match self {
            PasturePrimitiveType::U8 => 1,
            PasturePrimitiveType::I8 => 1,
            PasturePrimitiveType::U16 => 2,
            PasturePrimitiveType::I16 => 2,
            PasturePrimitiveType::U32 => 4,
            PasturePrimitiveType::I32 => 4,
            PasturePrimitiveType::U64 => 8,
            PasturePrimitiveType::I64 => 8,
            PasturePrimitiveType::F32 => 4,
            PasturePrimitiveType::F64 => 8,
            PasturePrimitiveType::Bool => 1,
            PasturePrimitiveType::Vec3u8 => 1,
            PasturePrimitiveType::Vec3u16 => 2,
            PasturePrimitiveType::Vec3f32 => 4,
            PasturePrimitiveType::Vec3f64 => 8,
            &PasturePrimitiveType::Vec4u8 => 1,
        }
    }

    fn size(&self) -> u64 {
        match self {
            PasturePrimitiveType::U8 => 1,
            PasturePrimitiveType::I8 => 1,
            PasturePrimitiveType::U16 => 2,
            PasturePrimitiveType::I16 => 2,
            PasturePrimitiveType::U32 => 4,
            PasturePrimitiveType::I32 => 4,
            PasturePrimitiveType::U64 => 8,
            PasturePrimitiveType::I64 => 8,
            PasturePrimitiveType::F32 => 4,
            PasturePrimitiveType::F64 => 8,
            PasturePrimitiveType::Bool => 1,
            PasturePrimitiveType::Vec3u8 => 3,
            PasturePrimitiveType::Vec3u16 => 6,
            PasturePrimitiveType::Vec3f32 => 12,
            PasturePrimitiveType::Vec3f64 => 24,
            &PasturePrimitiveType::Vec4u8 => 4,
        }
    }

    fn as_token_stream(&self) -> quote::__private::TokenStream {
        match self {
            PasturePrimitiveType::U8 => quote! {pasture_core::layout::PointAttributeDataType::U8},
            PasturePrimitiveType::I8 => quote! {pasture_core::layout::PointAttributeDataType::I8},
            PasturePrimitiveType::U16 => quote! {pasture_core::layout::PointAttributeDataType::U16},
            PasturePrimitiveType::I16 => quote! {pasture_core::layout::PointAttributeDataType::I16},
            PasturePrimitiveType::U32 => quote! {pasture_core::layout::PointAttributeDataType::U32},
            PasturePrimitiveType::I32 => quote! {pasture_core::layout::PointAttributeDataType::I32},
            PasturePrimitiveType::U64 => quote! {pasture_core::layout::PointAttributeDataType::U64},
            PasturePrimitiveType::I64 => quote! {pasture_core::layout::PointAttributeDataType::I64},
            PasturePrimitiveType::F32 => quote! {pasture_core::layout::PointAttributeDataType::F32},
            PasturePrimitiveType::F64 => quote! {pasture_core::layout::PointAttributeDataType::F64},
            PasturePrimitiveType::Bool => {
                quote! {pasture_core::layout::PointAttributeDataType::Bool}
            }
            PasturePrimitiveType::Vec3u8 => {
                quote! {pasture_core::layout::PointAttributeDataType::Vec3u8}
            }
            PasturePrimitiveType::Vec3u16 => {
                quote! {pasture_core::layout::PointAttributeDataType::Vec3u16}
            }
            PasturePrimitiveType::Vec3f32 => {
                quote! {pasture_core::layout::PointAttributeDataType::Vec3f32}
            }
            PasturePrimitiveType::Vec3f64 => {
                quote! {pasture_core::layout::PointAttributeDataType::Vec3f64}
            }
            PasturePrimitiveType::Vec4u8 => {
                quote! {pasture_core::layout::PointAttributeDataType::Vec4u8}
            }
        }
    }
}

fn get_primitive_type_for_ident_type(ident: &Ident) -> Result<PasturePrimitiveType> {
    let type_name = ident.to_string();
    match type_name.as_str() {
        "u8" => Ok(PasturePrimitiveType::U8),
        "u16" => Ok(PasturePrimitiveType::U16),
        "u32" => Ok(PasturePrimitiveType::U32),
        "u64" => Ok(PasturePrimitiveType::U64),
        "i8" => Ok(PasturePrimitiveType::I8),
        "i16" => Ok(PasturePrimitiveType::I16),
        "i32" => Ok(PasturePrimitiveType::I32),
        "i64" => Ok(PasturePrimitiveType::I64),
        "f32" => Ok(PasturePrimitiveType::F32),
        "f64" => Ok(PasturePrimitiveType::F64),
        "bool" => Ok(PasturePrimitiveType::Bool),
        _ => Err(Error::new_spanned(
            ident,
            format!("Type {} is no valid Pasture primitive type!", type_name),
        )),
    }
}

fn get_primitive_type_for_non_ident_type(type_path: &TypePath) -> Result<PasturePrimitiveType> {
    // Path should have an ident (Vector3, Vector4, ...), as well as one generic argument
    let valid_idents: HashSet<_> = ["Vector3", "Vector4"].iter().collect();

    let path_segment = type_path
        .path
        .segments
        .first()
        .ok_or_else(|| Error::new_spanned(&type_path.path, "Invalid type"))?;
    if !valid_idents.contains(&path_segment.ident.to_string().as_str()) {
        return Err(Error::new_spanned(&path_segment.ident, "Invalid type"));
    }

    let path_arg = match &path_segment.arguments {
        PathArguments::AngleBracketed(arg) => arg,
        _ => return Err(Error::new_spanned(&path_segment.arguments, "Invalid type")),
    };

    let first_generic_arg = path_arg
        .args
        .first()
        .ok_or_else(|| Error::new_spanned(path_arg, "Invalid type arguments"))?;

    let type_arg = match first_generic_arg {
        GenericArgument::Type(t) => t,
        _ => return Err(Error::new_spanned(first_generic_arg, "Invalid type")),
    };

    let type_path = match type_arg {
        Type::Path(p) => p,
        _ => return Err(Error::new_spanned(type_arg, "Invalid type")),
    };

    match type_path.path.get_ident() {
        Some(ident) => {
            // Not ALL primitive types are supported as generic arguments for Vector3
            let type_name = ident.to_string();
            match path_segment.ident.to_string().as_str() {
                "Vector3" => match type_name.as_str() {
                    "u8" => Ok(PasturePrimitiveType::Vec3u8),
                    "u16" => Ok(PasturePrimitiveType::Vec3u16),
                    "f32" => Ok(PasturePrimitiveType::Vec3f32),
                    "f64" => Ok(PasturePrimitiveType::Vec3f64),
                    _ => Err(Error::new_spanned(
                        ident,
                        format!("Vector3<{}> is no valid Pasture primitive type. Vector3 is supported, but only for generic argument(s) u8, u16, f32 or f64", type_name),
                    ))
                },
                "Vector4" => match type_name.as_str() {
                    "u8" => Ok(PasturePrimitiveType::Vec4u8),
                    _ => Err(Error::new_spanned(
                        ident,
                        format!("Vector4<{}> is no valid Pasture primitive type. Vector4 is supported, but only for generic argument(s) u8", type_name),
                    ))
                },
                _ => Err(Error::new_spanned(ident, format!("Invalid type"))),
            }
        }
        None => Err(Error::new_spanned(&type_path.path, "Invalid type")),
    }
}

fn type_path_to_primitive_type(type_path: &TypePath) -> Result<PasturePrimitiveType> {
    if type_path.qself.is_some() {
        return Err(Error::new_spanned(
            type_path,
            "Qualified types are illegal in a struct with #[derive(PointType)]",
        ));
    }

    let datatype = match type_path.path.get_ident() {
        Some(ident) => get_primitive_type_for_ident_type(ident),
        None => get_primitive_type_for_non_ident_type(type_path),
    }?;

    Ok(datatype)
    // let gen = quote! {
    //     pasture_core::layout::PointAttributeDataType::#datatype_name
    // };
    // Ok(gen)
}

fn get_attribute_name_from_field(field: &Field) -> Result<String> {
    if field.attrs.len() != 1 {
        return Err(Error::new_spanned(
            field,
            "derive(PointType) requires exactly one #[pasture] attribute per member!",
        ));
    }
    let pasture_attribute = &field.attrs[0];
    let meta = pasture_attribute.parse_meta()?;
    // TODO Better explanation of the builtin Pasture attributes in this error message!
    let malformed_field_error_msg = "#[pasture] attribute is malformed. Correct syntax is #[pasture(attribute = \"NAME\")] or #[pasture(BUILTIN_XXX)], where XXX matches any of the builtin attributes in Pasture.";

    // For now, we expect that 'meta' is a Meta::List containing a single entry
    // The entry should be a NameValue, corresponding to 'attribute = "NAME"', or a Path, corresponding to 'builtin_XXX', where XXX matches any of the basic
    // builtin attributes in Pasture (such as INTENSITY, POSITION_3D etc.)
    match &meta {
        syn::Meta::List(list) => {
            let first_list_entry = list
                .nested
                .first()
                .ok_or_else(|| Error::new_spanned(list, malformed_field_error_msg))?;
            let nested_meta = match first_list_entry {
                NestedMeta::Meta(nested_meta) => nested_meta,
                _ => return Err(Error::new_spanned(list, malformed_field_error_msg)),
            };

            match nested_meta {
                syn::Meta::Path(path) => {
                    let ident = path
                        .get_ident()
                        .ok_or_else(|| Error::new_spanned(path, malformed_field_error_msg))?;
                    let ident_as_str = ident.to_string();
                    match ident_as_str.as_str() {
                        "BUILTIN_POSITION_3D" => Ok("Position3D".into()),
                        "BUILTIN_INTENSITY" => Ok("Intensity".into()),
                        "BUILTIN_RETURN_NUMBER" => Ok("ReturnNumber".into()),
                        "BUILTIN_NUMBER_OF_RETURNS" => Ok("NumberOfReturns".into()),
                        "BUILTIN_CLASSIFICATION_FLAGS" => Ok("ClassificationFlags".into()),
                        "BUILTIN_SCANNER_CHANNEL" => Ok("ScannerChannel".into()),
                        "BUILTIN_SCAN_DIRECTION_FLAG" => Ok("ScanDirectionFlag".into()),
                        "BUILTIN_EDGE_OF_FLIGHT_LINE" => Ok("EdgeOfFlightLine".into()),
                        "BUILTIN_CLASSIFICATION" => Ok("Classification".into()),
                        "BUILTIN_SCAN_ANGLE_RANK" => Ok("ScanAngleRank".into()),
                        "BUILTIN_SCAN_ANGLE" => Ok("ScanAngle".into()),
                        "BUILTIN_USER_DATA" => Ok("UserData".into()),
                        "BUILTIN_POINT_SOURCE_ID" => Ok("PointSourceID".into()),
                        "BUILTIN_COLOR_RGB" => Ok("ColorRGB".into()),
                        "BUILTIN_GPS_TIME" => Ok("GpsTime".into()),
                        "BUILTIN_NIR" => Ok("NIR".into()),
                        "BUILTIN_WAVE_PACKET_DESCRIPTOR_INDEX" => {
                            Ok("WavePacketDescriptorIndex".into())
                        }
                        "BUILTIN_WAVEFORM_DATA_OFFSET" => Ok("WaveformDataOffset".into()),
                        "BUILTIN_WAVEFORM_PACKET_SIZE" => Ok("WaveformPacketSize".into()),
                        "BUILTIN_RETURN_POINT_WAVEFORM_LOCATION" => {
                            Ok("ReturnPointWaveformLocation".into())
                        }
                        "BUILTIN_WAVEFORM_PARAMETERS" => Ok("WaveformParameters".into()),
                        "BUILTIN_POINT_ID" => Ok("PointID".into()),
                        "BUILTIN_NORMAL" => Ok("Normal".into()),
                        // TODO Other attributes
                        _ => {
                            return Err(Error::new_spanned(
                                ident,
                                format!("Unrecognized attribute name {}", ident_as_str),
                            ))
                        }
                    }
                }
                syn::Meta::NameValue(name_value) => name_value
                    .path
                    .get_ident()
                    .and_then(|path| {
                        if path != "attribute" {
                            return None;
                        }

                        if let Lit::Str(ref attribute_name) = name_value.lit {
                            Some(attribute_name.value())
                        } else {
                            None
                        }
                    })
                    .ok_or_else(|| Error::new_spanned(name_value, malformed_field_error_msg)),
                bad => Err(Error::new_spanned(bad, malformed_field_error_msg)),
            }
        }
        bad => Err(Error::new_spanned(bad, malformed_field_error_msg)),
    }
}

/// Describes a single field within a `PointType` struct. Contains the name of the field, the point attribute
/// that the field maps to, as well as the primitive type of the field
struct FieldLayoutDescription {
    pub field_name: String,
    pub attribute_name: String,
    pub primitive_type: PasturePrimitiveType,
}

fn get_field_layout_descriptions(fields: &Fields) -> Result<Vec<FieldLayoutDescription>> {
    fields
        .iter()
        .map(|field| match field.ty {
            Type::Path(ref type_path) => {
                let field_name = field
                    .ident
                    .as_ref()
                    .map_or("_".into(), |ident| ident.to_string());
                let primitive_type = type_path_to_primitive_type(type_path)?;
                let attribute_name = get_attribute_name_from_field(field)?;

                Ok(FieldLayoutDescription {
                    field_name,
                    attribute_name,
                    primitive_type,
                })
            }
            ref bad => Err(Error::new_spanned(
                bad,
                format!("Invalid type in PointType struct"),
            )),
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
            format!("#[derive(PointType)] is only valid for structs"),
        )),
    }
}

fn calculate_offsets_and_alignment(
    fields: &[FieldLayoutDescription],
    data: &Data,
    ident: &Ident,
    type_attributes: &[Attribute],
) -> Result<(Vec<u64>, u64)> {
    let struct_data = match data {
        Data::Struct(struct_data) => struct_data,
        _ => {
            return Err(Error::new_spanned(
                ident,
                format!("#[derive(PointType)] is only valid for structs"),
            ))
        }
    };
    let struct_layout = get_struct_member_layout(type_attributes, struct_data)?;

    let mut current_offset = 0;
    let mut max_alignment = 1;
    let mut offsets = vec![];
    for field in fields {
        let min_alignment = match struct_layout {
            StructMemberLayout::C => field.primitive_type.min_alignment(),
            StructMemberLayout::Packed(max_alignment) => {
                std::cmp::min(max_alignment, field.primitive_type.min_alignment())
            }
        };
        max_alignment = std::cmp::max(min_alignment, max_alignment);

        let aligned_offset = ((current_offset + min_alignment - 1) / min_alignment) * min_alignment;
        offsets.push(aligned_offset);
        current_offset = aligned_offset + field.primitive_type.size();
    }

    Ok((offsets, max_alignment))
}

/// Custom `derive` macro that implements the [`PointType`](pasture_core::layout::PointType) trait for the type that it is applied to.
///
/// Any that that wants to implement `PointType` using this `derive` macro must fulfill the following requirements:
/// - It must be at least one of `#[repr(C)]` and `#[repr(packed)]`
/// - All its members may only be [Pasture primitive types](pasture_core::layout::PointAttributeDataType)
/// - Each member must contain an attribute `#[pasture(X)]`, where `X` is either one of the builtin attributes explained below, or `attribute = "name"` for a custom attribute named `name`
/// - No two members may share the same attribute name
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
/// # Custom attributes
///
/// To associate a member of a custom `PointType` with a point attribute with custom `name`, use the `#[pasture(attribute = "name")]` attribute
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
        // let err = quote_spanned! {
        //     input.generics.span() => compile_error!("derive(PointType) is not valid for generic types!")
        // };
        // return proc_macro::TokenStream::from(err);
    }

    let name = &input.ident;

    let fields = match field_parameters(&input.data, name) {
        Ok(inner) => inner,
        Err(why) => {
            return why.to_compile_error().into();
        }
    };
    let (offsets, type_alignment) =
        match calculate_offsets_and_alignment(&fields, &input.data, name, input.attrs.as_slice()) {
            Ok(inner) => inner,
            Err(why) => {
                return why.to_compile_error().into();
            }
        };

    let attribute_descriptions = fields.iter().zip(offsets.iter()).map(|(field, offset)| {
        let attribute_name = &field.attribute_name;
        let primitive_type = &field.primitive_type.as_token_stream();
        quote! {
            pasture_core::layout::PointAttributeDefinition::custom(#attribute_name, #primitive_type).at_offset_in_type(#offset)
        }
    });

    let gen = quote! {
        impl pasture_core::layout::PointType for #name {
            fn layout() -> pasture_core::layout::PointLayout {
                pasture_core::layout::PointLayout::from_members_and_alignment(&[
                    #(#attribute_descriptions ,)*
                ], #type_alignment)
            }
        }
    };

    gen.into()
}
