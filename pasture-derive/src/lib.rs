extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::spanned::Spanned;
use syn::{
    Attribute, Data, Error, Expr, ExprLit, Field, Fields, Ident, Index, Lit, Member, Result, Type,
    parse_macro_input,
};
use syn::{DeriveInput, Meta};

fn get_attribute_name_from_field(field: &Field) -> Result<String> {
    let pasture_attributes: Vec<&Attribute> = field
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("pasture"))
        .collect();
    if pasture_attributes.len() != 1 {
        return Err(Error::new_spanned(
            field,
            "derive(PointType) requires exactly one #[pasture] attribute per member!",
        ));
    }
    let pasture_attribute = pasture_attributes[0];

    let malformed_field_error_msg = "#[pasture] attribute is malformed. Correct syntax is #[pasture(attribute = \"NAME\")] or #[pasture(BUILTIN_XXX)], where XXX matches any of the builtin attributes in Pasture.";

    // TODO Better explanation of the builtin Pasture attributes in this error message!

    // For now, we expect that 'meta' is a Meta::List containing a single entry
    // The entry should be a NameValue, corresponding to 'attribute = "NAME"', or a Path, corresponding to 'builtin_XXX', where XXX matches any of the basic
    // builtin attributes in Pasture (such as INTENSITY, POSITION_3D etc.)
    match &pasture_attribute.meta {
        syn::Meta::List(list) => {
            let nested_meta = list.parse_args::<Meta>()?;
            match nested_meta {
                syn::Meta::Path(path) => {
                    let ident = path.get_ident().ok_or_else(|| {
                        Error::new_spanned(path.clone(), malformed_field_error_msg)
                    })?;
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
                        _ => Err(Error::new_spanned(
                            ident,
                            format!("Unrecognized attribute name {}", ident_as_str),
                        )),
                    }
                }
                syn::Meta::NameValue(name_value) => name_value
                    .path
                    .get_ident()
                    .and_then(|path| {
                        if path != "attribute" {
                            return None;
                        }

                        if let Expr::Lit(ExprLit {
                            lit: Lit::Str(attribute_name),
                            ..
                        }) = &name_value.value
                        {
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
    pub attribute_name: String,
    pub ty: Type,
    pub ident: Member,
}

fn get_field_layout_descriptions(fields: &Fields) -> Result<Vec<FieldLayoutDescription>> {
    fields
        .iter()
        .enumerate()
        .map(|(idx, field)| {
            let attribute_name = get_attribute_name_from_field(field)?;
            let ident = field.ident.clone().map_or_else(
                || {
                    Member::Unnamed(Index {
                        index: idx as u32,
                        span: field.span(),
                    })
                },
                Member::Named,
            );

            Ok(FieldLayoutDescription {
                attribute_name,
                ty: field.ty.clone(),
                ident,
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
    }

    let name = &input.ident;

    let fields = match field_parameters(&input.data, name) {
        Ok(inner) => inner,
        Err(why) => {
            return why.to_compile_error().into();
        }
    };

    let attribute_descriptions = fields.iter().map(|field| {
        let FieldLayoutDescription {
            attribute_name,
            ty,
            ident,
        } = field;
        quote! {
            pasture_core::layout::PointAttributeDefinition::custom(
                std::borrow::Cow::Borrowed(#attribute_name),
                <#ty as pasture_core::layout::PrimitiveType>::data_type()
            ).at_offset_in_type(offset_of!(#name, #ident))
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
        impl pasture_core::layout::PointType for #name {
            fn layout() -> pasture_core::layout::PointLayout {
                use core::mem::{align_of, offset_of};

                pasture_core::layout::PointLayout::from_members_and_alignment(&[
                    #(#attribute_descriptions ,)*
                ], align_of::<#name>())
            }
        }
    };

    r#gen.into()
}
