extern crate proc_macro;
//use anyhow::{anyhow, bail, Result};
use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::{
    parse_macro_input, spanned::Spanned, Attribute, Data, Error, Field, Ident, Lit, NestedMeta,
    Result, Type, TypePath,
};
use syn::{DataStruct, DeriveInput};

struct FieldLayoutDescription {
    pub field_name: String,
    pub attribute_name: String,
    pub primitive_type: quote::__private::TokenStream,
}

fn type_path_to_primitive_type(type_path: &TypePath) -> Result<quote::__private::TokenStream> {
    println!("Type: {:?}", type_path);

    let datatype_name = type_path.path.get_ident().map_or_else(
        || Err(Error::new_spanned(type_path, "Invalid type for a filed in a struct that implements the Pasture PointType trait! Only the Pasture primitive types are allowed!")),
        |ident| -> Result<quote::__private::TokenStream> {
            let type_name = ident.to_string();
            match type_name.as_str() {
                "u8" => Ok(quote! {U8}),
                "u16" => Ok(quote! {U16}),
                "u32" => Ok(quote! {U32}),
                "u64" => Ok(quote! {U64}),
                "i8" => Ok(quote! {I8}),
                "i16" => Ok(quote! {I16}),
                "i32" => Ok(quote! {I32}),
                "i64" => Ok(quote! {I64}),
                "f32" => Ok(quote! {F32}),
                "f64" => Ok(quote! {F64}),
                _ => Err(Error::new_spanned(ident, format!(
                    "Type {} is no valid Pasture primitive type!",
                    type_name)
                )),
            }
        },
    )?;

    let gen = quote! {
        pasture_core::layout::PointAttributeDataType::#datatype_name
    };
    Ok(gen)
}

fn get_attribute_name_from_field(field: &Field) -> Result<String> {
    if field.attrs.len() != 1 {
        return Err(Error::new_spanned(
            field,
            "derive(PointType) requires exactly one #[pasture] attribute per member!",
        ));
    }
    let pasture_attribute = &field.attrs[0];
    println!("pasture attribute tokens: {}", pasture_attribute.tokens);
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
                        "BUILTIN_POSITION_3D" => Ok("POSITION_3D".into()),
                        "BUILTIN_INTENSITY" => Ok("INTENSITY".into()),
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
        // syn::Meta::Path(path) => println!("Path arg: {:?}", path),
        // syn::Meta::NameValue(name_value) => println!("NameValue arg: {:?}", name_value),
    }

    //Ok("UNIMPLEMENTED".into())
}

fn get_field_layout_description(field: &Field) -> Result<FieldLayoutDescription> {
    match field.ty {
        Type::Path(ref type_path) => {
            let field_name = field
                .ident
                .as_ref()
                .map_or("_".into(), |ident| ident.to_string());
            let primitive_type = type_path_to_primitive_type(type_path)?;
            let attribute_name = get_attribute_name_from_field(field)?;

            // if primitive_type.is_none() {
            //     return Err(anyhow!("Field {} has invalid type. PointType struct can only have members that are of the primitive types defined by pasture-core!", field_name));
            // }

            Ok(FieldLayoutDescription {
                attribute_name,
                primitive_type,
                field_name,
            })
        }
        ref bad => Err(Error::new_spanned(
            bad,
            format!("Invalid type in PointType struct"),
        )),
    }
}

fn field_parameters(data: &Data, ident: &Ident) -> Result<Vec<FieldLayoutDescription>> {
    match data {
        Data::Struct(ref struct_data) => struct_data
            .fields
            .iter()
            .map(get_field_layout_description)
            .collect::<Result<Vec<_>>>(),
        _ => Err(Error::new_spanned(
            ident,
            format!("#[derive(PointType)] is only valid for structs"),
        )),
    }
}

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
        let err = quote_spanned! {
            input.span() => compile_error!("derive(PointType) is not valid for generic types!")
        };
        return proc_macro::TokenStream::from(err);
    }

    let name = &input.ident;

    let fields = field_parameters(&input.data, name);
    let gen = match fields {
        Err(why) => {
            why.to_compile_error()
            // let error_msg = why.to_string();
            // quote_spanned! {
            //     input.span() => compile_error!(#error_msg)
            // }
        }
        Ok(field_descriptions) => {
            let attribute_descriptions = field_descriptions
                .iter()
                .map(|desc| {
                    let attribute_name = &desc.attribute_name;
                    let primitive_type = &desc.primitive_type;
                    quote! {
                        pasture_core::layout::PointAttributeDefinition::custom(#attribute_name, #primitive_type)
                    }
                });
            quote! {
                impl pasture_core::layout::PointType for #name {
                    fn layout() -> pasture_core::layout::PointLayout {
                        pasture_core::layout::PointLayout::from_attributes(&[
                            #(#attribute_descriptions ,)*
                        ])
                    }
                }
            }
        }
    };

    // let gen2 = quote! {
    //     mod bla {}
    // };
    gen.into()
}

// #[proc_macro_attribute]
// pub fn point_attribute_custom(_attr: TokenStream, item: TokenStream) -> TokenStream {
//     println!("item: \"{}\"", item.to_string());
//     item
// }
