use syn::{Attribute, DataStruct, Error};
use syn::{NestedMeta, Result};

#[derive(Debug)]
pub(crate) enum StructMemberLayout {
    Rust,
    C,
    Packed(u64),
}

#[derive(Debug)]
pub(crate) enum StructAlignment {
    Default,
    AlignTo(u64),
}

/// Is the given Attribute a #[repr(...)] attribute?
fn is_repr_attribute(attribute: &Attribute) -> bool {
    attribute
        .path
        .get_ident()
        .map(|path| path == "repr")
        .unwrap_or(false)
}

fn arg_is_c(arg: &NestedMeta) -> bool {
    match arg {
        NestedMeta::Meta(meta) => match meta {
            syn::Meta::Path(path) => path.get_ident().map(|ident| ident == "C").unwrap_or(false),
            _ => false,
        },
        _ => false,
    }
}

fn arg_is_packed(arg: &NestedMeta) -> bool {
    match arg {
        NestedMeta::Meta(meta) => match meta {
            syn::Meta::Path(path) => path
                .get_ident()
                .map(|ident| ident == "packed")
                .unwrap_or(false),
            syn::Meta::List(list) => list
                .path
                .get_ident()
                .map(|ident| ident == "packed")
                .unwrap_or(false),
            _ => false,
        },
        _ => false,
    }
}

fn get_packing_from_arg(arg: &NestedMeta) -> Result<u64> {
    match arg {
        NestedMeta::Meta(meta) => {
            match meta {
                syn::Meta::Path(_) => Ok(1),
                syn::Meta::List(list) => {
                    let packed_value = match list.nested.first() {
                    Some(f) => f,
                    None => return Err(Error::new_spanned(arg, "Expected #[repr(packed(N))]. packed List attribute has no nested members")),
                };

                    match packed_value {
                        NestedMeta::Meta(meta) => match meta {
                            syn::Meta::Path(path) => {
                                let val_str = path
                                    .get_ident()
                                    .map(|ident| ident.to_string())
                                    .ok_or_else(|| {
                                        Error::new_spanned(
                                            arg,
                                            "Expected #[repr(packed(N))]. Could not get ident",
                                        )
                                    })?;
                                val_str.parse::<u64>().map_err(|_| {
                                    Error::new_spanned(
                                        arg,
                                        "Expected #[repr(packed(N))]. Could not parse number",
                                    )
                                })
                            }
                            _ => Err(Error::new_spanned(
                                arg,
                                "Expected #[repr(packed(N))]. Meta is no Path",
                            )),
                        },
                        NestedMeta::Lit(literal) => match literal {
                            syn::Lit::Int(int_literal) => Ok(int_literal.base10_parse::<u64>()?),
                            _ => Err(Error::new_spanned(
                                arg,
                                "Expected #[repr(packed(N))], but argument N is no integer literal",
                            )),
                        },
                    }
                }
                _ => Err(Error::new_spanned(arg, "Expected #[repr(packed(N))]")),
            }
        }
        _ => Err(Error::new_spanned(arg, "Expected #[repr(packed(N))]")),
    }

    //todo!()
}

pub(crate) fn get_struct_member_layout(
    struct_attributes: &[Attribute],
    data_struct: &DataStruct,
) -> Result<StructMemberLayout> {
    // Do we have a #[repr(...)] attribute?
    let maybe_repr_attribute = struct_attributes.iter().find(|a| is_repr_attribute(a));
    if maybe_repr_attribute.is_none() {
        //return Ok(StructMemberLayout::Rust);
        return Err(Error::new_spanned(data_struct.struct_token, "derive(PointType) is only valid for structs that are either #[repr(C)] or #[repr(packed)]"));
    }

    let repr_attribute = maybe_repr_attribute.unwrap();
    let attribute_as_meta = repr_attribute.parse_meta()?;

    match &attribute_as_meta {
        syn::Meta::List(list) => {
            // There are several possible arguments for #[repr] on structs:
            // - C
            // - transparent (on single-value structs)
            // - packed
            // - packed(N)
            // - align(N)
            // Several combinations of those are possible! So first we look for either
            // the 'C' or 'packed'/'packed(N)' arguments

            let maybe_packed = list.nested.iter().find(|arg| arg_is_packed(arg));
            if let Some(packed_arg) = maybe_packed {
                let packing = get_packing_from_arg(packed_arg)?;
                return Ok(StructMemberLayout::Packed(packing));
            }

            let maybe_repr_c = list.nested.iter().find(|arg| arg_is_c(arg));
            if maybe_repr_c.is_some() {
                return Ok(StructMemberLayout::C);
            }

            //Ok(StructMemberLayout::Rust)
            Err(Error::new_spanned(
                repr_attribute,
                "Unrecongized repr attribute",
            ))
        }
        _ => Err(Error::new_spanned(
            repr_attribute,
            "Could not parse repr attribute",
        )),
    }
}
