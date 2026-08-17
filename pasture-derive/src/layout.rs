use syn::punctuated::Punctuated;
use syn::{Attribute, DataStruct, Error, LitInt, Token};
use syn::{Meta, Result};

#[derive(Debug)]
pub(crate) enum StructMemberLayout {
    C,
    Packed(u64),
}

/// Is the given Attribute a #[repr(...)] attribute?
fn is_repr_attribute(attribute: &Attribute) -> bool {
    attribute.path().is_ident("repr")
}

fn arg_is_c(arg: &Meta) -> bool {
    match arg {
        syn::Meta::Path(path) => path.is_ident("C"),
        _ => false,
    }
}

fn arg_is_packed(arg: &Meta) -> bool {
    match arg {
        syn::Meta::Path(path) => path.is_ident("packed"),
        syn::Meta::List(list) => list.path.is_ident("packed"),
        _ => false,
    }
}

fn get_packing_from_arg(arg: &Meta) -> Result<u64> {
    match arg {
        syn::Meta::Path(_) => Ok(1),
        syn::Meta::List(list) => list.parse_args::<LitInt>()?.base10_parse::<u64>(),
        _ => Err(Error::new_spanned(arg, "Expected #[repr(packed(N))]")),
    }
}

pub(crate) fn get_struct_member_layout(
    struct_attributes: &[Attribute],
    data_struct: &DataStruct,
) -> Result<StructMemberLayout> {
    // Do we have a #[repr(...)] attribute?
    let maybe_repr_attribute = struct_attributes.iter().find(|a| is_repr_attribute(a));
    let repr_attribute = maybe_repr_attribute.ok_or_else(|| Error::new_spanned(
            data_struct.struct_token,
            "derive(PointType) is only valid for structs that are either #[repr(C)] or #[repr(packed)]",
        ))?;

    match &repr_attribute.meta {
        syn::Meta::List(list) => {
            let nested = list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)?;

            // There are several possible arguments for #[repr] on structs:
            // - C
            // - transparent (on single-value structs)
            // - packed
            // - packed(N)
            // - align(N)
            // Several combinations of those are possible! So first we look for either
            // the 'C' or 'packed'/'packed(N)' arguments

            let maybe_packed = nested.iter().find(|arg| arg_is_packed(arg));
            if let Some(packed_arg) = maybe_packed {
                let packing = get_packing_from_arg(packed_arg)?;
                return Ok(StructMemberLayout::Packed(packing));
            }

            let maybe_repr_c = nested.iter().find(|arg| arg_is_c(arg));
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
