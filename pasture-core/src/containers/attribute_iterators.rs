use crate::containers::{PerAttributePointBuffer, PerAttributePointBufferMut, PointBuffer};
use crate::layout::conversion::{get_converter_for_attributes, AttributeConversionFn};
use crate::layout::PointAttributeDefinition;
use crate::layout::PrimitiveType;
use crate::util::view_raw_bytes_mut;

use std::marker::PhantomData;
use std::mem::MaybeUninit;

// The iterators for a single point attribute are implemented without macros, because we want them to return just T instead of a tuple (T)

/// Contains iterators over a single point attribute
pub mod attr1 {
    use super::*;

    /// Iterator over a `PointBuffer` that yields strongly typed data by value for a specific attribute for each point
    pub struct AttributeIteratorByValue<'a, T: PrimitiveType, B: PointBuffer + ?Sized> {
        buffer: &'a B,
        attribute: &'a PointAttributeDefinition,
        current_index: usize,
        buffer_length: usize,
        internal_buffer: Vec<T>,
        index_in_internal_buffer: usize,
        _unused: PhantomData<T>,
    }

    impl<'a, T: PrimitiveType, B: PointBuffer + ?Sized> AttributeIteratorByValue<'a, T, B> {
        const INTERNAL_BUFFER_SIZE: usize = 50_000;

        pub fn new(buffer: &'a B, attribute: &'a PointAttributeDefinition) -> Self {
            if attribute.datatype() != T::data_type() {
                panic!("Type T does not match datatype of attribute {}", attribute);
            }
            if !buffer.point_layout().has_attribute(attribute) {
                panic!(
                    "Attribute {} not contained in PointLayout of buffer ({})",
                    attribute,
                    buffer.point_layout()
                );
            }
            Self {
                buffer,
                attribute,
                current_index: 0,
                buffer_length: buffer.len(),
                internal_buffer: Vec::new(),
                index_in_internal_buffer: 0,
                _unused: Default::default(),
            }
        }

        fn refill_internal_buffer(&mut self) {
            let remaining_points = std::cmp::min(
                Self::INTERNAL_BUFFER_SIZE,
                self.buffer_length - self.current_index,
            );

            if remaining_points > self.internal_buffer.len() {
                //Grow the vector without initializing the elements. This works because T is a `PrimitiveType`, which is `Copy`
                self.internal_buffer.reserve(remaining_points);
                unsafe {
                    self.internal_buffer.set_len(remaining_points);
                }
            }

            let buffer_slice = &mut self.internal_buffer[0..remaining_points];
            let buffer_slice_untyped = unsafe {
                std::slice::from_raw_parts_mut(
                    buffer_slice.as_mut_ptr() as *mut u8,
                    remaining_points * std::mem::size_of::<T>(),
                )
            };

            self.buffer.get_raw_attribute_range(
                self.current_index..(self.current_index + remaining_points),
                self.attribute,
                buffer_slice_untyped,
            );
            self.index_in_internal_buffer = 0;
        }
    }

    impl<'a, T: PrimitiveType, B: PointBuffer + ?Sized> Iterator
        for AttributeIteratorByValue<'a, T, B>
    {
        type Item = T;

        fn next(&mut self) -> Option<Self::Item> {
            if self.current_index == self.buffer_length {
                return None;
            }

            if self.index_in_internal_buffer >= self.internal_buffer.len() {
                self.refill_internal_buffer();
            }

            let ret = self.internal_buffer[self.index_in_internal_buffer];

            self.index_in_internal_buffer += 1;
            self.current_index += 1;

            Some(ret)
        }
    }

    pub struct AttributeIteratorByValueWithConversion<'a, T: PrimitiveType, B: PointBuffer + ?Sized> {
        buffer: &'a B,
        source_attribute: PointAttributeDefinition,
        current_index: usize,
        converter: AttributeConversionFn,
        source_attribute_buffer: Vec<u8>,
        _unused: PhantomData<T>,
    }

    impl<'a, T: PrimitiveType, B: PointBuffer + ?Sized>
        AttributeIteratorByValueWithConversion<'a, T, B>
    {
        pub fn new(buffer: &'a B, target_attribute: &'a PointAttributeDefinition) -> Self {
            let source_attribute = match buffer
                .point_layout()
                .get_attribute_by_name(target_attribute.name())
            {
                Some(a) => a,
                None => panic!(
                    "Attribute {} not contained in PointLayout of buffer ({})",
                    target_attribute,
                    buffer.point_layout()
                ),
            };

            let converter = match get_converter_for_attributes(&source_attribute.into(), target_attribute) {
                Some(c) => c,
                None => panic!("Can't convert from attribute {} to attribute {} because no valid conversion exists", source_attribute, target_attribute),
            };

            Self {
                buffer,
                source_attribute: source_attribute.into(),
                current_index: 0,
                converter,
                source_attribute_buffer: vec![0; source_attribute.size() as usize],
                _unused: Default::default(),
            }
        }
    }

    impl<'a, T: PrimitiveType, B: PointBuffer + ?Sized> Iterator
        for AttributeIteratorByValueWithConversion<'a, T, B>
    {
        type Item = T;

        fn next(&mut self) -> Option<Self::Item> {
            if self.current_index == self.buffer.len() {
                return None;
            }

            let mut target_attribute = MaybeUninit::<T>::uninit();
            unsafe {
                let target_attribute_byte_slice = std::slice::from_raw_parts_mut(
                    target_attribute.as_mut_ptr() as *mut u8,
                    std::mem::size_of::<T>(),
                );
                self.buffer.get_raw_attribute(
                    self.current_index,
                    &self.source_attribute,
                    self.source_attribute_buffer.as_mut_slice(),
                );

                (self.converter)(
                    self.source_attribute_buffer.as_slice(),
                    target_attribute_byte_slice,
                );
            }

            self.current_index += 1;

            unsafe { Some(target_attribute.assume_init()) }
        }
    }

    /// Iterator over a `PointBuffer` that yields strongly typed data by reference for a specific attribute for each point
    pub struct AttributeIteratorByRef<'a, T: PrimitiveType> {
        attribute_data: &'a [T],
        current_index: usize,
    }

    impl<'a, T: PrimitiveType> AttributeIteratorByRef<'a, T> {
        pub fn new<B: PerAttributePointBuffer + ?Sized>(
            buffer: &'a B,
            attribute: &'a PointAttributeDefinition,
        ) -> Self {
            if attribute.datatype() != T::data_type() {
                panic!("Type T does not match datatype of attribute {}", attribute);
            }
            if !buffer.point_layout().has_attribute(attribute) {
                panic!(
                    "Attribute {} not contained in PointLayout of buffer ({})",
                    attribute,
                    buffer.point_layout()
                );
            }

            let buffer_len = buffer.len();
            let attribute_data = unsafe {
                std::slice::from_raw_parts(
                    buffer
                        .get_raw_attribute_range_ref(0..buffer_len, attribute)
                        .as_ptr() as *const T,
                    buffer_len,
                )
            };
            Self {
                attribute_data,
                current_index: 0,
            }
        }
    }

    impl<'a, T: PrimitiveType> Iterator for AttributeIteratorByRef<'a, T> {
        type Item = &'a T;

        fn next(&mut self) -> Option<Self::Item> {
            if self.current_index == self.attribute_data.len() {
                return None;
            }

            let current_attribute = &self.attribute_data[self.current_index];
            self.current_index += 1;
            Some(current_attribute)
        }
    }

    pub struct AttributeIteratorByMut<'a, T: PrimitiveType> {
        attribute_data: &'a mut [T],
        current_index: usize,
    }

    impl<'a, T: PrimitiveType> AttributeIteratorByMut<'a, T> {
        pub fn new<'b, B: PerAttributePointBufferMut<'b> + ?Sized>(
            buffer: &'a mut B,
            attribute: &'a PointAttributeDefinition,
        ) -> Self {
            if attribute.datatype() != T::data_type() {
                panic!("Type T does not match datatype of attribute {}", attribute);
            }
            if !buffer.point_layout().has_attribute(attribute) {
                panic!(
                    "Attribute {} not contained in PointLayout of buffer ({})",
                    attribute,
                    buffer.point_layout()
                );
            }

            let buffer_len = buffer.len();
            let attribute_data = unsafe {
                std::slice::from_raw_parts_mut(
                    buffer
                        .get_raw_attribute_range_mut(0..buffer_len, attribute)
                        .as_mut_ptr() as *mut T,
                    buffer_len,
                )
            };
            Self {
                attribute_data,
                current_index: 0,
            }
        }
    }

    impl<'a, T: PrimitiveType> Iterator for AttributeIteratorByMut<'a, T> {
        type Item = &'a mut T;

        fn next(&mut self) -> Option<Self::Item> {
            if self.current_index == self.attribute_data.len() {
                return None;
            }

            // Returning mutable references seems to require unsafe...

            let current_index = self.current_index;
            self.current_index += 1;
            unsafe {
                let ptr_to_current_attribute = self.attribute_data.as_mut_ptr().add(current_index);
                let item = &mut *ptr_to_current_attribute;
                Some(item)
            }
        }
    }
}

// The iterators for multiple attributes are implemented using a macro, because Rust currently does not have variadic generics

macro_rules! extract_attributes {
    ($attributes:ident, $buffer:expr, $current_index:expr, $self_attributes:expr, $($idx:tt),+ ) => {{
        $(unsafe {
            let attribute_bytes = view_raw_bytes_mut(&mut $attributes.$idx);
            $buffer.get_raw_attribute($current_index, $self_attributes[$idx], attribute_bytes);
        })+
    }};
}

macro_rules! extract_and_convert_attributes {
    ($attributes:ident, $buffer:expr, $current_index:expr, $self_attributes:expr, $($idx:tt),+ ) => {{
        $(unsafe {
            let cur_attribute = &mut $self_attributes[$idx];
            let target_attribute_bytes = view_raw_bytes_mut(&mut $attributes.$idx);
            $buffer.get_raw_attribute($current_index, &cur_attribute.0, cur_attribute.2.as_mut_slice());
            let converter = cur_attribute.1;
            converter(cur_attribute.2.as_slice(), target_attribute_bytes);
        })+
    }};
}

macro_rules! attributes_iter {
    ($name:ident, $num_attributes:expr, $($t:ident), + and $($idx:tt),+ ) => {
        pub mod $name {
            use super::*;

            pub struct AttributeIteratorByValue<'a, $($t: PrimitiveType, )+> {
                buffer: &'a dyn PointBuffer,
                attributes: [&'a PointAttributeDefinition; $num_attributes],
                current_index: usize,
                _unused: PhantomData<( $($t),+ )>,
            }

            impl<'a, $($t: PrimitiveType + Default, )+> AttributeIteratorByValue<'a, $($t, )+> {
                pub fn new(buffer: &'a dyn PointBuffer, attributes: [&'a PointAttributeDefinition; $num_attributes]) -> Self {
                    $(if attributes[$idx].datatype() != $t::data_type() {
                        panic!("Type T does not match datatype of attribute {}", attributes[$idx]);
                    })+
                    for attribute in attributes.iter() {
                        if !buffer.point_layout().has_attribute(attribute) {
                            panic!("Attribute {} not contained in PointLayout of buffer ({})", attribute, buffer.point_layout());
                        }
                    }

                    Self {
                        buffer,
                        attributes,
                        current_index: 0,
                        _unused: Default::default(),
                    }
                }
            }

            impl<'a, $($t: PrimitiveType + Default, )+> Iterator for AttributeIteratorByValue<'a, $($t,)+> {
                type Item = ($($t,)+);

                fn next(&mut self) -> Option<Self::Item> {
                    if self.current_index == self.buffer.len() {
                        return None;
                    }

                    let mut attributes: Self::Item = Default::default();

                    extract_attributes!{attributes, self.buffer, self.current_index, self.attributes, $($idx),+ }

                    self.current_index += 1;

                    Some(attributes)
                }
            }

            pub struct AttributeIteratorByValueWithConversion<'a, $($t: PrimitiveType, )+> {
                buffer: &'a dyn PointBuffer,
                attributes_and_converters_and_buffers: Vec<(PointAttributeDefinition, AttributeConversionFn, Vec<u8>)>,
                current_index: usize,
                _unused: PhantomData<( $($t),+ )>,
            }

            impl<'a, $($t: PrimitiveType + Default, )+> AttributeIteratorByValueWithConversion<'a, $($t, )+> {
                pub fn new(buffer: &'a dyn PointBuffer, attributes: [&'a PointAttributeDefinition; $num_attributes]) -> Self {
                    let attributes_and_converters_and_buffers = attributes.iter().map(|target_attribute| { match buffer.point_layout().get_attribute_by_name(target_attribute.name()) {
                        Some(a) => {
                            let source_attribute : PointAttributeDefinition = a.into();
                            let converter = match get_converter_for_attributes(&source_attribute, target_attribute) {
                                Some(c) => c,
                                None => panic!("Can't convert from attribute {} to attribute {} because no valid conversion exists", source_attribute, target_attribute),
                            };
                            let buffer : Vec<u8> = vec![0; source_attribute.size() as usize];
                            (source_attribute, converter, buffer)
                        },
                        None => panic!("Attribute {} not contained in PointLayout of buffer ({})", target_attribute, buffer.point_layout()),
                    }}).collect();

                    Self {
                        buffer,
                        attributes_and_converters_and_buffers,
                        current_index: 0,
                        _unused: Default::default(),
                    }
                }
            }

            impl<'a, $($t: PrimitiveType + Default, )+> Iterator for AttributeIteratorByValueWithConversion<'a, $($t,)+> {
                type Item = ($($t,)+);

                fn next(&mut self) -> Option<Self::Item> {
                    if self.current_index == self.buffer.len() {
                        return None;
                    }

                    let mut attributes: Self::Item = Default::default();

                    extract_and_convert_attributes!{attributes, self.buffer, self.current_index, self.attributes_and_converters_and_buffers, $($idx),+ }

                    self.current_index += 1;

                    Some(attributes)
                }
            }

            pub struct AttributeIteratorByRef<'a, $($t: PrimitiveType, )+> {
                attribute_data: ($(&'a [$t],)+),
                current_index: usize,
            }

            impl<'a, $($t: PrimitiveType, )+> AttributeIteratorByRef<'a, $($t, )+> {
                pub fn new(
                    buffer: &'a dyn PerAttributePointBuffer,
                    attributes: [&'a PointAttributeDefinition; $num_attributes],
                ) -> Self {
                    $(if attributes[$idx].datatype() != $t::data_type() {
                        panic!("Type T does not match datatype of attribute {}", attributes[$idx]);
                    })+
                    let buffer_len = buffer.len();
                    let attribute_data = (
                        $(unsafe {
                        std::slice::from_raw_parts(
                            buffer
                                .get_raw_attribute_range_ref(0..buffer_len, attributes[$idx])
                                .as_ptr() as *const $t,
                            buffer_len,
                        )
                    }),+);
                    Self {
                        attribute_data,
                        current_index: 0,
                    }
                }
            }

            impl<'a, $($t: PrimitiveType, )+> Iterator for AttributeIteratorByRef<'a, $($t, )+> {
                type Item = ($(&'a $t,)+);

                fn next(&mut self) -> Option<Self::Item> {
                    if self.current_index == self.attribute_data.0.len() {
                        return None;
                    }

                    let current_attribute = (
                        $(&self.attribute_data.$idx[self.current_index],)+
                    );
                    self.current_index += 1;
                    Some(current_attribute)
                }
            }

            pub struct AttributeIteratorByMut<'a, $($t: PrimitiveType, )+> {
                attribute_data: ($(&'a mut [$t],)+),
                current_index: usize,
            }

            impl<'a, $($t: PrimitiveType, )+> AttributeIteratorByMut<'a, $($t, )+> {
                pub fn new(
                    buffer: &'a mut dyn PerAttributePointBufferMut,
                    attributes: [&'a PointAttributeDefinition; $num_attributes],
                ) -> Self {
                    $(if attributes[$idx].datatype() != $t::data_type() {
                        panic!("Type T does not match datatype of attribute {}", attributes[$idx]);
                    })+
                    let buffer_len = buffer.len();
                    let attribute_data = (
                        $(unsafe {
                        std::slice::from_raw_parts_mut(
                            buffer
                                .get_raw_attribute_range_mut(0..buffer_len, attributes[$idx])
                                .as_ptr() as *mut $t,
                            buffer_len,
                        )
                    }),+);
                    Self {
                        attribute_data,
                        current_index: 0,
                    }
                }
            }

            impl<'a, $($t: PrimitiveType, )+> Iterator for AttributeIteratorByMut<'a, $($t, )+> {
                type Item = ($(&'a mut $t,)+);

                fn next(&mut self) -> Option<Self::Item> {
                    if self.current_index == self.attribute_data.0.len() {
                        return None;
                    }

                    let current_index = self.current_index;
                    self.current_index += 1;

                    unsafe {
                        let ret = (
                            $(&mut *self.attribute_data.$idx.as_mut_ptr().add(current_index), )+
                        );

                        Some(ret)
                    }
                }
            }

        }
    };
}

// This weird syntax ('... and ...') exists to disambiguate between the type arguments T1, T2 etc. and the indices with which
// the attributes and tuples inside the iterator are indexed. Since we can't count in Rust macros, we have to instead expand
// a list of indices, which has to be passed to the macro

attributes_iter!(attr2, 2, T1, T2 and 0, 1);
attributes_iter!(attr3, 3, T1, T2, T3 and 0, 1, 2);
attributes_iter!(attr4, 4, T1, T2, T3, T4 and 0, 1, 2, 3);

/// Create an iterator over multiple attributes within a `PointBuffer`. This macro uses some special syntax  to determine the attributes
/// and their types:
///
/// `attributes!{ ATTRIBUTE_1_EXPR => ATTRIBUTE_1_TYPE, ATTRIBUTE_2_EXPR => ATTRIBUTE_2_TYPE, ..., buffer }`
///
/// `ATTRIBUTE_X_EXPR` must be an expression that evaluates to a `&PointAttributeDefinition` and `ATTRIBUTE_X_TYPE` must be the Pasture
/// `PrimitiveType` that the attribute will be returned as. The type must match the type that the attribute is stored with inside `buffer`.
/// The iterator will then return tuples of the type:
///
/// `(ATTRIBUTE_1_TYPE, ATTRIBUTE_2_TYPE, ...)`
///
/// *Note:* Currently, a maximum of 4 attributes at the same time are supported.
///
/// # Panics
///
/// Panics if any of the attributes are not contained within the `buffer`.
/// Panics if, for any attribute, the desired type does not match the `PointAttributeDataType` of that attribute.
#[macro_export]
macro_rules! attributes {
    ($attr1:expr => $t1:ty, $attr2:expr => $t2:ty, $buffer:expr) => {
        $crate::containers::attr2::AttributeIteratorByValue::<$t1, $t2>::new(
            $buffer,
            [$attr1, $attr2],
        )
    };
    ($attr1:expr => $t1:ty, $attr2:expr => $t2:ty, $attr3:expr => $t3:ty, $buffer:expr) => {
        $crate::containers::attr3::AttributeIteratorByValue::<$t1, $t2, $t3>::new(
            $buffer,
            [$attr1, $attr2, $attr3],
        )
    };
    ($attr1:expr => $t1:ty, $attr2:expr => $t2:ty, $attr3:expr => $t3:ty, $attr4:expr => $t4:ty, $buffer:expr) => {
        $crate::containers::attr3::AttributeIteratorByValue::<$t1, $t2, $t3, $t4>::new(
            $buffer,
            [$attr1, $attr2, $attr3, $attr4],
        )
    };
}

/// Create an iterator over multiple attributes within a `PointBuffer`, supporting type converisons. This macro uses some special syntax
/// to determine the attributes and their types:
///
/// `attributes_as!{ ATTRIBUTE_1_EXPR => ATTRIBUTE_1_TYPE, ATTRIBUTE_2_EXPR => ATTRIBUTE_2_TYPE, ..., buffer }`
///
/// `ATTRIBUTE_X_EXPR` must be an expression that evaluates to a `&PointAttributeDefinition` and `ATTRIBUTE_X_TYPE` must be the Pasture
/// `PrimitiveType` that the attribute will be returned as. This type must be convertible from the actual type that the attribute
/// is stored with inside `buffer`. The iterator will then return tuples of the form:
///
/// `(ATTRIBUTE_1_TYPE, ATTRIBUTE_2_TYPE, ...)`
///
/// *Note:* Currently, a maximum of 4 attributes at the same time are supported.
///
/// # Panics
///
/// Panics if any of the attributes are not contained within the `buffer`.
/// Panics if, for any attribute, no conversion exists between this attributes `PointAttributeDataType` and the desired type for this attribute.
#[macro_export]
macro_rules! attributes_as {
    ($attr1:expr => $t1:ty, $attr2:expr => $t2:ty, $buffer:expr) => {
        $crate::containers::attr2::AttributeIteratorByValueWithConversion::<$t1, $t2>::new(
            $buffer,
            [$attr1, $attr2],
        )
    };
    ($attr1:expr => $t1:ty, $attr2:expr => $t2:ty, $attr3:expr => $t3:ty, $buffer:expr) => {
        $crate::containers::attr3::AttributeIteratorByValueWithConversion::<$t1, $t2, $t3>::new(
            $buffer,
            [$attr1, $attr2, $attr3],
        )
    };
    ($attr1:expr => $t1:ty, $attr2:expr => $t2:ty, $attr3:expr => $t3:ty, $attr4:expr => $t4:ty, $buffer:expr) => {
        $crate::containers::attr3::AttributeIteratorByValueWithConversion::<$t1, $t2, $t3, $t4>::new(
            $buffer,
            [$attr1, $attr2, $attr3, $attr4],
        )
    };
}

/// Create an iterator over references to multiple attributes within a `PointBuffer`. Requires that the buffer implements
/// `PerAttributePointBuffer`. This macro uses some special syntax to determine the attributes and their types:
///
/// `attributes_ref!{ ATTRIBUTE_1_EXPR => ATTRIBUTE_1_TYPE, ATTRIBUTE_2_EXPR => ATTRIBUTE_2_TYPE, ..., buffer }`
///
/// `ATTRIBUTE_X_EXPR` must be an expression that evaluates to a `&PointAttributeDefinition` and `ATTRIBUTE_X_TYPE` must be the Pasture
/// `PrimitiveType` that the attribute reference will be returned as. The type must match the type that the attribute is stored with in
/// the `buffer`. The iterator will then return tuples of the form:
///
/// `(&ATTRIBUTE_1_TYPE, &ATTRIBUTE_2_TYPE, ...)`
///
/// *Note:* Currently, a maximum of 4 attributes at the same time are supported.
///
/// # Panics
///
/// Panics if any of the attributes are not contained within the `buffer`.
/// Panics if, for any attribute, the desired type does not match the `PointAttributeDataType` of that attribute.
#[macro_export]
macro_rules! attributes_ref {
    ($attr1:expr => $t1:ty, $attr2:expr => $t2:ty, $buffer:expr) => {
        $crate::containers::attr2::AttributeIteratorByRef::<$t1, $t2>::new(
            $buffer,
            [$attr1, $attr2],
        )
    };
    ($attr1:expr => $t1:ty, $attr2:expr => $t2:ty, $attr3:expr => $t3:ty, $buffer:expr) => {
        $crate::containers::attr3::AttributeIteratorByRef::<$t1, $t2, $t3>::new(
            $buffer,
            [$attr1, $attr2, $attr3],
        )
    };
    ($attr1:expr => $t1:ty, $attr2:expr => $t2:ty, $attr3:expr => $t3:ty, $attr4:expr => $t4:ty, $buffer:expr) => {
        $crate::containers::attr3::AttributeIteratorByRef::<$t1, $t2, $t3, $t4>::new(
            $buffer,
            [$attr1, $attr2, $attr3, $attr4],
        )
    };
}

/// Create an iterator over mutable references to multiple attributes within a `PointBuffer`. Requires that the buffer implements
/// `PerAttributePointBufferMut`. This macro uses some special syntax to determine the attributes and their types:
///
/// `attributes_mut!{ ATTRIBUTE_1_EXPR => ATTRIBUTE_1_TYPE, ATTRIBUTE_2_EXPR => ATTRIBUTE_2_TYPE, ..., buffer }`
///
/// `ATTRIBUTE_X_EXPR` must be an expression that evaluates to a `&PointAttributeDefinition` and `ATTRIBUTE_X_TYPE` must be the Pasture
/// `PrimitiveType` that the attribute reference will be returned as. The type must match the type that the attribute is stored with in
/// the `buffer`. The iterator will then return tuples of the form:
///
/// `(&mut ATTRIBUTE_1_TYPE, &mut ATTRIBUTE_2_TYPE, ...)`
///
/// *Note:* Currently, a maximum of 4 attributes at the same time are supported.
///
/// # Panics
///
/// Panics if any of the attributes are not contained within the `buffer`.
/// Panics if, for any attribute, the desired type does not match the `PointAttributeDataType` of that attribute.
#[macro_export]
macro_rules! attributes_mut {
    ($attr1:expr => $t1:ty, $attr2:expr => $t2:ty, $buffer:expr) => {
        $crate::containers::attr2::AttributeIteratorByMut::<$t1, $t2>::new(
            $buffer,
            [$attr1, $attr2],
        )
    };
    ($attr1:expr => $t1:ty, $attr2:expr => $t2:ty, $attr3:expr => $t3:ty, $buffer:expr) => {
        $crate::containers::attr3::AttributeIteratorByMut::<$t1, $t2, $t3>::new(
            $buffer,
            [$attr1, $attr2, $attr3],
        )
    };
    ($attr1:expr => $t1:ty, $attr2:expr => $t2:ty, $attr3:expr => $t3:ty, $attr4:expr => $t4:ty, $buffer:expr) => {
        $crate::containers::attr3::AttributeIteratorByMut::<$t1, $t2, $t3, $t4>::new(
            $buffer,
            [$attr1, $attr2, $attr3, $attr4],
        )
    };
}

#[cfg(test)]
mod tests {

    use crate::{containers::PointBufferExt, layout::attributes};
    use crate::{
        containers::{
            InterleavedVecPointStorage, PerAttributePointBufferExt, PerAttributePointBufferMutExt,
            PerAttributeVecPointStorage,
        },
        layout::attributes::POSITION_3D,
        layout::PointType,
    };
    use nalgebra::Vector3;
    use pasture_derive::PointType;

    // We need this, otherwise we can't use the derive(PointType) macro from within pasture_core because the macro
    // doesn't recognize the name 'pasture_core' :/
    use crate as pasture_core;

    #[derive(Debug, Copy, Clone, PartialEq, PointType)]
    #[repr(C)]
    struct TestPointType {
        #[pasture(BUILTIN_INTENSITY)]
        pub intensity: u16,
        #[pasture(BUILTIN_GPS_TIME)]
        pub gps_time: f64,
    }

    #[test]
    fn test_single_attribute_view_from_per_attribute() {
        let reference_points = vec![
            TestPointType {
                intensity: 42,
                gps_time: 0.123,
            },
            TestPointType {
                intensity: 43,
                gps_time: 0.456,
            },
        ];
        let mut storage = PerAttributeVecPointStorage::new(TestPointType::layout());
        storage.push_point(reference_points[0]);
        storage.push_point(reference_points[1]);

        {
            let first_attribute_mut_view =
                storage.iter_attribute_mut::<u16>(&attributes::INTENSITY);
            first_attribute_mut_view.for_each(|a| {
                *a *= 2;
            });
        }

        let modified_intensities = vec![84_u16, 86_u16];

        {
            let attribute_by_val_collected = storage
                .iter_attribute::<u16>(&attributes::INTENSITY)
                .collect::<Vec<_>>();
            assert_eq!(modified_intensities, attribute_by_val_collected);
        }

        {
            let attribute_by_ref_view = storage.iter_attribute_ref::<u16>(&attributes::INTENSITY);
            let attribute_by_ref_collected = attribute_by_ref_view.map(|a| *a).collect::<Vec<_>>();
            assert_eq!(modified_intensities, attribute_by_ref_collected);
        }
    }

    #[test]
    fn test_two_attributes_view_from_per_attribute() {
        let reference_points = vec![
            TestPointType {
                intensity: 42,
                gps_time: 0.123,
            },
            TestPointType {
                intensity: 43,
                gps_time: 0.456,
            },
        ];
        let mut storage = PerAttributeVecPointStorage::new(TestPointType::layout());
        storage.push_point(reference_points[0]);
        storage.push_point(reference_points[1]);

        {
            let attributes_mut_view = attributes_mut!(
                &attributes::INTENSITY => u16,
                &attributes::GPS_TIME => f64,
                &mut storage
            );
            attributes_mut_view.for_each(|(intensity, gps_time)| {
                *intensity *= 2;
                *gps_time += 1.0;
            });
        }

        let modified_data = vec![(84_u16, 1.123), (86_u16, 1.456)];

        {
            let attributes_by_val_view = attributes!(
                &attributes::INTENSITY => u16,
                &attributes::GPS_TIME => f64,
                &storage
            );
            let attributes_by_val_collected = attributes_by_val_view.collect::<Vec<_>>();
            assert_eq!(modified_data, attributes_by_val_collected);
        }

        {
            let attributes_by_ref_view = attributes_ref!(
                &attributes::INTENSITY => u16,
                &attributes::GPS_TIME => f64,
                &storage
            );
            let attributes_by_ref_collected = attributes_by_ref_view
                .map(|(&intensity, &gps_time)| (intensity, gps_time))
                .collect::<Vec<_>>();
            assert_eq!(modified_data, attributes_by_ref_collected);
        }
    }

    #[test]
    #[should_panic(expected = "not contained in PointLayout of buffer")]
    fn test_attributes_with_different_datatype_fails() {
        #[derive(Debug, Copy, Clone, PartialEq, PointType)]
        #[repr(C)]
        struct PositionLowp(#[pasture(BUILTIN_POSITION_3D)] Vector3<f32>);

        let mut storage = InterleavedVecPointStorage::new(PositionLowp::layout());
        storage.push_point(PositionLowp(Default::default()));

        storage
            .iter_attribute::<Vector3<f64>>(&POSITION_3D)
            .for_each(drop);
    }

    #[test]
    #[should_panic(expected = "not contained in PointLayout of buffer")]
    fn test_attributes_ref_with_different_datatype_fails() {
        #[derive(Debug, Copy, Clone, PartialEq, PointType)]
        #[repr(C)]
        struct PositionLowp(#[pasture(BUILTIN_POSITION_3D)] Vector3<f32>);

        let mut storage = PerAttributeVecPointStorage::new(PositionLowp::layout());
        storage.push_point(PositionLowp(Default::default()));

        storage
            .iter_attribute_ref::<Vector3<f64>>(&POSITION_3D)
            .for_each(drop);
    }

    #[test]
    #[should_panic(expected = "not contained in PointLayout of buffer")]
    fn test_attributes_mut_with_different_datatype_fails() {
        #[derive(Debug, Copy, Clone, PartialEq, PointType)]
        #[repr(C)]
        struct PositionLowp(#[pasture(BUILTIN_POSITION_3D)] Vector3<f32>);

        let mut storage = PerAttributeVecPointStorage::new(PositionLowp::layout());
        storage.push_point(PositionLowp(Default::default()));

        storage
            .iter_attribute_mut::<Vector3<f64>>(&POSITION_3D)
            .for_each(drop);
    }

    #[test]
    #[should_panic(expected = "Type T does not match datatype of attribute")]
    fn test_attribute_with_wrong_type_fails() {
        let storage = InterleavedVecPointStorage::new(TestPointType::layout());
        storage
            .iter_attribute::<u32>(&attributes::INTENSITY)
            .for_each(drop);
    }

    #[test]
    #[should_panic(expected = "Type T does not match datatype of attribute")]
    fn test_attribute_ref_with_wrong_type_fails() {
        let storage = PerAttributeVecPointStorage::new(TestPointType::layout());
        storage.iter_attribute_ref::<u32>(&attributes::INTENSITY);
    }

    #[test]
    #[should_panic(expected = "Type T does not match datatype of attribute")]
    fn test_attribute_mut_with_wrong_type_fails() {
        let mut storage = PerAttributeVecPointStorage::new(TestPointType::layout());
        storage.iter_attribute_mut::<u32>(&attributes::INTENSITY);
    }

    #[test]
    #[should_panic(expected = "Type T does not match datatype of attribute")]
    fn test_attributes_with_wrong_type_fails() {
        let storage = InterleavedVecPointStorage::new(TestPointType::layout());
        attributes!(
            &attributes::INTENSITY => u32,
            &attributes::GPS_TIME => f32,
            &storage
        );
    }

    #[test]
    #[should_panic(expected = "Type T does not match datatype of attribute")]
    fn test_attributes_ref_with_wrong_type_fails() {
        let storage = PerAttributeVecPointStorage::new(TestPointType::layout());
        attributes_ref!(
            &attributes::INTENSITY => u32,
            &attributes::GPS_TIME => f32,
            &storage
        );
    }

    #[test]
    #[should_panic(expected = "Type T does not match datatype of attribute")]
    fn test_attributes_mut_with_wrong_type_fails() {
        let mut storage = PerAttributeVecPointStorage::new(TestPointType::layout());
        attributes_mut!(
            &attributes::INTENSITY => u32,
            &attributes::GPS_TIME => f32,
            &mut storage
        );
    }
}
