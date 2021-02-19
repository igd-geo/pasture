use crate::containers::{PerAttributePointBuffer, PerAttributePointBufferMut, PointBuffer};
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
    pub struct AttributeIteratorByValue<'a, T: PrimitiveType> {
        buffer: &'a dyn PointBuffer,
        attribute: &'a PointAttributeDefinition,
        current_index: usize,
        _unused: PhantomData<T>,
    }

    impl<'a, T: PrimitiveType> AttributeIteratorByValue<'a, T> {
        pub fn new(buffer: &'a dyn PointBuffer, attribute: &'a PointAttributeDefinition) -> Self {
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
                _unused: Default::default(),
            }
        }
    }

    impl<'a, T: PrimitiveType> Iterator for AttributeIteratorByValue<'a, T> {
        type Item = T;

        fn next(&mut self) -> Option<Self::Item> {
            if self.current_index == self.buffer.len() {
                return None;
            }

            // Create an uninitialized T which is filled by the call to `buffer.get_point_by_copy`
            let mut attribute = MaybeUninit::<T>::uninit();
            unsafe {
                let attribute_byte_slice = std::slice::from_raw_parts_mut(
                    attribute.as_mut_ptr() as *mut u8,
                    std::mem::size_of::<T>(),
                );
                self.buffer.get_attribute_by_copy(
                    self.current_index,
                    self.attribute,
                    attribute_byte_slice,
                );
            }

            self.current_index += 1;

            unsafe { Some(attribute.assume_init()) }
        }
    }

    /// Iterator over a `PointBuffer` that yields strongly typed data by reference for a specific attribute for each point
    pub struct AttributeIteratorByRef<'a, T: PrimitiveType> {
        attribute_data: &'a [T],
        current_index: usize,
    }

    impl<'a, T: PrimitiveType> AttributeIteratorByRef<'a, T> {
        pub fn new(
            buffer: &'a dyn PerAttributePointBuffer,
            attribute: &'a PointAttributeDefinition,
        ) -> Self {
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
                        .get_attribute_range_ref(0..buffer_len, attribute)
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
        pub fn new(
            buffer: &'a mut dyn PerAttributePointBufferMut,
            attribute: &'a PointAttributeDefinition,
        ) -> Self {
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
                        .get_attribute_range_mut(0..buffer_len, attribute)
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
            $buffer.get_attribute_by_copy($current_index, $self_attributes[$idx], attribute_bytes);
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
                    // TODO Support converting iterators for when an attribute with the same name but different datatype exists!
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

            pub struct AttributeIteratorByRef<'a, $($t: PrimitiveType, )+> {
                attribute_data: ($(&'a [$t],)+),
                current_index: usize,
            }

            impl<'a, $($t: PrimitiveType, )+> AttributeIteratorByRef<'a, $($t, )+> {
                pub fn new(
                    buffer: &'a dyn PerAttributePointBuffer,
                    attributes: [&'a PointAttributeDefinition; $num_attributes],
                ) -> Self {
                    let buffer_len = buffer.len();
                    let attribute_data = (
                        $(unsafe {
                        std::slice::from_raw_parts(
                            buffer
                                .get_attribute_range_ref(0..buffer_len, attributes[$idx])
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
                    let buffer_len = buffer.len();
                    let attribute_data = (
                        $(unsafe {
                        std::slice::from_raw_parts_mut(
                            buffer
                                .get_attribute_range_mut(0..buffer_len, attributes[$idx])
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
