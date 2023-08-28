use std::marker::PhantomData;

use crate::layout::{PointAttributeDefinition, PointAttributeMember, PrimitiveType};

use super::point_buffer::{BorrowedBuffer, ColumnarBuffer, ColumnarBufferMut};

pub struct AttributeIteratorByValue<'a, T: PrimitiveType, B: BorrowedBuffer<'a>> {
    buffer: &'a B,
    attribute_member: &'a PointAttributeMember,
    current_index: usize,
    _phantom: PhantomData<T>,
}

impl<'a, T: PrimitiveType, B: BorrowedBuffer<'a>> AttributeIteratorByValue<'a, T, B> {
    pub(crate) fn new(buffer: &'a B, attribute: &PointAttributeDefinition) -> Self {
        Self {
            attribute_member: buffer
                .point_layout()
                .get_attribute(attribute)
                .expect("Attribute not found in PointLayout of buffer"),
            buffer,
            current_index: 0,
            _phantom: Default::default(),
        }
    }
}

impl<'a, T: PrimitiveType, B: BorrowedBuffer<'a>> Iterator for AttributeIteratorByValue<'a, T, B> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index == self.buffer.len() {
            None
        } else {
            let mut attribute = T::zeroed();
            let attribute_bytes = bytemuck::bytes_of_mut(&mut attribute);
            // This is safe because in `new` we obtain the `attribute_member` from the point layout of the buffer
            unsafe {
                self.buffer.get_attribute_unchecked(
                    self.attribute_member,
                    self.current_index,
                    attribute_bytes,
                );
            }
            self.current_index += 1;
            Some(attribute)
        }
    }
}

pub struct AttributeIteratorByRef<'a, T: PrimitiveType> {
    attribute_data: &'a [T],
    current_index: usize,
}

impl<'a, T: PrimitiveType> AttributeIteratorByRef<'a, T> {
    pub(crate) fn new<B: ColumnarBuffer<'a>>(
        buffer: &'a B,
        attribute: &PointAttributeDefinition,
    ) -> Self {
        let attribute_memory = buffer.get_attribute_range_ref(attribute, 0..buffer.len());
        Self {
            attribute_data: bytemuck::cast_slice(attribute_memory),
            current_index: 0,
        }
    }
}

impl<'a, T: PrimitiveType> Iterator for AttributeIteratorByRef<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index == self.attribute_data.len() {
            None
        } else {
            let ret = &self.attribute_data[self.current_index];
            self.current_index += 1;
            Some(ret)
        }
    }
}

pub struct AttributeIteratorByMut<'a, T: PrimitiveType> {
    attribute_data: &'a mut [T],
    current_index: usize,
}

impl<'a, T: PrimitiveType> AttributeIteratorByMut<'a, T> {
    pub(crate) fn new<B: ColumnarBufferMut<'a>>(
        buffer: &'a mut B,
        attribute: &PointAttributeDefinition,
    ) -> Self {
        let attribute_memory = buffer.get_attribute_range_mut(attribute, 0..buffer.len());
        Self {
            attribute_data: bytemuck::cast_slice_mut(attribute_memory),
            current_index: 0,
        }
    }
}

impl<'a, T: PrimitiveType> Iterator for AttributeIteratorByMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index == self.attribute_data.len() {
            None
        } else {
            unsafe {
                let attribute_ptr = self.attribute_data.as_mut_ptr().add(self.current_index);
                self.current_index += 1;
                Some(&mut *attribute_ptr)
            }
        }
    }
}
