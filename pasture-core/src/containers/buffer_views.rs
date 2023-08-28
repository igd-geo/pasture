use std::marker::PhantomData;

use crate::layout::{PointAttributeDefinition, PointAttributeMember, PointType, PrimitiveType};

use super::{
    attribute_iterators::{
        AttributeIteratorByMut, AttributeIteratorByRef, AttributeIteratorByValue,
    },
    point_buffer::{
        BorrowedBuffer, BorrowedMutBuffer, ColumnarBuffer, ColumnarBufferMut, InterleavedBuffer,
        InterleavedBufferMut,
    },
    point_iterators::{PointIteratorByMut, PointIteratorByRef, PointIteratorByValue},
    OwningBuffer,
};

pub struct PointView<'a, B: BorrowedBuffer<'a>, T: PointType> {
    buffer: &'a B,
    _phantom: PhantomData<T>,
}

impl<'a, B: BorrowedBuffer<'a>, T: PointType> PointView<'a, B, T> {
    pub(crate) fn new(buffer: &'a B) -> Self {
        Self {
            buffer,
            _phantom: Default::default(),
        }
    }

    pub fn at(&self, index: usize) -> T {
        let mut point = T::zeroed();
        self.buffer
            .get_point(index, bytemuck::bytes_of_mut(&mut point));
        point
    }
}

impl<'a, B: InterleavedBuffer<'a>, T: PointType> PointView<'a, B, T> {
    pub fn at_ref(&self, index: usize) -> &T {
        bytemuck::from_bytes(self.buffer.get_point_ref(index))
    }

    pub fn iter(&self) -> PointIteratorByRef<'a, T> {
        self.buffer.into()
    }
}

impl<'a, B: BorrowedBuffer<'a>, T: PointType> IntoIterator for PointView<'a, B, T> {
    type Item = T;
    type IntoIter = PointIteratorByValue<'a, T, B>;

    fn into_iter(self) -> Self::IntoIter {
        self.buffer.into()
    }
}

pub struct PointViewMut<'a, B: BorrowedMutBuffer<'a>, T: PointType> {
    buffer: &'a mut B,
    _phantom: PhantomData<T>,
}

impl<'a, B: BorrowedMutBuffer<'a>, T: PointType> PointViewMut<'a, B, T> {
    pub(crate) fn new(buffer: &'a mut B) -> Self {
        Self {
            buffer,
            _phantom: Default::default(),
        }
    }

    pub fn at(&self, index: usize) -> T {
        let mut point = T::zeroed();
        self.buffer
            .get_point(index, bytemuck::bytes_of_mut(&mut point));
        point
    }

    pub fn set_at(&mut self, index: usize, point: T) {
        self.buffer.set_point(index, bytemuck::bytes_of(&point));
    }
}

impl<'a, B: InterleavedBuffer<'a> + BorrowedMutBuffer<'a>, T: PointType> PointViewMut<'a, B, T> {
    pub fn at_ref(&'a self, index: usize) -> &'a T {
        bytemuck::from_bytes(self.buffer.get_point_ref(index))
    }

    pub fn iter(&'a self) -> PointIteratorByRef<'a, T> {
        (&*self.buffer).into()
    }
}

impl<'a, B: InterleavedBufferMut<'a>, T: PointType> PointViewMut<'a, B, T> {
    pub fn at_mut(&'a mut self, index: usize) -> &'a mut T {
        bytemuck::from_bytes_mut(self.buffer.get_point_mut(index))
    }

    pub fn iter_mut(&'a mut self) -> PointIteratorByMut<'a, T> {
        self.buffer.into()
    }
}

impl<'a, B: OwningBuffer<'a>, T: PointType> PointViewMut<'a, B, T> {
    pub fn push_point(&mut self, point: T) {
        self.buffer.push_point(bytemuck::bytes_of(&point));
    }
}

pub struct AttributeView<'a, B: BorrowedBuffer<'a>, T: PrimitiveType> {
    buffer: &'a B,
    attribute: &'a PointAttributeMember,
    _phantom: PhantomData<T>,
}

impl<'a, B: BorrowedBuffer<'a>, T: PrimitiveType> AttributeView<'a, B, T> {
    pub(crate) fn new(buffer: &'a B, attribute: &PointAttributeDefinition) -> Self {
        Self {
            attribute: buffer
                .point_layout()
                .get_attribute(attribute)
                .expect("Attribute not found in PointLayout of buffer"),
            buffer,
            _phantom: Default::default(),
        }
    }

    pub fn at(&self, index: usize) -> T {
        let mut attribute = T::zeroed();
        // Is safe because we get the attribute_member from the PointLayout of the buffer in `new`
        unsafe {
            self.buffer.get_attribute_unchecked(
                self.attribute,
                index,
                bytemuck::bytes_of_mut(&mut attribute),
            );
        }
        attribute
    }
}

impl<'a, B: ColumnarBuffer<'a>, T: PrimitiveType> AttributeView<'a, B, T> {
    pub fn at_ref(&self, index: usize) -> &'a T {
        bytemuck::from_bytes(
            self.buffer
                .get_attribute_ref(self.attribute.attribute_definition(), index),
        )
    }

    pub fn iter(&self) -> AttributeIteratorByRef<'a, T> {
        AttributeIteratorByRef::new(self.buffer, self.attribute.attribute_definition())
    }
}

impl<'a, B: BorrowedBuffer<'a>, T: PrimitiveType> IntoIterator for AttributeView<'a, B, T> {
    type Item = T;
    type IntoIter = AttributeIteratorByValue<'a, T, B>;

    fn into_iter(self) -> Self::IntoIter {
        AttributeIteratorByValue::new(self.buffer, self.attribute.attribute_definition())
    }
}

pub struct AttributeViewMut<'a, B: BorrowedMutBuffer<'a>, T: PrimitiveType> {
    buffer: &'a mut B,
    attribute: PointAttributeMember,
    _phantom: PhantomData<T>,
}

impl<'a, B: BorrowedMutBuffer<'a>, T: PrimitiveType> AttributeViewMut<'a, B, T> {
    pub(crate) fn new(buffer: &'a mut B, attribute: &PointAttributeDefinition) -> Self {
        Self {
            attribute: buffer
                .point_layout()
                .get_attribute(attribute)
                .expect("Attribute not found in PointLayout of buffer")
                .clone(),
            buffer,
            _phantom: Default::default(),
        }
    }

    pub fn at(&self, index: usize) -> T {
        let mut attribute = T::zeroed();
        // Is safe because we get the attribute_member from the PointLayout of the buffer in `new`
        unsafe {
            self.buffer.get_attribute_unchecked(
                &self.attribute,
                index,
                bytemuck::bytes_of_mut(&mut attribute),
            );
        }
        attribute
    }

    pub fn set_at(&mut self, index: usize, attribute_value: T) {
        self.buffer.set_attribute(
            self.attribute.attribute_definition(),
            index,
            bytemuck::bytes_of(&attribute_value),
        );
    }
}

impl<'a, B: ColumnarBuffer<'a> + BorrowedMutBuffer<'a>, T: PrimitiveType>
    AttributeViewMut<'a, B, T>
{
    pub fn at_ref(&'a self, index: usize) -> &'a T {
        bytemuck::from_bytes(
            self.buffer
                .get_attribute_ref(self.attribute.attribute_definition(), index),
        )
    }

    pub fn iter(&'a self) -> AttributeIteratorByRef<'a, T> {
        AttributeIteratorByRef::new(self.buffer, self.attribute.attribute_definition())
    }
}

impl<'a, B: ColumnarBufferMut<'a> + BorrowedMutBuffer<'a>, T: PrimitiveType>
    AttributeViewMut<'a, B, T>
{
    pub fn at_mut(&'a mut self, index: usize) -> &'a mut T {
        bytemuck::from_bytes_mut(
            self.buffer
                .get_attribute_mut(self.attribute.attribute_definition(), index),
        )
    }

    pub fn iter_mut(&'a mut self) -> AttributeIteratorByMut<'a, T> {
        AttributeIteratorByMut::new(self.buffer, self.attribute.attribute_definition())
    }
}

impl<'a, B: OwningBuffer<'a> + BorrowedMutBuffer<'a>, T: PrimitiveType> AttributeViewMut<'a, B, T> {
    pub fn push_point(&mut self, point: T) {
        self.buffer.push_point(bytemuck::bytes_of(&point));
    }
}
