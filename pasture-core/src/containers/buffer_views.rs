use anyhow::{anyhow, Result};
use std::{cell::RefCell, marker::PhantomData};

use crate::layout::{
    conversion::{get_converter_for_attributes, AttributeConversionFn},
    PointAttributeDefinition, PointAttributeMember, PointType, PrimitiveType,
};

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

pub struct PointView<'a, 'b, B: BorrowedBuffer<'a>, T: PointType>
where
    'a: 'b,
{
    buffer: &'b B,
    _phantom: PhantomData<&'a T>,
}

impl<'a, 'b, B: BorrowedBuffer<'a>, T: PointType> PointView<'a, 'b, B, T>
where
    'a: 'b,
{
    pub(crate) fn new(buffer: &'b B) -> Self {
        assert_eq!(T::layout(), *buffer.point_layout());
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

impl<'a, 'b, B: InterleavedBuffer<'a>, T: PointType> PointView<'a, 'b, B, T>
where
    'a: 'b,
{
    pub fn at_ref(&self, index: usize) -> &T {
        bytemuck::from_bytes(self.buffer.get_point_ref(index))
    }

    pub fn iter<'c>(&'c self) -> PointIteratorByRef<'c, T>
    where
        'b: 'c,
    {
        self.buffer.into()
    }
}

impl<'a, 'b, B: BorrowedBuffer<'a> + 'a, T: PointType> IntoIterator for PointView<'a, 'b, B, T>
where
    'a: 'b,
{
    type Item = T;
    type IntoIter = PointIteratorByValue<'a, 'b, T, B>;

    fn into_iter(self) -> Self::IntoIter {
        self.buffer.into()
    }
}

pub struct PointViewMut<'a, 'b, B: BorrowedMutBuffer<'a>, T: PointType>
where
    'a: 'b,
{
    buffer: &'b mut B,
    _phantom: PhantomData<&'a T>,
}

impl<'a, 'b, B: BorrowedMutBuffer<'a>, T: PointType> PointViewMut<'a, 'b, B, T> {
    pub(crate) fn new(buffer: &'b mut B) -> Self {
        assert_eq!(T::layout(), *buffer.point_layout());
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

impl<'a, 'b, B: InterleavedBuffer<'a> + BorrowedMutBuffer<'a>, T: PointType>
    PointViewMut<'a, 'b, B, T>
{
    pub fn at_ref<'c>(&'c self, index: usize) -> &'c T
    where
        'b: 'c,
    {
        bytemuck::from_bytes(self.buffer.get_point_ref(index))
    }

    pub fn iter<'c>(&'c self) -> PointIteratorByRef<'c, T>
    where
        'b: 'c,
    {
        (&*self.buffer).into()
    }
}

impl<'a, 'b, B: InterleavedBufferMut<'a>, T: PointType> PointViewMut<'a, 'b, B, T> {
    pub fn at_mut<'c>(&'c mut self, index: usize) -> &'c mut T
    where
        'b: 'c,
    {
        bytemuck::from_bytes_mut(self.buffer.get_point_mut(index))
    }

    pub fn iter_mut<'c>(&'c mut self) -> PointIteratorByMut<'c, T>
    where
        'b: 'c,
    {
        self.buffer.into()
    }
}

impl<'a, 'b, B: OwningBuffer<'a>, T: PointType> PointViewMut<'a, 'b, B, T> {
    pub fn push_point(&mut self, point: T) {
        // Safe because we know that a `PointViewMut` can never be created for a `T` that is different from
        // the `PointLayout` of the underlying buffer (see the check in `new`)
        unsafe {
            self.buffer.push_points(bytemuck::bytes_of(&point));
        }
    }
}

pub struct AttributeView<'a, 'b, B: BorrowedBuffer<'a>, T: PrimitiveType>
where
    'a: 'b,
{
    buffer: &'b B,
    attribute: &'b PointAttributeMember,
    _phantom: PhantomData<&'a T>,
}

impl<'a, 'b, B: BorrowedBuffer<'a>, T: PrimitiveType> AttributeView<'a, 'b, B, T> {
    pub(crate) fn new(buffer: &'b B, attribute: &PointAttributeDefinition) -> Self {
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

impl<'a, 'b, B: ColumnarBuffer<'a>, T: PrimitiveType> AttributeView<'a, 'b, B, T>
where
    'a: 'b,
{
    pub fn at_ref<'c>(&'c self, index: usize) -> &'c T
    where
        'b: 'c,
    {
        bytemuck::from_bytes(
            self.buffer
                .get_attribute_ref(self.attribute.attribute_definition(), index),
        )
    }

    pub fn iter<'c>(&'c self) -> AttributeIteratorByRef<'c, T>
    where
        'b: 'c,
    {
        AttributeIteratorByRef::new(self.buffer, self.attribute.attribute_definition())
    }
}

impl<'a, 'b, B: BorrowedBuffer<'a> + 'a, T: PrimitiveType> IntoIterator
    for AttributeView<'a, 'b, B, T>
{
    type Item = T;
    type IntoIter = AttributeIteratorByValue<'a, 'b, T, B>;

    fn into_iter(self) -> Self::IntoIter {
        AttributeIteratorByValue::new(self.buffer, self.attribute.attribute_definition())
    }
}

pub struct AttributeViewMut<'a, 'b, B: BorrowedMutBuffer<'a>, T: PrimitiveType>
where
    'a: 'b,
{
    buffer: &'b mut B,
    attribute: PointAttributeMember,
    _phantom: PhantomData<&'a T>,
}

impl<'a, 'b, B: BorrowedMutBuffer<'a>, T: PrimitiveType> AttributeViewMut<'a, 'b, B, T>
where
    'a: 'b,
{
    pub(crate) fn new(buffer: &'b mut B, attribute: &PointAttributeDefinition) -> Self {
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

impl<'a, 'b, B: ColumnarBuffer<'a> + BorrowedMutBuffer<'a>, T: PrimitiveType>
    AttributeViewMut<'a, 'b, B, T>
where
    'a: 'b,
{
    pub fn at_ref(&'b self, index: usize) -> &'b T {
        bytemuck::from_bytes(
            self.buffer
                .get_attribute_ref(self.attribute.attribute_definition(), index),
        )
    }

    pub fn iter(&'b self) -> AttributeIteratorByRef<'b, T> {
        AttributeIteratorByRef::new(self.buffer, self.attribute.attribute_definition())
    }
}

impl<'a, 'b, B: ColumnarBufferMut<'a> + BorrowedMutBuffer<'a>, T: PrimitiveType>
    AttributeViewMut<'a, 'b, B, T>
{
    pub fn at_mut(&'b mut self, index: usize) -> &'b mut T {
        bytemuck::from_bytes_mut(
            self.buffer
                .get_attribute_mut(self.attribute.attribute_definition(), index),
        )
    }

    pub fn iter_mut(&'b mut self) -> AttributeIteratorByMut<'b, T> {
        AttributeIteratorByMut::new(self.buffer, self.attribute.attribute_definition())
    }
}

// impl<'a, B: OwningBuffer<'a> + BorrowedMutBuffer<'a>, T: PrimitiveType> AttributeViewMut<'a, B, T> {
//     pub fn push_point(&mut self, point: T) {
//         self.buffer.push_point(bytemuck::bytes_of(&point));
//     }
// }

/// A view over a strongly typed point attribute that supports type conversion. This means that the
/// `PointAttributeDataType` of the attribute must not match the type `T` that this view returns
pub struct AttributeViewConverting<'a, 'b, B: BorrowedBuffer<'a>, T: PrimitiveType>
where
    'a: 'b,
{
    buffer: &'b B,
    attribute: PointAttributeMember,
    converter_fn: AttributeConversionFn,
    converter_buffer: RefCell<Vec<u8>>,
    _phantom: PhantomData<&'a T>,
}

impl<'a, 'b, B: BorrowedBuffer<'a>, T: PrimitiveType> AttributeViewConverting<'a, 'b, B, T> {
    pub(crate) fn new(buffer: &'b B, attribute: &PointAttributeDefinition) -> Result<Self> {
        let attribute_in_layout: &PointAttributeMember = buffer
            .point_layout()
            .get_attribute_by_name(attribute.name())
            .expect("Attribute not found in PointLayout of buffer");
        let converter_fn = get_converter_for_attributes(
            attribute_in_layout.attribute_definition(),
            &attribute.with_custom_datatype(T::data_type()),
        )
        .ok_or(anyhow!("Conversion between attribute types is impossible"))?;
        let converter_buffer = vec![0; T::data_type().size() as usize];
        Ok(Self {
            attribute: attribute_in_layout.clone(),
            buffer,
            converter_fn,
            converter_buffer: RefCell::new(converter_buffer),
            _phantom: Default::default(),
        })
    }

    pub fn at(&self, index: usize) -> T {
        let mut value = T::zeroed();
        // Is safe because we took 'attribute' from the point layout of the buffer
        // conversion is safe because we checked the source and destination types in `new`
        unsafe {
            self.buffer.get_attribute_unchecked(
                &self.attribute,
                index,
                self.converter_buffer.borrow_mut().as_mut_slice(),
            );
            (self.converter_fn)(
                self.converter_buffer.borrow().as_slice(),
                bytemuck::bytes_of_mut(&mut value),
            );
        }
        value
    }
}

impl<'a, 'b, B: BorrowedBuffer<'a>, T: PrimitiveType> IntoIterator
    for AttributeViewConverting<'a, 'b, B, T>
{
    type Item = T;
    type IntoIter = AttributeViewConvertingIterator<'a, 'b, B, T>;

    fn into_iter(self) -> Self::IntoIter {
        AttributeViewConvertingIterator {
            current_index: 0,
            view: self,
        }
    }
}

pub struct AttributeViewConvertingIterator<'a, 'b, B: BorrowedBuffer<'a>, T: PrimitiveType> {
    view: AttributeViewConverting<'a, 'b, B, T>,
    current_index: usize,
}

impl<'a, 'b, B: BorrowedBuffer<'a>, T: PrimitiveType> Iterator
    for AttributeViewConvertingIterator<'a, 'b, B, T>
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index == self.view.buffer.len() {
            None
        } else {
            let ret = self.view.at(self.current_index);
            self.current_index += 1;
            Some(ret)
        }
    }
}
