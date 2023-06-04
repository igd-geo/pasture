use std::{iter::FromIterator, ops::Range};

use crate::{
    layout::{PointAttributeDefinition, PointLayout, PointType, PrimitiveType},
    util::view_raw_bytes,
};

use super::{
    AttributeViewMut, AttributeViewRef, BufferStorage, BufferStorageMut, BufferViewMut,
    BufferViewRef, IndexBuffer,
};

/// Non-owning range of points
pub trait PointSlice: Sized {
    fn point_layout(&self) -> &PointLayout;
    fn storage(&self) -> &dyn BufferStorage;
    fn len(&self) -> usize {
        self.storage().len()
    }
}

impl<'a, T: PointSlice> PointSlice for &'a T {
    fn point_layout(&self) -> &PointLayout {
        (*self).point_layout()
    }

    fn storage(&self) -> &dyn BufferStorage {
        (*self).storage()
    }
}

pub trait PointSliceTyped<T: PointType>: PointSlice + IntoIterator {
    fn at(&self, index: usize) -> T;
}

pub trait PointSliceTypedByRef<T: PointType>: PointSliceTyped<T> {
    fn at(&self, index: usize) -> &T;
}

pub trait PointSliceTypedByMut<T: PointType>: PointSliceTypedByRef<T> {
    fn at_mut(&mut self, index: usize) -> &mut T;
}

pub trait AttributeSliceTyped<T: PrimitiveType> {
    fn at(&self, index: usize) -> T;
    fn len(&self) -> usize;
}

pub trait AttributeSliceTypedByRef<T: PrimitiveType>: AttributeSliceTyped<T> {
    fn at(&self, index: usize) -> &T;
}

pub trait AttributeSliceTypedByMut<T: PrimitiveType>: AttributeSliceTypedByRef<T> {
    fn at_mut(&mut self, index: usize) -> &mut T;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PointBuffer<T: BufferStorage> {
    storage: T,
    point_layout: PointLayout,
}

impl<T: BufferStorage> PointBuffer<T> {
    pub fn new(storage: T, point_layout: PointLayout) -> Self {
        Self {
            storage,
            point_layout,
        }
    }

    pub fn storage_mut(&mut self) -> &mut T {
        &mut self.storage
    }

    pub fn view<U: PointType>(&self) -> BufferViewRef<'_, U, T> {
        BufferViewRef::from_storage_and_layout(&self.storage, &self.point_layout)
    }

    pub fn view_mut<U: PointType>(&mut self) -> BufferViewMut<'_, U, T> {
        BufferViewMut::from_storage_and_layout(&mut self.storage, &self.point_layout)
    }

    pub fn view_attribute<'a, U: PrimitiveType>(
        &'a self,
        attribute: &PointAttributeDefinition,
    ) -> AttributeViewRef<'a, U, T> {
        let attribute_member = self
            .point_layout
            .get_attribute(attribute)
            .expect("Attribute not found in PointLayout");
        AttributeViewRef::from_storage_and_attribute(&self.storage, attribute_member)
    }

    pub fn view_attribute_mut<'a, U: PrimitiveType>(
        &'a mut self,
        attribute: &PointAttributeDefinition,
    ) -> AttributeViewMut<'a, U, T> {
        let attribute_member = self
            .point_layout
            .get_attribute(attribute)
            .expect("Attribute not found in PointLayout");
        AttributeViewMut::from_storage_and_attribute(&mut self.storage, attribute_member)
    }

    pub fn slice<'a>(&'a self, range: Range<usize>) -> PointBuffer<<T as IndexBuffer<'a>>::Output>
    where
        T: IndexBuffer<'a>,
    {
        if range.start >= range.end {
            panic!("empty range is not supported in call to slice()");
        }
        if range.end > self.len() {
            panic!("Range end is out of bounds");
        }
        PointBuffer::new(self.storage.index(range), self.point_layout.clone())
    }
}

impl<T: BufferStorageMut> PointBuffer<T> {
    pub fn push<U: PointType>(&mut self, point: U) {
        if U::layout() != self.point_layout {
            panic!("PointLayout of PointType U does not match the PointLayout of this buffer");
        }
        // Is safe because we checked the PointLayout of U
        unsafe {
            self.storage.push(view_raw_bytes(&point));
        }
    }

    pub fn extend<I: IntoIterator<Item = U>, U: PointType>(&mut self, iter: I) {
        if U::layout() != self.point_layout {
            panic!("PointLayout of PointType U does not match the PointLayout of this buffer");
        }
        unsafe {
            iter.into_iter()
                .for_each(|point| self.storage.push(view_raw_bytes(&point)));
        }
    }
}

impl<T: BufferStorage> PointSlice for PointBuffer<T> {
    fn point_layout(&self) -> &PointLayout {
        &self.point_layout
    }

    fn storage(&self) -> &dyn BufferStorage {
        &self.storage
    }
}

impl<P: PointType, T: BufferStorage + FromIterator<P>> FromIterator<P> for PointBuffer<T> {
    fn from_iter<U: IntoIterator<Item = P>>(iter: U) -> Self {
        let point_layout = P::layout();
        Self {
            point_layout,
            storage: iter.into_iter().collect(),
        }
    }
}
