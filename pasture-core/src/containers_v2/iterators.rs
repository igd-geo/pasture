use std::marker::PhantomData;

use crate::layout::{PointType, PrimitiveType};

use super::{
    AttributeSliceTyped, AttributeSliceTypedByMut, AttributeSliceTypedByRef, PointSliceTyped,
    PointSliceTypedByMut, PointSliceTypedByRef,
};

/// Iterator over strongly typed points in a PointSlice by value
pub struct PointIteratorByValue<T: PointType, S: PointSliceTyped<T>> {
    slice: S,
    current_index: usize,
    length: usize,
    _phantom: PhantomData<T>,
}

impl<T: PointType, S: PointSliceTyped<T>> From<S> for PointIteratorByValue<T, S> {
    fn from(slice: S) -> Self {
        if T::layout() != *slice.point_layout() {
            panic!("PointLayout of T does not match PointLayout of slice");
        }

        let length = slice.storage().len();
        Self {
            slice,
            current_index: 0,
            length,
            _phantom: Default::default(),
        }
    }
}

impl<T: PointType, S: PointSliceTyped<T>> Iterator for PointIteratorByValue<T, S> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index == self.length {
            None
        } else {
            let index = self.current_index;
            self.current_index += 1;
            Some(PointSliceTyped::at(&self.slice, index))
        }
    }
}

pub struct PointIteratorByRef<'a, T: PointType, S: PointSliceTypedByRef<T> + 'a> {
    slice: S,
    current_index: usize,
    length: usize,
    _phantom: PhantomData<(T, &'a S)>,
}

impl<'a, T: PointType, S: PointSliceTypedByRef<T> + 'a> From<S> for PointIteratorByRef<'a, T, S> {
    fn from(slice: S) -> Self {
        if T::layout() != *slice.point_layout() {
            panic!("PointLayout of T does not match PointLayout of slice");
        }

        let length = slice.storage().len();
        Self {
            slice,
            current_index: 0,
            length,
            _phantom: Default::default(),
        }
    }
}

impl<'a, T: PointType + 'a, S: PointSliceTypedByRef<T> + 'a> Iterator
    for PointIteratorByRef<'a, T, S>
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index == self.length {
            None
        } else {
            let index = self.current_index;
            self.current_index += 1;
            // This should be safe because we know that the iterator can never live longer than 'a, because
            // we constrained the type S with lifetime 'a. The implicit lifetime of '&mut self' is not known,
            // but it can never exceed 'a
            unsafe {
                let point_ref = PointSliceTypedByRef::at(&self.slice, index);
                Some(&*(point_ref as *const T))
            }
        }
    }
}

pub struct PointIteratorByMut<'a, T: PointType, S: PointSliceTypedByMut<T> + 'a> {
    slice: S,
    current_index: usize,
    length: usize,
    _phantom: PhantomData<(T, &'a mut S)>,
}

impl<'a, T: PointType, S: PointSliceTypedByMut<T> + 'a> From<S> for PointIteratorByMut<'a, T, S> {
    fn from(slice: S) -> Self {
        if T::layout() != *slice.point_layout() {
            panic!("PointLayout of T does not match PointLayout of slice");
        }

        let length = slice.storage().len();
        Self {
            slice,
            current_index: 0,
            length,
            _phantom: Default::default(),
        }
    }
}

impl<'a, T: PointType + 'a, S: PointSliceTypedByMut<T> + 'a> Iterator
    for PointIteratorByMut<'a, T, S>
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index == self.length {
            None
        } else {
            let index = self.current_index;
            self.current_index += 1;
            // This should be safe because we know that the iterator can never live longer than 'a, because
            // we constrained the type S with lifetime 'a. The implicit lifetime of '&mut self' is not known,
            // but it can never exceed 'a
            unsafe {
                let point_mut = self.slice.at_mut(index);
                Some(&mut *(point_mut as *mut T))
            }
        }
    }
}

pub struct AttributeIteratorByValue<T: PrimitiveType, S: AttributeSliceTyped<T>> {
    slice: S,
    current_index: usize,
    length: usize,
    _phantom: PhantomData<T>,
}

impl<T: PrimitiveType, S: AttributeSliceTyped<T>> From<S> for AttributeIteratorByValue<T, S> {
    fn from(slice: S) -> Self {
        let length = slice.len();
        Self {
            slice,
            current_index: 0,
            length,
            _phantom: Default::default(),
        }
    }
}

impl<T: PrimitiveType, S: AttributeSliceTyped<T>> Iterator for AttributeIteratorByValue<T, S> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index == self.length {
            None
        } else {
            let index = self.current_index;
            self.current_index += 1;
            Some(self.slice.at(index))
        }
    }
}

pub struct AttributeIteratorByRef<'a, T: PrimitiveType, S: AttributeSliceTypedByRef<T> + 'a> {
    slice: S,
    current_index: usize,
    length: usize,
    _phantom: PhantomData<(T, &'a S)>,
}

impl<'a, T: PrimitiveType, S: AttributeSliceTypedByRef<T> + 'a> From<S>
    for AttributeIteratorByRef<'a, T, S>
{
    fn from(slice: S) -> Self {
        let length = slice.len();
        Self {
            slice,
            current_index: 0,
            length,
            _phantom: Default::default(),
        }
    }
}

impl<'a, T: PrimitiveType + 'a, S: AttributeSliceTypedByRef<T> + 'a> Iterator
    for AttributeIteratorByRef<'a, T, S>
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index == self.length {
            None
        } else {
            let index = self.current_index;
            self.current_index += 1;
            // This should be safe because we know that the iterator can never live longer than 'a, because
            // we constrained the type S with lifetime 'a. The implicit lifetime of '&self' is not known,
            // but it can never exceed 'a
            unsafe {
                let attribute_ref = AttributeSliceTypedByRef::at(&self.slice, index);
                Some(&*(attribute_ref as *const T))
            }
        }
    }
}

pub struct AttributeIteratorByMut<'a, T: PrimitiveType, S: AttributeSliceTypedByMut<T> + 'a> {
    slice: S,
    current_index: usize,
    length: usize,
    _phantom: PhantomData<(T, &'a mut S)>,
}

impl<'a, T: PrimitiveType, S: AttributeSliceTypedByMut<T> + 'a> From<S>
    for AttributeIteratorByMut<'a, T, S>
{
    fn from(slice: S) -> Self {
        let length = slice.len();
        Self {
            slice,
            current_index: 0,
            length,
            _phantom: Default::default(),
        }
    }
}

impl<'a, T: PrimitiveType + 'a, S: AttributeSliceTypedByMut<T> + 'a> Iterator
    for AttributeIteratorByMut<'a, T, S>
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index == self.length {
            None
        } else {
            let index = self.current_index;
            self.current_index += 1;
            // This should be safe because we know that the iterator can never live longer than 'a, because
            // we constrained the type S with lifetime 'a. The implicit lifetime of '&self' is not known,
            // but it can never exceed 'a
            unsafe {
                let attribute_ref = AttributeSliceTypedByMut::at_mut(&mut self.slice, index);
                Some(&mut *(attribute_ref as *mut T))
            }
        }
    }
}
