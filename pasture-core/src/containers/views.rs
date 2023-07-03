use std::{cell::RefCell, cmp::Ordering, marker::PhantomData, mem::MaybeUninit};

use crate::layout::{
    conversion::{get_converter_for_datatype_and_target_attribute, AttributeConversionFn},
    PointAttributeMember, PointLayout, PointType, PrimitiveType,
};

use super::{
    AttributeIteratorByMut, AttributeIteratorByRef, AttributeIteratorByValue, AttributeSliceTyped,
    AttributeSliceTypedByMut, AttributeSliceTypedByRef, BufferStorage, BufferStorageColumnar,
    BufferStorageColumnarMut, BufferStorageRowWise, BufferStorageRowWiseMut, PointIteratorByMut,
    PointIteratorByRef, PointIteratorByValue, PointSlice, PointSliceTyped, PointSliceTypedByMut,
    PointSliceTypedByRef,
};

pub struct BufferViewRef<'a, T: PointType, S: BufferStorage> {
    storage: &'a S,
    point_layout: &'a PointLayout,
    _phantom: PhantomData<(T, S)>,
}

impl<'a, T: PointType, S: BufferStorage> BufferViewRef<'a, T, S> {
    pub(crate) fn from_storage_and_layout(storage: &'a S, point_layout: &'a PointLayout) -> Self {
        if T::layout() != *point_layout {
            panic!("PointLayout of T does not match given PointLayout");
        }
        Self {
            storage,
            point_layout,
            _phantom: Default::default(),
        }
    }
}

impl<'a, T: PointType, S: BufferStorageRowWise> BufferViewRef<'a, T, S> {
    pub fn iter(self) -> PointIteratorByRef<'a, T, Self> {
        self.into()
    }
}

impl<'a, T: PointType, S: BufferStorage> PointSlice for BufferViewRef<'a, T, S> {
    fn point_layout(&self) -> &PointLayout {
        self.point_layout
    }

    fn storage(&self) -> &dyn BufferStorage {
        self.storage
    }
}

impl<'a, T: PointType, S: BufferStorage> PointSliceTyped<T> for BufferViewRef<'a, T, S> {
    fn at(&self, index: usize) -> T {
        let mut point = MaybeUninit::<T>::uninit();

        unsafe {
            self.storage.get(
                index,
                std::slice::from_raw_parts_mut(
                    point.as_mut_ptr() as *mut u8,
                    std::mem::size_of::<T>(),
                ),
            );
            point.assume_init()
        }
    }
}

impl<'a, T: PointType, S: BufferStorageRowWise> PointSliceTypedByRef<T>
    for BufferViewRef<'a, T, S>
{
    fn at_ref(&self, index: usize) -> &T {
        let raw_point = self.storage.get_ref(index);
        unsafe {
            let ptr = raw_point.as_ptr() as *const T;
            ptr.as_ref().expect("raw_point pointer was null")
        }
    }
}

impl<'a, T: PointType, S: BufferStorage> IntoIterator for BufferViewRef<'a, T, S> {
    type Item = T;
    type IntoIter = PointIteratorByValue<T, BufferViewRef<'a, T, S>>;

    fn into_iter(self) -> Self::IntoIter {
        self.into()
    }
}

pub struct BufferViewMut<'a, T: PointType, S: BufferStorage> {
    storage: &'a mut S,
    point_layout: &'a PointLayout,
    _phantom: PhantomData<T>,
}

impl<'a, T: PointType, S: BufferStorage> BufferViewMut<'a, T, S> {
    pub(crate) fn from_storage_and_layout(
        storage: &'a mut S,
        point_layout: &'a PointLayout,
    ) -> Self {
        if T::layout() != *point_layout {
            panic!("PointLayout of T does not match given PointLayout");
        }
        Self {
            storage,
            point_layout,
            _phantom: Default::default(),
        }
    }
}

impl<'a, T: PointType, S: BufferStorageRowWiseMut> BufferViewMut<'a, T, S> {
    pub fn iter_mut(self) -> PointIteratorByMut<'a, T, BufferViewMut<'a, T, S>> {
        self.into()
    }

    pub fn sort_by<C>(&mut self, compare: C)
    where
        C: Fn(&T, &T) -> Ordering,
    {
        // It is safe to call sort_by<T>, because `BufferViewMut` guarantees that `T::point_layout` and
        // the point layout of the storage match (see BufferViewMut::from_storage_and_layout)
        unsafe {
            self.storage.sort_by(compare);
        }
    }
}

impl<'a, T: PointType, S: BufferStorage> PointSlice for BufferViewMut<'a, T, S> {
    fn point_layout(&self) -> &PointLayout {
        self.point_layout
    }

    fn storage(&self) -> &dyn BufferStorage {
        self.storage
    }
}

impl<'a, T: PointType, S: BufferStorage> PointSliceTyped<T> for BufferViewMut<'a, T, S> {
    fn at(&self, index: usize) -> T {
        let mut point = MaybeUninit::<T>::uninit();

        unsafe {
            self.storage.get(
                index,
                std::slice::from_raw_parts_mut(
                    point.as_mut_ptr() as *mut u8,
                    std::mem::size_of::<T>(),
                ),
            );
            point.assume_init()
        }
    }
}

impl<'a, T: PointType, S: BufferStorageRowWise> PointSliceTypedByRef<T>
    for BufferViewMut<'a, T, S>
{
    fn at_ref(&self, index: usize) -> &T {
        let raw_point = self.storage.get_ref(index);
        unsafe {
            let ptr = raw_point.as_ptr() as *const T;
            ptr.as_ref().expect("raw_point pointer was null")
        }
    }
}

impl<'a, T: PointType, S: BufferStorageRowWiseMut> PointSliceTypedByMut<T>
    for BufferViewMut<'a, T, S>
{
    fn at_mut(&mut self, index: usize) -> &mut T {
        let raw_point = self.storage.get_mut(index);
        unsafe {
            let ptr = raw_point.as_ptr() as *mut T;
            ptr.as_mut().expect("raw_point pointer was null")
        }
    }
}

impl<'a, T: PointType, S: BufferStorage> IntoIterator for BufferViewMut<'a, T, S> {
    type Item = T;
    type IntoIter = PointIteratorByValue<T, BufferViewMut<'a, T, S>>;

    fn into_iter(self) -> Self::IntoIter {
        self.into()
    }
}

pub struct AttributeViewRef<'a, T: PrimitiveType, B: BufferStorage> {
    storage: &'a B,
    attribute: &'a PointAttributeMember,
    _phantom: PhantomData<T>,
}

impl<'a, T: PrimitiveType, B: BufferStorage> AttributeViewRef<'a, T, B> {
    pub(crate) fn from_storage_and_attribute(
        storage: &'a B,
        attribute: &'a PointAttributeMember,
    ) -> Self {
        Self {
            storage,
            attribute,
            _phantom: Default::default(),
        }
    }
}

impl<'a, T: PrimitiveType, B: BufferStorageColumnar> AttributeViewRef<'a, T, B> {
    pub fn iter(self) -> AttributeIteratorByRef<'a, T, Self> {
        self.into()
    }
}

impl<'a, T: PrimitiveType, B: BufferStorage> AttributeSliceTyped<T> for AttributeViewRef<'a, T, B> {
    fn at(&self, index: usize) -> T {
        let mut attribute = MaybeUninit::<T>::uninit();

        unsafe {
            self.storage.get_attribute(
                self.attribute,
                index,
                std::slice::from_raw_parts_mut(
                    attribute.as_mut_ptr() as *mut u8,
                    std::mem::size_of::<T>(),
                ),
            );
            attribute.assume_init()
        }
    }

    fn len(&self) -> usize {
        self.storage.len()
    }
}

impl<'a, T: PrimitiveType, B: BufferStorageColumnar> AttributeSliceTypedByRef<T>
    for AttributeViewRef<'a, T, B>
{
    fn at(&self, index: usize) -> &T {
        let raw_attribute = self.storage.get_attribute_ref(self.attribute, index);
        unsafe {
            let attribute: *const T = raw_attribute.as_ptr() as *const T;
            attribute.as_ref().expect("raw_attribute pointer was null")
        }
    }
}

impl<'a, T: PrimitiveType, B: BufferStorage> IntoIterator for AttributeViewRef<'a, T, B> {
    type Item = T;
    type IntoIter = AttributeIteratorByValue<T, AttributeViewRef<'a, T, B>>;

    fn into_iter(self) -> Self::IntoIter {
        self.into()
    }
}

pub struct AttributeViewMut<'a, T: PrimitiveType, B: BufferStorage> {
    storage: &'a mut B,
    attribute: &'a PointAttributeMember,
    _phantom: PhantomData<T>,
}

impl<'a, T: PrimitiveType, B: BufferStorage> AttributeViewMut<'a, T, B> {
    pub(crate) fn from_storage_and_attribute(
        storage: &'a mut B,
        attribute: &'a PointAttributeMember,
    ) -> Self {
        Self {
            storage,
            attribute,
            _phantom: Default::default(),
        }
    }
}

impl<'a, T: PrimitiveType, B: BufferStorageColumnarMut> AttributeViewMut<'a, T, B> {
    pub fn iter_mut(self) -> AttributeIteratorByMut<'a, T, AttributeViewMut<'a, T, B>> {
        self.into()
    }
}

impl<'a, T: PrimitiveType, B: BufferStorage> AttributeSliceTyped<T> for AttributeViewMut<'a, T, B> {
    fn at(&self, index: usize) -> T {
        let mut attribute = MaybeUninit::<T>::uninit();

        unsafe {
            self.storage.get_attribute(
                self.attribute,
                index,
                std::slice::from_raw_parts_mut(
                    attribute.as_mut_ptr() as *mut u8,
                    std::mem::size_of::<T>(),
                ),
            );
            attribute.assume_init()
        }
    }

    fn len(&self) -> usize {
        self.storage.len()
    }
}

impl<'a, T: PrimitiveType, B: BufferStorageColumnar> AttributeSliceTypedByRef<T>
    for AttributeViewMut<'a, T, B>
{
    fn at(&self, index: usize) -> &T {
        let raw_attribute = self.storage.get_attribute_ref(self.attribute, index);
        unsafe {
            let attribute: *const T = raw_attribute.as_ptr() as *const T;
            attribute.as_ref().expect("raw_attribute pointer was null")
        }
    }
}

impl<'a, T: PrimitiveType, B: BufferStorageColumnarMut> AttributeSliceTypedByMut<T>
    for AttributeViewMut<'a, T, B>
{
    fn at_mut(&mut self, index: usize) -> &mut T {
        let raw_attribute = self.storage.get_attribute_mut(self.attribute, index);
        unsafe {
            let attribute: *mut T = raw_attribute.as_ptr() as *mut T;
            attribute.as_mut().expect("raw_attribute pointer was null")
        }
    }
}

impl<'a, T: PrimitiveType, B: BufferStorage> IntoIterator for AttributeViewMut<'a, T, B> {
    type Item = T;
    type IntoIter = AttributeIteratorByValue<T, AttributeViewMut<'a, T, B>>;

    fn into_iter(self) -> Self::IntoIter {
        self.into()
    }
}

pub struct AttributeViewConverting<'a, T: PrimitiveType, B: BufferStorage> {
    storage: &'a mut B,
    attribute: &'a PointAttributeMember,
    converter: AttributeConversionFn,
    copy_buffer: RefCell<Vec<u8>>,
    _phantom: PhantomData<T>,
}

impl<'a, T: PrimitiveType, B: BufferStorage> AttributeViewConverting<'a, T, B> {
    pub(crate) fn from_storage_and_attribute(
        storage: &'a mut B,
        attribute: &'a PointAttributeMember,
    ) -> Option<Self> {
        let converter = get_converter_for_datatype_and_target_attribute(
            attribute.datatype(),
            attribute.name(),
            T::data_type(),
        )?;
        Some(Self {
            storage,
            attribute,
            converter,
            copy_buffer: RefCell::new(vec![0; attribute.size() as usize]),
            _phantom: Default::default(),
        })
    }
}

impl<'a, T: PrimitiveType, B: BufferStorage> AttributeSliceTyped<T>
    for AttributeViewConverting<'a, T, B>
{
    fn at(&self, index: usize) -> T {
        let mut copy_buffer = self.copy_buffer.borrow_mut();
        self.storage
            .get_attribute(self.attribute, index, copy_buffer.as_mut_slice());
        let mut attribute = MaybeUninit::<T>::uninit();
        unsafe {
            (self.converter)(
                copy_buffer.as_slice(),
                std::slice::from_raw_parts_mut(
                    attribute.as_mut_ptr() as *mut u8,
                    std::mem::size_of::<T>(),
                ),
            );
            attribute.assume_init()
        }
    }

    fn len(&self) -> usize {
        self.storage.len()
    }
}

#[cfg(test)]
mod tests {
    use std::iter::FromIterator;

    use nalgebra::Vector3;
    use pasture_derive::PointType;
    use rand::{thread_rng, Rng};

    use crate::containers::{PointBuffer, VectorStorage};

    use super::*;

    #[derive(PointType, Copy, Clone, Debug)]
    #[repr(C, packed)]
    struct TestPoint {
        #[pasture(BUILTIN_POSITION_3D)]
        pub position: Vector3<f64>,
        #[pasture(BUILTIN_INTENSITY)]
        pub intensity: i16,
        #[pasture(BUILTIN_GPS_TIME)]
        pub gps_time: f64,
    }

    fn gen_random_points<S: BufferStorage + FromIterator<TestPoint>>(
        count: usize,
    ) -> PointBuffer<S> {
        let mut rng = thread_rng();
        (0..count)
            .map(|_| TestPoint {
                gps_time: rng.gen(),
                intensity: rng.gen(),
                position: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
            })
            .collect()
    }

    #[test]
    fn test_sort() {
        let mut test_points = gen_random_points::<VectorStorage>(1024);

        test_points.view_mut::<TestPoint>().sort_by(|a, b| {
            let intensity_a = a.intensity;
            let intensity_b = b.intensity;
            intensity_a.cmp(&intensity_b)
        });

        for (low, high) in test_points
            .view::<TestPoint>()
            .iter()
            .zip(test_points.view::<TestPoint>().iter().skip(1))
        {
            let low_intensity = low.intensity;
            let high_intensity = high.intensity;
            assert!(low_intensity <= high_intensity);
        }
    }
}
