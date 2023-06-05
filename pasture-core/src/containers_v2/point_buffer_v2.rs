use std::{iter::FromIterator, ops::Range};

use crate::{
    layout::{PointAttributeDefinition, PointLayout, PointType, PrimitiveType},
    util::view_raw_bytes,
};

use super::{
    AttributeViewMut, AttributeViewRef, BufferStorage, BufferStorageMut, BufferViewMut,
    BufferViewRef, DefaultFromPointLayout, IndexBuffer, IndexBufferMut,
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
    fn at_ref(&self, index: usize) -> &T;
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

    pub fn for_point_type<U: PointType>() -> Self
    where
        T: DefaultFromPointLayout,
    {
        let layout = U::layout();
        Self::new(T::from_point_layout(&layout), layout)
    }

    pub fn from_layout(point_layout: PointLayout) -> Self
    where
        T: DefaultFromPointLayout,
    {
        Self::new(T::from_point_layout(&point_layout), point_layout)
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

    pub fn slice_mut<'a>(
        &'a mut self,
        range: Range<usize>,
    ) -> PointBuffer<<T as IndexBufferMut<'a>>::OutputMut>
    where
        T: IndexBufferMut<'a>,
    {
        if range.start >= range.end {
            panic!("empty range is not supported in call to slice()");
        }
        if range.end > self.len() {
            panic!("Range end is out of bounds");
        }
        PointBuffer::new(self.storage.index_mut(range), self.point_layout.clone())
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

    pub fn clear(&mut self) {
        self.storage.clear();
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

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use nalgebra::Vector3;
    use pasture_derive::PointType;
    use rand::{distributions::Standard, prelude::Distribution, thread_rng, Rng};

    use crate::{
        containers_v2::VectorStorage,
        layout::attributes::{GPS_TIME, INTENSITY, POSITION_3D},
    };

    use super::*;

    #[derive(PointType, Debug, Copy, Clone, PartialEq)]
    #[repr(C, packed)]
    struct ExamplePoint {
        #[pasture(BUILTIN_POSITION_3D)]
        position: Vector3<f64>,
        #[pasture(BUILTIN_INTENSITY)]
        intensity: u16,
        #[pasture(BUILTIN_GPS_TIME)]
        gps_time: f64,
    }

    impl PartialOrd for ExamplePoint {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            let self_intensity = self.intensity;
            let other_intensity = other.intensity;
            self_intensity.partial_cmp(&other_intensity)
        }
    }

    impl Distribution<ExamplePoint> for Standard {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> ExamplePoint {
            ExamplePoint {
                position: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
                intensity: rng.gen(),
                gps_time: rng.gen(),
            }
        }
    }

    fn reference_data<T: PointType>(count: usize) -> Vec<T>
    where
        Standard: Distribution<T>,
    {
        let mut rng = thread_rng();
        (0..count).map(|_| rng.gen::<T>()).collect()
    }

    #[test]
    fn test_point_buffer_default() {
        let mut buffer: PointBuffer<VectorStorage> = PointBuffer::for_point_type::<ExamplePoint>();
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.view::<ExamplePoint>().len(), 0);
        assert_eq!(buffer.view_mut::<ExamplePoint>().len(), 0);
        assert_eq!(buffer.point_layout(), &ExamplePoint::layout());

        assert_eq!(buffer.view_attribute::<Vector3<f64>>(&POSITION_3D).len(), 0);
        assert_eq!(
            buffer
                .view_attribute_mut::<Vector3<f64>>(&POSITION_3D)
                .len(),
            0
        );
        assert_eq!(buffer.view_attribute::<u16>(&INTENSITY).len(), 0);
        assert_eq!(buffer.view_attribute_mut::<u16>(&INTENSITY).len(), 0);
        assert_eq!(buffer.view_attribute::<f64>(&GPS_TIME).len(), 0);
        assert_eq!(buffer.view_attribute_mut::<f64>(&GPS_TIME).len(), 0);
    }

    #[test]
    fn test_point_buffer_from_iterator() {
        const COUNT: usize = 64;
        let expected_data = reference_data::<ExamplePoint>(COUNT);

        let buffer: PointBuffer<VectorStorage> = expected_data.iter().copied().collect();
        assert_eq!(buffer.len(), expected_data.len());
        assert_eq!(buffer.point_layout(), &ExamplePoint::layout());

        let actual_data = buffer
            .view::<ExamplePoint>()
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(expected_data, actual_data);
    }

    #[test]
    fn test_point_buffer_push() {
        let mut buffer: PointBuffer<VectorStorage> = PointBuffer::for_point_type::<ExamplePoint>();

        let example_point = ExamplePoint {
            gps_time: 1.0,
            intensity: 10,
            position: Vector3::new(2.0, 3.0, 4.0),
        };
        buffer.push(example_point);

        assert_eq!(buffer.len(), 1);
        assert_eq!(example_point, buffer.view::<ExamplePoint>().at(0));

        buffer.clear();
        assert_eq!(buffer.len(), 0);

        let many_points = reference_data::<ExamplePoint>(16);
        buffer.extend(many_points.iter().copied());
        assert_eq!(buffer.len(), many_points.len());

        let actual_points = buffer
            .view::<ExamplePoint>()
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(actual_points, many_points);
    }
}
