use pasture_core::containers::{
    PointSlice, PolymorphicStorage, PolymorphicStorageMut, VectorStorage,
};
use pasture_core::layout::PointType;
use pasture_core::{containers::PointBuffer, nalgebra::Vector3};
use pasture_derive::PointType;

#[repr(C)]
#[derive(PointType, Debug, Default, Clone)]
struct XYZIntensity {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: u16,
}

#[repr(C)]
#[derive(PointType, Debug, Default, Clone)]
struct XYZRGBIntensity {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_COLOR_RGB)]
    pub rgb: Vector3<u16>,
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: u16,
}

fn read_as_point_buffer<T: PointType + Default + Clone>() -> PointBuffer<PolymorphicStorage> {
    let layout = T::layout();
    let mut buffer = PointBuffer::new(VectorStorage::from_layout(&layout), layout);
    buffer.extend(std::iter::repeat(<T as Default>::default()).take(2));
    buffer.into_polymorphic_buffer()
}

fn read_as_point_buffer_mut<T: PointType + Default + Clone>() -> PointBuffer<PolymorphicStorageMut>
{
    let layout = T::layout();
    let mut buffer = PointBuffer::new(VectorStorage::from_layout(&layout), layout);
    buffer.extend(std::iter::repeat(<T as Default>::default()).take(2));
    buffer.into_polymorphic_buffer_mut()
}

// fn read_as_interleaved_buffer<T: PointType + Default>() -> Box<dyn InterleavedPointBufferMut> {
//     let mut buffer = InterleavedVecPointStorage::new(T::layout());
//     buffer.push_points::<T>(&[Default::default(), Default::default()]);
//     Box::new(buffer)
// }

// fn read_as_per_attribute_buffer<T: PointType + Default>(
// ) -> Box<dyn PerAttributePointBufferMut<'static>> {
//     let mut buffer = PerAttributeVecPointStorage::new(T::layout());
//     buffer.push_points::<T>(&[Default::default(), Default::default()]);
//     Box::new(buffer)
// }

fn main() {
    // In the previous example `basic_point_buffers.rs`, we knew the specific type of `PointBuffer` that we were
    // working with. If you can do that, this option is preferred, however there are cases where you will have to
    // work with a polymorphic `PointBuffer` type. Let's see how that works:

    {
        // One example where you might get a polymorphic `PointBuffer` is by using one of the readers in the `pasture-io`
        // crate. We emulate this behaviour here:
        let poly_buffer = read_as_point_buffer::<XYZIntensity>();

        //`PointBuffer` is the most basic trait. It supports getting the number of points in the buffer...
        println!("Number of points: {}", poly_buffer.len());

        //...and the `PointLayout`...
        println!("Point layout: {}", poly_buffer.point_layout());

        //...and has the usual accessor methods to view the buffer as strongly typed points
        let points = poly_buffer.view::<XYZIntensity>();
        let point_with_max_intensity = points.into_iter().max_by_key(|p| p.intensity).unwrap();
        println!(
            "Point with maximum intensity: {:?}",
            point_with_max_intensity
        );

        // With this polymorphic buffer type, we unfortunately don't know what the underlying memory layout is, so we can't
        // iterate over points by reference for example, so this code won't compile:
        // poly_buffer.view::<XYZIntensity>().iter();

        // We also can't mutate the buffer, because `PolymorphicStorage` only implements `BufferStorage` but not `BufferStorageMut`
        // The next example shows how to mutate the data!
    }

    {
        // There is also a `PolymorphicStorageMut` that allows us to modify the contents of a polymorphic point buffer:
        let mut poly_buffer_mut = read_as_point_buffer_mut::<XYZIntensity>();

        println!("Number of points: {}", poly_buffer_mut.len());
        println!("Point layout: {}", poly_buffer_mut.point_layout());

        // We can use the typical methods that the `BufferStorageMut` trait allows, so pushing points, extending the
        // buffer, clearing it etc.
        poly_buffer_mut.push(XYZIntensity {
            position: Vector3::new(1.0, 2.0, 3.0),
            intensity: 10,
        });
        poly_buffer_mut.extend(std::iter::repeat(XYZIntensity::default()).take(16));
        poly_buffer_mut.clear();
    }

    // With the current pasture version, polymorphic buffers with specific memory layouts are not yet supported
}
