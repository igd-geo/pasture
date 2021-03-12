use pasture_core::{
    attributes_mut,
    containers::PointBuffer,
    containers::{
        InterleavedPointBufferExt, InterleavedPointBufferMut, InterleavedPointBufferMutExt,
        InterleavedPointView, PerAttributePointBufferMut, PerAttributeVecPointStorage,
        PointBufferExt, PointBufferWriteable,
    },
    layout::attributes::COLOR_RGB,
    nalgebra::Vector3,
};
use pasture_core::{
    containers::InterleavedVecPointStorage,
    layout::{attributes::INTENSITY, PointType},
};
use pasture_derive::PointType;

#[repr(C)]
#[derive(PointType, Debug, Default)]
struct XYZIntensity {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: u16,
}

#[repr(C)]
#[derive(PointType, Debug, Default)]
struct XYZRGBIntensity {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_COLOR_RGB)]
    pub rgb: Vector3<u16>,
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: u16,
}

fn read_as_point_buffer<T: PointType + Default>() -> Box<dyn PointBuffer> {
    let mut buffer = InterleavedVecPointStorage::new(T::layout());
    buffer.push_points::<T>(&[Default::default(), Default::default()]);
    Box::new(buffer)
}

fn read_as_point_buffer_mut<T: PointType + Default>() -> Box<dyn PointBufferWriteable> {
    let mut buffer = InterleavedVecPointStorage::new(T::layout());
    buffer.push_points::<T>(&[Default::default(), Default::default()]);
    Box::new(buffer)
}

fn read_as_interleaved_buffer<T: PointType + Default>() -> Box<dyn InterleavedPointBufferMut> {
    let mut buffer = InterleavedVecPointStorage::new(T::layout());
    buffer.push_points::<T>(&[Default::default(), Default::default()]);
    Box::new(buffer)
}

fn read_as_per_attribute_buffer<T: PointType + Default>(
) -> Box<dyn PerAttributePointBufferMut<'static>> {
    let mut buffer = PerAttributeVecPointStorage::new(T::layout());
    buffer.push_points::<T>(&[Default::default(), Default::default()]);
    Box::new(buffer)
}

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

        //...the `PointLayout`...
        println!("Point layout: {}", poly_buffer.point_layout());

        //...and some low-level methods to access untyped point data
        let mut untyped_point_data =
            vec![0; poly_buffer.point_layout().size_of_point_entry() as usize];
        poly_buffer.get_raw_point(0, untyped_point_data.as_mut());

        // You will rarely have to use these low-level methods to access untyped data. Instead, even with a polymorphic
        // `PointBuffer`, we can use some of the helper methods that we have seen in the previous example:
        let points = poly_buffer.as_ref().iter_point::<XYZIntensity>();
        let point_with_max_intensity = points.max_by_key(|p| p.intensity).unwrap();
        println!(
            "Point with maximum intensity: {:?}",
            point_with_max_intensity
        );

        // Since `PointBuffer` is the most basic buffer trait, it makes no assumptions over the actual memory layout of
        // the point data that it stores. We don't know whether it is Interleaved or PerAttribute format. This means that
        // we can't use the `_ref`/`_mut` variants of the `points` and `attribute` helper methods. Iterating over strongly
        // typed points by reference requires that the data is stored in Interleaved format, whereas strongly typed attribute
        // references are only supported in PerAttribute format. So the following code won't compile:

        // let ref_points = containers::points_ref::<XYZIntensity>(poly_buffer.as_ref());

        // We will see shortly what can be done about that. For now, let's move on to the next, more powerful `PointBuffer` trait
    }

    {
        // The next trait in the hierarchy of `PointBuffer` traits is `PointBufferWriteable`:
        let mut poly_buffer_mut = read_as_point_buffer_mut::<XYZIntensity>();

        // Every type that implements `PointBufferWriteable` also implements `PointBuffer`, so the basic operations like
        // getting the number of points or the `PointLayout` are supported as well:
        println!("Number of points: {}", poly_buffer_mut.len());
        println!("Point layout: {}", poly_buffer_mut.point_layout());

        // While `PointBufferWriteable` still makes no assumptions about the memory layout, it does support writing point
        // data to it. We can push some points...
        poly_buffer_mut.push(&InterleavedPointView::from_slice(&[XYZIntensity {
            position: Vector3::new(1.0, 2.0, 3.0),
            intensity: 10,
        }]));

        // ...and we can overwrite some points...
        poly_buffer_mut.splice(
            0..1,
            &InterleavedPointView::from_slice(&[XYZIntensity {
                position: Vector3::new(2.0, 3.0, 4.0),
                intensity: 20,
            }]),
        );

        // The syntax is a bit verbose because we only have a trait object of our buffer, and we can't call generic methods
        // on trait objects, so both 'push' and 'splice' accept not a `&[T]` but instead a `&dyn PointBuffer`. To quickly get a
        // `PointBuffer` from a slice of strongly typed points, we were using the `InterleavedPointView` struct, which is a
        // wrapper that exposes a `&[T]` as a `PointBuffer`.

        // We can also clear the buffer, removing all points
        poly_buffer_mut.clear();
    }

    {
        // Lastly, there are even more specialized traits that define the actual memory layout of the buffer:
        // `InterleavedPointBuffer` and `PerAttributePointBuffer`. Both also come in a `...Mut` variant that provides
        // mutable access to the data. Let's take a closer look at those traits:

        let mut interleaved_buffer = read_as_interleaved_buffer::<XYZIntensity>();
        // `InterleavedPointBuffer` provides some low-level methods for accessing point data by reference, which should be rarely
        // used by users. Instead, we can use the `InterleavedPointBufferExt` extension trait to get an iterator over strongly
        // typed point references:
        for point_ref in interleaved_buffer.iter_point_ref::<XYZIntensity>() {
            println!("Point: {:?}", *point_ref);
        }

        // Iterating over point data by reference is substantially faster than by value! If we want to mutate our points, we can
        // use the `InterleavedPointBufferMutExt` trait, which provides the `iter_point_mut<T>` method:
        for point_mut in interleaved_buffer.iter_point_mut::<XYZIntensity>() {
            point_mut.intensity *= 2;
        }

        // As we have seen in the `basic_point_buffers.rs` example, a specific memory layout enables some optimizations, such as iterating
        // over certain attributes of the whole point data. We saw the `attribute` helper method for iterating over a single attribute, there
        // is also `attributes` for iterating over multiple attributes at once. Since Rust does not yet have variadic generics, `attributes` is
        // a macro instead:
        let mut per_attribute_buffer = read_as_per_attribute_buffer::<XYZRGBIntensity>();
        for (color_mut, intensity) in attributes_mut! {
            &COLOR_RGB => Vector3<u16>,
            &INTENSITY => u16,
            per_attribute_buffer.as_mut()
        } {
            // An easy way to calculate greyscale colors
            *color_mut = Vector3::new(*intensity, *intensity, *intensity);
        }

        // It is a bit harder to read, so here is an annotated version:
        // attributes_mut! {
        //  &COLOR_RGB => Vector3<u16>,     // Attribute (COLOR_RGB) as type (=>) Vector3<u16>
        //  &INTENSITY => u16,              // Attribute (INTENSITY) as type (=>) u16
        //  per_attribute_buffer.as_mut()   // Buffer to iterate over
        // }
    }
}
