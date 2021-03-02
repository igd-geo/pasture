use pasture_core::{
    containers::{self, PerAttributeVecPointStorage, PointBuffer, PointBufferWriteable},
    nalgebra::Vector3,
};
use pasture_core::{
    containers::{InterleavedVecPointStorage, PerAttributePointBuffer},
    layout::{
        attributes::{INTENSITY, POSITION_3D},
        PointType,
    },
};
use pasture_derive::PointType;

#[repr(C)]
#[derive(PointType, Debug)]
struct SimplePoint {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: u16,
}

fn read_as_point_buffer() -> Box<dyn PointBuffer> {
    todo!()
}

fn read_as_point_buffer_mut() -> Box<dyn PointBufferWriteable> {
    todo!()
}

fn main() {
    // In the previous example `basic_point_buffers.rs`, we knew the specific type of `PointBuffer` that we were
    // working with. If you can do that, this option is preferred, however there are cases where you will have to
    // work with a polymorphic `PointBuffer` type. Let's see how that works:

    {
        // One example where you might get a polymorphic `PointBuffer` is by using one of the readers in the `pasture-io`
        // crate. We emulate this behaviour here:
        let poly_buffer = read_as_point_buffer();

        //`PointBuffer` is the most basic trait. It supports getting the number of points in the buffer...
        println!("Number of points: {}", poly_buffer.len());

        //...the `PointLayout`...
        println!("Point layout: {}", poly_buffer.point_layout());

        //...and some low-level methods to access untyped point data
        let mut untyped_point_data =
            vec![0; poly_buffer.point_layout().size_of_point_entry() as usize];
        poly_buffer.get_point_by_copy(0, untyped_point_data.as_mut());

        // You will rarely have to use these low-level methods to access untyped data. Instead, even with a polymorphic
        // `PointBuffer`, we can use some of the helper methods that we have seen in the previous example:
        let points = containers::points::<SimplePoint>(poly_buffer.as_ref());
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

        // let ref_points = containers::points_ref::<SimplePoint>(poly_buffer.as_ref());

        // We will see shortly what can be done about that. For now, let's move on to the next, more powerful `PointBuffer` trait
    }

    {
        // The next trait in the hierarchy of `PointBuffer` traits is `PointBufferWriteable`:
        let poly_buffer_mut = read_as_point_buffer_mut();

        // Every type that implements `PointBufferWriteable` also implements `PointBuffer`, so the basic operations like
        // getting the number of points or the `PointLayout` are supported as well:
        println!("Number of points: {}", poly_buffer_mut.len());
        println!("Point layout: {}", poly_buffer_mut.point_layout());

        // While `PointBufferWriteable` still makes no assumptions about the memory layout, it does support writing point
        // data to it
        //poly_buffer_mut.push()

        // TODO And here we have the first inconsistency (but its easy to fix): PointBufferWriteable should have not only
        // 'push' operations, but also 'insert' operations so that we can overwrite points at existing indices
        // Also, it is confusing that there is 'push' but also 'extend' which take slightly different types!
        // TODO 'extend_from_...' could be a single 'extend' method that takes a 'dyn PointBuffer' and uses that 'as_interleaved'/'as_per_attribute'
        // methods internally
        //
        // TODO Figure out what I have to implement so that I can 'collect' the point/attribute iterators into PointBuffers
    }
}
