use pasture_core::{
    containers::{ColumnarStorage, PointBuffer, VectorStorage},
    layout::attributes::{INTENSITY, POSITION_3D},
    nalgebra::Vector3,
};
use pasture_derive::PointType;

/// We define a simple point type here that has two attributes: 3D position and intensity
#[repr(C)]
#[derive(PointType, Debug, Clone, Copy)]
struct SimplePoint {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: u16,
}

fn main() {
    // Create some points
    let points = vec![
        SimplePoint {
            position: Vector3::new(1.0, 2.0, 3.0),
            intensity: 42,
        },
        SimplePoint {
            position: Vector3::new(-1.0, -2.0, -3.0),
            intensity: 84,
        },
    ];

    // By default, our data is in row-wise format, because a struct is a form of row-wise data. So
    // let's create a buffer to hold our points:
    {
        // The simplest row-wise storage is a vector:
        let mut buffer: PointBuffer<VectorStorage> = PointBuffer::for_point_type::<SimplePoint>();
        // We can add point data like so:
        buffer.extend(points.iter().copied());

        println!("Iterating over interleaved points:");
        // The buffer itself is not strongly typed, but there are some helper methods to create *views* over strongly typed data
        for point in buffer.view::<SimplePoint>() {
            println!("{:?}", point);
        }

        // PointBuffer::view by default iterates over points by value. If we want to mutate the points, we can use PointBuffer::view_mut,
        // but we also have to explicitly state that we want a mutable iterator
        for point_mut in buffer.view_mut::<SimplePoint>().iter_mut() {
            point_mut.intensity *= 2;
        }

        // We can also directly slice our buffer (also see the docs of the `slice` method which explains the syntax)
        println!("Iterating over interleaved points slice:");
        let sliced = buffer.slice(1..2);
        for point in sliced.view::<SimplePoint>() {
            println!("{:?}", point);
        }
    }

    // There are several different types of point buffers. Most code in pasture can deal with any of these buffer types, though
    // sometimes this is not possible due to memory layout concerns or general performance.
    // Let's try a different type of buffer:
    {
        // Instead of storing data row-wise, we can store it in a column-oriented format:
        let mut buffer: PointBuffer<ColumnarStorage> = PointBuffer::for_point_type::<SimplePoint>();
        // We can still add row-wise (i.e. struct) data to this buffer
        buffer.extend(points.iter().copied());

        //... and iterate it:
        println!("Iterating over columnar points:");
        for point in buffer.view::<SimplePoint>() {
            println!("{:?}", point);
        }

        // With the columnar memory layout, we can iterate over specific attributes and even mutate them, instead of always
        // iterating over the whole point. This can give better performance in many cases.
        // As the buffer is not strongly typed, we need to specify the type of the attribute, similar to the call to `view<T>`
        // before. In addition, we have to give Pasture an 'attribute specifier' to determine which attribute we want:
        println!("Iterating over a single attribute:");
        for position in buffer.view_attribute::<Vector3<f64>>(&POSITION_3D) {
            // Notice that `view_attribute<T>` returns `T` by value. It is available for all point buffer types, at the expense of
            // only receiving a copy of the attribute.
            println!("Position: {:?}", position);
        }

        // There are several builtin attribute specifiers in the namespace `pasture_core::layout::attributes`.These are the ones that
        // are used when you `#[derive(PointType)]` and say `#[pasture(BUILTIN_XYZ)]`. An attribute specifier internally uses a unique
        // name to identify the attribute, as well as the default datatype of the attribute. Using the builtin specifiers guarantees that
        // all attributes are always correctly addressed.

        // Let's try mutating a specific attribute. This is only possible for a buffer that stores data in columnar memory layout
        for intensity in buffer.view_attribute_mut::<u16>(&INTENSITY).iter_mut() {
            *intensity *= 2;
        }

        // Just as with the row-wise buffer, we can slice:
        println!("Iterating over columnar point slice:");
        let sliced = buffer.slice(1..2);
        for point in sliced.view::<SimplePoint>() {
            println!("{:?}", point);
        }
    }
}
