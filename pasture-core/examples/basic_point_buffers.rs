use pasture_core::containers::{
    BorrowedBuffer, BorrowedMutBuffer, HashMapBuffer, MakeBufferFromLayout, SliceBuffer,
    SliceBufferMut,
};
use pasture_core::nalgebra::Vector3;
use pasture_core::{
    containers::VectorBuffer,
    layout::{
        attributes::{INTENSITY, POSITION_3D},
        PointType,
    },
};
use pasture_derive::PointType;

/// We define a simple point type here that has two attributes: 3D position and intensity
#[repr(C, packed)]
#[derive(Copy, Clone, PointType, Debug, bytemuck::NoUninit, bytemuck::AnyBitPattern)]
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

    // By default, our data is in interleaved format, because a struct is a form of interleaved data. So
    // let's create a buffer to hold our points:
    {
        let mut buffer = VectorBuffer::new_from_layout(SimplePoint::layout());
        // We can add interleaved data like so:
        buffer.view_mut().push_point(points[0]);
        buffer.view_mut().push_point(points[1]);

        // More elegant is to collect from an iterator:
        buffer = points.iter().copied().collect::<VectorBuffer>();

        println!("Iterating over interleaved points:");
        // The buffer itself is not strongly typed, but we already saw the `view_mut` method to get a strongly typed
        // view of our buffer. If we don't need mutable access, `view` is sufficient. The view implements `IntoIterator`
        // so that we can iterate over strongly typed points:
        for point in buffer.view::<SimplePoint>() {
            println!("{:?}", point);
        }

        // The iterator over a regular view returns points by value, similar to how iterating over a `Vec` value gives
        // an iterator by value. If we want to iterate by reference, we can use the `iter` function on the view:
        for point_ref in buffer.view::<SimplePoint>().iter() {
            println!("{:?}", point_ref);
        }

        // Using `iter_mut`, we can also mutate our point data. This requires a mutable view, so we have to use `view_mut`
        // as well:
        for point_mut in buffer.view_mut::<SimplePoint>().iter_mut() {
            point_mut.intensity *= 2;
        }

        // Note that iterating by (mutable) reference only works because our `buffer` has interleaved memory layout! This
        // is what 'interleaved' means: We can get (mutable) references to the strongly typed point data, since all data
        // for a single point is stored contiguously in memory!

        // Just like arrays and vectors, our buffers can also be sliced. Unfortunately, the current constraints of the `Index`
        // trait prevent us from implementing it for the pasture point buffers, so we can't slice our buffers using the
        // `[range]` syntax. Instead, use the `slice` and `slice_mut` methods:
        println!("Iterating over interleaved points slice:");
        let sliced = buffer.slice(1..2);
        for point in sliced.view::<SimplePoint>() {
            println!("{:?}", point);
        }

        let mut sliced_mut = buffer.slice_mut(1..2);
        for point_mut in sliced_mut.view_mut::<SimplePoint>().iter_mut() {
            point_mut.intensity *= 2;
        }
    }

    // There are several different types of point buffers. Most code in Pasture can deal with any of these buffer types, though
    // sometimes this is not possible due to memory layout concerns or general performance.
    // Let's try a different type of buffer:
    {
        let mut buffer = HashMapBuffer::new_from_layout(SimplePoint::layout());
        // This buffer stores points with a different memory layout internally (Columnar as opposed to Interleaved). We can
        // still add our strongly typed points to it:
        buffer.view_mut().push_point(points[0]);
        buffer.view_mut().push_point(points[1]);

        // ...or collect it from an iterator of strongly typed points:
        buffer = points.into_iter().collect::<HashMapBuffer>();

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
            // Notice that `view_attribute<T>` converts to an iterator that returns `T` by value.
            // It is available for all point buffer types, at the expense of only receiving a copy of the attribute.
            println!("Position: {:?}", position);
        }

        // There are several builtin attribute specifiers in the namespace `pasture_core::layout::attributes`.These are the ones that
        // are used when you `#[derive(PointType)]` and say `#[pasture(BUILTIN_XYZ)]`. An attribute specifier internally uses a unique
        // name to identify the attribute, as well as the default datatype of the attribute. Using the builtin specifiers guarantees that
        // all attributes are always correctly addressed.

        // Let's try mutating a specific attribute. This is only possible for a buffer that stores data in columnar memory layout.
        // To mutate the data, we use the `view_attribute_mut` function, together with `iter_mut`
        for intensity in buffer.view_attribute_mut::<u16>(&INTENSITY).iter_mut() {
            *intensity *= 2;
        }

        // Just as with the Interleaved buffer, we can slice:
        println!("Iterating over columnar point slice:");
        let sliced = buffer.slice(1..2);
        for point in sliced.view::<SimplePoint>() {
            println!("{:?}", point);
        }
    }
}
