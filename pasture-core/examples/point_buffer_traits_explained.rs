use nalgebra::Vector3;
use pasture_core::{
    containers::{
        BorrowedBuffer, BorrowedBufferExt, BorrowedMutBuffer, BorrowedMutBufferExt, ColumnarBuffer,
        ColumnarBufferMut, ExternalMemoryBuffer, HashMapBuffer, InterleavedBuffer,
        InterleavedBufferMut, OwningBuffer, VectorBuffer,
    },
    layout::{
        attributes::{INTENSITY, POSITION_3D},
        PointType,
    },
};
use pasture_derive::PointType;

#[repr(C, packed)]
#[derive(Copy, Clone, PointType, Debug, bytemuck::NoUninit, bytemuck::AnyBitPattern, PartialEq)]
struct SimplePoint {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: u16,
}

fn get_default_points() -> Vec<SimplePoint> {
    vec![
        SimplePoint {
            position: Vector3::new(1.0, 2.0, 3.0),
            intensity: 123,
        },
        SimplePoint {
            position: Vector3::new(4.0, 5.0, 6.0),
            intensity: 456,
        },
    ]
}

fn main() {
    // pasture defines a bunch of traits for point buffers. Some of these are explained implicitly in the
    // `basic_point_buffers` example. In this example, we will look at all the traits in more detail. In
    // addition you are encouraged to read the module documentation of `pasture_core::containers` as well

    let points = get_default_points();

    // pasture currently provides the following point buffer implementations:
    let mut vector_buffer: VectorBuffer = points.iter().copied().collect();
    let mut hashmap_buffer: HashMapBuffer = points.iter().copied().collect();

    let memory = vec![0; SimplePoint::layout().size_of_point_entry() as usize];
    let external_memory_buffer = ExternalMemoryBuffer::new(&memory, SimplePoint::layout());

    // What makes these buffers different, and when would you use which one? Let's ignore the `ExternalMemoryBuffer`
    // for now and focus on `VectorBuffer` and `HashMapBuffer`. If you look at the trait implementations for
    // `VectorBuffer` (https://docs.rs/pasture-core/latest/pasture_core/containers/struct.VectorBuffer.html#trait-implementations)
    // you will see that it implements many different traits with `Buffer` in their name. There are two hierarchies
    // of point buffer traits in pasture: One defines the memory ownership model of the buffer, the other defines the
    // memory layout of points within the buffer. First we will look at the memory ownership traits, starting with
    // `BorrowedBuffer`:

    // `BorrowedBuffer` is the most abstract trait. It makes no assumptions about the memory layout of points
    // and only assumes that the memory of the point buffer is borrowed somehow. Given this, how can we access point
    // data within a `BorrowedBuffer`? A point cloud in pasture is defined as a collection of tuples of attribute
    // values (where each tuple has the same attributes). So we need ways to access specific tuples (i.e. points) as
    // well as specific tuple elements (i.e. attributes of a point). This is precisely what `BorrowedBuffer` does,
    // as shown in the following code (using explicit trait function names instead of the dot operator for clarity):

    let mut memory_for_one_point: Vec<u8> =
        vec![0; SimplePoint::layout().size_of_point_entry() as usize];
    BorrowedBuffer::get_point(&vector_buffer, 0, &mut memory_for_one_point);
    BorrowedBuffer::get_point(&hashmap_buffer, 0, &mut memory_for_one_point);

    let mut memory_for_one_position: Vec<u8> = vec![0; POSITION_3D.size() as usize];
    BorrowedBuffer::get_attribute(
        &vector_buffer,
        &POSITION_3D,
        0,
        &mut memory_for_one_position,
    );
    BorrowedBuffer::get_attribute(
        &hashmap_buffer,
        &POSITION_3D,
        0,
        &mut memory_for_one_position,
    );

    // By design, all buffer traits in pasture work on raw binary data, typically in the form of byte slices (`[u8]`).
    // This enables handling point clouds where the number and types of point attributes is only known at runtime.
    // The `view` methods shown in the `basic_point_buffers` example provide more convenient ways to get point and
    // attribute data with strong typing, instead of byte slices. Under the hood, views use the raw `get_...` APIs from
    // the point buffer traits, such as `BorrowedBuffer`.

    // There are some caveats with the API from `BorrowedBuffer`:
    // 1) Accessing a point or an attribute of a point requires a copy into some buffer
    // 2) We can't mutate points or point attributes
    // These issues correspond to a lack of knowledge about the memory layout (1) and memory ownership (2) of the
    // point buffer. We require copy operations because `BorrowedBuffer` doesn't know what the actual memory layout
    // of the point data is, point attributes might be stored at non-adjacent memory locations, which might even be
    // unaligned, so getting a reference to point/attribute data is impossible.
    // Mutation is impossible because `BorrowedBuffer` assumes that the underlying memory is borrowed immutably!
    // Introduce `BorrowedMutBuffer`:

    let new_point = SimplePoint {
        position: Vector3::new(1.1, 2.2, 3.3),
        intensity: 555,
    };
    unsafe {
        let raw_memory_of_new_point = bytemuck::bytes_of(&new_point);
        BorrowedMutBuffer::set_point(&mut vector_buffer, 0, raw_memory_of_new_point);
    }

    let new_intensity: i16 = 1024;
    unsafe {
        let raw_memory_of_new_intensity = bytemuck::bytes_of(&new_intensity);
        BorrowedMutBuffer::set_attribute(
            &mut vector_buffer,
            &INTENSITY,
            0,
            raw_memory_of_new_intensity,
        );
    }

    // With `BorrowedMutBuffer`, we know that the underlying memory is borrowed mutably, so we can mutate the point
    // and attribute data. These functions also operate on byte slices, and in this case pasture can't check whether
    // the incoming byte slice contains valid memory, so these functions are unsafe! In principle, since all pasture
    // `PrimitiveType`s implement `bytemuck::AnyBitPattern`, it is not possible to create undefined behavior with
    // these `set_...` functions, so the unsafety is more of a marker to the user that care must be taken when using
    // these functions.

    // Notice that `BorrowedMutBuffer` has more strict guarantees about the memory ownership than `BorrowedBuffer`,
    // i.e. any type implementing `BorrowedMutBuffer` also implements `BorrowedBuffer`. The API of `BorrowedMutBuffer`
    // is still restricted, for example resizing of the memory is not supported. For this, there is the last trait in
    // the memory ownership hierarchy in pasture: `OwningBuffer`:

    let old_size = vector_buffer.len();
    unsafe {
        let bytes_of_new_point = bytemuck::bytes_of(&new_point);
        OwningBuffer::push_points(&mut vector_buffer, bytes_of_new_point);
    }
    assert_eq!(old_size + 1, vector_buffer.len());

    // Instead of using these raw APIs, using views is often the better choice, and there are corresponding functions
    // on all the views:

    {
        let point_view = hashmap_buffer.view::<SimplePoint>();
        assert_eq!(point_view.at(0), points[0]);

        let positions_view = hashmap_buffer.view_attribute::<Vector3<f64>>(&POSITION_3D);
        let expected_position = points[0].position;
        assert_eq!(positions_view.at(0), expected_position);
    }
    {
        let mut point_mut_view = hashmap_buffer.view_mut::<SimplePoint>();
        point_mut_view.set_at(0, points[1]);
        point_mut_view.push_point(points[1]);
    }

    // Now we look at the other hierarchy of point buffer traits, which relates to the memory layout of buffers.
    // Again the base trait is `BorrowedBuffer`, which makes no assumptions about the memory layout, which means
    // that point attributes can be stored at arbitrary addresses, or even computed on the fly. Beyond that, pasture
    // knows two specific memory layouts, called *interleaved* and *columnar*. This can be illustrated using the
    // `SimplePoint` type at the top of this file. It has two attributes: A position, as a `Vector3<f64>`, and an
    // intensity, as an `i16`. Given four points, the memory layouts will look like this:

    // Interleaved  : [p_1,i_1,p_2,i_2,p_3,i_3,p_4,i_4]
    // Columnar     : [p_1,p_2,p_3,p_4,i_1,i_2,i_3,i_4]

    // The interleaved memory layout stores all data for a single point together in memory, which makes it possible
    // to obtain a reference to memory for an individual point, or even for a range of points. This is the memory
    // layout you would expect a `Vec<SimplePoint>` to have.
    // The columnar memory layout stores all data for the same attribute together in memory, which makes it possible
    // to obtain a referecne to memory for an individual attribute of a point, or range of points.
    // The interleaved memory layout is sometimes called 'array-of-structs', whereas the columnar memory layout is
    // sometimes called 'struct-of-arrays', to illustrate how these layouts might be implemented in a C-like language.

    // pasture defines traits for buffers that guarantee a specific memory layout. The first is `InterleavedBuffer`:

    let _first_point: &[u8] = InterleavedBuffer::get_point_ref(&vector_buffer, 0);
    let _first_two_points: &[u8] = InterleavedBuffer::get_point_range_ref(&vector_buffer, 0..2);

    // `VectorBuffer` supports interleaved memory layout, so we can get references to the raw point memory without
    // any copying. This is also why it is possible to iterate over the points in an `InterleavedBuffer` by reference:

    for point_ref in vector_buffer.view::<SimplePoint>().iter() {
        println!("Point ref: {point_ref:?}");
    }

    // For mutating data, there is also `InterleavedBufferMut`:

    let _first_point_mut: &mut [u8] = InterleavedBufferMut::get_point_mut(&mut vector_buffer, 0);

    // Columnar memory layout buffers will implement `ColumnarBuffer`:

    let _first_position: &[u8] =
        ColumnarBuffer::get_attribute_ref(&hashmap_buffer, &POSITION_3D, 0);
    let _all_intensities: &[u8] = ColumnarBuffer::get_attribute_range_ref(
        &hashmap_buffer,
        &INTENSITY,
        0..hashmap_buffer.len(),
    );

    // `HashMapBuffer` supports columnar memory layout, so we can get references to the raw memory of a specific
    // attribute, or range of that attribute. Where we could iterate over points by reference in an `InterleavedBuffer`,
    // we can iterate over attributes by reference in a `ColumnarBuffer`:

    for position_ref in hashmap_buffer
        .view_attribute::<Vector3<f64>>(&POSITION_3D)
        .iter()
    {
        println!("Position ref: {position_ref}");
    }

    // There is also `ColumnarBufferMut` for mutating attribute values:

    let _first_position_mut: &mut [u8] =
        ColumnarBufferMut::get_attribute_mut(&mut hashmap_buffer, &POSITION_3D, 0);

    // Note that interleaved and columnar memory layouts are generally mutually exclusive. There are hypothetical edge
    // cases, such as a buffer holding just one point, but even then the attributes might be misaligned, preventing
    // access by reference to attribute memory. Unfortunately, the Rust language does not support negative trait bounds,
    // which make some code more complicated/less flexible than it could be. In particular we cannot do compile-time dispatch
    // based on the memory layout of a given buffer type. Instead, we have to do that at runtime, like this:

    fn accepts_any_buffer<'a, B: BorrowedBuffer<'a>>(buffer: &'a B) {
        // We can't statically dispatch to an implementation for `B: InterleavedBuffer` or `B: ColumnarBuffer`, but
        // we can use runtime polymorphism for this
        if let Some(interleaved) = buffer.as_interleaved() {
            for point in interleaved.view::<SimplePoint>().iter() {
                println!("{point:?}");
            }
        } else if let Some(columnar) = buffer.as_columnar() {
            for position in columnar.view_attribute::<Vector3<f64>>(&POSITION_3D).iter() {
                println!("{position}");
            }
        }
    }
    accepts_any_buffer(&vector_buffer);
    accepts_any_buffer(&hashmap_buffer);
    accepts_any_buffer(&external_memory_buffer);
}
