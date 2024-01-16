use nalgebra::Vector3;
use pasture_core::{
    containers::{
        BorrowedBuffer, BorrowedBufferExt, BorrowedMutBuffer, BorrowedMutBufferExt, ColumnarBuffer,
        ColumnarBufferMut, HashMapBuffer, InterleavedBuffer, InterleavedBufferMut, SliceBuffer,
        SliceBufferMut, VectorBuffer,
    },
    layout::attributes::{INTENSITY, POSITION_3D},
};
use pasture_derive::PointType;

#[repr(C, packed)]
#[derive(Copy, Clone, PointType, Debug, bytemuck::NoUninit, bytemuck::AnyBitPattern)]
struct SimplePoint {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: u16,
}

fn test_points() -> Vec<SimplePoint> {
    vec![
        SimplePoint {
            intensity: 1,
            position: Vector3::new(1.0, 1.0, 1.0),
        },
        SimplePoint {
            intensity: 2,
            position: Vector3::new(2.0, 2.0, 2.0),
        },
        SimplePoint {
            intensity: 3,
            position: Vector3::new(3.0, 3.0, 3.0),
        },
        SimplePoint {
            intensity: 4,
            position: Vector3::new(4.0, 4.0, 4.0),
        },
    ]
}

fn print_points_by_value<'a, B: BorrowedBufferExt<'a> + ?Sized>(buffer: &'a B) {
    for point in buffer.view::<SimplePoint>() {
        println!("{point:?}");
    }
}

fn print_points_by_ref<'a, B: BorrowedBufferExt<'a> + InterleavedBuffer<'a> + ?Sized>(
    buffer: &'a B,
) {
    for point in buffer.view::<SimplePoint>().iter() {
        println!("{point:?}");
    }
}

fn print_positions_by_value<'a, B: BorrowedBufferExt<'a> + ?Sized>(buffer: &'a B) {
    for position in buffer.view_attribute::<Vector3<f64>>(&POSITION_3D) {
        println!("{position}");
    }
}

fn print_positions_by_ref<'a, B: BorrowedBufferExt<'a> + ColumnarBuffer<'a> + ?Sized>(
    buffer: &'a B,
) {
    for position in buffer.view_attribute::<Vector3<f64>>(&POSITION_3D).iter() {
        println!("{position}");
    }
}

fn slice_vector_buffer() {
    let buffer: VectorBuffer = test_points().into_iter().collect();

    // Buffers can be sliced by using `.slice(range), similar to `[range]`
    let first_two_points = buffer.slice(0..2);
    print_points_by_ref(&first_two_points);

    // Slices of slices are possible
    let slice_of_slice = first_two_points.slice(0..1);
    print_points_by_ref(&slice_of_slice);

    // Slices are also possible if the buffer is a trait object
    let buffer_trait_object: &dyn InterleavedBuffer = &buffer;
    let first_two_points_dynamic = buffer_trait_object.slice(0..2);
    print_points_by_ref(&first_two_points_dynamic);

    // Slices of trait objects retain their memory layout properties. If the trait object has an unknown memory
    // layout, so does the slice, and we can only view points/attributes by value
    let buffer_trait_object_unknown_memory_layout: &dyn BorrowedBuffer = &buffer;
    let first_two_points_dynamic_unknown_memory_layout =
        buffer_trait_object_unknown_memory_layout.slice(0..2);
    print_points_by_value(&first_two_points_dynamic_unknown_memory_layout);
}

fn slice_vector_buffer_mut() {
    let mut buffer: VectorBuffer = test_points().into_iter().collect();

    let mut first_two_points_mut = buffer.slice_mut(0..2);
    for point in first_two_points_mut.view_mut::<SimplePoint>().iter_mut() {
        *point = test_points()[0];
    }

    let mut slice_of_slice_mut = first_two_points_mut.slice_mut(0..1);
    for point in slice_of_slice_mut.view_mut::<SimplePoint>().iter_mut() {
        *point = test_points()[1];
    }

    let buffer_trait_object_mut: &mut dyn InterleavedBufferMut = &mut buffer;
    let mut first_two_points_dynamic_mut = buffer_trait_object_mut.slice_mut(0..2);
    for point in first_two_points_dynamic_mut
        .view_mut::<SimplePoint>()
        .iter_mut()
    {
        *point = test_points()[2];
    }

    let buffer_trait_object_unknown_memory_layout: &mut dyn BorrowedMutBuffer = &mut buffer;
    let mut first_two_points_dynamic_unknown_memory_layout_mut =
        buffer_trait_object_unknown_memory_layout.slice_mut(0..2);
    let mut view = first_two_points_dynamic_unknown_memory_layout_mut.view_mut::<SimplePoint>();
    view.set_at(0, test_points()[2]);
}

fn slice_hashmap_buffer() {
    let buffer: HashMapBuffer = test_points().into_iter().collect();

    let first_two_points = buffer.slice(0..2);
    print_positions_by_ref(&first_two_points);

    let slice_of_slice = first_two_points.slice(0..1);
    print_positions_by_ref(&slice_of_slice);

    let buffer_trait_object: &dyn ColumnarBuffer = &buffer;
    let first_two_points_dynamic = buffer_trait_object.slice(0..2);
    print_positions_by_ref(&first_two_points_dynamic);

    let buffer_trait_object_unknown_memory_layout: &dyn BorrowedBuffer = &buffer;
    let first_two_points_unknown_memory_layout =
        buffer_trait_object_unknown_memory_layout.slice(0..2);
    print_positions_by_value(&first_two_points_unknown_memory_layout);
}

fn slice_hashmap_buffer_mut() {
    let mut buffer: HashMapBuffer = test_points().into_iter().collect();

    let mut first_two_points_mut = buffer.slice_mut(0..2);
    for intensity in first_two_points_mut
        .view_attribute_mut::<u16>(&INTENSITY)
        .iter_mut()
    {
        *intensity = 42;
    }

    let mut slice_of_slice_mut = first_two_points_mut.slice_mut(0..1);
    for intensity in slice_of_slice_mut
        .view_attribute_mut::<u16>(&INTENSITY)
        .iter_mut()
    {
        *intensity = 64;
    }

    let buffer_trait_object_mut: &mut dyn ColumnarBufferMut = &mut buffer;
    let mut first_two_points_dynamic_mut = buffer_trait_object_mut.slice_mut(0..2);
    for intensity in first_two_points_dynamic_mut
        .view_attribute_mut::<u16>(&INTENSITY)
        .iter_mut()
    {
        *intensity = 96;
    }

    let buffer_trait_object_unknown_memory_layout: &mut dyn BorrowedMutBuffer = &mut buffer;
    let mut first_two_points_dynamic_unknown_memory_layout_mut =
        buffer_trait_object_unknown_memory_layout.slice_mut(0..2);
    let mut view =
        first_two_points_dynamic_unknown_memory_layout_mut.view_attribute_mut::<u16>(&INTENSITY);
    view.set_at(0, 128);
}

fn main() {
    // Demonstrates how to use the `SliceBuffer` and `SliceBufferMut` traits to get slices to point buffers
    slice_vector_buffer();
    slice_vector_buffer_mut();
    slice_hashmap_buffer();
    slice_hashmap_buffer_mut();
}
