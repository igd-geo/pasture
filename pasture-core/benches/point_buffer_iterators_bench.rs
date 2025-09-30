use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use pasture_core::{
    containers::{
        BorrowedBuffer, BorrowedBufferExt, BorrowedMutBufferExt, ColumnarBuffer, HashMapBuffer,
        InterleavedBuffer, VectorBuffer,
    },
    layout::PointType,
    layout::attributes::POSITION_3D,
    layout::{PointAttributeDefinition, PrimitiveType},
    nalgebra::Vector3,
};
use pasture_derive::PointType;
use rand::{Rng, distr::Uniform, rng};

#[derive(PointType, Default, Copy, Clone, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
#[repr(C, packed)]
struct CustomPointTypeSmall {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_CLASSIFICATION)]
    pub classification: u8,
}

fn random_custom_point_small<R: Rng + ?Sized>(rng: &mut R) -> CustomPointTypeSmall {
    let distribution_xyz = Uniform::new(-100.0, 100.0).unwrap();
    let distribution_classification = Uniform::new(0u8, 8).unwrap();
    CustomPointTypeSmall {
        position: Vector3::new(
            rng.sample(distribution_xyz),
            rng.sample(distribution_xyz),
            rng.sample(distribution_xyz),
        ),
        classification: rng.sample(distribution_classification),
    }
}

fn get_dummy_points_custom_format_small_interleaved() -> VectorBuffer {
    const NUM_POINTS: usize = 1_000;
    let mut buffer = VectorBuffer::with_capacity(NUM_POINTS, CustomPointTypeSmall::layout());
    let mut rng = rng();
    for _ in 0..NUM_POINTS {
        buffer
            .view_mut()
            .push_point(random_custom_point_small(&mut rng));
    }
    buffer
}

fn get_dummy_points_custom_format_small_perattribute() -> HashMapBuffer {
    const NUM_POINTS: usize = 1_000;
    let mut buffer = HashMapBuffer::with_capacity(NUM_POINTS, CustomPointTypeSmall::layout());
    let mut rng = rng();
    for _ in 0..NUM_POINTS {
        buffer
            .view_mut()
            .push_point(random_custom_point_small(&mut rng));
    }
    buffer
}

fn points_iterator_performance_opaque_buffer<T: PointType + Default, B: BorrowedBuffer>(
    buffer: &B,
) {
    for point in buffer.view::<T>().into_iter() {
        black_box(point);
    }
}

fn points_iterator_performance_interleaved_buffer<T: PointType + Default, B: InterleavedBuffer>(
    buffer: &B,
) {
    for point in buffer.view::<T>().iter() {
        black_box(point);
    }
}

fn points_iterator_performance_per_attribute_buffer<T: PointType + Default, B: ColumnarBuffer>(
    buffer: &B,
) {
    for point in buffer.view::<T>() {
        black_box(point);
    }
}

fn points_ref_iterator_performance_small_type(buffer: &impl InterleavedBuffer) {
    for point in buffer.view::<CustomPointTypeSmall>().iter() {
        black_box(point.position);
    }
}

fn points_ref_iterator_performance_with_trait_object(buffer: &dyn InterleavedBuffer) {
    for point in buffer.view::<CustomPointTypeSmall>().iter() {
        black_box(point.position);
    }
}

fn attribute_iterator_performance_opaque_buffer<T: PrimitiveType + Default>(
    buffer: &impl BorrowedBuffer,
    attribute: &PointAttributeDefinition,
) {
    for val in buffer.view_attribute::<T>(attribute) {
        black_box(val);
    }
}

fn attribute_iterator_performance_interleaved_buffer<
    T: PrimitiveType + Default,
    B: InterleavedBuffer,
>(
    buffer: &B,
    attribute: &PointAttributeDefinition,
) {
    for val in buffer.view_attribute::<T>(attribute) {
        black_box(val);
    }
}

fn attribute_iterator_performance_perattribute_buffer<
    T: PrimitiveType + Default,
    B: ColumnarBuffer,
>(
    buffer: &B,
    attribute: &PointAttributeDefinition,
) {
    for val in buffer.view_attribute::<T>(attribute).iter() {
        black_box(val);
    }
}

fn attribute_ref_iterator_performance_small_type(buffer: &impl ColumnarBuffer) {
    for position in buffer.view_attribute::<Vector3<f64>>(&POSITION_3D).iter() {
        black_box(position);
    }
}

fn bench(c: &mut Criterion) {
    let dummy_points_small_interleaved = get_dummy_points_custom_format_small_interleaved();
    let dummy_points_small_perattribute = get_dummy_points_custom_format_small_perattribute();

    c.bench_function(
        "points_iterator_interleaved_opaque_buffer_small_type",
        |b| {
            b.iter(|| {
                points_iterator_performance_opaque_buffer::<CustomPointTypeSmall, _>(
                    &dummy_points_small_interleaved,
                )
            })
        },
    );
    c.bench_function(
        "points_iterator_perattribute_opaque_buffer_small_type",
        |b| {
            b.iter(|| {
                points_iterator_performance_opaque_buffer::<CustomPointTypeSmall, _>(
                    &dummy_points_small_perattribute,
                )
            })
        },
    );
    c.bench_function("points_iterator_interleaved_typed_buffer_small_type", |b| {
        b.iter(|| {
            points_iterator_performance_interleaved_buffer::<CustomPointTypeSmall, _>(
                &dummy_points_small_interleaved,
            )
        })
    });
    c.bench_function(
        "points_iterator_perattribute_typed_buffer_small_type",
        |b| {
            b.iter(|| {
                points_iterator_performance_per_attribute_buffer::<CustomPointTypeSmall, _>(
                    &dummy_points_small_perattribute,
                )
            })
        },
    );
    c.bench_function("points_ref_iterator_small_type", |b| {
        b.iter(|| points_ref_iterator_performance_small_type(&dummy_points_small_interleaved))
    });
    c.bench_function("points_ref_iterator_small_type_with_trait_object", |b| {
        b.iter(|| {
            points_ref_iterator_performance_with_trait_object(&dummy_points_small_interleaved)
        })
    });

    c.bench_function("attribute_iterator_interleaved_opaque_buffer", |b| {
        b.iter(|| {
            attribute_iterator_performance_opaque_buffer::<Vector3<f64>>(
                &dummy_points_small_interleaved,
                &POSITION_3D,
            )
        })
    });
    c.bench_function("attribute_iterator_perattribute_opaque_buffer", |b| {
        b.iter(|| {
            attribute_iterator_performance_opaque_buffer::<Vector3<f64>>(
                &dummy_points_small_perattribute,
                &POSITION_3D,
            )
        })
    });
    c.bench_function("attribute_iterator_interleaved_typed_buffer", |b| {
        b.iter(|| {
            attribute_iterator_performance_interleaved_buffer::<Vector3<f64>, _>(
                &dummy_points_small_interleaved,
                &POSITION_3D,
            )
        })
    });
    c.bench_function("attribute_iterator_perattribute_typed_buffer", |b| {
        b.iter(|| {
            attribute_iterator_performance_perattribute_buffer::<Vector3<f64>, _>(
                &dummy_points_small_perattribute,
                &POSITION_3D,
            )
        })
    });
    c.bench_function("attribute_ref_iterator_small_type", |b| {
        b.iter(|| attribute_ref_iterator_performance_small_type(&dummy_points_small_perattribute))
    });
}

criterion_group! {
    name = point_buffer_iterators;
    config = Criterion::default().sample_size(40);
    targets = bench
}
criterion_main!(point_buffer_iterators);
