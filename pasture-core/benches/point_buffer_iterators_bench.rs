use criterion::{criterion_group, criterion_main, Criterion};
use pasture_core::{
    containers::{
        BorrowedBuffer, BorrowedBufferExt, ColumnarBuffer, HashMapBuffer, InterleavedBuffer,
        MatrixBuffer, VectorBuffer,
    },
    layout::attributes::POSITION_3D,
    layout::PointType,
    layout::{PointAttributeDefinition, PrimitiveType},
    nalgebra::Vector3,
};
use pasture_derive::PointType;
use rand::{distributions::Uniform, thread_rng, Rng};

#[derive(PointType, Default, Copy, Clone, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
#[repr(C, packed)]
struct CustomPointTypeSmall {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_CLASSIFICATION)]
    pub classification: u8,
}

#[derive(PointType, Default, Copy, Clone, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
#[repr(C, packed)]
struct CustomPointTypeBig {
    #[pasture(BUILTIN_GPS_TIME)]
    pub gps_time: f64,
    #[pasture(BUILTIN_COLOR_RGB)]
    pub color: Vector3<u16>,
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_CLASSIFICATION)]
    pub classification: u8,
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: i16,
}

fn random_custom_point_small<R: Rng + ?Sized>(rng: &mut R) -> CustomPointTypeSmall {
    CustomPointTypeSmall {
        position: Vector3::new(
            rng.sample(Uniform::new(-100.0, 100.0)),
            rng.sample(Uniform::new(-100.0, 100.0)),
            rng.sample(Uniform::new(-100.0, 100.0)),
        ),
        classification: rng.sample(Uniform::new(0u8, 8)),
    }
}

fn get_dummy_points_small_vector_buffer() -> VectorBuffer {
    const NUM_POINTS: usize = 1_000;
    let mut rng = thread_rng();
    (0..NUM_POINTS)
        .map(|_| random_custom_point_small(&mut rng))
        .collect()
}

fn get_dummy_points_small_hashmap_buffer() -> HashMapBuffer {
    const NUM_POINTS: usize = 1_000;
    let mut rng = thread_rng();
    (0..NUM_POINTS)
        .map(|_| random_custom_point_small(&mut rng))
        .collect()
}

fn get_dummy_points_small_matrix_buffer() -> MatrixBuffer {
    const NUM_POINTS: usize = 1_000;
    let mut rng = thread_rng();
    (0..NUM_POINTS)
        .map(|_| random_custom_point_small(&mut rng))
        .collect()
}

fn points_iterator_performance_opaque_buffer<'a, T: PointType + Default, B: BorrowedBuffer<'a>>(
    buffer: &'a B,
) {
    for point in buffer.view::<T>().into_iter() {
        criterion::black_box(point);
    }
}

fn points_iterator_performance_interleaved_buffer<
    'a,
    T: PointType + Default,
    B: InterleavedBuffer<'a>,
>(
    buffer: &'a B,
) {
    for point in buffer.view::<T>().iter() {
        criterion::black_box(point);
    }
}

fn points_iterator_performance_per_attribute_buffer<
    'a,
    T: PointType + Default,
    B: ColumnarBuffer<'a>,
>(
    buffer: &'a B,
) {
    for point in buffer.view::<T>() {
        criterion::black_box(point);
    }
}

fn points_ref_iterator_performance_small_type<'a>(buffer: &'a impl InterleavedBuffer<'a>) {
    for point in buffer.view::<CustomPointTypeSmall>().iter() {
        criterion::black_box(point.position);
    }
}

fn points_ref_iterator_performance_with_trait_object<'a>(buffer: &'a dyn InterleavedBuffer<'a>) {
    for point in buffer.view::<CustomPointTypeSmall>().iter() {
        criterion::black_box(point.position);
    }
}

fn attribute_iterator_performance_opaque_buffer<'a, T: PrimitiveType + Default>(
    buffer: &'a impl BorrowedBuffer<'a>,
    attribute: &PointAttributeDefinition,
) {
    for val in buffer.view_attribute::<T>(attribute) {
        criterion::black_box(val);
    }
}

fn attribute_iterator_performance_interleaved_buffer<
    'a,
    T: PrimitiveType + Default,
    B: InterleavedBuffer<'a>,
>(
    buffer: &'a B,
    attribute: &PointAttributeDefinition,
) {
    for val in buffer.view_attribute::<T>(attribute) {
        criterion::black_box(val);
    }
}

fn attribute_iterator_performance_perattribute_buffer<
    'a,
    T: PrimitiveType + Default,
    B: ColumnarBuffer<'a>,
>(
    buffer: &'a B,
    attribute: &PointAttributeDefinition,
) {
    for val in buffer.view_attribute::<T>(attribute).iter() {
        criterion::black_box(val);
    }
}

fn attribute_ref_iterator_performance_small_type<'a>(buffer: &'a impl ColumnarBuffer<'a>) {
    for position in buffer.view_attribute::<Vector3<f64>>(&POSITION_3D).iter() {
        criterion::black_box(position);
    }
}

fn bench(c: &mut Criterion) {
    let dummy_points_small_vector = get_dummy_points_small_vector_buffer();
    let dummy_points_small_hashmap = get_dummy_points_small_hashmap_buffer();
    let dummy_points_small_matrix = get_dummy_points_small_matrix_buffer();

    c.bench_function(
        "points_iterator_interleaved_opaque_buffer_small_type",
        |b| {
            b.iter(|| {
                points_iterator_performance_opaque_buffer::<CustomPointTypeSmall, _>(
                    &dummy_points_small_vector,
                )
            })
        },
    );
    c.bench_function(
        "points_iterator_perattribute_opaque_buffer_small_type",
        |b| {
            b.iter(|| {
                points_iterator_performance_opaque_buffer::<CustomPointTypeSmall, _>(
                    &dummy_points_small_hashmap,
                )
            })
        },
    );
    c.bench_function("points_iterator_matrix_buffer_opaque_small_type", |b| {
        b.iter(|| {
            points_iterator_performance_opaque_buffer::<CustomPointTypeSmall, _>(
                &dummy_points_small_matrix,
            )
        })
    });
    c.bench_function("points_iterator_interleaved_typed_buffer_small_type", |b| {
        b.iter(|| {
            points_iterator_performance_interleaved_buffer::<CustomPointTypeSmall, _>(
                &dummy_points_small_vector,
            )
        })
    });
    c.bench_function(
        "points_iterator_perattribute_typed_buffer_small_type",
        |b| {
            b.iter(|| {
                points_iterator_performance_per_attribute_buffer::<CustomPointTypeSmall, _>(
                    &dummy_points_small_hashmap,
                )
            })
        },
    );
    c.bench_function("points_iterator_matrix_buffer_columnar_small_type", |b| {
        b.iter(|| {
            points_iterator_performance_per_attribute_buffer::<CustomPointTypeSmall, _>(
                &dummy_points_small_matrix,
            )
        })
    });
    c.bench_function("points_ref_iterator_small_type", |b| {
        b.iter(|| points_ref_iterator_performance_small_type(&dummy_points_small_vector))
    });
    c.bench_function("points_ref_iterator_small_type_with_trait_object", |b| {
        b.iter(|| points_ref_iterator_performance_with_trait_object(&dummy_points_small_vector))
    });

    c.bench_function("attribute_iterator_interleaved_opaque_buffer", |b| {
        b.iter(|| {
            attribute_iterator_performance_opaque_buffer::<Vector3<f64>>(
                &dummy_points_small_vector,
                &POSITION_3D,
            )
        })
    });
    c.bench_function("attribute_iterator_perattribute_opaque_buffer", |b| {
        b.iter(|| {
            attribute_iterator_performance_opaque_buffer::<Vector3<f64>>(
                &dummy_points_small_hashmap,
                &POSITION_3D,
            )
        })
    });
    c.bench_function("attribute_iterator_matrix_buffer_opaque_buffer", |b| {
        b.iter(|| {
            attribute_iterator_performance_opaque_buffer::<Vector3<f64>>(
                &dummy_points_small_matrix,
                &POSITION_3D,
            )
        })
    });
    c.bench_function("attribute_iterator_interleaved_typed_buffer", |b| {
        b.iter(|| {
            attribute_iterator_performance_interleaved_buffer::<Vector3<f64>, _>(
                &dummy_points_small_vector,
                &POSITION_3D,
            )
        })
    });
    c.bench_function("attribute_iterator_perattribute_typed_buffer", |b| {
        b.iter(|| {
            attribute_iterator_performance_perattribute_buffer::<Vector3<f64>, _>(
                &dummy_points_small_hashmap,
                &POSITION_3D,
            )
        })
    });
    c.bench_function("attribute_iterator_matrix_buffer_typed_buffer", |b| {
        b.iter(|| {
            attribute_iterator_performance_perattribute_buffer::<Vector3<f64>, _>(
                &dummy_points_small_matrix,
                &POSITION_3D,
            )
        })
    });
    c.bench_function("attribute_ref_iterator_small_type", |b| {
        b.iter(|| attribute_ref_iterator_performance_small_type(&dummy_points_small_hashmap))
    });
    c.bench_function("attribute_ref_iterator_matrix_buffer_small_type", |b| {
        b.iter(|| attribute_ref_iterator_performance_small_type(&dummy_points_small_matrix))
    });
}

criterion_group! {
    name = point_buffer_iterators;
    config = Criterion::default().sample_size(40);
    targets = bench
}
criterion_main!(point_buffer_iterators);
