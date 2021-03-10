use criterion::{criterion_group, criterion_main, Criterion};
use pasture_core::{
    containers::{
        InterleavedPointBuffer, InterleavedPointBufferExt, InterleavedVecPointStorage,
        PerAttributePointBuffer, PerAttributePointBufferExt, PerAttributeVecPointStorage,
        PointBuffer, PointBufferExt,
    },
    layout::attributes::POSITION_3D,
    layout::PointType,
    layout::{PointAttributeDefinition, PrimitiveType},
    nalgebra::Vector3,
};
use pasture_derive::PointType;
use rand::{distributions::Uniform, thread_rng, Rng};

#[derive(PointType, Default)]
#[repr(C)]
struct CustomPointTypeSmall {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_CLASSIFICATION)]
    pub classification: u8,
}

#[derive(PointType, Default)]
#[repr(C)]
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

fn get_dummy_points_custom_format_small_interleaved() -> InterleavedVecPointStorage {
    const NUM_POINTS: usize = 1_000;
    let mut buffer =
        InterleavedVecPointStorage::with_capacity(NUM_POINTS, CustomPointTypeSmall::layout());
    let mut rng = thread_rng();
    for _ in 0..NUM_POINTS {
        buffer.push_point(random_custom_point_small(&mut rng));
    }
    buffer
}

fn get_dummy_points_custom_format_small_perattribute() -> PerAttributeVecPointStorage {
    const NUM_POINTS: usize = 1_000;
    let mut buffer =
        PerAttributeVecPointStorage::with_capacity(NUM_POINTS, CustomPointTypeSmall::layout());
    let mut rng = thread_rng();
    for _ in 0..NUM_POINTS {
        buffer.push_point(random_custom_point_small(&mut rng));
    }
    buffer
}

fn points_iterator_performance_opaque_buffer<T: PointType + Default>(
    buffer: &dyn PointBuffer,
) -> T {
    let mut ret = Default::default();
    for point in buffer.iter_point::<T>() {
        ret = point;
    }
    ret
}

fn points_iterator_performance_interleaved_buffer<
    T: PointType + Default,
    B: InterleavedPointBuffer,
>(
    buffer: &B,
) -> T {
    let mut ret = Default::default();
    for point in buffer.iter_point::<T>() {
        ret = point;
    }
    ret
}

fn points_iterator_performance_per_attribute_buffer<
    T: PointType + Default,
    B: PerAttributePointBuffer,
>(
    buffer: &B,
) -> T {
    let mut ret = Default::default();
    for point in buffer.iter_point::<T>() {
        ret = point;
    }
    ret
}

fn points_ref_iterator_performance_small_type(buffer: &dyn InterleavedPointBuffer) -> Vector3<f64> {
    let mut position = Vector3::new(0.0, 0.0, 0.0);
    for point in buffer.iter_point_ref::<CustomPointTypeSmall>() {
        position = point.position.clone();
    }
    position
}

fn attribute_iterator_performance_opaque_buffer<T: PrimitiveType + Default>(
    buffer: &dyn PointBuffer,
    attribute: &PointAttributeDefinition,
) -> T {
    let mut ret: T = Default::default();
    for val in buffer.iter_attribute::<T>(attribute) {
        ret = val;
    }
    ret
}

fn attribute_iterator_performance_interleaved_buffer<
    T: PrimitiveType + Default,
    B: InterleavedPointBuffer,
>(
    buffer: &B,
    attribute: &PointAttributeDefinition,
) -> T {
    let mut ret: T = Default::default();
    for val in buffer.iter_attribute::<T>(attribute) {
        ret = val;
    }
    ret
}

fn attribute_iterator_performance_perattribute_buffer<
    T: PrimitiveType + Default,
    B: PerAttributePointBuffer,
>(
    buffer: &B,
    attribute: &PointAttributeDefinition,
) -> T {
    let mut ret: T = Default::default();
    for val in buffer.iter_attribute::<T>(attribute) {
        ret = val;
    }
    ret
}

fn attribute_ref_iterator_performance_small_type(
    buffer: &dyn PerAttributePointBuffer,
) -> Vector3<f64> {
    let mut ret = Vector3::new(0.0, 0.0, 0.0);
    for position in buffer.iter_attribute_ref::<Vector3<f64>>(&POSITION_3D) {
        ret = position.clone();
    }
    ret
}

fn bench(c: &mut Criterion) {
    let dummy_points_small_interleaved = get_dummy_points_custom_format_small_interleaved();
    let dummy_points_small_perattribute = get_dummy_points_custom_format_small_perattribute();

    c.bench_function(
        "points_iterator_interleaved_opaque_buffer_small_type",
        |b| {
            b.iter(|| {
                points_iterator_performance_opaque_buffer::<CustomPointTypeSmall>(
                    &dummy_points_small_interleaved,
                )
            })
        },
    );
    c.bench_function(
        "points_iterator_perattribute_opaque_buffer_small_type",
        |b| {
            b.iter(|| {
                points_iterator_performance_opaque_buffer::<CustomPointTypeSmall>(
                    &dummy_points_small_perattribute,
                )
            })
        },
    );
    c.bench_function("points_iterator_interleaved_typed_buffer_small_type", |b| {
        b.iter(|| -> CustomPointTypeSmall {
            points_iterator_performance_interleaved_buffer(&dummy_points_small_interleaved)
        })
    });
    c.bench_function(
        "points_iterator_perattribute_typed_buffer_small_type",
        |b| {
            b.iter(|| -> CustomPointTypeSmall {
                points_iterator_performance_per_attribute_buffer(&dummy_points_small_perattribute)
            })
        },
    );
    c.bench_function("points_ref_iterator_small_type", |b| {
        b.iter(|| points_ref_iterator_performance_small_type(&dummy_points_small_interleaved))
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
        b.iter(|| -> Vector3<f64> {
            attribute_iterator_performance_interleaved_buffer(
                &dummy_points_small_interleaved,
                &POSITION_3D,
            )
        })
    });
    c.bench_function("attribute_iterator_perattribute_typed_buffer", |b| {
        b.iter(|| -> Vector3<f64> {
            attribute_iterator_performance_perattribute_buffer(
                &dummy_points_small_perattribute,
                &POSITION_3D,
            )
        })
    });
    c.bench_function("attribute_ref_iterator_small_type", |b| {
        b.iter(|| -> Vector3<f64> {
            attribute_ref_iterator_performance_small_type(&dummy_points_small_perattribute)
        })
    });
}

criterion_group! {
    name = point_buffer_iterators;
    config = Criterion::default().sample_size(40);
    targets = bench
}
criterion_main!(point_buffer_iterators);
