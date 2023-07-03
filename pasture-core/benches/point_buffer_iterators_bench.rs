use std::iter::FromIterator;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pasture_core::{
    containers::{
        BufferStorage, BufferStorageColumnar, BufferStorageRowWise, ColumnarStorage, PointBuffer,
        PolymorphicStorage, VectorStorage,
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

fn get_dummy_points_custom_format_small<T: BufferStorage + FromIterator<CustomPointTypeSmall>>(
) -> PointBuffer<T> {
    const NUM_POINTS: usize = 1_000;
    let mut rng = thread_rng();
    (0..NUM_POINTS)
        .map(|_| random_custom_point_small(&mut rng))
        .collect::<PointBuffer<T>>()
}

// fn get_dummy_points_custom_format_small_perattribute() -> PerAttributeVecPointStorage {
//     const NUM_POINTS: usize = 1_000;
//     let mut buffer =
//         PerAttributeVecPointStorage::with_capacity(NUM_POINTS, CustomPointTypeSmall::layout());
//     let mut rng = thread_rng();
//     for _ in 0..NUM_POINTS {
//         buffer.push_point(random_custom_point_small(&mut rng));
//     }
//     buffer
// }

fn points_iterator_performance_polymorphic_storage<T: PointType + Default>(
    buffer: &PointBuffer<PolymorphicStorage>,
) {
    for point in buffer.view::<T>() {
        black_box(point);
    }
}

fn points_iterator_performance_rowwise_storage<T: PointType + Default, S: BufferStorageRowWise>(
    buffer: &PointBuffer<S>,
) {
    for point in buffer.view::<T>().iter() {
        black_box(point);
    }
}

fn points_iterator_performance_columnar_storage<
    T: PointType + Default,
    S: BufferStorageColumnar,
>(
    buffer: &PointBuffer<S>,
) {
    for point in buffer.view::<T>() {
        black_box(point);
    }
}

fn attribute_iterator_performance_polymorphic_storage<T: PrimitiveType + Default>(
    buffer: &PointBuffer<PolymorphicStorage>,
    attribute: &PointAttributeDefinition,
) {
    for val in buffer.view_attribute::<T>(attribute) {
        black_box(val);
    }
}

fn attribute_iterator_performance_rowwise_storage<
    T: PrimitiveType + Default,
    B: BufferStorageRowWise,
>(
    buffer: &PointBuffer<B>,
    attribute: &PointAttributeDefinition,
) {
    for val in buffer.view_attribute::<T>(attribute) {
        black_box(val);
    }
}

fn attribute_iterator_performance_columnar_storage<
    T: PrimitiveType + Default,
    B: BufferStorageColumnar,
>(
    buffer: &PointBuffer<B>,
    attribute: &PointAttributeDefinition,
) {
    for val in buffer.view_attribute::<T>(attribute).iter() {
        black_box(val);
    }
}

fn bench(c: &mut Criterion) {
    let dummy_points_small_vector_storage = get_dummy_points_custom_format_small::<VectorStorage>();
    let dummy_points_small_columnar_storage =
        get_dummy_points_custom_format_small::<ColumnarStorage>();

    c.bench_function(
        "points_iterator_performance_polymorphic_storage vector storage",
        |b| {
            let polymorphic_storage = dummy_points_small_vector_storage
                .clone()
                .into_polymorphic_buffer();
            b.iter(|| {
                points_iterator_performance_polymorphic_storage::<CustomPointTypeSmall>(
                    &polymorphic_storage,
                )
            })
        },
    );
    c.bench_function(
        "points_iterator_performance_polymorphic_storage columnar storage",
        |b| {
            let polymorphic_storage = dummy_points_small_columnar_storage
                .clone()
                .into_polymorphic_buffer();
            b.iter(|| {
                points_iterator_performance_polymorphic_storage::<CustomPointTypeSmall>(
                    &polymorphic_storage,
                )
            })
        },
    );
    c.bench_function("points_iterator_performance_rowwise_storage", |b| {
        b.iter(|| {
            points_iterator_performance_rowwise_storage::<CustomPointTypeSmall, _>(
                &dummy_points_small_vector_storage,
            )
        })
    });
    c.bench_function("points_iterator_performance_columnar_storage", |b| {
        b.iter(|| {
            points_iterator_performance_columnar_storage::<CustomPointTypeSmall, _>(
                &dummy_points_small_columnar_storage,
            )
        })
    });

    c.bench_function(
        "attribute_iterator_performance_polymorphic_storage vector storage",
        |b| {
            let polymorphic_storage = dummy_points_small_vector_storage
                .clone()
                .into_polymorphic_buffer();
            b.iter(|| {
                attribute_iterator_performance_polymorphic_storage::<Vector3<f64>>(
                    &polymorphic_storage,
                    &POSITION_3D,
                )
            })
        },
    );
    c.bench_function(
        "attribute_iterator_performance_polymorphic_storage columnar storage",
        |b| {
            b.iter(|| {
                let polymorphic_storage = dummy_points_small_columnar_storage
                    .clone()
                    .into_polymorphic_buffer();
                attribute_iterator_performance_polymorphic_storage::<Vector3<f64>>(
                    &polymorphic_storage,
                    &POSITION_3D,
                )
            })
        },
    );
    c.bench_function("attribute_iterator_performance_rowwise_storage", |b| {
        b.iter(|| {
            attribute_iterator_performance_rowwise_storage::<Vector3<f64>, _>(
                &dummy_points_small_vector_storage,
                &POSITION_3D,
            )
        })
    });
    c.bench_function("attribute_iterator_performance_columnar_storage", |b| {
        b.iter(|| {
            attribute_iterator_performance_columnar_storage::<Vector3<f64>, _>(
                &dummy_points_small_columnar_storage,
                &POSITION_3D,
            )
        })
    });
}

criterion_group! {
    name = point_buffer_iterators;
    config = Criterion::default().sample_size(40);
    targets = bench
}
criterion_main!(point_buffer_iterators);
