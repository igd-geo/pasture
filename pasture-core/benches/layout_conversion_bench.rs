use std::iter::FromIterator;

use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::Vector3;
use pasture_core::{
    containers::{
        BorrowedBufferExt, ColumnarBuffer, HashMapBuffer, InterleavedBuffer, MakeBufferFromLayout,
        OwningBuffer, VectorBuffer,
    },
    layout::{conversion::BufferLayoutConverter, PointType},
};
use pasture_derive::PointType;
use rand::{prelude::Distribution, thread_rng, Rng};

#[derive(Debug, Copy, Clone, bytemuck::Zeroable, bytemuck::Pod, PointType)]
#[repr(C, packed)]
struct PointTypeSource {
    #[pasture(BUILTIN_POSITION_3D)]
    position: Vector3<f64>,
    #[pasture(BUILTIN_CLASSIFICATION)]
    classification: u8,
    #[pasture(BUILTIN_INTENSITY)]
    intensity: u16,
    #[pasture(BUILTIN_GPS_TIME)]
    gps_time: f64,
}

#[derive(Debug, Copy, Clone, bytemuck::Zeroable, bytemuck::Pod, PointType)]
#[repr(C, packed)]
struct PointTypeTarget {
    #[pasture(BUILTIN_GPS_TIME)]
    gps_time: f64,
    #[pasture(BUILTIN_POSITION_3D)]
    position: Vector3<f32>,
    #[pasture(BUILTIN_CLASSIFICATION)]
    classification: u32,
    #[pasture(BUILTIN_INTENSITY)]
    intensity: u8,
}

struct PointDistribution;

impl Distribution<PointTypeSource> for PointDistribution {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> PointTypeSource {
        PointTypeSource {
            position: Vector3::new(
                rng.gen_range(0.0..1000.0),
                rng.gen_range(0.0..1000.0),
                rng.gen_range(0.0..1000.0),
            ),
            classification: rng.gen(),
            intensity: rng.gen(),
            gps_time: rng.gen_range(0.0..1000.0),
        }
    }
}

fn gen_random_points<
    B: for<'a> OwningBuffer<'a> + for<'a> MakeBufferFromLayout<'a> + FromIterator<PointTypeSource>,
>(
    count: usize,
) -> B {
    thread_rng()
        .sample_iter::<PointTypeSource, _>(PointDistribution)
        .take(count)
        .collect::<B>()
}

fn convert_interleaved_to_interleaved<B: for<'a> InterleavedBuffer<'a>>(
    buffer: &B,
    converter: &BufferLayoutConverter,
) {
    let converted = converter.convert::<VectorBuffer, _>(buffer);
    criterion::black_box(converted);
}

fn convert_interleaved_to_columnar<B: for<'a> InterleavedBuffer<'a>>(
    buffer: &B,
    converter: &BufferLayoutConverter,
) {
    let converted = converter.convert::<HashMapBuffer, _>(buffer);
    criterion::black_box(converted);
}

fn convert_columnar_to_interleaved<B: for<'a> ColumnarBuffer<'a>>(
    buffer: &B,
    converter: &BufferLayoutConverter,
) {
    let converted = converter.convert::<VectorBuffer, _>(buffer);
    criterion::black_box(converted);
}

fn convert_columnar_to_columnar<B: for<'a> ColumnarBuffer<'a>>(
    buffer: &B,
    converter: &BufferLayoutConverter,
) {
    let converted = converter.convert::<HashMapBuffer, _>(buffer);
    criterion::black_box(converted);
}

fn convert_baseline<B: for<'a> InterleavedBuffer<'a>>(buffer: &B) {
    let converted = buffer
        .view::<PointTypeSource>()
        .iter()
        .map(|point| {
            let position = point.position;
            PointTypeTarget {
                classification: point.classification as u32,
                gps_time: point.gps_time,
                intensity: point.intensity as u8,
                position: Vector3::new(position.x as f32, position.y as f32, position.z as f32),
            }
        })
        .collect::<VectorBuffer>();
    criterion::black_box(converted);
}

fn bench(c: &mut Criterion) {
    const COUNT: usize = 1024;
    let test_points_interleaved = gen_random_points::<VectorBuffer>(COUNT);
    let test_points_columnar = gen_random_points::<HashMapBuffer>(COUNT);

    let source_layout = PointTypeSource::layout();
    let target_layout = PointTypeTarget::layout();
    let converter = BufferLayoutConverter::for_layouts_with_default(&source_layout, &target_layout);

    // Bench all conversions against a baseline function that calls `map` on a point iterator
    c.bench_function("convert_baseline", |b| {
        b.iter(|| convert_baseline(&test_points_interleaved))
    });

    c.bench_function("convert_interleaved_to_interleaved", |b| {
        b.iter(|| convert_interleaved_to_interleaved(&test_points_interleaved, &converter))
    });

    c.bench_function("convert_interleaved_to_columnar", |b| {
        b.iter(|| convert_interleaved_to_columnar(&test_points_interleaved, &converter))
    });

    c.bench_function("convert_columnar_to_interleaved", |b| {
        b.iter(|| convert_columnar_to_interleaved(&test_points_columnar, &converter))
    });

    c.bench_function("convert_columnar_to_columnar", |b| {
        b.iter(|| convert_columnar_to_columnar(&test_points_columnar, &converter))
    });
}

criterion_group! {
    name = layout_conversion;
    config = Criterion::default().sample_size(40);
    targets = bench
}
criterion_main!(layout_conversion);
