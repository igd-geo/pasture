use std::{fs::File, io::BufWriter};

use criterion::{criterion_group, criterion_main, Criterion};
use las::Builder;
use pasture_core::{
    containers::{InterleavedVecPointStorage, PointBuffer, PointBufferWriteable, OwningPointBuffer},
    layout::PointType,
    nalgebra::Vector3,
};
use pasture_derive::PointType;
use pasture_io::{
    base::{PointReader, PointWriter},
    las::{LASReader, LASWriter, LasPointFormat0},
};
use rand::{distributions::Uniform, thread_rng, Rng};
use scopeguard::defer;

const LAS_PATH: &'static str = "las_bench_file.las";
const LAZ_PATH: &'static str = "laz_bench_file.laz";
const WRITE_DUMMY_FILE: &'static str = "write_dummy.las";

#[derive(PointType)]
#[repr(C)]
struct CustomPointType {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_CLASSIFICATION)]
    pub classification: u8,
}

fn random_las_point<R: Rng + ?Sized>(rng: &mut R) -> LasPointFormat0 {
    LasPointFormat0 {
        classification: rng.sample(Uniform::new(0u8, 8)),
        edge_of_flight_line: rng.gen::<bool>(),
        intensity: rng.gen::<u16>(),
        number_of_returns: rng.sample(Uniform::new(0u8, 5)),
        point_source_id: 0,
        return_number: rng.sample(Uniform::new(0u8, 5)),
        position: Vector3::new(
            rng.sample(Uniform::new(-100.0, 100.0)),
            rng.sample(Uniform::new(-100.0, 100.0)),
            rng.sample(Uniform::new(-100.0, 100.0)),
        ),
        scan_angle_rank: rng.gen::<i8>(),
        scan_direction_flag: rng.gen::<bool>(),
        user_data: 0,
    }
}

fn random_custom_point<R: Rng + ?Sized>(rng: &mut R) -> CustomPointType {
    CustomPointType {
        position: Vector3::new(
            rng.sample(Uniform::new(-100.0, 100.0)),
            rng.sample(Uniform::new(-100.0, 100.0)),
            rng.sample(Uniform::new(-100.0, 100.0)),
        ),
        classification: rng.sample(Uniform::new(0u8, 8)),
    }
}

fn get_dummy_points() -> InterleavedVecPointStorage {
    const NUM_POINTS: usize = 1_000_000;
    let mut buffer =
        InterleavedVecPointStorage::with_capacity(NUM_POINTS, LasPointFormat0::layout());
    let mut rng = thread_rng();
    for _ in 0..NUM_POINTS {
        buffer.push_point(random_las_point(&mut rng));
    }
    buffer
}

fn get_dummy_points_custom_format() -> InterleavedVecPointStorage {
    const NUM_POINTS: usize = 1_000_000;
    let mut buffer =
        InterleavedVecPointStorage::with_capacity(NUM_POINTS, CustomPointType::layout());
    let mut rng = thread_rng();
    for _ in 0..NUM_POINTS {
        buffer.push_point(random_custom_point(&mut rng));
    }
    buffer
}

fn create_dummy_files() {
    let buffer = get_dummy_points();

    let header = Builder::from((1, 4)).into_header().unwrap();
    {
        let mut writer = LASWriter::from_path_and_header(LAS_PATH, header.clone()).unwrap();
        writer.write(&buffer).unwrap();
        writer.flush().unwrap();
    }
    {
        let mut writer = LASWriter::from_path_and_header(LAZ_PATH, header).unwrap();
        writer.write(&buffer).unwrap();
        writer.flush().unwrap();
    }
}

fn remove_dummy_files() {
    std::fs::remove_file(LAS_PATH).unwrap();
    std::fs::remove_file(LAZ_PATH).unwrap();
    std::fs::remove_file(WRITE_DUMMY_FILE).unwrap();
}

fn read_performance(path: &str) {
    let mut reader = LASReader::from_path(path).unwrap();
    let count = reader.remaining_points();
    reader.read(count).unwrap();
}

fn read_performance_custom_format(buffer: &mut dyn PointBufferWriteable, path: &str) {
    buffer.clear();
    let mut reader = LASReader::from_path(path).unwrap();
    let count = reader.remaining_points();
    reader.read_into(buffer, count).unwrap();
}

fn write_performance(points: &dyn PointBuffer, compressed: bool) {
    let writer = BufWriter::new(File::create(WRITE_DUMMY_FILE).unwrap());
    let header = Builder::from((1, 4)).into_header().unwrap();
    let mut writer = LASWriter::from_writer_and_header(writer, header, compressed).unwrap();
    writer.write(points).unwrap();
    writer.flush().unwrap();
}

fn bench(c: &mut Criterion) {
    create_dummy_files();
    defer! {
        remove_dummy_files();
    }
    c.bench_function("las_read", |b| b.iter(|| read_performance(LAS_PATH)));
    c.bench_function("laz_read", |b| b.iter(|| read_performance(LAZ_PATH)));

    {
        let mut read_buffer =
            InterleavedVecPointStorage::with_capacity(1_000_000, CustomPointType::layout());
        c.bench_function("las_read_custom_format", |b| {
            b.iter(|| read_performance_custom_format(&mut read_buffer, LAS_PATH))
        });
        c.bench_function("laz_read_custom_format", |b| {
            b.iter(|| read_performance_custom_format(&mut read_buffer, LAZ_PATH))
        });
    }

    {
        let write_data = get_dummy_points();
        c.bench_function("las_write", |b| {
            b.iter(|| write_performance(&write_data, false))
        });
        c.bench_function("laz_write", |b| {
            b.iter(|| write_performance(&write_data, true))
        });
    }

    {
        let write_data_custom_format = get_dummy_points_custom_format();
        c.bench_function("las_write_custom_format", |b| {
            b.iter(|| write_performance(&write_data_custom_format, false))
        });
        c.bench_function("laz_write_custom_format", |b| {
            b.iter(|| write_performance(&write_data_custom_format, true))
        });
    }
}

criterion_group! {
    name = las;
    config = Criterion::default().sample_size(20);
    targets = bench
}
criterion_main!(las);
