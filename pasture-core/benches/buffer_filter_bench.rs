use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;
use nalgebra::Vector3;
use pasture_core::containers::{
    BorrowedBuffer, HashMapBuffer, InterleavedBufferMut, OwningBuffer, VectorBuffer,
};
use pasture_derive::PointType;
use rand::{distributions::Standard, prelude::Distribution, thread_rng, Rng};

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

struct DefaultPointDistribution;

impl Distribution<CustomPointTypeBig> for DefaultPointDistribution {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> CustomPointTypeBig {
        CustomPointTypeBig {
            classification: rng.gen(),
            position: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
            color: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
            gps_time: rng.gen(),
            intensity: rng.gen(),
        }
    }
}

fn gen_random_points(count: usize) -> HashMapBuffer {
    thread_rng()
        .sample_iter::<CustomPointTypeBig, _>(DefaultPointDistribution)
        .take(count)
        .collect()
}

fn filter_with_get_point(buffer: &HashMapBuffer, random_matches: &[bool]) {
    let num_matches = random_matches.iter().filter(|b| **b).count();
    let mut filtered = VectorBuffer::with_capacity(num_matches, buffer.point_layout().clone());
    filtered.resize(num_matches);
    let target_points = filtered.get_point_range_mut(0..num_matches);
    for ((idx, _), target_point) in random_matches
        .iter()
        .enumerate()
        .filter(|(_, b)| **b)
        .zip(target_points.chunks_mut(buffer.point_layout().size_of_point_entry() as usize))
    {
        buffer.get_point(idx, target_point);
    }
    black_box(filtered);
}

fn filter_with_filter_function(buffer: &HashMapBuffer, random_matches: &[bool]) {
    let filtered = buffer.filter::<VectorBuffer, _>(|idx| random_matches[idx]);
    black_box(filtered);
}

fn bench(c: &mut Criterion) {
    let random_points = gen_random_points(4096);
    let random_matches = thread_rng().sample_iter(Standard).take(4096).collect_vec();

    c.bench_function("filter_with_get_point", |b| {
        b.iter(|| filter_with_get_point(&random_points, &random_matches));
    });
    c.bench_function("filter_with_filter_function", |b| {
        b.iter(|| filter_with_filter_function(&random_points, &random_matches));
    });
}

criterion_group! {
    name = buffer_filter;
    config = Criterion::default().sample_size(40);
    targets = bench
}
criterion_main!(buffer_filter);
