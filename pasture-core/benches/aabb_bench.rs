use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use nalgebra::Vector3;
use pasture_core::math::AABB;
use rand::{RngExt, rng};

fn gen_random_positions(count: usize) -> Vec<Vector3<f64>> {
    let mut rng = rng();
    (0..count)
        .map(|_| Vector3::new(rng.random(), rng.random(), rng.random()))
        .collect()
}

fn aabb_from_iter(positions: &[Vector3<f64>]) {
    let bounds: AABB<f64> = positions.iter().copied().collect();
    black_box(bounds);
}

fn aabb_by_extending(positions: &[Vector3<f64>]) {
    let mut bounds = AABB::from_min_max_unchecked(positions[0].into(), positions[0].into());
    for position in positions.iter().copied().skip(1) {
        bounds = AABB::extend_with_point(&bounds, &position.into());
    }
    black_box(bounds);
}

fn bench(c: &mut Criterion) {
    let random_positions = gen_random_positions(4096);

    c.bench_function("aabb_by_extending", |b| {
        b.iter(|| aabb_by_extending(&random_positions));
    });
    c.bench_function("aabb_from_iter", |b| {
        b.iter(|| aabb_from_iter(&random_positions));
    });
}

criterion_group! {
    name = aabb;
    config = Criterion::default().sample_size(40);
    targets = bench
}
criterion_main!(aabb);
