use criterion::{criterion_group, criterion_main, Criterion};
use pasture_algorithms::convexhull;
use pasture_core::{
    containers::{BorrowedMutBuffer, HashMapBuffer, VectorBuffer},
    layout::PointType,
    nalgebra::Vector3,
};
use pasture_derive::PointType;
use rand::{distributions::Uniform, thread_rng, Rng};

#[derive(PointType, Default, Clone, Copy, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
#[repr(C, packed)]
struct CustomPointTypeSmall {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_CLASSIFICATION)]
    pub classification: u8,
}

const NUM_POINTS_SMALL: usize = 1000;
const NUM_POINTS_MEDIUM: usize = 10000;
const NUM_POINTS_BIG: usize = 100000;

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

fn get_dummy_points_custom_format_small_interleaved(num_points: usize) -> VectorBuffer {
    let mut buffer = VectorBuffer::with_capacity(num_points, CustomPointTypeSmall::layout());
    let mut rng = thread_rng();
    for _ in 0..num_points {
        buffer
            .view_mut()
            .push_point(random_custom_point_small(&mut rng));
    }
    buffer
}

fn get_dummy_points_custom_format_small_columnar(num_points: usize) -> HashMapBuffer {
    let mut buffer = HashMapBuffer::with_capacity(num_points, CustomPointTypeSmall::layout());
    let mut rng = thread_rng();
    for _ in 0..num_points {
        buffer
            .view_mut()
            .push_point(random_custom_point_small(&mut rng));
    }
    buffer
}

fn bench(c: &mut Criterion) {
    let mut testname;
    let mut dummy_points_small_interleaved;
    let mut dummy_points_small_perattribute;
    for i in 0..3 {
        match i {
            0 => {
                testname = "small";
                dummy_points_small_interleaved =
                    get_dummy_points_custom_format_small_interleaved(NUM_POINTS_SMALL);
                dummy_points_small_perattribute =
                    get_dummy_points_custom_format_small_columnar(NUM_POINTS_SMALL);
            }
            1 => {
                testname = "medium";
                dummy_points_small_interleaved =
                    get_dummy_points_custom_format_small_interleaved(NUM_POINTS_MEDIUM);
                dummy_points_small_perattribute =
                    get_dummy_points_custom_format_small_columnar(NUM_POINTS_MEDIUM);
            }
            _ => {
                testname = "big";
                dummy_points_small_interleaved =
                    get_dummy_points_custom_format_small_interleaved(NUM_POINTS_BIG);
                dummy_points_small_perattribute =
                    get_dummy_points_custom_format_small_columnar(NUM_POINTS_BIG);
            }
        }
        let mut testname1 = String::from("convexhull_as_points_performance_interleaved_buffer_");
        testname1.push_str(testname);
        c.bench_function(&testname1, |b| {
            b.iter(|| convexhull::convex_hull_as_points(&dummy_points_small_interleaved))
        });
        let mut testname2 =
            String::from("convexhull_as_triangle_mesh_performance_interleaved_buffer_");
        testname2.push_str(testname);
        c.bench_function(&testname2, |b| {
            b.iter(|| convexhull::convex_hull_as_triangle_mesh(&dummy_points_small_interleaved))
        });
        let mut testname3 = String::from("convexhull_as_points_performance_perattribute_buffer_");
        testname3.push_str(testname);
        c.bench_function(&testname3, |b| {
            b.iter(|| convexhull::convex_hull_as_points(&dummy_points_small_perattribute))
        });
        let mut testname4 =
            String::from("convexhull_as_triangle_mesh_performance_perattribute_buffer_");
        testname4.push_str(testname);
        c.bench_function(&testname4, |b| {
            b.iter(|| convexhull::convex_hull_as_triangle_mesh(&dummy_points_small_perattribute))
        });
    }
}

criterion_group! {
    name = convexhull;
    config = Criterion::default().sample_size(40);
    targets = bench
}
criterion_main!(convexhull);
