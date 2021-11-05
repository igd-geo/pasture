use kd_tree::{self, KdPoint};
use pasture_algorithms::normal_estimation::compute_normals;
use pasture_core::nalgebra::Vector3;
use pasture_core::{containers::InterleavedVecPointStorage, layout::PointType};
use pasture_derive::PointType;
use typenum;

#[repr(C)]
#[derive(PointType, Debug, Clone, Copy)]
struct SimplePoint {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: u16,
}
impl KdPoint for SimplePoint {
    type Scalar = f64;
    type Dim = typenum::U3;
    fn at(&self, k: usize) -> f64 {
        return self.position[k];
    }
}
fn main() {
    let points = vec![
        SimplePoint {
            position: Vector3::new(1.0, 22.0, 0.0),
            intensity: 42,
        },
        SimplePoint {
            position: Vector3::new(12.0, 23.0, 0.0),
            intensity: 84,
        },
        SimplePoint {
            position: Vector3::new(103.0, 84.0, 2.0),
            intensity: 84,
        },
        SimplePoint {
            position: Vector3::new(101.0, 0.0, 1.0),
            intensity: 84,
        },
    ];

    let mut interleaved = InterleavedVecPointStorage::new(SimplePoint::layout());

    interleaved.push_points(points.as_slice());

    let solution_vec = compute_normals::<InterleavedVecPointStorage, SimplePoint>(&interleaved, 4);
    for solution in solution_vec {
        println!(
            "Normals: n_x: {}, n_y: {}, n_z: {}, curvature: {}",
            solution.0[0], solution.0[1], solution.0[2], solution.1
        );
    }
}
