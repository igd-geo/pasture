use kd_tree::{self, KdPoint};
use pasture_algorithms::normal_estimation::compute_normals;
use pasture_core::containers::VectorBuffer;
use pasture_core::nalgebra::Vector3;
use pasture_derive::PointType;

#[repr(C, packed)]
#[derive(PointType, Debug, Clone, Copy, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
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
        let position = self.position;
        position[k]
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

    let interleaved = points.into_iter().collect::<VectorBuffer>();

    let solution_vec = compute_normals::<VectorBuffer, SimplePoint>(&interleaved, 4);
    for solution in solution_vec {
        println!(
            "Normals: n_x: {}, n_y: {}, n_z: {}, curvature: {}",
            solution.0[0], solution.0[1], solution.0[2], solution.1
        );
    }
}
