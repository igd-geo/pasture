#[cfg(not(target_arch = "wasm32"))]
mod ex {
    use pasture_algorithms::reprojection::reproject_point_cloud_within;
    use pasture_core::containers::{BorrowedBuffer, VectorBuffer};
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

    pub fn main() {
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
                position: Vector3::new(10.0, 8.0, 2.0),
                intensity: 84,
            },
            SimplePoint {
                position: Vector3::new(10.0, 0.0, 1.0),
                intensity: 84,
            },
        ];

        let mut interleaved = points.into_iter().collect::<VectorBuffer>();

        reproject_point_cloud_within(&mut interleaved, "EPSG:4326", "EPSG:3309");

        for point in interleaved.view::<SimplePoint>() {
            println!("{:?}", point);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    ex::main();
}

#[cfg(target_arch = "wasm32")]
fn main() {}
