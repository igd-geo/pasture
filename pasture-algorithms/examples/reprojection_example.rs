use pasture_algorithms::reprojection::reproject_point_cloud_within;
use pasture_core::containers::InterleavedVecPointStorage;
use pasture_core::containers::PointBufferExt;
use pasture_core::layout::PointType;
use pasture_core::nalgebra::Vector3;
use pasture_derive::PointType;

#[repr(C)]
#[derive(PointType, Debug, Clone, Copy)]
struct SimplePoint {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: u16,
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
            position: Vector3::new(10.0, 8.0, 2.0),
            intensity: 84,
        },
        SimplePoint {
            position: Vector3::new(10.0, 0.0, 1.0),
            intensity: 84,
        },
    ];

    let mut interleaved = InterleavedVecPointStorage::new(SimplePoint::layout());

    interleaved.push_points(points.as_slice());
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

    let mut interleaved = InterleavedVecPointStorage::new(SimplePoint::layout());

    interleaved.push_points(points.as_slice());

    reproject_point_cloud_within::<InterleavedVecPointStorage>(
        &mut interleaved,
        "EPSG:4326",
        "EPSG:3309",
    );

    for point in interleaved.iter_point::<SimplePoint>() {
        println!("{:?}", point);
    }
}
