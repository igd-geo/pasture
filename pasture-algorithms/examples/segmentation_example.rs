use pasture_algorithms::segmentation::{
    ransac_line_par, ransac_line_serial, ransac_plane_par, ransac_plane_serial,
};
use pasture_core::{
    attributes_mut,
    containers::PerAttributeVecPointStorage,
    layout::{
        attributes::{INTENSITY, POSITION_3D},
        PointType,
    },
    nalgebra::Vector3,
};
use pasture_derive::PointType;
use rand::Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[repr(C)]
#[derive(PointType, Debug)]
pub struct SimplePoint {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: u16,
}

fn main() -> () {
    let mut buffer = PerAttributeVecPointStorage::new(SimplePoint::layout());

    //generate random points for the pointcloud
    let points: Vec<SimplePoint> = (0..20000)
        .into_par_iter()
        .map(|p| {
            let mut rng = rand::thread_rng();
            //generate plane points (along x- and y-axis)
            let mut point = SimplePoint {
                position: Vector3::new(rng.gen_range(0.0..100.0), rng.gen_range(0.0..100.0), 1.0),
                intensity: 1,
            };
            //generate z-axis points for the line
            if p % 4 == 0 {
                point.position = Vector3::new(0.0, 0.0, rng.gen_range(0.0..200.0));
            }
            //generate outliers
            if p % 50 == 0 {
                point.position.z = rng.gen_range(-50.0..50.2);
            }
            point
        })
        .collect();

    buffer.push_points(&points);
    println!("done generating pointcloud");
    let plane_and_points = ransac_plane_par::<PerAttributeVecPointStorage>(&buffer, 0.01, 50);
    println!("done ransac_plane");
    println!("{:?}", plane_and_points.0);
    let plane_and_points_ser =
        ransac_plane_serial::<PerAttributeVecPointStorage>(&buffer, 0.01, 50);
    println!("done ransac_plane_ser");
    println!("{:?}", plane_and_points_ser.0);
    let line_and_points = ransac_line_par::<PerAttributeVecPointStorage>(&buffer, 0.01, 50);
    println!("done ransac_line");
    println!("{:?}", line_and_points.0);
    let line_and_points_ser = ransac_line_serial::<PerAttributeVecPointStorage>(&buffer, 0.01, 50);
    println!("done ransac_line_ser");
    println!("{:?}", line_and_points_ser.0);

    // change intensity for the line and plane points
    for (index, (_position, intensity)) in
        attributes_mut![&POSITION_3D => Vector3<f64>, &INTENSITY => u16, &mut buffer].enumerate()
    {
        if line_and_points.1.contains(&index) {
            *intensity = 500;
        } else if plane_and_points.1.contains(&index) {
            *intensity = 700;
        } else {
            *intensity = 300;
        }
    }
    println!("changed intensity");

    // write into file
    // let writer = BufWriter::new(File::create("testCloud.las").unwrap());
    // let header = Builder::from((1, 4)).into_header().unwrap();
    // let mut writer = LASWriter::from_writer_and_header(writer, header, false).unwrap();
    // writer.write(&buffer).unwrap();
}
