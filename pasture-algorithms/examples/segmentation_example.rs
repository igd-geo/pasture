use pasture_algorithms::segmentation::{
    ransac_line_par, ransac_line_serial, ransac_plane_par, ransac_plane_serial,
};
use pasture_core::{
    containers::{BorrowedMutBufferExt, HashMapBuffer},
    layout::attributes::INTENSITY,
    nalgebra::Vector3,
};
use pasture_derive::PointType;
use rand::Rng;

#[repr(C, packed)]
#[derive(PointType, Debug, Copy, Clone, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
pub struct SimplePoint {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: u16,
}

fn main() {
    //generate random points for the pointcloud
    let mut buffer = (0..20000)
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
                let position = point.position;
                point.position = Vector3::new(position.x, position.y, rng.gen_range(-50.0..50.2));
            }
            point
        })
        .collect::<HashMapBuffer>();

    println!("done generating pointcloud");
    let plane_and_points = ransac_plane_par(&buffer, 0.01, 50);
    println!("done ransac_plane");
    println!("{:?}", plane_and_points.0);
    let plane_and_points_ser = ransac_plane_serial(&buffer, 0.01, 50);
    println!("done ransac_plane_ser");
    println!("{:?}", plane_and_points_ser.0);
    let line_and_points = ransac_line_par(&buffer, 0.01, 50);
    println!("done ransac_line");
    println!("{:?}", line_and_points.0);
    let line_and_points_ser = ransac_line_serial(&buffer, 0.01, 50);
    println!("done ransac_line_ser");
    println!("{:?}", line_and_points_ser.0);

    // change intensity for the line and plane points
    for (index, intensity) in buffer
        .view_attribute_mut::<u16>(&INTENSITY)
        .iter_mut()
        .enumerate()
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
