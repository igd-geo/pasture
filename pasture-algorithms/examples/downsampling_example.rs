use pasture_algorithms::voxel_grid::voxelgrid_filter;
use pasture_core::{
    containers::{PerAttributeVecPointStorage, PointBuffer, OwningPointBuffer},
    layout::PointType,
    nalgebra::Vector3,
};
use pasture_derive::PointType;
use rand::{prelude::ThreadRng, Rng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[repr(C)]
#[derive(PointType, Debug)]
pub struct SimplePoint {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: u16,
    #[pasture(BUILTIN_RETURN_NUMBER)]
    pub return_number: u8,
    #[pasture(BUILTIN_NUMBER_OF_RETURNS)]
    pub num_of_returns: u8,
    #[pasture(BUILTIN_CLASSIFICATION_FLAGS)]
    pub classification_flags: u8,
    #[pasture(BUILTIN_SCANNER_CHANNEL)]
    pub scanner_channel: u8,
    #[pasture(BUILTIN_SCAN_DIRECTION_FLAG)]
    pub scan_dir_flag: bool,
    #[pasture(BUILTIN_EDGE_OF_FLIGHT_LINE)]
    pub edge_of_flight_line: bool,
    #[pasture(BUILTIN_CLASSIFICATION)]
    pub classification: u8,
    #[pasture(BUILTIN_SCAN_ANGLE_RANK)]
    pub scan_angle_rank: i8,
    #[pasture(BUILTIN_SCAN_ANGLE)]
    pub scan_angle: i16,
    #[pasture(BUILTIN_USER_DATA)]
    pub user_data: u8,
    #[pasture(BUILTIN_POINT_SOURCE_ID)]
    pub point_source_id: u16,
    #[pasture(BUILTIN_COLOR_RGB)]
    pub color_rgb: Vector3<u16>,
    #[pasture(BUILTIN_GPS_TIME)]
    pub gps_time: f64,
    #[pasture(BUILTIN_NIR)]
    pub nir: u16,
    // #[pasture(BUILTIN_WAVE_PACKET_DESCRIPTOR_INDEX)]
    // pub wave_packet_descriptor_index: u8,
    // #[pasture(BUILTIN_WAVEFORM_DATA_OFFSET)]
    // pub waveform_data_offset: u64,
    // #[pasture(BUILTIN_WAVEFORM_PACKET_SIZE)]
    // pub waveform_packet_size: u32,
    // #[pasture(BUILTIN_RETURN_POINT_WAVEFORM_LOCATION)]
    // pub return_point_waveform_location: f32,
    // #[pasture(BUILTIN_WAVEFORM_PARAMETERS)]
    // pub waveform_parameters: Vector3<f32>,
}

fn _generate_vec3f32(rng: &mut ThreadRng) -> Vector3<f32> {
    Vector3::new(rng.gen_range(-30.0..10.0), rng.gen_range(-11.1..10.0), 31.0)
}
fn generate_vec3f64(rng: &mut ThreadRng) -> Vector3<f64> {
    Vector3::new(rng.gen_range(0.0..10.0), rng.gen_range(0.0..10.0), 1.0)
}
fn generate_vec3u16(rng: &mut ThreadRng) -> Vector3<u16> {
    Vector3::new(rng.gen_range(11..120), rng.gen_range(11..120), 42)
}

fn main() -> () {
    let mut buffer = PerAttributeVecPointStorage::new(SimplePoint::layout());

    //generate random points for the pointcloud
    let points: Vec<SimplePoint> = (0..100000)
        .into_par_iter()
        .map(|p| {
            let mut rng = rand::thread_rng();
            //generate plane points (along x- and y-axis)
            let mut point = SimplePoint {
                position: generate_vec3f64(&mut rng),
                intensity: rng.gen_range(200..800),
                return_number: rng.gen_range(20..80),
                num_of_returns: rng.gen_range(20..80),
                classification_flags: rng.gen_range(7..20),
                scanner_channel: rng.gen_range(7..20),
                scan_dir_flag: rng.gen_bool(0.47),
                edge_of_flight_line: rng.gen_bool(0.81),
                classification: rng.gen_range(121..200),
                scan_angle_rank: rng.gen_range(-121..20),
                scan_angle: rng.gen_range(-21..8),
                user_data: rng.gen_range(1..8),
                point_source_id: rng.gen_range(9..89),
                color_rgb: generate_vec3u16(&mut rng),
                gps_time: rng.gen_range(-22.4..81.3),
                nir: rng.gen_range(4..82),
                // wave_packet_descriptor_index: rng.gen_range(2..42),
                // waveform_data_offset: rng.gen_range(1..31),
                // waveform_packet_size: rng.gen_range(32..64),
                // return_point_waveform_location: rng.gen_range(-32.2..64.1),
                // waveform_parameters: generate_vec3f32(&mut rng),
            };
            //generate z-axis points for the line
            if p % 4 == 0 {
                point.position = Vector3::new(0.0, 0.0, rng.gen_range(0.0..20.0));
                point.intensity = 200
            }
            //generate outliers
            if p % 50 == 0 {
                point.position.z = rng.gen_range(5.0..7.3);
                point.intensity = 100
            }
            point
        })
        .collect();

    buffer.push_points(&points);
    println!("done generating pointcloud, size: {}", buffer.len());
    let mut filtered = PerAttributeVecPointStorage::new(buffer.point_layout().clone());
    voxelgrid_filter(&buffer, 0.9, 0.9, 0.9, &mut filtered);
    println!("done filtering pointcloud");
    println!("filtered cloud size: {:?}", filtered.len());

    // let writer = BufWriter::new(File::create("filtered_cloud.las").unwrap());
    // let writer2 = BufWriter::new(File::create("unfiltered_cloud.las").unwrap());
    // let header = Builder::from((1, 4)).into_header().unwrap();
    // let header2 = header.clone();
    // let mut writer = LASWriter::from_writer_and_header(writer, header, false).unwrap();
    // let mut writer2 = LASWriter::from_writer_and_header(writer2, header2, false).unwrap();
    // println!("start writing unfiltered pointcloud");
    // writer2.write(&buffer).unwrap();
    // println!("start writing filtered pointcloud");
    // writer.write(&filtered).unwrap();
    // println!("done writing pointclouds");
}
