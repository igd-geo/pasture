#[cfg(feature = "gpu")]
mod ex {

    use pasture_core::containers::{InterleavedVecPointStorage, PointBuffer, PointBufferExt};
    use pasture_core::gpu;
    use pasture_core::gpu::GpuPointBufferInterleaved;
    use pasture_core::layout::PointType;
    use pasture_core::layout::{attributes, PointAttributeDataType, PointAttributeDefinition};
    use pasture_core::nalgebra::Vector3;
    use pasture_derive::PointType;

    #[repr(C)]
    #[derive(PointType, Debug)]
    struct MyPointType {
        #[pasture(BUILTIN_POSITION_3D)]
        pub position: Vector3<f64>,
        #[pasture(BUILTIN_COLOR_RGB)]
        pub icolor: Vector3<u16>,
        #[pasture(attribute = "MyColorF32")]
        pub fcolor: Vector3<f32>,
        #[pasture(attribute = "MyVec3U8")]
        pub byte_vec: Vector3<u8>,
        #[pasture(BUILTIN_CLASSIFICATION)]
        pub classification: u8,
        #[pasture(BUILTIN_INTENSITY)]
        pub intensity: u16,
        #[pasture(BUILTIN_SCAN_ANGLE)]
        pub scan_angle: i16,
        #[pasture(BUILTIN_SCAN_DIRECTION_FLAG)]
        pub scan_dir_flag: bool,
        #[pasture(attribute = "MyInt32")]
        pub my_int: i32,
        #[pasture(BUILTIN_WAVEFORM_PACKET_SIZE)]
        pub packet_size: u32,
        #[pasture(BUILTIN_RETURN_POINT_WAVEFORM_LOCATION)]
        pub ret_point_loc: f32,
        #[pasture(BUILTIN_GPS_TIME)]
        pub gps_time: f64,
    }

    pub fn main() {
        futures::executor::block_on(run());
    }

    async fn run() {
        // == Init point buffer ======================================================================

        let points = vec![
            MyPointType {
                position: Vector3::new(1.0, 0.0, 0.0),
                icolor: Vector3::new(255, 0, 0),
                fcolor: Vector3::new(1.0, 1.0, 1.0),
                byte_vec: Vector3::new(1, 0, 0),
                classification: 1,
                intensity: 1,
                scan_angle: -1,
                scan_dir_flag: true,
                my_int: -100000,
                packet_size: 1,
                ret_point_loc: 1.0,
                gps_time: 1.0,
            },
            MyPointType {
                position: Vector3::new(0.0, 1.0, 0.0),
                icolor: Vector3::new(0, 255, 0),
                fcolor: Vector3::new(0.0, 1.0, 0.0),
                byte_vec: Vector3::new(0, 1, 0),
                classification: 2,
                intensity: 2,
                scan_angle: -2,
                scan_dir_flag: false,
                my_int: -200000,
                packet_size: 2,
                ret_point_loc: 2.0,
                gps_time: 2.0,
            },
            MyPointType {
                position: Vector3::new(0.0, 0.0, 1.0),
                icolor: Vector3::new(0, 0, 255),
                fcolor: Vector3::new(0.0, 0.0, 1.0),
                byte_vec: Vector3::new(0, 0, 1),
                classification: 3,
                intensity: 3,
                scan_angle: -3,
                scan_dir_flag: true,
                my_int: -300000,
                packet_size: 3,
                ret_point_loc: 3.0,
                gps_time: 3.0,
            },
        ];
        let a: f64 = 1.0;
        println!("{:?}", a.to_be_bytes());
        let layout = MyPointType::layout();
        let mut point_buffer = InterleavedVecPointStorage::new(layout);
        point_buffer.push_points(points.as_slice());

        let mut octree =
            pasture_tools::acceleration_structures::GpuOctree::new(&point_buffer).await;
        octree.construct(MyPointType::layout());
    }
}

#[cfg(feature = "gpu")]
fn main() {
    ex::main();
}

#[cfg(not(feature = "gpu"))]
fn main() {}
