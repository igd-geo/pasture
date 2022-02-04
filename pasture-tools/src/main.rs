#[macro_use]
extern crate log;

mod ex {

    use pasture_core::containers::InterleavedVecPointStorage;
    use pasture_core::layout::PointType;
    use pasture_core::nalgebra::Vector3;
    use pasture_derive::PointType;
    use pasture_io::base::PointReader;
    use pasture_io::las::LASReader;
    use pasture_io::las::LasPointFormat0;

    use anyhow::Result;
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

    async fn run() -> Result<()> {
        // == Init point buffer ======================================================================
        env_logger::init();
        info!("starting up");
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

        let layout = MyPointType::layout();
        let mut point_buffer = InterleavedVecPointStorage::new(layout);
        point_buffer.push_points(points.as_slice());

        let mut reader = LASReader::from_path(
            //"/home/jnoice/dev/pasture/pasture-io/examples/in/10_points_format_1.las",
            "/home/jnoice/Downloads/WSV_Pointcloud_Tile-3-1.laz",
            //"/home/jnoice/Downloads/interesting.las",
            //"/home/jnoice/Downloads/20150930_matsch_flight2_rgb_densified_point_cloud_part_1 - Cloud.las",
            //"/home/jnoice/Downloads/45123H3316.laz",
            //"/home/jnoice/Downloads/OR_Camp_Creek_OLC_2008_000001.laz",
            //"/home/jnoice/Downloads/tirol.las",
        )?;
        let count = reader.remaining_points();
        let mut buffer =
            InterleavedVecPointStorage::with_capacity(count, LasPointFormat0::layout());
        reader.read_into(&mut buffer, count)?;

        let bounds = reader.get_metadata().bounds().unwrap();

        // let device = gpu::Device::new(gpu::DeviceOptions {
        //     device_power: gpu::DevicePower::High,
        //     device_backend: gpu::DeviceBackend::Vulkan,
        //     use_adapter_features: true,
        //     use_adapter_limits: true,
        // })
        // .await;
        //
        // let mut device = match device {
        //     Ok(d) => d,
        //     Err(_) => {
        //         println!("Failed to request device. Aborting.");
        //         return Ok(());
        //     }
        // };
        //
        // device.print_device_info();
        // device.print_active_features();
        // device.print_active_limits();
        // println!("\n");
        //
        // let attribs = &[attributes::POSITION_3D];
        //
        // let buffer_info_interleaved = gpu::BufferInfoInterleaved {
        //     attributes: attribs,
        //     binding: 0,
        // };
        //
        // let mut gpu_point_buffer = GpuPointBufferInterleaved::new();
        // gpu_point_buffer.malloc(
        //     count as u64,
        //     &buffer_info_interleaved,
        //     &mut device.wgpu_device,
        // );
        // gpu_point_buffer.upload(
        //     &buffer,
        //     0..buffer.len(),
        //     &buffer_info_interleaved,
        //     &mut device.wgpu_device,
        //     &device.wgpu_queue,
        // );
        //
        // device.set_bind_group(
        //     0,
        //     gpu_point_buffer.bind_group_layout.as_ref().unwrap(),
        //     gpu_point_buffer.bind_group.as_ref().unwrap(),
        // );
        // device.set_compute_shader_glsl(include_str!(
        //     "acceleration_structures/shaders/interleaved.comp"
        // ));
        // device.compute(1, 1, 1);
        //
        // println!("\n===== COMPUTE =====\n");
        //
        // println!("Before:");
        // for point in point_buffer.iter_point::<LasPointFormat0>().take(5) {
        //     println!("{:?}", point);
        // }
        // println!();
        //
        // gpu_point_buffer
        //     .download_into_interleaved(
        //         &mut buffer,
        //         0..count,
        //         &buffer_info_interleaved,
        //         &device.wgpu_device,
        //     )
        //     .await;
        //
        // println!("After:");
        // for point in point_buffer.iter_point::<LasPointFormat0>().take(5) {
        //     println!("{:?}", point);
        // }

        let mut octree =
            pasture_tools::acceleration_structures::GpuOctree::new(&buffer, bounds, 500).await;
        let mut octree = match octree {
            Ok(a) => a,
            Err(b) => {
                println!("{:?}", b);
                return Ok(());
            }
        };

        octree.construct().await;
        Ok(())
    }
}

#[cfg(feature = "gpu")]
fn main() {
    ex::main();
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("Whoops");
}
