use pasture_core::gpu;
use pasture_core::nalgebra::Vector3;
use pasture_core::containers::{PerAttributeVecPointStorage, PointBufferExt};
use pasture_derive::PointType;
use pasture_core::layout::{attributes, PointAttributeDefinition, PointAttributeDataType};
use pasture_core::layout::PointType;
use pasture_core::gpu::{GpuPointBufferPerAttribute};

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

fn main() {
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
            gps_time: 1.0
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
            gps_time: 2.0
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
            gps_time: 3.0
        },
    ];

    // Can use per-attribute layout...
    let layout = MyPointType::layout();
    let mut point_buffer = PerAttributeVecPointStorage::new(layout);
    point_buffer.push_points(points.as_slice());

    // ... or interleaved layout (comment out to try per-attribute)
    // let layout = MyPointType::layout();
    // let mut point_buffer = InterleavedVecPointStorage::new(layout);
    // point_buffer.push_points(points.as_slice());

    let custom_color_attrib =
        PointAttributeDefinition::custom("MyColorF32", PointAttributeDataType::Vec3f32);

    let custom_byte_vec_attrib =
        PointAttributeDefinition::custom("MyVec3U8", PointAttributeDataType::Vec3u8);

    let custom_int_attrib =
        PointAttributeDefinition::custom("MyInt32", PointAttributeDataType::I32);

    // == GPU ====================================================================================

    // Create a device with defaults...
    let device = gpu::Device::default().await;
    let device = match device {
        Ok(d) => d ,
        Err(_) => {
            println!("Failed to request device. Aborting.");
            return;
        }
    };
    device.print_device_info();

    // ... or custom options
    let device = gpu::Device::new(
        gpu::DeviceOptions {
            device_power: gpu::DevicePower::High,
            device_backend: gpu::DeviceBackend::Vulkan,
            use_adapter_features: true,
            use_adapter_limits: true,
        }
    ).await;

    let mut device = match device {
        Ok(d) => d ,
        Err(_) => {
            println!("Failed to request device. Aborting.");
            return;
        }
    };

    device.print_device_info();
    device.print_active_features();
    device.print_active_limits();
    println!("\n");

    // Connects point buffer attributes to shader bindings
    let buffer_infos = vec![
        gpu::BufferInfoPerAttribute {
            attribute: &attributes::POSITION_3D,
            binding: 0,
        },
        gpu::BufferInfoPerAttribute {
            attribute: &attributes::COLOR_RGB,
            binding: 1,
        },
        gpu::BufferInfoPerAttribute {
            attribute: &custom_color_attrib,
            binding: 2,
        },
        gpu::BufferInfoPerAttribute {
            attribute: &custom_byte_vec_attrib,
            binding: 3,
        },
        gpu::BufferInfoPerAttribute {
            attribute: &attributes::CLASSIFICATION,
            binding: 4,
        },
        gpu::BufferInfoPerAttribute {
            attribute: &attributes::INTENSITY,
            binding: 5,
        },
        gpu::BufferInfoPerAttribute {
            attribute: &attributes::SCAN_ANGLE,
            binding: 6,
        },
        gpu::BufferInfoPerAttribute {
            attribute: &attributes::SCAN_DIRECTION_FLAG,
            binding: 7,
        },
        gpu::BufferInfoPerAttribute {
            attribute: &custom_int_attrib,
            binding: 8,
        },
        gpu::BufferInfoPerAttribute {
            attribute: &attributes::WAVEFORM_PACKET_SIZE,
            binding: 9,
        },
        gpu::BufferInfoPerAttribute {
            attribute: &attributes::RETURN_POINT_WAVEFORM_LOCATION,
            binding: 10,
        },
        gpu::BufferInfoPerAttribute {
            attribute: &attributes::GPS_TIME,
            binding: 11,
        },
    ];

    let mut gpu_point_buffer = GpuPointBufferPerAttribute::new();
    gpu_point_buffer.malloc(3, &buffer_infos, &mut device.wgpu_device);
    gpu_point_buffer.upload(&mut point_buffer, 0..3, &buffer_infos, &mut device.wgpu_device, &device.wgpu_queue);

    device.set_bind_group(0, gpu_point_buffer.bind_group_layout.as_ref().unwrap(), gpu_point_buffer.bind_group.as_ref().unwrap());
    // device.set_compute_shader_glsl(include_str!("shaders/device.comp"));
    device.compute(1, 1, 1);
    println!("\n===== COMPUTE =====\n");

    println!("Before:");
    for point in point_buffer.iter_point::<MyPointType>() {
        println!("{:?}", point);
    }
    println!();

    gpu_point_buffer.download_into_per_attribute(&mut point_buffer, 0..3, &buffer_infos, &device.wgpu_device).await;

    println!("After:");
    for point in point_buffer.iter_point::<MyPointType>() {
        println!("{:?}", point);
    }
}