use pasture_core::gpu;
use pasture_core::nalgebra::Vector3;
use pasture_core::containers::{PerAttributeVecPointStorage};
use pasture_derive::PointType;
use pasture_core::layout::{attributes, PointAttributeDefinition, PointAttributeDataType};
use pasture_core::layout::PointType;
use bytemuck::__core::convert::TryInto;

// Custom PointLayout
#[derive(PointType, Debug)]
#[repr(C)]
struct MyPointType {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_COLOR_RGB)]
    pub icolor: Vector3<u16>,
    #[pasture(attribute = "MyColorF32")]
    pub fcolor: Vector3<f32>,
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
        },
        MyPointType {
            position: Vector3::new(0.0, 1.0, 0.0),
            icolor: Vector3::new(0, 255, 0),
            fcolor: Vector3::new(0.0, 1.0, 0.0),
        },
        MyPointType {
            position: Vector3::new(0.0, 0.0, 1.0),
            icolor: Vector3::new(0, 0, 255),
            fcolor: Vector3::new(0.0, 0.0, 1.0),
        },
    ];

    let layout = MyPointType::layout();
    let mut point_buffer = PerAttributeVecPointStorage::new(layout);
    point_buffer.push_points(points.as_slice());

    let custom_color_attrib =
        PointAttributeDefinition::custom("MyColorF32", PointAttributeDataType::Vec3f32);

    // == GPU ====================================================================================

    // Create a device

    let device = gpu::Device::default().await;
    device.print_device_info();

    let mut device = gpu::Device::new(
        gpu::DeviceOptions {
            device_power: gpu::DevicePower::High,
            device_backend: gpu::DeviceBackend::Vulkan,
        }
    ).await;
    device.print_device_info();

    // TODO: this may be useful when trying to create buffer from just the layout
    // println!("{:?}", point_buffer.point_layout());

    let buffer_infos = vec![
        gpu::BufferInfo {
            attribute: &attributes::POSITION_3D,
            binding: 0
        },
        gpu::BufferInfo {
            attribute: &attributes::COLOR_RGB,
            binding: 1
        },
        gpu::BufferInfo {
            attribute: &custom_color_attrib,
            binding: 2
        },
    ];

    // device.upload(device_buffers);
    device.upload(&mut point_buffer, buffer_infos);
    device.set_compute_shader(include_str!("device.comp"));
    device.compute(1, 1, 1);
    println!("\n===== COMPUTE =====\n");

    let results_as_bytes = device.download().await;

    let pos_result_vec: Vec<f64> = results_as_bytes[0]
        .chunks_exact(8)
        .map(|b| f64::from_ne_bytes(b.try_into().unwrap()))
        .collect();
    println!("Positions: {:?}", pos_result_vec);

    // TODO: need to convert back to u16
    let icol_result_vec: Vec<u32> = results_as_bytes[1]
        .chunks_exact(4)
        .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
        .collect();
    println!("Colors (u16): {:?}", icol_result_vec);

    let fcol_result_vec: Vec<f32> = results_as_bytes[2]
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
        .collect();
    println!("Colors (f32): {:?}", fcol_result_vec);
}