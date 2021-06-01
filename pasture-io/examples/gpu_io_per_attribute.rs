use pasture_io::las::{LASReader, LASWriter};
use std::path::Path;
use pasture_io::base::{SeekToPoint, PointReader, PointWriter};
use pasture_core::containers::{PerAttributeVecPointStorage, PointBuffer};
use pasture_core::gpu;
use pasture_core::layout::attributes;
use pasture_core::gpu::GpuPointBufferPerAttribute;
use crevice::std140::AsStd140;

// To run this example you have to enable a feature flag: --features="io_gpu_example"

// log + env_logging give improved wgpu error messages.
// See https://crates.io/crates/env_logger
extern crate log;

// Important: keep in mind that uniforms use a std140 alignment, so you might have to pad your data.
// See: https://sotrh.github.io/learn-wgpu/showcase/alignment/#alignments
// This example uses 'crevice' to take care of the alignment. An alternative would be 'glsl_layout'.
#[repr(C)]
#[derive(Debug, Copy, Clone, AsStd140)]
struct PointUniform {
    point_count: u32,
    model: mint::ColumnMatrix4<f32>,
}

fn main() {
    env_logger::init();

    futures::executor::block_on(run());
}

async fn run() {
    // == Read LAS =================================================================================

    let duration_start = chrono::Utc::now().timestamp_millis();

    // If you decide to try out your own point cloud data, remember to adjust the shaders and the attributes.
    let path = Path::new("pasture-io/examples/in/10_points_format_1.las");
    let mut las_reader = match LASReader::from_path(path) {
        Ok(reader) => { println!("Ok {:?}", reader.header()); reader}
        Err(e) => { println!("Error: {}", e); return; }
    };

    let count = las_reader.point_count().unwrap();
    let mut point_buffer = PerAttributeVecPointStorage::with_capacity(
        count,
        las_reader.get_default_point_layout().clone()
    );
    las_reader.read_into(&mut point_buffer, count).unwrap();

    let duration_end = chrono::Utc::now().timestamp_millis();
    let elapsed_time = (duration_end - duration_start) as f32 / 1000.0;
    println!("Time elapsed (Read LAS): {}s", elapsed_time);

    // == GPU ======================================================================================

    let duration_start = chrono::Utc::now().timestamp_millis();

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

    // Connects point buffer attributes to shader bindings
    let buffer_infos = vec![
        gpu::BufferInfoPerAttribute {
            attribute: &attributes::POSITION_3D,
            binding: 0,
        },
        gpu::BufferInfoPerAttribute {
            attribute: &attributes::INTENSITY,
            binding: 1,
        },
    ];

    let point_count = point_buffer.len();
    println!("Number of points: {}", point_count);
    
    // Fill uniform data and generate its bind group and bind group layout
    // Note the conversion to std140
    let point_uniform = PointUniform {
        point_count: point_count as u32,
        // Scale y by half
        model: mint::ColumnMatrix4::from([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    }.as_std140();

    let uniform_as_bytes: &[u8] = bytemuck::bytes_of(&point_uniform);
    let (uniform_bind_group_layout, uniform_bind_group) = device.create_uniform_bind_group(uniform_as_bytes, 0);

    // Allocate memory for point buffer and queue it for upload onto the GPU
    let mut gpu_point_buffer = GpuPointBufferPerAttribute::new();
    gpu_point_buffer.malloc(point_count as u64, &buffer_infos, &mut device.wgpu_device);
    gpu_point_buffer.upload(&mut point_buffer, 0..point_count, &buffer_infos, &mut device.wgpu_device, &device.wgpu_queue);

    // Here: GpuPointBuffer -> "set=0",
    //       PointUniform   -> "set=1"
    device.set_bind_group(0, gpu_point_buffer.bind_group_layout.as_ref().unwrap(), gpu_point_buffer.bind_group.as_ref().unwrap());
    device.set_bind_group(1, &uniform_bind_group_layout, &uniform_bind_group);

    device.set_compute_shader_glsl(include_str!("shaders/io_per_attribute.comp"));
    device.compute(((point_count / 128) + 1) as u32, 1, 1);

    gpu_point_buffer.download_into_per_attribute(&mut point_buffer, 0..point_count, &buffer_infos, &device.wgpu_device).await;

    let duration_end = chrono::Utc::now().timestamp_millis();
    let elapsed_time = (duration_end - duration_start) as f32 / 1000.0;
    println!("Time elapsed (GPU): {}s", elapsed_time);

    // == Write LAS ================================================================================

    let duration_start = chrono::Utc::now().timestamp_millis();

    let out_path = "pasture-io/examples/out/test_output.las";
    let las_writer = LASWriter::from_path_and_header(out_path, las_reader.header().clone());
    las_writer.unwrap().write(&point_buffer).unwrap();

    let duration_end = chrono::Utc::now().timestamp_millis();
    let elapsed_time = (duration_end - duration_start) as f32 / 1000.0;
    println!("Time elapsed (Write LAS): {}s", elapsed_time);
}