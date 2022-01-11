#[cfg(feature = "io_gpu_examples")]
mod ex {

    use crevice::std140::AsStd140;
    use pasture_core::containers::{InterleavedVecPointStorage, PointBuffer};
    use pasture_core::gpu;
    use pasture_core::gpu::GpuPointBufferInterleaved;
    use pasture_core::layout::attributes;
    use pasture_io::base::{PointReader, PointWriter, SeekToPoint};
    use pasture_io::las::{LASReader, LASWriter};
    use std::path::Path;

    // To run this example you have to enable a feature flag: --features="io_gpu_examples"

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
        // Note: this example uses a .wgsl shader which is more strict than .glsl shaders.
        // Since our positions are 64-bit floats we have to make sure our matrix elements are also 64-bit,
        // otherwise we can't multiply the two together.
        model: mint::ColumnMatrix4<f64>,
    }

    // This example uses an interleaved format. It is similar to gpu_io_per_attribute.rs.
    // Storing data in interleaved format on the GPU reduces the number of storage buffers needed,
    // but depending on how the buffer is structured it can be much more wasteful with memory because
    // of the additional padding, even with std430. This also leads to longer computation time.
    //
    // For these reasons you're probably better off sticking to the per-attribute format for now.
    pub fn main() {
        env_logger::init();

        futures::executor::block_on(run());
    }

    async fn run() {
        // == Read LAS =================================================================================

        let duration_start = chrono::Utc::now().timestamp_millis();

        // If you decide to try out your own point cloud data, remember to adjust the shaders and the attributes.
        let path = Path::new("pasture-io/examples/in/10_points_format_1.las");
        let mut las_reader = match LASReader::from_path(path) {
            Ok(reader) => {
                println!("Ok {:?}", reader.header());
                reader
            }
            Err(e) => {
                println!("Error: {}", e);
                return;
            }
        };

        let count = las_reader.point_count().unwrap();
        let mut point_buffer = InterleavedVecPointStorage::with_capacity(
            count,
            las_reader.get_default_point_layout().clone(),
        );
        las_reader.read_into(&mut point_buffer, count).unwrap();

        let duration_end = chrono::Utc::now().timestamp_millis();
        let elapsed_time = (duration_end - duration_start) as f32 / 1000.0;
        println!("Time elapsed (Read LAS): {}s", elapsed_time);

        // == GPU ======================================================================================

        let duration_start = chrono::Utc::now().timestamp_millis();

        let device = gpu::Device::new(gpu::DeviceOptions {
            device_power: gpu::DevicePower::High,
            device_backend: gpu::DeviceBackend::Vulkan,
            use_adapter_features: true,
            use_adapter_limits: true,
        })
        .await;

        let mut device = match device {
            Ok(d) => d,
            Err(_) => {
                println!("Failed to request device. Aborting.");
                return;
            }
        };

        // Connects point buffer to shader bindings
        let buffer_info = gpu::BufferInfoInterleaved {
            // Same order as in shader
            attributes: &[attributes::POSITION_3D, attributes::INTENSITY],
            binding: 0,
        };

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
        }
        .as_std140();

        let uniform_as_bytes: &[u8] = bytemuck::bytes_of(&point_uniform);
        let (uniform_bind_group_layout, uniform_bind_group) =
            device.create_uniform_bind_group(uniform_as_bytes, 0);

        // Allocate memory for point buffer and queue it for upload onto the GPU
        let mut gpu_point_buffer = GpuPointBufferInterleaved::new();
        gpu_point_buffer.malloc(point_count as u64, &buffer_info, &mut device.wgpu_device);
        gpu_point_buffer.upload(
            &mut point_buffer,
            0..point_count,
            &buffer_info,
            &mut device.wgpu_device,
            &device.wgpu_queue,
        );

        // Here: GpuPointBuffer -> "set=0",
        //       PointUniform   -> "set=1"
        device.set_bind_group(
            0,
            gpu_point_buffer.bind_group_layout.as_ref().unwrap(),
            gpu_point_buffer.bind_group.as_ref().unwrap(),
        );
        device.set_bind_group(1, &uniform_bind_group_layout, &uniform_bind_group);

        // device.set_compute_shader_glsl(include_str!("shaders/io_interleaved.comp"));
        device.set_compute_shader_wgsl(include_str!("shaders/io_interleaved.wgsl"));
        device.compute(((point_count / 128) + 1) as u32, 1, 1);

        gpu_point_buffer
            .download_into_interleaved(
                &mut point_buffer,
                0..point_count,
                &buffer_info,
                &device.wgpu_device,
            )
            .await;

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
}

#[cfg(feature = "io_gpu_examples")]
fn main() {
    ex::main();
}

#[cfg(not(feature = "io_gpu_examples"))]
fn main() {}
