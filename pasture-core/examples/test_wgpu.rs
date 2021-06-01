use pasture_core::containers::{PerAttributeVecPointStorage, PointBufferExt, PerAttributePointBufferMutExt, PointBuffer};
use pasture_core::layout::{PointType, attributes, PointAttributeDefinition, PointAttributeDataType};
use pasture_core::nalgebra::Vector3;
use pasture_derive::PointType;

use std::convert::TryInto;
use wgpu::util::{DeviceExt, BufferInitDescriptor};

// Custom PointLayout
#[derive(PointType, Debug)]
#[repr(C)]
struct MyPointType {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(attribute="Size")]
    pub size: u32,
}


// Goal of this example:
// Use PointLayout and PointBuffer to create some "pasture points"
// Send some of the point data to the GPU and manipulate them in some way
// Retrieve manipulated point data and show results
fn main() {
    futures::executor::block_on(run());
}

async fn run() {
    // Define some points
    let points = vec![
        MyPointType {
            position: Vector3::new(1.0, 2.0, 3.0),
            size: 10,
        },
        MyPointType {
            position: Vector3::new(5.0, 5.0, 5.0),
            size: 5
        },
    ];

    // Put them into a buffer (per attribute)
    let layout = MyPointType::layout();
    let mut buffer = PerAttributeVecPointStorage::new(layout);
    buffer.push_points(points.as_slice());

    println!("Points: ");
    for point in buffer.iter_point::<MyPointType>() {
        println!("\t{:?}", point);
    }

    on_gpu(&mut buffer).await;
}

async fn on_gpu(pasture_buffer: &mut PerAttributeVecPointStorage) {
    // == Get device ==============================================================================

    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);

    // The adapter gives us a handle to the actual device.
    // We can query some GPU information, such as the device name, its type (discrete vs integrated)
    // or the backend that is being used.
    let adapter = instance.request_adapter(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance, // or LowPower for iGPU (encouraged unless performance really needed)
            compatible_surface: None
        }
    ).await.unwrap();
    println!("{:?}", adapter.get_info());
    // println!("{:?}", adapter.limits());
    // println!("{:?}", adapter.features());

    println!("{:?}", wgpu::PowerPreference::default());

    // The device gives us access to the core of the WebGPU API.
    // The queue allows us to asynchronously send work to the GPU.
    let (device, queue) = adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            features: wgpu::Features::empty(),
            limits: wgpu::Limits::default()
        },
        None,
    ).await.unwrap();

    // == Setup shader ============================================================================

    // WebGPU wants its shaders pre-compiled in binary SPIR-V format.
    // So we'll take the source code of our compute shader and compile it
    // with the help of the shaderc crate.
    let cs_src = include_str!("shader.comp");
    let mut compiler = shaderc::Compiler::new().unwrap();
    let cs_spirv = compiler.compile_into_spirv(
        cs_src,
        shaderc::ShaderKind::Compute,
        "shader.comp",
        "main",
        None,
    ).unwrap();
    let cs_data = wgpu::util::make_spirv(cs_spirv.as_binary_u8());

    // Now with the binary data we can create our ShaderModule,
    // which will be executed on the GPU within our compute pipeline.
    let cs_module = device.create_shader_module(
        &wgpu::ShaderModuleDescriptor {
            label: None,
            source: cs_data,
            flags: wgpu::ShaderFlags::default()
        }
    );

    // == Setup data ==============================================================================

    // Separate attributes into their own lists so that we can send them to the GPU.
    // TODO: how to avoid doing this? Will be problematic for different/unknown PointLayouts
    let nr_points = (*pasture_buffer).len();
    let mut positions: Vec<f64> = vec![];
    let mut sizes: Vec<u32> = vec![];
    for point in (*pasture_buffer).iter_point::<MyPointType>() {
        positions.push(point.position.x);
        positions.push(point.position.y);
        positions.push(point.position.z);
        sizes.push(point.size);
    }

    // First the positions (note the *3)
    let data_size_pos_attrib = (3 * nr_points * std::mem::size_of::<f64>()) as wgpu::BufferAddress;
    let pos_buffer = device.create_buffer_init(
        &BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&positions),
            usage: wgpu::BufferUsage::STORAGE
                | wgpu::BufferUsage::COPY_SRC
                | wgpu::BufferUsage::COPY_DST,
        }
    );

    // Then the sizes
    let data_size_size_attrib = (nr_points * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
    let size_buffer = device.create_buffer_init(
        &BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&sizes),
            usage: wgpu::BufferUsage::STORAGE
                | wgpu::BufferUsage::COPY_SRC
                | wgpu::BufferUsage::COPY_DST,
        }
    );

    // The buffer to write the position updates to
    let result_pos_buffer = device.create_buffer(
        &wgpu::BufferDescriptor {
            label: None,
            size: data_size_pos_attrib,
            usage: wgpu::BufferUsage::COPY_DST
                | wgpu::BufferUsage::MAP_READ,
            mapped_at_creation: false
        }
    );

    // The buffer to write the size updates to
    let result_size_buffer = device.create_buffer(
        &wgpu::BufferDescriptor {
            label: None,
            size: data_size_size_attrib,
            usage: wgpu::BufferUsage::COPY_DST
                | wgpu::BufferUsage::MAP_READ,
            mapped_at_creation: false
        }
    );

    // TODO: ideally use this, but separation returned 0 for sizes...
    let _result_buffer = device.create_buffer(  // Prefix with _ to suppress "unused" warning
                                                &wgpu::BufferDescriptor {
                                                    label: None,
                                                    size: data_size_pos_attrib + data_size_size_attrib,
                                                    usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::MAP_READ,
                                                    mapped_at_creation: false
                                                }
    );

    // == Setup bind groups =======================================================================

    // First define the layout
    let bind_group_layout = device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // Positions for binding 0: can be written to
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None
                    },
                    count: None
                },
                // Sizes for binding 1: can also be written to
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None
                    },
                    count: None
                },
            ]
        }
    );

    // Then the actual bind group
    let bind_group = device.create_bind_group(
        &wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pos_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: size_buffer.as_entire_binding(),
                }
            ]
        }
    );

    // == Setup pipeline ==========================================================================

    let compute_pipeline_layout = device.create_pipeline_layout(
        &wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[]
        }
    );

    let compute_pipeline = device.create_compute_pipeline(
        &wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&compute_pipeline_layout),
            module: &cs_module,
            entry_point: "main"
        }
    );

    // == Compute =================================================================================

    // Use a CommandEncoder to batch all commands that you wish to send to the GPU to execute.
    // The resulting CommandBuffer can then be submitted to the GPU via a Queue.
    // Signal the end of the batch with CommandEncoder#finish().
    let mut encoder = device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: None
        }
    );

    {
        // The compute pass will start ("dispatch") our compute shader.
        let mut compute_pass = encoder.begin_compute_pass(
            &wgpu::ComputePassDescriptor {
                label: None
            }
        );
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.insert_debug_marker("Pasture Compute Debug");
        compute_pass.dispatch(1, 1, 1);
    }

    {
        // Copy positions
        encoder.copy_buffer_to_buffer(
            &pos_buffer, 0,
            &result_pos_buffer, 0,
            data_size_pos_attrib
        );
    }

    {
        // Copy sizes
        encoder.copy_buffer_to_buffer(
            &size_buffer, 0,
            &result_size_buffer, 0,
            data_size_size_attrib
        );
    }

    queue.submit(Some(encoder.finish()));

    // == Retrieve results ========================================================================

    let result_buffer_slice = result_pos_buffer.slice(..);
    let result_buffer_future = result_buffer_slice.map_async(wgpu::MapMode::Read);
    device.poll(wgpu::Maintain::Wait); // Should be called in event loop or other thread ...
    if let Ok(()) = result_buffer_future.await {
        let result_as_bytes = result_buffer_slice.get_mapped_range();
        let pos_result: Vec<f64> = result_as_bytes
            .chunks_exact(8)
            .map(|b| f64::from_ne_bytes(b.try_into().unwrap()))
            .collect();

        // Drop all mapped views before unmapping buffer
        drop(result_as_bytes);
        result_pos_buffer.unmap();

        let pos_attribs = (*pasture_buffer).get_attribute_range_mut::<Vector3<f64>>(
            0..nr_points, &attributes::POSITION_3D
        );

        for i in 0..pos_attribs.len() {
            pos_attribs[i].x = pos_result[i * 3 + 0];
            pos_attribs[i].y = pos_result[i * 3 + 1];
            pos_attribs[i].z = pos_result[i * 3 + 2];
        }
    }

    let result_buffer_slice = result_size_buffer.slice(..);
    let result_buffer_future = result_buffer_slice.map_async(wgpu::MapMode::Read);
    device.poll(wgpu::Maintain::Wait); // Should be called in event loop or other thread ...
    if let Ok(()) = result_buffer_future.await {
        let result_as_bytes = result_buffer_slice.get_mapped_range();
        let size_result: Vec<u32> = result_as_bytes
            .chunks_exact(4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
            .collect();

        // Drop all mapped views before unmapping buffer
        drop(result_as_bytes);
        result_size_buffer.unmap();

        // TODO: how to not redefine custom type
        let custom_size_attrib = PointAttributeDefinition::custom(
            "Size",
            PointAttributeDataType::U32
        );

        let size_attribs = (*pasture_buffer).get_attribute_range_mut::<u32>(
            0..nr_points, &custom_size_attrib
        );

        for i in 0..size_attribs.len() {
            size_attribs[i] = size_result[i];
        }
    }

    println!("Updated points: ");
    for point in (*pasture_buffer).iter_point::<MyPointType>() {
        println!("\t{:?}", point);
    }
}