use pasture_core::containers::attr1::AttributeIteratorByValue;
use pasture_core::{
    containers::{
        InterleavedPointBufferMut, InterleavedVecPointStorage, PointBuffer, PointBufferExt,
    },
    gpu,
    layout::{attributes, PointAttributeDataType, PointAttributeDefinition, PointLayout},
    nalgebra::Vector3,
};
use pasture_derive::PointType;
use std::convert::TryInto;

use wgpu;

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

struct OctreeRegion {
    bounds: [Vector3<f64>; 8],
    partition: Vec<usize>,
    points_per_partition: Vec<usize>,
    start: usize,
    end: usize,
    children: Option<[Box<OctreeNode>; 8]>,
}

struct OctreeLeaf {
    point: Vector3<f64>,
}

enum OctreeNode {
    OctreeRegion,
    OctreeLeaf,
}

pub struct GpuOctree<'a> {
    device: &'a mut gpu::Device<'a>,
    buffer_bind_group: Option<wgpu::BindGroup>,
    buffer_bind_group_layout: Option<wgpu::BindGroupLayout>,
    point_buffer: &'a dyn PointBuffer,
    root_node: Option<OctreeRegion>,
}

impl OctreeRegion {}

impl<'a> GpuOctree<'a> {
    pub fn new(point_buffer: &'a dyn PointBuffer, device: &'a mut gpu::Device<'a>) -> Self {
        GpuOctree {
            device: device,
            buffer_bind_group: None,
            buffer_bind_group_layout: None,
            point_buffer: point_buffer,
            root_node: None,
        }
    }
    pub async fn construct(&'a mut self, layout: PointLayout) {
        let point_count = self.point_buffer.len();
        let mut points: Vec<Vector3<f64>> = Vec::new();
        let point_iterator: AttributeIteratorByValue<Vector3<f64>, dyn PointBuffer> =
            self.point_buffer.iter_attribute(&attributes::POSITION_3D);
        let mut raw_points = vec![0u8; 24 * point_count];
        self.point_buffer.get_raw_attribute_range(
            0..point_count,
            &attributes::POSITION_3D,
            raw_points.as_mut_slice(),
        );
        let mut blah: Vec<f64> = raw_points
            .chunks_exact(8)
            .map(|b| f64::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        for point in point_iterator {
            points.push(point);
        }
        println!("{:?}", blah);

        let max = self.find_max(&raw_points, point_count).await;
        println!("{}", max);
        // let mut root_node = OctreeRegion {
        //     bounds: [Vector3::new(f64::MIN, f64::MAX, f64::MIN), Vector3::new(f64::MAX, f64::MAX, f64::MIN),]
        //     partition: vec![point_count],
        //     points_per_partition: vec![point_count],
        //     start: 0,
        //     end: point_count - 1,
        //     children: None,
        // };
        //let mut current_nodes = vec![&root_node];
        let tree_depth = 1;
        let new_nodes_created = false;
        let num_leaves = 0;
        loop {
            if !new_nodes_created {
                break;
            }
        }
    }

    async fn find_max(&'a mut self, points: &[u8], count: usize) -> u32 {
        let mut result: u32 = 0;
        let mut max_value_buffer = wgpu::util::DeviceExt::create_buffer_init(
            &self.device.wgpu_device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("max_value_buffer"),
                contents: &result.to_ne_bytes(),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::MAP_READ
                    | wgpu::BufferUsages::MAP_WRITE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            },
        );

        let buffer = Some(wgpu::util::DeviceExt::create_buffer_init(
            &self.device.wgpu_device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("gpu_point_buffer"),
                contents: points,
                usage: wgpu::BufferUsages::STORAGE,
            },
        ));

        self.buffer_bind_group_layout = Some(self.device.wgpu_device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("compute_bind_group_layout"),
            },
        ));

        self.buffer_bind_group = Some(self.device.wgpu_device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("storage_bind_group"),
                layout: self.buffer_bind_group_layout.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: max_value_buffer.as_entire_binding(),
                    },
                ],
            },
        ));
        self.device.set_bind_group(
            0,
            self.buffer_bind_group_layout.as_ref().unwrap(),
            self.buffer_bind_group.as_ref().unwrap(),
        );

        self.device
            .set_compute_shader_glsl(include_str!("shaders/find_max_values.comp"));
        self.device.compute(count as u32, 1, 1);

        let max_value_buffer_slice = max_value_buffer.slice(..);
        let mapped_future = max_value_buffer_slice.map_async(wgpu::MapMode::Read);
        self.device.wgpu_device.poll(wgpu::Maintain::Wait);

        if let Ok(()) = mapped_future.await {
            let mapped_max_value_buffer = max_value_buffer.slice(..).get_mapped_range().to_vec();
            println!("{:?}", mapped_max_value_buffer);
            println!("----------");
            println!("{:?}", points);
            result = mapped_max_value_buffer
                .chunks_exact(4)
                .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
                .last()
                .unwrap();
        }
        result
    }
}
