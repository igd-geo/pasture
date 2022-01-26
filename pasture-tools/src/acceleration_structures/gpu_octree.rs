use pasture_core::containers::attr1::AttributeIteratorByValue;
use pasture_core::math::AABB;
use pasture_core::nalgebra::Point3;
use pasture_core::{
    containers::{PointBuffer, PointBufferExt},
    layout::{attributes, PointLayout},
    nalgebra::Vector3,
};
use pasture_derive::PointType;
use std::borrow::Cow;
use std::convert::TryInto;
use std::fs::File;
use std::io::prelude::*;

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
#[derive(Debug, Clone)]
struct OctreeNode {
    bounds: AABB<f64>,
    children: Option<Box<[OctreeNode]>>,
    node_partitioning: [u32; 8],
    points_per_partition: [u32; 8],
    point_start: u32,
    point_end: u32,
}

pub struct GpuOctree<'a> {
    gpu_device: wgpu::Device,
    gpu_queue: wgpu::Queue,
    point_buffer: &'a dyn PointBuffer,
    point_partitioning: Vec<u32>,
    root_node: Option<OctreeNode>,
    bounds: AABB<f64>,
    points_per_node: u32,
}

impl OctreeNode {
    fn is_leaf(&self, points_per_node: u32) -> bool {
        // println!(
        //     "\npoint start: {}, point end: {}\n",
        //     self.point_start, self.point_end
        // );
        return self.point_end - self.point_start <= points_per_node;
    }
    fn into_raw(&self) -> Vec<u8> {
        let mut raw_node: Vec<u8> = Vec::new();
        for coord in self.bounds.min().iter() {
            raw_node.append(&mut coord.to_le_bytes().to_vec());
        }
        for coord in self.bounds.max().iter() {
            raw_node.append(&mut coord.to_le_bytes().to_vec());
        }
        raw_node.append(
            &mut self
                .node_partitioning
                .map(|x| x.to_le_bytes())
                .to_vec()
                .into_iter()
                .flatten()
                .collect(),
        );
        raw_node.append(
            &mut self
                .points_per_partition
                .map(|x| x.to_le_bytes())
                .to_vec()
                .into_iter()
                .flatten()
                .collect(),
        );
        raw_node.append(&mut self.point_start.to_le_bytes().to_vec());
        //[0u8; 4].iter().for_each(|&x| raw_node.push(x));
        raw_node.append(&mut self.point_end.to_le_bytes().to_vec());
        //[0u8; 4].iter().for_each(|&x| raw_node.push(x));
        raw_node
    }
    fn from_raw(mut data: Vec<u8>) -> Self {
        let raw_bounds: Vec<u8> = data.drain(..24).collect();
        let bounds_iter = raw_bounds.chunks_exact(8);
        let bounds_min: Point3<f64> = Point3 {
            coords: Vector3::from_vec(
                bounds_iter
                    .take(3)
                    .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
            ),
        };
        let raw_bounds: Vec<u8> = data.drain(..24).collect();
        let bounds_iter = raw_bounds.chunks_exact(8);
        let bounds_max: Point3<f64> = Point3 {
            coords: Vector3::from_vec(
                bounds_iter
                    .take(3)
                    .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
            ),
        };
        let mut rest_data: Vec<u32> = data
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let mut rest_iter = rest_data.iter_mut();
        let mut node_partitioning = [0u32; 8];
        for i in 0..8 {
            node_partitioning[i] = *rest_iter.next().unwrap();
        }
        let mut points_per_partition = [0u32; 8];
        for i in 0..8 {
            points_per_partition[i] = *rest_iter.next().unwrap();
        }
        let points_start = *rest_iter.next().unwrap();
        //rest_iter.next();
        let points_end = *rest_iter.next().unwrap();

        OctreeNode {
            bounds: AABB::from_min_max(bounds_min, bounds_max),
            children: None,
            node_partitioning,
            points_per_partition,
            point_start: points_start,
            point_end: points_end,
        }
    }
}

impl<'a> GpuOctree<'a> {
    pub async fn new(
        point_buffer: &'a dyn PointBuffer,
        max_bounds: AABB<f64>,
        points_per_node: u32,
    ) -> Result<GpuOctree<'a>, wgpu::RequestDeviceError> {
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: adapter.features(),
                    limits: adapter.limits(),
                    label: Some("Octree_Device"),
                },
                None,
            )
            .await?;
        println!("GPU Adapter limits: {:?}", adapter.limits());
        Ok(GpuOctree {
            gpu_device: device,
            gpu_queue: queue,
            point_buffer,
            point_partitioning: (0..point_buffer.len() as u32).collect(),
            root_node: None,
            bounds: max_bounds,
            points_per_node,
        })
    }
    pub fn print_tree(&self) {
        println!("{:?}", self.root_node);
    }
    pub async fn construct(&mut self, layout: PointLayout) {
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
        for point in point_iterator {
            points.push(point);
        }

        let mut construction_shader = File::open(
            "/home/jnoice/dev/pasture/pasture-tools/src/acceleration_structures/shaders/comp.spv",
        )
        .unwrap();
        let mut shader_bytes: Vec<u8> = Vec::new();
        for byte in construction_shader.bytes() {
            shader_bytes.push(byte.unwrap());
        }
        let shader_words: Vec<u32> = shader_bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let mut compiler = shaderc::Compiler::new().unwrap();
        // let shader = self
        //     .gpu_device
        //     .create_shader_module(&wgpu::ShaderModuleDescriptor {
        //         label: Some("NodeGenerationShader"),
        //         source: wgpu::ShaderSource::SpirV(Cow::from(shader_words.as_slice())),
        //     });
        let comp_shader = include_str!("shaders/generate_nodes.comp");
        let comp_spirv = compiler.compile_into_spirv(
            comp_shader,
            shaderc::ShaderKind::Compute,
            "ComputeShader",
            "main",
            None
        )
        .unwrap();
        let comp_data = wgpu::util::make_spirv(comp_spirv.as_binary_u8());
        let shader = self.gpu_device.create_shader_module(&wgpu::ShaderModuleDescriptor{
            label: Some("ModeGenerationShader"),
            source: comp_data,
        });
        let points_bind_group_layout =
            self.gpu_device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                    label: Some("PointBufferBindGroupLayout"),
                });
        let mut nodes_bind_group_layout =
            self.gpu_device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("NodesBindGroupLayout"),
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
                });

        let compute_pipeline_layout =
            self.gpu_device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ConstructionPipelineLayout"),
                    bind_group_layouts: &[&points_bind_group_layout, &nodes_bind_group_layout],
                    push_constant_ranges: &[],
                });
        let compute_pipeline =
            self.gpu_device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("ConstructionPipeline"),
                    layout: Some(&compute_pipeline_layout),
                    module: &shader,
                    entry_point: "main",
                });

        let gpu_point_buffer = wgpu::util::DeviceExt::create_buffer_init(
            &self.gpu_device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("PointBuffer"),
                contents: raw_points.as_slice(),
                usage: wgpu::BufferUsages::MAP_READ
                    | wgpu::BufferUsages::MAP_WRITE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::STORAGE,
            },
        );

        let mut root_node = OctreeNode {
            bounds: self.bounds,
            children: None,
            node_partitioning: [0; 8],
            points_per_partition: [0; 8],
            point_start: 0,
            point_end: point_count as u32 - 1,
        };
        root_node.node_partitioning[0] = point_count as u32 - 1;
        root_node.points_per_partition[0] = point_count as u32 - 1;
        let xdiff = &root_node.bounds.max().x - &root_node.bounds.min().x;
        let ydiff = &root_node.bounds.max().y - &root_node.bounds.min().y;
        let zdiff = &root_node.bounds.max().z - &root_node.bounds.min().z;
        println!("Point count: {}", point_count);
        println!("xdiff {}", xdiff);
        println!("ydiff {}", ydiff);
        println!("zdiff {}", zdiff);
        let xpartition = &root_node.bounds.min().x + 0.5 * xdiff;
        let ypartition = &root_node.bounds.min().y + 0.5 * ydiff;
        let zpartition = &root_node.bounds.min().z + 0.5 * zdiff;
        println!("x_partition {}", xpartition);
        println!("y_partition {}", ypartition);
        println!("z_partition {}", zpartition);

        let mut tree_depth = 1;
        let mut num_leaves: u32 = 0;
        let mut num_nodes: u32 = 1;

        let mut current_nodes = vec![&mut root_node];
        let mut children_nodes: Vec<Box<[OctreeNode]>> = Vec::new();

        let mut raw_indeces: Vec<u8> = (0u32..(point_count - 1) as u32)
            .flat_map(|x| x.to_le_bytes().to_vec())
            .collect();

        while !current_nodes.is_empty() {
            //for i in 0..2 {
            //let num_new_nodes = 8u64.pow(tree_depth) - num_leaves as u64;

            let point_index_buffer = wgpu::util::DeviceExt::create_buffer_init(
                &self.gpu_device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some("IndexBuffer"),
                    contents: raw_indeces.as_slice(),
                    usage: wgpu::BufferUsages::MAP_READ
                        | wgpu::BufferUsages::MAP_WRITE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::STORAGE,
                },
            );
            let debug_buffer = self.gpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("DebugBuffer"),
                size: (3 * 4 + 8 * 4 + 4 + 2 * 4) * current_nodes.len() as u64,
                usage: wgpu::BufferUsages::MAP_READ
                    | wgpu::BufferUsages::MAP_WRITE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });

            let new_nodes_buffer = self.gpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("NewNodesBuffer"),
                size: //(mem::size_of::<OctreeNode>() - mem::size_of::<Box<[OctreeNode]>>()) as u64
                    128
                    * current_nodes.len() as u64 * 8 as u64,
                usage: wgpu::BufferUsages::MAP_READ
                    | wgpu::BufferUsages::MAP_WRITE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let points_bind_group = self
                .gpu_device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("PointBufferBindGroup"),
                    layout: &points_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: gpu_point_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: point_index_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: debug_buffer.as_entire_binding(),
                        },
                    ],
                });
            let mut parent_nodes_raw = Vec::new();
            for node in &current_nodes {
                parent_nodes_raw.append(&mut node.into_raw());
            }

            let parent_nodes_buffer = wgpu::util::DeviceExt::create_buffer_init(
                &self.gpu_device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some("ParentNodesBuffer"),
                    contents: parent_nodes_raw.as_slice(),
                    usage: wgpu::BufferUsages::MAP_READ
                        | wgpu::BufferUsages::MAP_WRITE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::STORAGE,
                },
            );

            let nodes_bind_group = self
                .gpu_device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("NodesBindGroup"),
                    layout: &nodes_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: parent_nodes_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: new_nodes_buffer.as_entire_binding(),
                        },
                    ],
                });
            let mut encoder =
                self.gpu_device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("CommandEncoder"),
                    });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ConstructionComputePass"),
                });
                compute_pass.set_pipeline(&compute_pipeline);
                compute_pass.set_bind_group(0, &points_bind_group, &[]);
                compute_pass.set_bind_group(1, &nodes_bind_group, &[]);
                println!(
                    "Starting gpu computation with {} threads",
                    current_nodes.len()
                );
                compute_pass.insert_debug_marker("Pasture Compute Debug");
                compute_pass.dispatch(current_nodes.len() as u32, 1, 1);
            }
            self.gpu_queue.submit(Some(encoder.finish()));

            let mapped_future = point_index_buffer.slice(..).map_async(wgpu::MapMode::Read);
            self.gpu_device.poll(wgpu::Maintain::Wait);
            if let Ok(()) = mapped_future.await {
                let mapped_index_buffer = point_index_buffer.slice(..).get_mapped_range().to_vec();
                let indices: Vec<u32> = mapped_index_buffer
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .collect();

                // self.point_partitioning = indices.clone();
                // indices.iter().for_each(|x| {
                //     println!("{}", x);
                // });
                raw_indeces = mapped_index_buffer.clone();
            }
            point_index_buffer.unmap();
            let mapped_debug = debug_buffer.slice(..).map_async(wgpu::MapMode::Read);
            self.gpu_device.poll(wgpu::Maintain::Wait);
            if let Ok(()) = mapped_debug.await {
                let mut debug: Vec<u32> = debug_buffer
                    .slice(..)
                    .get_mapped_range()
                    .to_vec()
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                for k in 0..current_nodes.len() {
                    let partition_order: Vec<u32> = debug.drain(..3).collect();
                    let borders: Vec<u32> = debug.drain(..8).collect();
                    let thread_id: Vec<u32> = debug.drain(..1).collect();
                    let start_end: Vec<u32> = debug.drain(..2).collect();
                    println!(
                        " Node {} | Partition Order: {:?} \n Partition borders: {:?}\n thread index: {}\n start/end: {:?}",
                         k, partition_order, borders, thread_id.first().unwrap(), start_end,
                    );
                }
            }

            let mapped_children = new_nodes_buffer.slice(..).map_async(wgpu::MapMode::Read);
            self.gpu_device.poll(wgpu::Maintain::Wait);

            let mapped_parents = parent_nodes_buffer.slice(..).map_async(wgpu::MapMode::Read);
            self.gpu_device.poll(wgpu::Maintain::Wait);

            if matches!(mapped_parents.await, Ok(())) && matches!(mapped_children.await, Ok(())) {
                let mapped_node_buffer = parent_nodes_buffer.slice(..).get_mapped_range().to_vec();
                let nodes: Vec<OctreeNode> = mapped_node_buffer
                    .chunks_exact(120)
                    .map(|b| OctreeNode::from_raw(b.to_vec()))
                    .collect();
                let mapped_children = new_nodes_buffer.slice(..).get_mapped_range().to_vec();
                let mut children: Vec<OctreeNode> = mapped_children
                    .chunks_exact(120)
                    .map(|b| OctreeNode::from_raw(b.to_vec()))
                    .collect();
                let mut generated_children: Vec<&mut OctreeNode> = Vec::new();
                for mut node in nodes {
                    let children_sizes = node.points_per_partition.clone();
                    // println!("=========== Node ===========");
                    // println!("{:?}", node);
                    let mut local_children: Vec<OctreeNode> = children.drain(..8).collect();

                    node.children = Some(local_children.into_boxed_slice());
                    let mut node_ref = current_nodes.swap_remove(0);
                    // println!("Number of parents left: {}", current_nodes.len());
                    *node_ref = node;
                    // println!(
                    //     "Parent node point start: {}, point end: {}",
                    //     &node_ref.point_start, &node_ref.point_end
                    // );
                    // println!(
                    //     "Points per partition in parent : {:?}",
                    //     &node_ref.points_per_partition
                    // );
                    // println!("Parent node bounds: {:?}", &node_ref.bounds);
                    let mut children: &mut Box<[OctreeNode]> = node_ref.children.as_mut().unwrap();
                    let iter = (*children).iter_mut();

                    let mut child_index = 0;
                    for child in iter {
                        // println!("========== Child ===========");
                        // println!("{:?}", child);
                        // println!(
                        //     "{}, {}",
                        //     children_sizes[child_index] != 0,
                        //     !child.is_leaf(self.points_per_node)
                        // );
                        if children_sizes[child_index] != 0 && !child.is_leaf(self.points_per_node)
                        {
                            generated_children.push(child);
                        } else {
                            num_leaves += 1;
                            // println!(
                            //     "Child {} is LEAF. Point Count: {}",
                            //     child_index, children_sizes[child_index]
                            // );
                        }

                        num_nodes += 1;
                        child_index += 1;
                    }
                }
                current_nodes.append(&mut generated_children);
            }
            parent_nodes_buffer.unmap();
            new_nodes_buffer.unmap();
            // println!(
            //     "Num Nodes: {}, Num New Parents: {}, Num Leaves: {}",
            //     num_nodes,
            //     current_nodes.len(),
            //     num_leaves
            // );

            tree_depth += 1;

            // println!("{}", tree_depth);
            // println!("{:?}", current_nodes);
        }

        self.root_node = Some(root_node);
        println!("{:?}", self.root_node);
        // println!("Tree depth = {}", tree_depth);
    }
}
