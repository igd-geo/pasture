use pasture_core::containers::attr1::AttributeIteratorByValue;
use pasture_core::math::AABB;
use pasture_core::nalgebra::Point3;
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
use std::mem;
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

struct OctreeNode {
    bounds: AABB<f64>,
    children: Option<[Box<OctreeNode>; 8]>,
    node_partitioning: [u32; 8],
    points_per_partition: [u32; 8],
    point_start: u32,
    point_end: u32,
}

pub struct GpuOctree<'a> {
    point_buffer: &'a dyn PointBuffer,
    point_partitioning: Vec<u32>,
    root_node: Option<OctreeNode>,
    bounds: AABB<f64>,
    points_per_node: u32,
}

impl OctreeNode {
    fn is_leaf(&self) -> bool {
        return self.children.is_none();
    }
    fn into_raw(&self) -> Vec<u8> {
        let mut raw_node: Vec<u8> = Vec::new();
        for coord in self.bounds.min().iter() {
            raw_node.append(&mut coord.to_ne_bytes().to_vec());
        }
        for coord in self.bounds.max().iter() {
            raw_node.append(&mut coord.to_ne_bytes().to_vec());
        }
        raw_node.append(
            &mut self
                .node_partitioning
                .map(|x| x.to_ne_bytes())
                .to_vec()
                .into_iter()
                .flatten()
                .collect(),
        );
        raw_node.append(
            &mut self
                .points_per_partition
                .map(|x| x.to_ne_bytes())
                .to_vec()
                .into_iter()
                .flatten()
                .collect(),
        );
        raw_node.append(&mut self.point_start.to_ne_bytes().to_vec());
        raw_node.append(&mut self.point_end.to_ne_bytes().to_vec());

        raw_node
    }
    fn from_raw(mut data: Vec<u8>) -> Self {
        let raw_bounds: Vec<u8> = data.drain(..24).collect();
        let bounds_iter = raw_bounds.chunks_exact(8);
        let bounds_min: Point3<f64> = Point3 {
            coords: Vector3::from_vec(
                bounds_iter
                    .take(3)
                    .map(|b| f64::from_ne_bytes(b.try_into().unwrap()))
                    .collect(),
            ),
        };
        let raw_bounds: Vec<u8> = data.drain(..24).collect();
        let bounds_iter = raw_bounds.chunks_exact(8);
        let bounds_max: Point3<f64> = Point3 {
            coords: Vector3::from_vec(
                bounds_iter
                    .take(3)
                    .map(|b| f64::from_ne_bytes(b.try_into().unwrap()))
                    .collect(),
            ),
        };
        let mut rest_data: Vec<u32> = data
            .chunks_exact(4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
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
    pub fn new(
        point_buffer: &'a dyn PointBuffer,
        max_bounds: AABB<f64>,
        points_per_node: u32,
    ) -> Self {
        GpuOctree {
            point_buffer,
            point_partitioning: (0..point_buffer.len() as u32).collect(),
            root_node: None,
            bounds: max_bounds,
            points_per_node,
        }
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

        let gpu_point_buffer = wgpu::util::DeviceExt::create_buffer_init(
            &device.wgpu_device,
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
        let raw_indeces: Vec<u8> = (0u32..(point_count - 1) as u32)
            .flat_map(|x| x.to_le_bytes().to_vec())
            .collect();
        let point_index_buffer = wgpu::util::DeviceExt::create_buffer_init(
            &device.wgpu_device,
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
        let mut root_node = OctreeNode {
            bounds: self.bounds,
            children: None,
            node_partitioning: [1; 8],
            points_per_partition: [2; 8],
            point_start: 0,
            point_end: point_count as u32 - 1,
        };
        let xdiff = &root_node.bounds.max().x - &root_node.bounds.min().x;
        let ydiff = &root_node.bounds.max().y - &root_node.bounds.min().y;
        let zdiff = &root_node.bounds.max().z - &root_node.bounds.min().z;
        println!("xdiff {}", xdiff);
        println!("ydiff {}", ydiff);
        println!("zdiff {}", zdiff);
        let xpartition = &root_node.bounds.min().x + 0.5 * xdiff;
        let ypartition = &root_node.bounds.min().y + 0.5 * ydiff;
        let zpartition = &root_node.bounds.min().z + 0.5 * zdiff;
        println!("x_partition {}", xpartition);
        println!("y_partition {}", ypartition);
        println!("z_partition {}", zpartition);
        let points_bind_group_layout =
            device
                .wgpu_device
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
                    ],
                    label: Some("PointBufferBindGroupLayout"),
                });
        let points_bind_group = device
            .wgpu_device
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
                ],
            });
        device.set_bind_group(1, &points_bind_group_layout, &points_bind_group);

        let tree_depth = 1;
        let num_leaves: u32 = 0;
        let num_nodes: u32 = 1;
        let nodes_counter = wgpu::util::DeviceExt::create_buffer_init(
            &device.wgpu_device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("NodeCounterBuffer"),
                contents: &num_nodes.to_le_bytes(),
                usage: wgpu::BufferUsages::MAP_READ
                    | wgpu::BufferUsages::MAP_WRITE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::STORAGE,
            },
        );
        let leaves_counter = wgpu::util::DeviceExt::create_buffer_init(
            &device.wgpu_device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("LeafCounterBuffer"),
                contents: &num_leaves.to_le_bytes(),
                usage: wgpu::BufferUsages::MAP_READ
                    | wgpu::BufferUsages::MAP_WRITE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::STORAGE,
            },
        );
        let counter_bind_group_layout =
            device
                .wgpu_device
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
                    ],
                    label: Some("CounterBindGroupLayout"),
                });
        let counter_bind_group = device
            .wgpu_device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("CounterBindGroup"),
                layout: &counter_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: nodes_counter.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: leaves_counter.as_entire_binding(),
                    },
                ],
            });
        device.set_bind_group(0, &counter_bind_group_layout, &counter_bind_group);
        let mut current_nodes = vec![&root_node];
        loop {
            let num_new_nodes = 8u32.pow(tree_depth) - num_leaves;
            let new_nodes_buffer = device.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("NewNodesBuffer"),
                size: (mem::size_of::<OctreeNode>() as u64
                    - mem::size_of::<[Box<OctreeNode>; 8]>() as u64)
                    * num_new_nodes as u64,
                usage: wgpu::BufferUsages::MAP_READ
                    | wgpu::BufferUsages::MAP_WRITE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let mut parent_nodes_raw: Vec<u8> = Vec::new();
            for node in &current_nodes {
                parent_nodes_raw.append(&mut node.into_raw());
            }

            let parent_nodes_buffer = wgpu::util::DeviceExt::create_buffer_init(
                &device.wgpu_device,
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
            let nodes_bind_group_layout =
                device
                    .wgpu_device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("NodesBindGroupLayout"),
                        entries: &[
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
                                binding: 0,
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
            let nodes_bind_group =
                device
                    .wgpu_device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("NodesBindGroup"),
                        layout: &nodes_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: parent_nodes_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: new_nodes_buffer.as_entire_binding(),
                            },
                        ],
                    });
            device.set_bind_group(2, &nodes_bind_group_layout, &nodes_bind_group);

            device.set_compute_shader_glsl(include_str!("shaders/generate_nodes.comp"));
            device.compute(current_nodes.len() as u32, 1, 1);

            let mapped_future = point_index_buffer.slice(..).map_async(wgpu::MapMode::Read);
            device.wgpu_device.poll(wgpu::Maintain::Wait);

            if let Ok(()) = mapped_future.await {
                let mapped_index_buffer = point_index_buffer.slice(..).get_mapped_range().to_vec();
                let indices: Vec<u32> = mapped_index_buffer
                    .chunks_exact(4)
                    .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
                    .collect();

                self.point_partitioning = indices.clone();

                let mapped_counter = nodes_counter.slice(..).map_async(wgpu::MapMode::Read);
                device.wgpu_device.poll(wgpu::Maintain::Wait);
                if let Ok(()) = mapped_counter.await {
                    let mapped_counter = nodes_counter.slice(..).get_mapped_range().to_vec();
                    let x: Vec<u32> = mapped_counter
                        .chunks_exact(4)
                        .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
                        .collect();
                    // println!(
                    //     "Point at middle: {:?}",
                    //     points[indices[x[0] as usize] as usize]
                    // );
                    // println!(
                    //     "Point after middle: {:?}",
                    //     points[indices[x[0] as usize + 1] as usize]
                    // );
                    let mut index = 0;
                    for i in indices.iter().take(200) {
                        let p = points[*i as usize];
                        println!("index: {}, partition_index: {}, point: {:?}", index, i, p);
                        index += 1;
                    }
                }
            }
            let mapped_future = parent_nodes_buffer.slice(..).map_async(wgpu::MapMode::Read);
            device.wgpu_device.poll(wgpu::Maintain::Wait);

            if let Ok(()) = mapped_future.await {
                let mapped_node_buffer = parent_nodes_buffer.slice(..).get_mapped_range().to_vec();
                let nodes = OctreeNode::from_raw(mapped_node_buffer);
                println!("{:?}", nodes.node_partitioning);
                println!("{:?}", nodes.points_per_partition);
                let indices = self.point_partitioning.as_slice();
                for i in 0..nodes.node_partitioning[0] as usize {
                    let point = points[indices[i] as usize];
                    if point.x > xpartition || point.y > ypartition || point.z > zpartition {
                        println!("ERROR at point {} {:?}", indices[i], point);
                    }
                }
                for i in self.point_partitioning[0] as usize..nodes.node_partitioning[1] as usize {
                    let point = points[indices[i] as usize];
                    if point.x > xpartition || point.y > ypartition || point.z < zpartition {
                        println!("ERROR at point {} {:?}", indices[i], point);
                    }
                }
                for i in self.point_partitioning[1] as usize..nodes.node_partitioning[2] as usize {
                    let point = points[indices[i] as usize];
                    if point.x < xpartition || point.y > ypartition || point.z > zpartition {
                        println!("ERROR at point {} {:?}", indices[i], point);
                    }
                }
            }

            break;
        }
    }
}
