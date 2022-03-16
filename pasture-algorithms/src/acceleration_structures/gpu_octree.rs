use pasture_core::{
    containers::{attr1::AttributeIteratorByValue, PointBuffer, PointBufferExt, PerAttributePointBufferExt},
    layout::attributes,
    math::{AABB, DynamicMortonIndex, Octant, MortonIndexNaming, MortonIndex64},
    nalgebra::{Point3, Vector3},
};
use priority_queue::DoublePriorityQueue;
use ordered_float::OrderedFloat;
use std::convert::TryInto;
use std::fmt;
use std::mem;
use wgpu::util::DeviceExt;

#[derive(Debug, Clone)]
pub struct OctreeNode {
    bounds: AABB<f64>,
    children: Option<Box<[OctreeNode; 8]>>,
    node_partitioning: [u32; 8],
    points_per_partition: [u32; 8],
    point_start: u32,
    point_end: u32,
}

pub struct GpuOctree{
    gpu_device: wgpu::Device,
    gpu_queue: wgpu::Queue,
    point_buffer: Vec<Vector3<f64>>,
    raw_points: Vec<u8>,
    point_partitioning: Vec<u32>,
    root_node: Option<OctreeNode>,
    depth: u32,
    bounds: AABB<f64>,
    points_per_node: u32,
    morton_code: DynamicMortonIndex,
}

enum OctreeRelation {
    In,
    Out,
    Partial,
}

impl OctreeNode {
    /// Get the number of bytes, a node allocates on the gpu.
    /// Because the `children` pointer is not required for GPU node creation,
    /// it's size is neglected.
    fn size() -> usize {
        let mut size = mem::size_of::<OctreeNode>();
        size -= mem::size_of::<Option<Box<[OctreeNode; 8]>>>();
        size
    }
    /// Checks if the given node has less than or equal to `points_per_node` points.
    /// If yes, the node is a leaf.
    fn is_leaf(&self, points_per_node: u32) -> bool {
        let diff: i64 = self.point_end as i64 - self.point_start as i64;
        return diff <= points_per_node as i64;
    }
    fn is_empty(&self) -> bool {
        self.point_start == self.point_end && self.points_per_partition[0] == 0
    }
    /// Returns a vector of the nodes raw data. As with `size(), the field
    /// `children`is not included, as it is not necessary for GPU computation.
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
        raw_node.append(&mut self.point_end.to_le_bytes().to_vec());

        raw_node
    }
    /// Converts a vector of raw data back into `OctreeNode`.
    /// Panics, if the vector has not enough data.
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
        use std::sync::{mpsc::channel, Arc, Mutex};
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

    /// Specifies the relation between the query point `pos` and the Bounding Box of the Node.
    /// OctreeRelation::In      ==> the whole node sits inside the radius of the query
    /// OctreeRelation::Partial ==> node and query intersect or node contains whole query
    /// OctreeRelation::Out     ==> node and query are disjoint
    fn relation_to_point(&self, pos: &Vector3<f64>, radius: f64) -> OctreeRelation {
        let node_extent = self.bounds.extent();
        let node_center = self.bounds.center().coords;
        let x_diff = (pos.x - node_center.x).abs();
        let y_diff = (pos.y - node_center.y).abs();
        let z_diff = (pos.z - node_center.z).abs();

        // Point and radius outside of node
        let max_diff = Vector3::new(
            node_extent.x / 2. + radius,
            node_extent.y / 2. + radius,
            node_extent.z / 2. + radius
        );
        if x_diff >= max_diff.x || y_diff >= max_diff.y || z_diff >= max_diff.z {
            return OctreeRelation::Out;
        }
        let radius_squared = radius * radius;
        if x_diff <= node_extent.x || y_diff <= node_extent.y || z_diff <= node_extent.z {
            let radius_squared = radius * radius;
            let distance_squared = f64::powi(x_diff + node_extent.x * 0.5, 2) + f64::powi(y_diff + node_extent.y * 0.5, 2) + f64::powi(z_diff + node_extent.z * 0.5, 2);
            // Whole Node lies within radius
            if radius_squared >= distance_squared {
                return OctreeRelation::In;
            }
            return OctreeRelation::Partial;
        }
        let distance_squared = f64::powi(x_diff - node_extent.x * 0.5, 2) + f64::powi(y_diff - node_extent.y * 0.5, 2) + f64::powi(z_diff - node_extent.z * 0.5, 2);
        if radius_squared >= distance_squared{
            return OctreeRelation::Partial;
        }
        return OctreeRelation::Out;
    }
    
}

impl fmt::Display for OctreeNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "####### Octree Node #######\n");
        write!(f, "Bounds: {:?}\n", self.bounds);
        write!(f, "Start: {}, End: {}\n", self.point_start, self.point_end);
        write!(f, "Node Partitioning: {:?}\n", self.node_partitioning);
        write!(f, "Points per partition: {:?}\n", self.points_per_partition);
        write!(f, "Chilren: ");
        if let Some(c) = &self.children {
            c.iter().for_each(|x| {
                write!(f, "  {}", x);
            });
        } else {
            write!(f, "None\n");
        }
        write!(f, "##########\n")
    }
}

impl GpuOctree {
    /// Creates an empty Octree accelerated by the GPU.
    /// 
    /// `point_buffer`: pasture buffer containing the point cloud data
    /// 
    /// `max_bounds`: boundary of the point cloud
    /// 
    /// `points_per_node`: threshold for a node becoming a leaf
    /// 
    /// The generated instance has no constructed octree. To get the octree,
    /// run `construct()`.
    pub async fn new(
        point_buffer: & dyn PointBuffer,
        max_bounds: AABB<f64>,
        points_per_node: u32,
    ) -> Result<GpuOctree, wgpu::RequestDeviceError> {
        if points_per_node < 1 {
            panic!("Cannot build octree with less than 1 point per node!")
        }
        let instance = wgpu::Instance::new(wgpu::Backends::VULKAN);
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
            

        // Points are read from the Pasture PointBuffer to allow for faster access of individual points.,
        // Without this, onw would need to get each raw point individually and convert it
        let mut points: Vec<Vector3<f64>> = Vec::new();
        let point_iter = AttributeIteratorByValue::new(point_buffer, &attributes::POSITION_3D);
        for point in point_iter {
            points.push(point);
        }

        // raw points read here, so that it must not be done while construction
        let point_count = point_buffer.len();
        let mut raw_points = vec![0u8; 24 * point_count];
        point_buffer.get_raw_attribute_range(
            0..point_count,
            &attributes::POSITION_3D,
            raw_points.as_mut_slice(),
        );
        let morton_code = DynamicMortonIndex::from_octants(&[]);

        Ok(GpuOctree {
            gpu_device: device,
            gpu_queue: queue,
            point_buffer: points,
            raw_points,
            point_partitioning: (0..point_buffer.len() as u32).collect(),
            root_node: None,
            depth: 0,
            bounds: max_bounds,
            points_per_node,
            morton_code
        })
    }

    pub fn print_tree(&self) {
        println!("Num Points: {}", self.point_buffer.len());
        println!("Tree Depth: {}", self.depth);
        println!("{}", self.root_node.as_ref().unwrap());
    }

    /// Prints the morton index for given point inside the AABB of the octree
    pub fn print_morton_code(&self, point: &Point3<f64>) {
        if let Some(root) = self.root_node.as_ref() {
            println!("{}", 
            MortonIndex64::from_point_in_bounds(&point, &root.bounds)
            .to_string(MortonIndexNaming::AsOctantConcatenationWithRoot));
        }
        else {
            println!("Octree not constructed yet");
        }
    }

    // }
    /// Run top-down construction of the octree.
    /// 
    /// Starting from the root, on each level the children of all current leaves
    /// are computed and put into the next compute stage, if these children are big enough.
    pub async fn construct(&mut self) {
        let point_count = self.point_buffer.len();
        
        let mut raw_points = &self.raw_points;
        
        let mut compiler = shaderc::Compiler::new().unwrap();
        let comp_shader = include_str!("shaders/generate_nodes.comp");
        let comp_spirv = compiler
            .compile_into_spirv(
                comp_shader,
                shaderc::ShaderKind::Compute,
                "ComputeShader",
                "main",
                None,
            )
            .unwrap();

        let comp_data = wgpu::util::make_spirv(comp_spirv.as_binary_u8());
        let shader = self
            .gpu_device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: Some("NodeGenerationShader"),
                source: comp_data,
            });

        // 2 Bind groups are used
        // - points_bind_group for point cloud data and point indices
        // - nodes_bind_group for parent nodes and children nodes computed by GPU
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
                    ],
                    label: Some("PointBufferBindGroupLayout"),
                });

        let nodes_bind_group_layout =
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
                    bind_group_layouts: &[&nodes_bind_group_layout, &points_bind_group_layout],
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

        let gpu_point_buffer =
            self.gpu_device
                .create_buffer(&wgpu::BufferDescriptor {
                    label: Some("PointBuffer"),
                    size: (point_count * mem::size_of::<Vector3<f64>>()) as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
        self.gpu_queue.write_buffer(&gpu_point_buffer, 0, self.raw_points.as_slice());
        let mut root_node = OctreeNode {
            bounds: self.bounds,
            children: None,
            node_partitioning: [0; 8],
            points_per_partition: [0; 8],
            point_start: 0,
            point_end: point_count as u32,
        };
        root_node.node_partitioning[0] = point_count as u32;
        root_node.points_per_partition[0] = point_count as u32;

        let mut tree_depth = 1;
        let mut num_leaves: u32 = 0;
        let mut num_nodes: u32 = 1;

        let mut current_nodes = vec![&mut root_node];

        let index_range: Vec<u32> = (0u32..point_count as u32).map(u32::from).collect::<Vec<u32>>();
        let raw_indeces: &[u8] = bytemuck::cast_slice(index_range.as_slice());
        let point_index_buffer =
            self.gpu_device
                .create_buffer(&wgpu::BufferDescriptor {
                    label: Some("IndexBuffer"),
                    size: (point_count * mem::size_of::<u32>()) as u64,
                    usage: wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::MAP_READ
                        | wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
        self.gpu_queue.write_buffer(&point_index_buffer, 0, raw_indeces);

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
                ],
            });
        while !current_nodes.is_empty() {
            let num_blocks = current_nodes.len();

            // Nodes buffers are created inside the loop, as their size changes per iteration
            // Staging Buffers do not reside on GPU and are used for reading Compure results
            let child_buffer_size = 8 * (OctreeNode::size() * current_nodes.len()) as u64; 
            let child_nodes_buffer_staging =
                self.gpu_device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("CPU_NewNodesBuffer"),
                    size: child_buffer_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            let child_nodes_buffer = self.gpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("NewNodesBuffer"),
                size: 
                    child_buffer_size,
                usage: wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            
            let mut parent_nodes_raw = Vec::new();
            for node in &current_nodes {
                parent_nodes_raw.append(&mut node.into_raw());
            }
            let parent_nodes_buffer_staging =
                self.gpu_device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("CPU_ParentNodesBuffer"),
                    size: parent_nodes_raw.len() as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            let parent_nodes_buffer =
                self.gpu_device
                    .create_buffer(&wgpu::BufferDescriptor {
                        label: Some("ParentNodesBuffer"),
                        size: current_nodes.len() as u64 * OctreeNode::size() as u64,
                        usage: wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    });
            self.gpu_queue.write_buffer(&parent_nodes_buffer, 0, parent_nodes_raw.as_slice());

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
                            resource: child_nodes_buffer.as_entire_binding(),
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

                compute_pass.set_bind_group(0, &nodes_bind_group, &[]);
                compute_pass.set_bind_group(1, &points_bind_group, &[]);

                compute_pass.insert_debug_marker("Pasture Compute Debug");
                compute_pass.dispatch(num_blocks as u32, 1, 1);
            }
            // Copy computed Nodes into CPU staging buffers for mapped reading
            encoder.copy_buffer_to_buffer(
                &child_nodes_buffer,
                0,
                &child_nodes_buffer_staging,
                0,
                child_buffer_size,
            );
            encoder.copy_buffer_to_buffer(
                &parent_nodes_buffer,
                0,
                &parent_nodes_buffer_staging,
                0,
                parent_nodes_raw.len() as u64,
            );
            

            self.gpu_queue.submit(Some(encoder.finish()));
            
            let parents_slice = parent_nodes_buffer_staging.slice(..);
            let parents_future = parents_slice.map_async(wgpu::MapMode::Read);
            let children_slice = child_nodes_buffer_staging.slice(..);
            let children_future = children_slice.map_async(wgpu::MapMode::Read);
            
            self.gpu_device.poll(wgpu::Maintain::Wait);
            if let Ok(()) = parents_future.await {
                let mapped_nodes_data = parents_slice.get_mapped_range();
                let mapped_node_buffer = mapped_nodes_data.to_vec();
                let nodes: Vec<OctreeNode> = mapped_node_buffer
                    .chunks_exact(OctreeNode::size())
                    .map(|b| OctreeNode::from_raw(b.to_vec()))
                    .collect();

                if let Ok(()) = children_future.await {
                    let mapped_children_data = children_slice.get_mapped_range();
                    let mapped_children_buffer = mapped_children_data.to_vec();
                    let mut children: Vec<OctreeNode> = mapped_children_buffer
                        .chunks_exact(OctreeNode::size())
                        .map(|b| OctreeNode::from_raw(b.to_vec()))
                        .collect();
                    for mut node in nodes {
                        let children_sizes = node.points_per_partition.clone();

                        let local_children: Vec<OctreeNode> = children.drain(..8).collect();

                        let child_array: [OctreeNode; 8] = local_children.try_into().unwrap();
                        node.children = Some(Box::new(child_array));

                        let node_ref = current_nodes.remove(0);
                        *node_ref = node;
                        
                        let children: &mut Box<[OctreeNode; 8]> =
                            node_ref.children.as_mut().unwrap();

                        let iter = children.iter_mut();

                        let mut child_index = 0;

                        for child in iter {
                            if children_sizes[child_index] != 0
                                && !child.is_leaf(self.points_per_node)
                            {
                                current_nodes.push(child);
                            } else {
                                num_leaves += 1;
                            }

                            num_nodes += 1;
                            child_index += 1;
                        }
                        let morton_octants: Vec<Octant> = vec![
                            0, 1, 2, 3, 4, 5, 6, 7
                        ]
                        .iter()
                        .map(|&raw_octant| (raw_octant as u8).try_into().unwrap())
                        .collect();

                        morton_octants.iter()
                        .for_each(|octant| self.morton_code.add_octant(*octant));
                    }
                    drop(mapped_nodes_data);
                    parent_nodes_buffer_staging.unmap();
                    drop(mapped_children_data);
                    child_nodes_buffer_staging.unmap();
                    parent_nodes_buffer.destroy();
                    child_nodes_buffer.destroy();
                    parent_nodes_buffer_staging.destroy();
                    child_nodes_buffer_staging.destroy();
                }
            }

            tree_depth += 1;
        }
        // Point indices are read after compute loop to reduce data copying and runtime
        let index_slice = point_index_buffer.slice(..);
        let mapped_future = index_slice.map_async(wgpu::MapMode::Read);

        self.gpu_device.poll(wgpu::Maintain::Wait);
        // Read in the changes of the global point partitioning
        if let Ok(()) = mapped_future.await {
            let mapped_index_buffer = index_slice.get_mapped_range();
            let index_vec = mapped_index_buffer.to_vec();
            let indices: Vec<u32> = index_vec
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();

            self.point_partitioning = indices.clone();

            drop(mapped_index_buffer);
            point_index_buffer.unmap();
        }
        gpu_point_buffer.destroy();
        point_index_buffer.destroy();
        self.root_node = Some(root_node);
        self.depth = tree_depth;
    }

    /// Returns a Vec containing the indices of all points belonging to `point`
    fn get_points(&self, node: &OctreeNode) -> Vec<u32> {
        let indices =
            self.point_partitioning[node.point_start as usize..node.point_end as usize].to_vec();
        return indices;
    }

    /// Computes the `k` nearest neighbors of `point` that are within `radius`. 
    /// Returns a Vec containing the indices to the neighbor points in increasing order of distance 
    pub fn k_nearest_neighbors(&self, pos: Vector3<f64>, radius: f64, k: usize) -> Vec<u32> {
        if k < 1 {
            return vec![];
        }
        // We use a Vec<&Octree> as working queue for the nodes that we are visitting
        // We use a priority queue so that the found indices are already
        // being sorted by their distance to pos
        let node = self.root_node.as_ref().unwrap();
        let mut worklist = vec![node];
        let point_buffer = &self.point_buffer;
        let mut points = DoublePriorityQueue::new();

        if pos.x - radius > node.bounds.max().x ||
            pos.x + radius < node.bounds.min().x ||
            pos.y - radius > node.bounds.max().y ||
            pos.y + radius < node.bounds.min().y ||
            pos.z - radius > node.bounds.max().z ||
            pos.z + radius < node.bounds.min().z 
        {
            return vec![];
        }
        let radius_squared = radius * radius;

        while !worklist.is_empty() {
            let node = worklist.pop().unwrap();
            // When node is leaf we need to search it's points
            if node.is_leaf(self.points_per_node) {
                let point_indices = self.get_points(node);
                for i in point_indices.iter() {
                    let point = point_buffer[*i as usize];
                    let curr_dist = point - pos;
                    let curr_dist = f64::powi(curr_dist.x, 2) + f64::powi(curr_dist.y, 2) + f64::powi(curr_dist.z, 2);
                    if  curr_dist <= radius_squared{
                        if points.len() >= k {
                            let (_, dist) = points.peek_max().unwrap();
                            if *dist > OrderedFloat(curr_dist) {
                                points.pop_max();
                                points.push(i.clone(), OrderedFloat(curr_dist));
                            }
                        }
                        else {
                            points.push(i.clone(), OrderedFloat(curr_dist));
                        }
                    }
                }
                let point_indices: Vec<u32> = self.get_points(node);
            }
            else {
                // When node is not leaf, we must check if the Bounding Box of the node is in range of the query
                // If so, we check if the whole node is inside the radius to possibly reduce step downs
                // If node and query only intersect, children of the node are inspected
                match node.relation_to_point(&pos, radius) {
                    OctreeRelation::In => {
                        let point_indices = self.get_points(node);
                        for i in point_indices.iter() {
                            let point = point_buffer[*i as usize];
                            let curr_dist = point - pos;
                            let curr_dist = f64::powi(curr_dist.x, 2) + f64::powi(curr_dist.y, 2) + f64::powi(curr_dist.z, 2);
                            if  curr_dist <= radius_squared {
                                if points.len() >= k {
                                    let (_, dist) = points.peek_max().unwrap();
                                    if *dist > OrderedFloat(curr_dist) {
                                        points.pop_max();
                                        points.push(i.clone(), OrderedFloat(curr_dist));
                                    }
                                }
                                else {
                                    points.push(i.clone(), OrderedFloat(curr_dist));
                                }
                            }
                        }
                    }
                    OctreeRelation::Partial => {
                        if let Some(children) = node.children.as_ref() {
                            children.iter().for_each(|c| {
                                if !c.is_empty(){ 
                                    worklist.push(c);
                                }
                            });
                        }
                    }
                    OctreeRelation::Out => {}
                }; 
            }
        }
        if points.is_empty() {
            return vec![];
        }
        let mut nearest = points.into_ascending_sorted_vec();
        nearest.truncate(k);
        
        return nearest;
        
    }

}

#[cfg(test)]
mod tests {
    use crate::acceleration_structures::GpuOctree;
    use crate::acceleration_structures::gpu_octree::OctreeNode;
    use pasture_core::containers::InterleavedVecPointStorage;
    use pasture_core::containers::PointBufferExt;
    use pasture_core::layout::PointType;
    use pasture_core::nalgebra::Vector3;
    use pasture_io::base::PointReader;
    use pasture_io::las::LASReader;
    use pasture_io::las::LasPointFormat0;

    use tokio;

    static FILE: &'static str = //"/home/jnoice/Downloads/WSV_Pointcloud_Tile-3-1.laz"
                                //"/home/jnoice/Downloads/interesting.las"
                                //"/home/jnoice/Downloads/45123H3316.laz"
                                //"/home/jnoice/Downloads/OR_Camp_Creek_OLC_2008_000001.laz"
                                "/home/jnoice/Downloads/portland.laz"
                                ;
    #[tokio::test]
    async fn check_correct_bounds() {
        let reader = LASReader::from_path(FILE);
        let mut reader = match reader {
            Ok(a) => a,
            Err(_) => panic!("Could not create LAS Reader"),
        };
        let count = reader.remaining_points();
        let mut buffer =
            InterleavedVecPointStorage::with_capacity(count, LasPointFormat0::layout());
        let _data_read = match reader.read_into(&mut buffer, count) {
            Ok(a) => a,
            Err(_) => panic!("Could not write Point Buffer"),
        };
        let bounds = reader.get_metadata().bounds().unwrap();

        let octree = GpuOctree::new(&buffer, bounds, 75).await;
        let mut octree = match octree {
            Ok(a) => a,
            Err(b) => {
                println!("{:?}", b);
                panic!("Could not create GPU Device for Octree")
            }
        };
        octree.construct().await;
        let node = octree.root_node.as_ref().unwrap();
        let mut nodes_to_visit: Vec<&OctreeNode> = vec![node];
        while !nodes_to_visit.is_empty() {
            let current_node = nodes_to_visit.remove(0);
            println!("Partition {:?}", current_node.node_partitioning);
            assert!((current_node.point_start == 0 &&
                current_node.point_end == 0 &&
                current_node.node_partitioning == [0; 8]) || 
                current_node.node_partitioning != [0; 8]);
            let current_bounds = current_node.bounds;
            let point_ids = octree.get_points(&current_node).into_iter();
            for id in point_ids {
                let point = buffer.get_point::<LasPointFormat0>(id as usize);
                let pos: Vector3<f64> = Vector3::from(point.position);
                
                assert!(
                    current_bounds.min().x <= pos.x
                        && current_bounds.max().x >= pos.x
                        && current_bounds.min().y <= pos.y
                        && current_bounds.max().y >= pos.y
                        && current_bounds.min().z <= pos.z
                        && current_bounds.max().z >= pos.z
                );
                
            }
            if let Some(children) = current_node.children.as_ref() {
                (*children).iter().for_each(|x| nodes_to_visit.push(x));
            }
        }
    }

    #[tokio::test]
    async fn check_point_count() {
        let reader = LASReader::from_path(FILE);
        let mut reader = match reader {
            Ok(a) => a,
            Err(_) => panic!("Could not create LAS Reader"),
        };
        let count = reader.remaining_points();
        let mut buffer =
            InterleavedVecPointStorage::with_capacity(count, LasPointFormat0::layout());
        let _data_read = match reader.read_into(&mut buffer, count) {
            Ok(a) => a,
            Err(_) => panic!("Could not write Point Buffer"),
        };
        let bounds = reader.get_metadata().bounds().unwrap();

        let octree = GpuOctree::new(&buffer, bounds, 50).await;
        let mut octree = match octree {
            Ok(a) => a,
            Err(b) => {
                println!("{:?}", b);
                panic!("Could not create GPU Device for Octree")
            }
        };
        octree.construct().await;
        let node = octree.root_node.as_ref().unwrap();
        let mut nodes_to_visit: Vec<&OctreeNode> = vec![node];
        let mut point_count: usize = 0;
        while !nodes_to_visit.is_empty() {
            let current_node = nodes_to_visit.pop().unwrap();
            if let None = current_node.children {
                println!("{}", current_node);
                point_count += current_node.points_per_partition[0] as usize;
            } else {
                let children = current_node.children.as_ref().unwrap();
                (*children).iter().for_each(|x| nodes_to_visit.push(x));
            }
        }
        println!(
            "Point count of octree: {}, Point Count of Buffer {}",
            point_count, count
        );
        assert!(point_count == count);
    }
    #[tokio::test]
    async fn check_point_partitioning_duplicates() {
        let reader = LASReader::from_path(FILE);
        let mut reader = match reader {
            Ok(a) => a,
            Err(_) => panic!("Could not create LAS Reader"),
        };
        let count = reader.remaining_points();
        let mut buffer =
            InterleavedVecPointStorage::with_capacity(count, LasPointFormat0::layout());
        let _data_read = match reader.read_into(&mut buffer, count) {
            Ok(a) => a,
            Err(_) => panic!("Could not write Point Buffer"),
        };
        let bounds = reader.get_metadata().bounds().unwrap();

        let octree = GpuOctree::new(&buffer, bounds, 50).await;
        let mut octree = match octree {
            Ok(a) => a,
            Err(b) => {
                println!("{:?}", b);
                panic!("Could not create GPU Device for Octree")
            }
        };
        octree.construct().await;
        let mut indices = octree.point_partitioning.clone();
        indices.sort();
        indices.dedup();
        assert!(indices.len() == octree.point_partitioning.len());
    }
    #[tokio::test]
    async fn check_node_overflows() {
        let reader = LASReader::from_path(FILE);
        let mut reader = match reader {
            Ok(a) => a,
            Err(_) => panic!("Could not create LAS Reader"),
        };
        let count = reader.remaining_points();
        let mut buffer =
            InterleavedVecPointStorage::with_capacity(count, LasPointFormat0::layout());
        let _data_read = match reader.read_into(&mut buffer, count) {
            Ok(a) => a,
            Err(_) => panic!("Could not write Point Buffer"),
        };
        let bounds = reader.get_metadata().bounds().unwrap();

        let octree = GpuOctree::new(&buffer, bounds, 50).await;
        let mut octree = match octree {
            Ok(a) => a,
            Err(b) => {
                println!("{:?}", b);
                panic!("Could not create GPU Device for Octree")
            }
        };
        octree.construct().await;
        let node = octree.root_node.as_ref().unwrap();
        let mut nodes_to_visit: Vec<&OctreeNode> = vec![node];
        while !nodes_to_visit.is_empty() {
            let current_node = nodes_to_visit.pop().unwrap();
            assert!(current_node.point_start <= current_node.point_end);
            if let Some(children) = &current_node.children {
                (*children).iter().for_each(|x| nodes_to_visit.push(x));
            }
        }
    }
    
}
