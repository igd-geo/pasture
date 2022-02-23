use pasture_core::{
    containers::{attr1::AttributeIteratorByValue, PointBuffer, PointBufferExt, PerAttributePointBufferExt},
    layout::attributes,
    math::AABB,
    nalgebra::{Point3, Vector3},
};

use std::convert::TryInto;
use std::fmt;
use std::mem;
use std::time;
use wgpu::util::DeviceExt;
use bitvec::prelude::*;

#[derive(Debug, Clone)]
pub struct OctreeNode {
    bounds: AABB<f64>,
    children: Option<Box<[OctreeNode; 8]>>,
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
    depth: u32,
    bounds: AABB<f64>,
    points_per_node: u32,
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
    /// Checks if `pos` is within the bounds of the node.
    fn contains(&self, pos: Point3<f64>) -> bool {
        self.bounds.contains(&pos)
    }

    fn get_closest_child(pos: Point3<f64>) -> u32 {
        let child_id = 0;

        child_id
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

impl<'a> GpuOctree<'a> {
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
        point_buffer: &'a dyn PointBuffer,
        max_bounds: AABB<f64>,
        points_per_node: u32,
    ) -> Result<GpuOctree<'a>, wgpu::RequestDeviceError> {
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
            
        Ok(GpuOctree {
            gpu_device: device,
            gpu_queue: queue,
            point_buffer,
            point_partitioning: (0..point_buffer.len() as u32).collect(),
            root_node: None,
            depth: 0,
            bounds: max_bounds,
            points_per_node,
        })
    }

    pub fn print_tree(&self) {
        println!("Num Points: {}", self.point_buffer.len());
        println!("Tree Depth: {}", self.depth);
        println!("{}", self.root_node.as_ref().unwrap());
    }
    /// Run top-down construction of the octree.
    /// 
    /// Starting from the root, on each level the children of all current leaves
    /// are computed and put into the next compute stage, if these children are big enough.
    pub async fn construct(&mut self) {
        let point_count = self.point_buffer.len();
        
        // point cloud data, later uploaded to GPU
        let mut raw_points = vec![0u8; 24 * point_count];
        let now = time::Instant::now();
        self.point_buffer.get_raw_attribute_range(
            0..point_count,
            &attributes::POSITION_3D,
            raw_points.as_mut_slice(),
        );
        let elapsed = now.elapsed();
        println!("Octree - Getting raw point data took {} ms", elapsed.as_millis());

        let now = time::Instant::now();
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
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("PointBuffer"),
                    contents: &raw_points.as_slice(),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                });

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

        let raw_indeces: Vec<u8> = (0u32..point_count as u32)
            .flat_map(|x| x.to_le_bytes().to_vec())
            .collect();

        let point_index_buffer =
            self.gpu_device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("IndexBuffer"),
                    contents: &raw_indeces.as_slice(),
                    usage: wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::STORAGE,
                });
        let index_buffer_staging = self.gpu_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CPU_IndexBuffer"),
            size: raw_indeces.len() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
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
                ],
            });
            let elapsed = now.elapsed();
            println!("Octree - GPU prep took {} ms", elapsed.as_millis());
        let now_compute = time::Instant::now();
        while !current_nodes.is_empty() {
            let now = time::Instant::now();
            // Nodes buffers are created inside the loop, as their size changes per iteration
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
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("ParentNodesBuffer"),
                        contents: parent_nodes_raw.as_slice(),
                        usage: wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::STORAGE,
                    });
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
                compute_pass.dispatch(current_nodes.len() as u32, 1, 1);
            }
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
            encoder.copy_buffer_to_buffer(
                &point_index_buffer,
                0,
                &index_buffer_staging,
                0,
                raw_indeces.len() as u64,
            );

            self.gpu_queue.submit(Some(encoder.finish()));

            let index_slice = index_buffer_staging.slice(..);
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
                index_buffer_staging.unmap();
            }

            let parents_slice = parent_nodes_buffer_staging.slice(..);
            let parents_future = parents_slice.map_async(wgpu::MapMode::Read);

            self.gpu_device.poll(wgpu::Maintain::Wait);
            if let Ok(()) = parents_future.await {
                let mapped_nodes_data = parents_slice.get_mapped_range();
                let mapped_node_buffer = mapped_nodes_data.to_vec();
                let nodes: Vec<OctreeNode> = mapped_node_buffer
                    .chunks_exact(OctreeNode::size())
                    .map(|b| OctreeNode::from_raw(b.to_vec()))
                    .collect();

                let children_slice = child_nodes_buffer_staging.slice(..);
                let children_future = children_slice.map_async(wgpu::MapMode::Read);
                self.gpu_device.poll(wgpu::Maintain::Wait);

                if let Ok(()) = children_future.await {
                    let mapped_children_data = children_slice.get_mapped_range();
                    let mapped_children_buffer = mapped_children_data.to_vec();
                    let mut children: Vec<OctreeNode> = mapped_children_buffer
                        .chunks_exact(OctreeNode::size())
                        .map(|b| OctreeNode::from_raw(b.to_vec()))
                        .collect();
                    let mut generated_children: Vec<&mut OctreeNode> = Vec::new();
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
                                generated_children.push(child);
                            } else {
                                num_leaves += 1;
                            }

                            num_nodes += 1;
                            child_index += 1;
                        }
                    }
                    current_nodes.append(&mut generated_children);
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
            let work_done = self.gpu_queue.on_submitted_work_done();
            work_done.await;

            tree_depth += 1;
            let elapsed = now.elapsed();
            println!("Octree - Compute Pass took {} ms", elapsed.as_millis());
        }
        let elapsed = now_compute.elapsed();
        println!("Octree - Whole Computation loop took {} ms", elapsed.as_millis());
        gpu_point_buffer.destroy();
        point_index_buffer.destroy();
        index_buffer_staging.destroy();
        self.root_node = Some(root_node);
        self.depth = tree_depth;
    }

    fn get_points(&self, node: &OctreeNode) -> Vec<u32> {
        let indices =
            self.point_partitioning[node.point_start as usize..node.point_end as usize].to_vec();
        return indices;
    }

    fn deepest_octant(&self, node: &'a OctreeNode, pos: Point3<f64>, max_distance: f64) -> &'a OctreeNode {
        if let Some(children) = node.children.as_ref() {
            for child in children.iter() {
                let bounds_extent = child.bounds.extent();
                if !child.is_leaf(self.points_per_node)
                    && child.contains(pos)
                    && bounds_extent.x >= max_distance * 2.0
                    && bounds_extent.y >= max_distance * 2.0
                    && bounds_extent.z >= max_distance * 2.0
                    {
                    return self.deepest_octant(child, pos, max_distance);
                }
            }
        }
        node
    }
    fn nearest_neighbor_helper(&self, pos: &Point3<f64>, dist: f64, node: &OctreeNode) -> Option<u32> {
        let mut axes: Vector3<f64> = 0.5 * (node.bounds.max() - node.bounds.min());
        axes += Vector3::new(node.bounds.min().x, node.bounds.min().y, node.bounds.min().z);
        let pos_vector = Vector3::new(pos.x, pos.y, pos.z);
        let mut nearest_index: Option<u32> = None;
        let mut shortest_distance = f64::MAX;
        if let Some(children) = node.children.as_ref() {
            // Sort children according to proximity to pos
            let mut sorted_children: Vec<&OctreeNode> = Vec::new();
            for i in 0..8 {
                let mut k = 0;
                let mut center = 0.5 * (children[i].bounds.max() - children[i].bounds.min());
                center += Vector3::new(children[i].bounds.min().x, children[i].bounds.min().y, children[i].bounds.min().z);
                let curr_dist = center.metric_distance(&pos_vector);
                for c in sorted_children.iter(){
                    let sorted_center = 0.5 * (c.bounds.max() - c.bounds.min()) + Vector3::new(c.bounds.min().x, c.bounds.min().y, c.bounds.min().z);
                    let sorted_dist = sorted_center.metric_distance(&pos_vector);
                    if sorted_dist < curr_dist {
                        k += 1;
                    }
                }
                sorted_children.insert(k, &children[i]);
                
            }
            for c in sorted_children.iter(){
                if let None = nearest_index {
                    nearest_index = self.nearest_neighbor_helper(pos, dist, c);
                    if let Some(index) = nearest_index {
                        let point = self.point_buffer.get_attribute::<Vector3<f64>>(&attributes::POSITION_3D, index as usize);
                        shortest_distance = point.metric_distance(&pos_vector);
                    }
                }
                else {
                    let current_nearest = self.nearest_neighbor_helper(pos, dist, c);
                    if let Some(index) = current_nearest {
                        let point = self.point_buffer.get_attribute::<Vector3<f64>>(&attributes::POSITION_3D, current_nearest.unwrap() as usize);
                        let curr_dist = point.metric_distance(&pos_vector);
                        if curr_dist < shortest_distance {
                            nearest_index = current_nearest;
                            shortest_distance = curr_dist;
                        }
                    }
                }
            }
            
        }
        else if !node.is_empty(){
            let point_indices = self.get_points(node);
            for i in point_indices.iter() {
                let point = self.point_buffer.get_attribute::<Vector3<f64>>(&attributes::POSITION_3D, *i as usize);
                let curr_dist = point.metric_distance(&pos_vector);
                
                if  curr_dist <= dist && curr_dist < shortest_distance {
                    shortest_distance = curr_dist;
                    nearest_index = Some(i.clone())
                }
            }
        }
        nearest_index
    }

    pub fn nearest_neighbor(&self, pos: Point3<f64>, max_distance: f64) -> Option<u32> {
        let node = self.deepest_octant(self.root_node.as_ref().unwrap(), pos, max_distance);
        let neighbor = self.nearest_neighbor_helper(&pos, max_distance, &node);
        neighbor
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
                                "/home/jnoice/Downloads/interesting.las"
                                //"/home/jnoice/Downloads/45123H3316.laz"
                                //"/home/jnoice/Downloads/OR_Camp_Creek_OLC_2008_000001.laz"
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
