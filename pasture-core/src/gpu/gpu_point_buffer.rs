use crate::layout::{PointAttributeDataType, PointAttributeDefinition};
use bytemuck::__core::convert::TryInto;
use crate::containers::{PointBuffer, PerAttributePointBufferMutExt, PerAttributePointBufferMut, InterleavedPointBufferMut, InterleavedVecPointStorage};
use crate::gpu::{BufferInfoInterleaved, BufferInfoPerAttribute};
use std::collections::HashMap;
use crate::nalgebra::{Vector3, Vector4};

trait GpuPointBuffer {
    fn alignment_per_element(&self, datatype: PointAttributeDataType) -> usize {
        // Assuming no extensions and GLSL:
        // - Only 32-bit integers (signed or unsigned) on shader side
        // - 32-bit and 64-bit floating point numbers
        // - vec3's are treated as vec4's
        //
        // Hence a u8 takes up 4 bytes (32 bits) and a Vec3u8 takes up 16 bytes (4x 32 bits).
        let alignment = match datatype {
            PointAttributeDataType::U8 => { 4 }
            PointAttributeDataType::I8 => { 4 }
            PointAttributeDataType::U16 => { 4 }
            PointAttributeDataType::I16 => { 4 }
            PointAttributeDataType::U32 => { 4 }
            PointAttributeDataType::I32 => { 4 }
            PointAttributeDataType::U64 => { 8 }    // Currently not supported on shader side
            PointAttributeDataType::I64 => { 8 }    // Currently not supported on shader side
            PointAttributeDataType::F32 => { 4 }
            PointAttributeDataType::F64 => { 8 }
            PointAttributeDataType::Bool => { 4 }
            PointAttributeDataType::Vec3u8 => { 16 }
            PointAttributeDataType::Vec3u16 => { 16 }
            PointAttributeDataType::Vec3f32 => { 16 }
            PointAttributeDataType::Vec3f64 => { 32 }
            PointAttributeDataType::Vec4u8 => { 16 }
        };

        alignment
    }

    fn align_slice(&self, slice: &[u8], datatype: PointAttributeDataType, offset: &mut usize) -> Vec<u8> {
        let mut ret_bytes: Vec<u8> = Vec::new();

        let num_bytes = slice.len();

        // Remember, signed and unsigned have the same bit-level behavior.
        // When casting to a larger type, first the size will change (zero or sign extension),
        // and only after that the type. Eg.
        //      (1) 0xff: u8 -> cast to u32 -> 0x000000ff [-> 255]
        //      (2) 0xff: i8 -> cast to u32 -> 0xffffffff [-> UINT_MAX]
        // Since we're only interested at what the numbers look like on the bit-level,
        // this allows us to combine the patterns for unsigned and signed types of the same size.
        match datatype {
            PointAttributeDataType::U8 | PointAttributeDataType::I8 | PointAttributeDataType::Bool => {
                // Treating as u32
                for i in 0..num_bytes {
                    // Alignment is 4 bytes
                    while *offset % 4 != 0 {
                        ret_bytes.push(0);
                        *offset += 1;
                    }

                    let current = (slice[i] as u32).to_ne_bytes();
                    ret_bytes.extend_from_slice(&current);
                    *offset += current.len();
                }
            }
            PointAttributeDataType::U16 | PointAttributeDataType::I16 => {
                // Treating as u32
                let stride = datatype.size() as usize;
                let num_elements = num_bytes / stride;

                for i in 0..num_elements {
                    // Alignment is 4 bytes
                    while *offset % 4 != 0 {
                        ret_bytes.push(0);
                        *offset += 1;
                    }

                    let begin = i * stride;
                    let end = (i * stride) + stride;
                    let current = u16::from_ne_bytes(slice[begin..end].try_into().unwrap());

                    let current = (current as u32).to_ne_bytes();
                    ret_bytes.extend_from_slice(&current);
                    *offset += std::mem::size_of::<u32>();
                }
            }
            PointAttributeDataType::U32 | PointAttributeDataType::I32 => {
                // Alignment is 4 bytes
                while *offset % 4 != 0 {
                    ret_bytes.push(0);
                    *offset += 1;
                }

                ret_bytes.extend_from_slice(&slice);
                *offset += num_bytes;
            }
            PointAttributeDataType::U64 | PointAttributeDataType::I64 => {
                // Trouble: no 64-bit integer types on GPU
                // TODO: consider extensions for GLSL that allow 64-bit integer types, eg. u64int
                panic!("Uploading 64-bit integer types to the GPU is not supported.")
            }
            PointAttributeDataType::F32 => {
                // Alignment is 4 bytes
                while *offset % 4 != 0 {
                    ret_bytes.push(0);
                    *offset += 1;
                }

                ret_bytes.extend_from_slice(&slice);
                *offset += num_bytes;
            }
            PointAttributeDataType::F64 => {
                // Alignment is 8 bytes
                while *offset % 8 != 0 {
                    ret_bytes.push(0);
                    *offset += 1;
                }

                ret_bytes.extend_from_slice(&slice);
                *offset += num_bytes;
            }
            PointAttributeDataType::Vec3u8 | PointAttributeDataType::Vec4u8 => {
                // Treating as Vec4u32
                let one_as_bytes = 1_u32.to_ne_bytes();

                // Each entry is 8 bits, ie. 1 byte -> each Vec3 has 3 bytes
                let stride = datatype.size() as usize;
                let num_elements = num_bytes / stride;
                let num_components = stride;

                // Iteration over each Vec3 or Vec4
                for i in 0..num_elements {
                    // Alignment is 16 bytes, need to add padding for Vec3
                    while *offset % 16 != 0 {
                        ret_bytes.push(0);
                        *offset += 1;
                    }

                    // Extend each entry to 32 bits
                    for j in 0..num_components {
                        let begin = (i * stride) + j;
                        let end = (i * stride) + j + 1;

                        let current = u8::from_ne_bytes(slice[begin..end].try_into().unwrap());
                        let current = (current as u32).to_ne_bytes();
                        ret_bytes.extend_from_slice(&current);
                        *offset += current.len();
                    }

                    // Append fourth coordinate
                    ret_bytes.extend_from_slice(&one_as_bytes);
                    *offset += one_as_bytes.len();
                }
            }
            PointAttributeDataType::Vec3u16 => {
                // Treating as Vec4u32
                let one_as_bytes = 1_u32.to_ne_bytes();

                // Each entry is 16 bits, ie. 2 bytes -> each Vec3 has 3*2 = 6 bytes
                let stride = datatype.size() as usize;   // = 6
                let num_elements = num_bytes / stride;

                // Iteration over each Vec3
                for i in 0..num_elements {
                    // Alignment is 16 bytes
                    while *offset % 16 != 0 {
                        ret_bytes.push(0);
                        *offset += 1;
                    }

                    // Extend each entry to 32 bits
                    for j in 0..3 {
                        let begin = (i * stride) + j * 2;
                        let end = (i * stride) + (j * 2) + 2;

                        let current = u16::from_ne_bytes(slice[begin..end].try_into().unwrap());
                        let current = (current as u32).to_ne_bytes();
                        ret_bytes.extend_from_slice(&current);
                        *offset += current.len();
                    }

                    // Append fourth coordinate
                    ret_bytes.extend_from_slice(&one_as_bytes);
                    *offset += one_as_bytes.len();
                }
            }
            PointAttributeDataType::Vec3f32 => {
                // Make Vec4f32 by appending 1.0
                let one_as_bytes = 1.0_f32.to_ne_bytes();

                // Each entry is 64 bits and hence consists of 8 bytes -> a Vec3 has 24 bytes
                let stride = datatype.size() as usize;   // = 24
                let num_elements = num_bytes / stride;

                for i in 0..num_elements {
                    // Alignment is 16 bytes
                    while *offset % 16 != 0 {
                        ret_bytes.push(0);
                        *offset += 1;
                    }

                    let begin = i * stride;
                    let end = (i * stride) + stride;

                    // Push current Vec3
                    let current = &slice[begin..end];
                    ret_bytes.extend_from_slice(current);
                    *offset += current.len();

                    // Push 1 as fourth coordinate
                    ret_bytes.extend_from_slice(&one_as_bytes);
                    *offset += one_as_bytes.len();
                }
            }
            PointAttributeDataType::Vec3f64 => {
                // Make Vec4f64 by appending 1.0
                let one_as_bytes = 1.0_f64.to_ne_bytes();

                // Each entry is 64 bits and hence consists of 8 bytes -> a Vec3 has 24 bytes
                let stride = datatype.size() as usize;   // = 24
                let num_elements = num_bytes / stride;

                for i in 0..num_elements {
                    // Alignment is 32 bytes
                    while *offset % 32 != 0 {
                        ret_bytes.push(0);
                        *offset += 1;
                    }

                    let begin = i * stride;
                    let end = (i * stride) + stride;

                    // Push current Vec3
                    let current = &slice[begin..end];
                    ret_bytes.extend_from_slice(current);
                    *offset += current.len();

                    // Push 1 as fourth coordinate
                    ret_bytes.extend_from_slice(&one_as_bytes);
                    *offset += one_as_bytes.len();
                }
            }
        }

        return ret_bytes;
    }

    // TODO: see if this can be done better with less duplication (the offset parameter is also ugly)
    fn calc_size(&self, num_bytes: usize, datatype: PointAttributeDataType, offset: &mut usize) {
        match datatype {
            PointAttributeDataType::U8 | PointAttributeDataType::I8 | PointAttributeDataType::Bool => {
                // Treating as u32
                for _ in 0..num_bytes {
                    // Alignment is 4 bytes
                    while *offset % 4 != 0 {
                        *offset += 1;
                    }
                    *offset += std::mem::size_of::<u32>();
                }
            }
            PointAttributeDataType::U16 | PointAttributeDataType::I16 => {
                // Treating as u32
                let stride = datatype.size() as usize;
                let num_elements = num_bytes / stride;

                for _ in 0..num_elements {
                    // Alignment is 4 bytes
                    while *offset % 4 != 0 {
                        *offset += 1;
                    }
                    *offset += std::mem::size_of::<u32>();
                }
            }
            PointAttributeDataType::U32 | PointAttributeDataType::I32 => {
                // Alignment is 4 bytes
                while *offset % 4 != 0 {
                    *offset += 1;
                }
                *offset += num_bytes;
            }
            PointAttributeDataType::U64 | PointAttributeDataType::I64 => {
                // Trouble: no 64-bit integer types on GPU
                // TODO: consider extensions for GLSL that allow 64-bit integer types, eg. u64int
                panic!("Uploading 64-bit integer types to the GPU is not supported.")
            }
            PointAttributeDataType::F32 => {
                // Alignment is 4 bytes
                while *offset % 4 != 0 {
                    *offset += 1;
                }
                *offset += num_bytes;
            }
            PointAttributeDataType::F64 => {
                // Alignment is 8 bytes
                while *offset % 8 != 0 {
                    *offset += 1;
                }
                *offset += num_bytes;
            }
            PointAttributeDataType::Vec3u8 => {
                // Each entry is 8 bits, ie. 1 byte -> each Vec3 has 3 bytes
                let stride = datatype.size() as usize;
                let num_elements = num_bytes / stride;

                // Iteration over each Vec3
                for _ in 0..num_elements {
                    // Alignment is 16 bytes
                    while *offset % 16 != 0 {
                        *offset += 1;
                    }

                    // Treating a Vec4u32: [x y z w]
                    *offset += 4 * std::mem::size_of::<u32>();
                }
            }
            PointAttributeDataType::Vec4u8 => {
                // Alignment is 16 bytes
                while *offset % 16 != 0 {
                    *offset += 1;
                }

                // Each entry is 8 bits, ie. 1 byte -> each Vec4 has 4 bytes
                let stride = datatype.size() as usize;
                let num_elements = num_bytes / stride;

                // Treating a Vec4u32: [x y z w]
                *offset += 4 * std::mem::size_of::<u32>() * num_elements;
            }
            PointAttributeDataType::Vec3u16 => {
                // Each entry is 16 bits, ie. 2 bytes -> each Vec3 has 3*2 = 6 bytes
                let stride = datatype.size() as usize;   // = 6
                let num_elements = num_bytes / stride;

                // Iteration over each Vec3
                for _ in 0..num_elements {
                    // Alignment is 16 bytes
                    while *offset % 16 != 0 {
                        *offset += 1;
                    }

                    // Treating a Vec4u32: [x y z w]
                    *offset += 4 * std::mem::size_of::<u32>();
                }
            }
            PointAttributeDataType::Vec3f32 => {
                // Each entry is 64 bits and hence consists of 8 bytes -> a Vec3 has 24 bytes
                let stride = datatype.size() as usize;   // = 24
                let num_elements = num_bytes / stride;

                for _ in 0..num_elements {
                    // Alignment is 16 bytes
                    while *offset % 16 != 0 {
                        *offset += 1;
                    }

                    // Treating a Vec4f32: [x y z w]
                    *offset += 4 * std::mem::size_of::<f32>();
                }
            }
            PointAttributeDataType::Vec3f64 => {
                // Each entry is 64 bits and hence consists of 8 bytes -> a Vec3 has 24 bytes
                let stride = datatype.size() as usize;   // = 24
                let num_elements = num_bytes / stride;

                for _ in 0..num_elements {
                    // Alignment is 32 bytes
                    while *offset % 32 != 0 {
                        *offset += 1;
                    }

                    // Treating a Vec4f64: [x y z w]
                    *offset += 4 * std::mem::size_of::<f64>();
                }
            }
        }
    }
}

/// Manages point buffer data that is to be stored in interleaved format on the GPU.
///
/// Make sure to allocate enough memory before trying to upload anything.
pub struct GpuPointBufferInterleaved {
    /// The [BindGroupLayout](wgpu::BindGroupLayout) that needs to be passed to the [Device](gpu::Device).
    /// It will be set with a call to [upload()](GpuPointBufferInterleaved::upload).
    pub bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// The [BindGroup](wgpu::BindGroup) that needs to be passed to the [Device](gpu::Device).
    /// It will be set with a call to [upload()](GpuPointBufferInterleaved::upload).
    pub bind_group: Option<wgpu::BindGroup>,

    buffer: Option<wgpu::Buffer>,
    buffer_size: Option<wgpu::BufferAddress>,
    buffer_binding: Option<u32>,

    offsets: Vec<HashMap<PointAttributeDataType, usize>>,
}

impl GpuPointBuffer for GpuPointBufferInterleaved {}

impl GpuPointBufferInterleaved {
    pub fn new() -> GpuPointBufferInterleaved {
        GpuPointBufferInterleaved {
            bind_group_layout: None,
            bind_group: None,
            buffer: None,
            buffer_size: None,
            buffer_binding: None,
            offsets: vec![],
        }
    }

    /// Allocates enough memory on the device to hold `num_points` many points that are structured
    /// as described in `buffer_info`.
    pub fn malloc(&mut self, num_points: u64, buffer_info: &BufferInfoInterleaved,
                  wgpu_device: &mut wgpu::Device, allow_download: bool) {
        // Determine struct alignment
        let struct_alignment =  self.struct_alignment(&buffer_info);

        let mut offset: usize = 0;
        for _ in 0..num_points {
            // Align to struct
            offset += self.padding_for_struct_alignment(offset, struct_alignment);

            let mut datatype_offset_map: HashMap<PointAttributeDataType, usize> = HashMap::new();
            for attrib in buffer_info.attributes {
                let num_bytes = attrib.datatype().size() as usize;
                self.calc_size(num_bytes, attrib.datatype(), &mut offset);

                let start_offset = offset - self.alignment_per_element(attrib.datatype());
                datatype_offset_map.insert(attrib.datatype(), start_offset);
            }
            self.offsets.push(datatype_offset_map);
        }

        let size = offset as wgpu::BufferAddress;
        self.buffer_size = Some(size);

        self.buffer_binding = Some(buffer_info.binding);

        let mut usage =
            wgpu::BufferUsages::STORAGE |
            wgpu::BufferUsages::COPY_SRC |
            wgpu::BufferUsages::COPY_DST |
            wgpu::BufferUsages::VERTEX; // for renderer

        if allow_download {
            // TODO: warning message from wgpu
            //  Feature MAPPABLE_PRIMARY_BUFFERS enabled on a discrete gpu.
            //  This is a massive performance footgun and likely not what you wanted.
            usage = usage |
                wgpu::BufferUsages::MAP_READ |
                wgpu::BufferUsages::MAP_WRITE;
        }

        self.buffer = Some(wgpu_device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("storage_buffer"),
                size,
                usage,
                mapped_at_creation: false
            }
        ));
    }

    // TODO: check if points_range valid etc.
    /// Queues the points in `points_range` within the `point_buffer` for upload onto the GPU device
    /// and sets the bind group together with its layout.
    /// The actual upload will occur once work is submitted to the GPU.
    ///
    /// Padding is inserted as necessary, depending on how the point data is structured in `buffer_info`.
    ///
    ///
    ///
    /// # Arguments
    /// * `point_buffer` - contains the point data that should be uploaded. While this could be an
    ///                    per-attribute buffer, you should only use interleaved buffers for now
    ///                    because currently there's only an option to download data into an
    ///                    interleaved buffer.
    /// * `points_range` - which points to upload.
    /// * `buffer_info`  - tells us how the point structure is defined on the shader side.
    /// * `wgpu_device`  - can be obtained from [Device](gpu::Device).
    /// * `wgpu_queue`   - can be obtained from [Device](gpu::Device).
    ///
    /// # Panics
    /// If no memory or not enough memory has been allocated previously via
    /// [malloc()](GpuPointBufferInterleaved::malloc), this method will panic.
    pub fn upload(
        &mut self,
        point_buffer: &dyn PointBuffer,
        points_range: std::ops::Range<usize>,
        buffer_info: &BufferInfoInterleaved,
        wgpu_device: &mut wgpu::Device,
        wgpu_queue: &wgpu::Queue)
    {
        let pt_rng = &points_range;

        // Determine struct alignment
        let struct_alignment = self.struct_alignment(&buffer_info);

        // Get bytes in interleaved format
        let mut offset: usize = 0;
        let mut bytes_to_write: Vec<u8> = vec![];
        for i in pt_rng.start..pt_rng.end {
            // Align to struct
            let required_padding = self.padding_for_struct_alignment(offset, struct_alignment);
            offset += required_padding;
            bytes_to_write.append(&mut vec![0_u8; required_padding]);

            for attrib in buffer_info.attributes {
                // Store the bytes for the current attribute in 'bytes_for_attrib'
                let bytes_per_element = attrib.datatype().size() as usize;
                let mut bytes_for_attrib: Vec<u8> = vec![0; bytes_per_element];
                point_buffer.get_raw_attribute(i, attrib, &mut *bytes_for_attrib);

                // Align each attribute
                let bytes_for_attrib: &[u8] = &*bytes_for_attrib;
                let mut bytes_for_attrib = self.align_slice(bytes_for_attrib, attrib.datatype(), &mut offset);

                bytes_to_write.append(&mut bytes_for_attrib);
            }
        }

        // Change Vec<u8> to &[u8]
        let bytes_to_write: &[u8] = &*bytes_to_write;

        // Schedule write to GPU memory
        let mut offset: usize = 0;
        for _ in 0..pt_rng.start {
            for attrib in buffer_info.attributes {
                let bytes_per_element = attrib.datatype().size() as usize;
                self.calc_size(bytes_per_element, attrib.datatype(), &mut offset);
            }
        }
        offset += self.padding_for_struct_alignment(offset, struct_alignment);

        let gpu_buffer = self.buffer.as_ref().unwrap();
        wgpu_queue.write_buffer(&gpu_buffer, offset as wgpu::BufferAddress, bytes_to_write);
    }

    /// Writes the contents of the GPU buffer into `point_buffer`, which is in interleaved format,
    /// within the `points_range` range.
    pub async fn download_into_interleaved(
        &self,
        point_buffer: &mut InterleavedVecPointStorage,
        points_range: std::ops::Range<usize>,
        buffer_info: &BufferInfoInterleaved<'_>,
        wgpu_device: &wgpu::Device)
    {
        let gpu_buffer = self.buffer.as_ref().unwrap();

        let gpu_buffer_slice = gpu_buffer.slice(..);
        let mapped_future = gpu_buffer_slice.map_async(wgpu::MapMode::Read);
        wgpu_device.poll(wgpu::Maintain::Wait); // TODO: "Should be called in event loop or other thread ..."

        if let Ok(()) = mapped_future.await {
            let mapped_view = gpu_buffer_slice.get_mapped_range();
            let result_as_bytes = mapped_view.to_vec();

            // Used to determine the offset of an attribute
            let point_layout = point_buffer.point_layout().clone();

            for j in points_range {
                let point_as_bytes = point_buffer.get_raw_point_mut(j);
                let datatype_offset_map = self.offsets.get(j).unwrap();

                for attrib in buffer_info.attributes {
                    let attrib_offset = point_layout.get_attribute(&attrib).unwrap().offset() as usize;
                    let offset = *datatype_offset_map.get(&attrib.datatype()).unwrap();
                    let size = self.alignment_per_element(attrib.datatype());

                    match attrib.datatype() {
                        PointAttributeDataType::Bool => {
                            let result: Vec<bool> = result_as_bytes[offset..(offset + size)]
                                .chunks_exact(4)
                                .map(|b| u32::from_ne_bytes(b.try_into().unwrap()) != 0)
                                .collect();

                            for i in attrib_offset..(attrib_offset + attrib.size() as usize) {
                                point_as_bytes[i] = result[i - attrib_offset] as u8;
                            }
                        },
                        PointAttributeDataType::U8 => {
                            let result: Vec<u8> = result_as_bytes[offset..(offset + size)]
                                .chunks_exact(4)
                                .map(|b| u32::from_ne_bytes(b.try_into().unwrap()) as u8)
                                .collect();

                            for i in attrib_offset..(attrib_offset + attrib.size() as usize) {
                                point_as_bytes[i] = result[i - attrib_offset];
                            }
                        },
                        PointAttributeDataType::I8 => {
                            let result: Vec<i8> = result_as_bytes[offset..(offset + size)]
                                .chunks_exact(4)
                                .map(|b| i32::from_ne_bytes(b.try_into().unwrap()) as i8)
                                .collect();

                            let bytes: &[u8] = bytemuck::cast_slice(result.as_slice());
                            for i in attrib_offset..(attrib_offset + attrib.size() as usize) {
                                point_as_bytes[i] = bytes[i - attrib_offset];
                            }
                        },
                        PointAttributeDataType::U16 => {
                            let result: Vec<u16> = result_as_bytes[offset..(offset + size)]
                                .chunks_exact(4)
                                .map(|b| u32::from_ne_bytes(b.try_into().unwrap()) as u16)
                                .collect();

                            let bytes: &[u8] = bytemuck::cast_slice(result.as_slice());
                            for i in attrib_offset..(attrib_offset + attrib.size() as usize) {
                                point_as_bytes[i] = bytes[i - attrib_offset];
                            }
                        },
                        PointAttributeDataType::I16 => {
                            let result: Vec<i16> = result_as_bytes[offset..(offset + size)]
                                .chunks_exact(4)
                                .map(|b| i32::from_ne_bytes(b.try_into().unwrap()) as i16)
                                .collect();

                            let bytes: &[u8] = bytemuck::cast_slice(result.as_slice());
                            for i in attrib_offset..(attrib_offset + attrib.size() as usize) {
                                point_as_bytes[i] = bytes[i - attrib_offset];
                            }
                        },
                        PointAttributeDataType::U32 => {
                            let result: Vec<u32> = result_as_bytes[offset..(offset + size)]
                                .chunks_exact(4)
                                .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
                                .collect();

                            let bytes: &[u8] = bytemuck::cast_slice(result.as_slice());
                            for i in attrib_offset..(attrib_offset + attrib.size() as usize) {
                                point_as_bytes[i] = bytes[i - attrib_offset];
                            }
                        },
                        PointAttributeDataType::I32 => {
                            let result: Vec<i32> = result_as_bytes[offset..(offset + size)]
                                .chunks_exact(4)
                                .map(|b| i32::from_ne_bytes(b.try_into().unwrap()))
                                .collect();

                            let bytes: &[u8] = bytemuck::cast_slice(result.as_slice());
                            for i in attrib_offset..(attrib_offset + attrib.size() as usize) {
                                point_as_bytes[i] = bytes[i - attrib_offset];
                            }
                        },
                        PointAttributeDataType::U64 => { /* Currently not supported */ },
                        PointAttributeDataType::I64 => { /* Currently not supported */ },
                        PointAttributeDataType::F32 => {
                            let result: Vec<f32> = result_as_bytes[offset..(offset + size)]
                                .chunks_exact(4)
                                .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                                .collect();

                            let bytes: &[u8] = bytemuck::cast_slice(result.as_slice());
                            for i in attrib_offset..(attrib_offset + attrib.size() as usize) {
                                point_as_bytes[i] = bytes[i - attrib_offset];
                            }
                        },
                        PointAttributeDataType::F64 => {
                            let result: Vec<f64> = result_as_bytes[offset..(offset + size)]
                                .chunks_exact(8)
                                .map(|b| f64::from_ne_bytes(b.try_into().unwrap()))
                                .collect();

                            let bytes: &[u8] = bytemuck::cast_slice(result.as_slice());
                            for i in attrib_offset..(attrib_offset + attrib.size() as usize) {
                                point_as_bytes[i] = bytes[i - attrib_offset];
                            }
                        },
                        PointAttributeDataType::Vec3u8 => {
                            let result4d: Vec<u8> = result_as_bytes[offset..(offset + size)]
                                .chunks_exact(4)
                                .map(|b| u32::from_ne_bytes(b.try_into().unwrap()) as u8)
                                .collect();

                            // Throw 4th coordinate away
                            let mut result: Vec<u8> = vec![];
                            for i in 0..result4d.len() {
                                if (i + 1) % 4 == 0 {
                                    continue;
                                }

                                result.push(result4d[i]);
                            }

                            for i in attrib_offset..(attrib_offset + attrib.size() as usize) {
                                point_as_bytes[i] = result[i - attrib_offset];
                            }
                        },
                        PointAttributeDataType::Vec4u8 => {
                            let result4d: Vec<u8> = result_as_bytes[offset..(offset + size)]
                                .chunks_exact(4)
                                .map(|b| u32::from_ne_bytes(b.try_into().unwrap()) as u8)
                                .collect();

                            let mut result: Vec<u8> = vec![];
                            for i in 0..result4d.len() {
                                result.push(result4d[i]);
                            }

                            for i in attrib_offset..(attrib_offset + attrib.size() as usize) {
                                point_as_bytes[i] = result[i - attrib_offset];
                            }
                        },
                        PointAttributeDataType::Vec3u16 => {
                            let result4d: Vec<u16> = result_as_bytes[offset..(offset + size)]
                                .chunks_exact(4)
                                .map(|b| u32::from_ne_bytes(b.try_into().unwrap()) as u16)
                                .collect();

                            // Throw 4th coordinate away
                            let mut result: Vec<u16> = vec![];
                            for i in 0..result4d.len() {
                                if (i + 1) % 4 == 0 {
                                    continue;
                                }

                                result.push(result4d[i]);
                            }

                            let bytes: &[u8] = bytemuck::cast_slice(result.as_slice());
                            for i in attrib_offset..(attrib_offset + attrib.size() as usize) {
                                point_as_bytes[i] = bytes[i - attrib_offset];
                            }
                        },
                        PointAttributeDataType::Vec3f32 => {
                            let result4d: Vec<f32> = result_as_bytes[offset..(offset + size)]
                                .chunks_exact(4)
                                .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                                .collect();

                            // Throw 4th coordinate away
                            let mut result: Vec<f32> = vec![];
                            for i in 0..result4d.len() {
                                if (i + 1) % 4 == 0 {
                                    continue;
                                }

                                result.push(result4d[i]);
                            }

                            let bytes: &[u8] = bytemuck::cast_slice(result.as_slice());
                            for i in attrib_offset..(attrib_offset + attrib.size() as usize) {
                                point_as_bytes[i] = bytes[i - attrib_offset];
                            }
                        },
                        PointAttributeDataType::Vec3f64 => {
                            let result4d: Vec<f64> = result_as_bytes[offset..(offset + size)]
                                .chunks_exact(8)
                                .map(|b| f64::from_ne_bytes(b.try_into().unwrap()))
                                .collect();

                            // Throw 4th coordinate away
                            let mut result: Vec<f64> = vec![];
                            for i in 0..result4d.len() {
                                if (i + 1) % 4 == 0 {
                                    continue;
                                }

                                result.push(result4d[i]);
                            }

                            let bytes: &[u8] = bytemuck::cast_slice(result.as_slice());
                            for i in attrib_offset..(attrib_offset + attrib.size() as usize) {
                                point_as_bytes[i] = bytes[i - attrib_offset];
                            }
                        },
                    }
                }
            }

            drop(mapped_view);
            gpu_buffer.unmap();
        }
    }

    /// Needs to be called before using the buffer in a shader via a bound group.
    pub fn create_bind_group(&mut self, wgpu_device: &mut wgpu::Device) {
        let bind_group_layout = wgpu_device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("storage_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: self.buffer_binding.unwrap(),
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: false
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None
                        },
                        count: None
                    }
                ],
            }
        );

        let bind_group = wgpu_device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("storage_bind_group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: self.buffer_binding.unwrap(),
                        resource: self.buffer.as_ref().unwrap().as_entire_binding(),
                    }
                ],
            }
        );

        self.bind_group_layout = Some(bind_group_layout);
        self.bind_group = Some(bind_group);
    }

    fn struct_alignment(&self, buffer_info: &BufferInfoInterleaved) -> usize {
        let mut struct_alignment: usize = 0;

        for attrib in buffer_info.attributes {
            let alignment = self.alignment_per_element(attrib.datatype());
            if alignment > struct_alignment {
                struct_alignment = alignment;
            }
        }

        struct_alignment
    }

    fn padding_for_struct_alignment(&self, offset: usize, struct_alignment: usize) -> usize {
        let mut padding: usize = 0;

        while (offset + padding) % struct_alignment != 0 {
            padding += 1;
        }

        padding
    }
}

/// Manages point buffer data that is to be stored in per-attribute format on the GPU.
///
/// Make sure to allocate enough memory before trying to upload anything.
pub struct GpuPointBufferPerAttribute<'a> {
    /// The [BindGroupLayout](wgpu::BindGroupLayout) that needs to be passed to the [Device](gpu::Device).
    /// It will be set with a call to [upload()](GpuPointBufferPerAttribute::upload).
    pub bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// The [BindGroup](wgpu::BindGroup) that needs to be passed to the [Device](gpu::Device).
    /// It will be set with a call to [upload()](GpuPointBufferPerAttribute::upload).
    pub bind_group: Option<wgpu::BindGroup>,

    // String: name of the attribute, eg. "POSITION_3D"
    // Consider (String, PointAttributeDataType)?, eg. ("POSITION_3D", Vec3f64)
    pub buffers: HashMap<String, wgpu::Buffer>,
    buffer_sizes: HashMap<String, wgpu::BufferAddress>,
    buffer_bindings: HashMap<String, u32>,
    buffer_keys: Vec<&'a PointAttributeDefinition>,   // For now need order (because download code in device_compute depends on it)
}

impl GpuPointBuffer for GpuPointBufferPerAttribute<'_> {}

impl<'a> GpuPointBufferPerAttribute<'a> {
    pub fn new() -> GpuPointBufferPerAttribute<'a> {
        GpuPointBufferPerAttribute {
            bind_group_layout: None,
            bind_group: None,
            buffers: HashMap::new(),
            buffer_sizes: HashMap::new(),
            buffer_bindings: HashMap::new(),
            buffer_keys: vec![]
        }
    }

    /// Allocates enough memory on the device to hold `num_points` many points that are structured
    /// as described in `buffer_info`.
    pub fn malloc(&mut self, num_points: u64, buffer_infos: &'a[BufferInfoPerAttribute],
                  wgpu_device: &mut wgpu::Device, allow_download: bool) {
        for info in buffer_infos {
            let size = (num_points as usize) * self.alignment_per_element(info.attribute.datatype());

            let key = info.attribute;
            self.buffer_keys.push(key);

            // HashMap need trait bound Hash, which PointAttributeDefinition does not have
            // So use String instead
            let key = String::from(info.attribute.name());
            self.buffer_sizes.insert(key.clone(), size as wgpu::BufferAddress);
            self.buffer_bindings.insert(key.clone(), info.binding);

            let mut usage =
                wgpu::BufferUsages::STORAGE |
                wgpu::BufferUsages::COPY_SRC |
                wgpu::BufferUsages::COPY_DST |
                wgpu::BufferUsages::VERTEX; // for renderer

            if allow_download {
                // TODO: warning message from wgpu
                //  Feature MAPPABLE_PRIMARY_BUFFERS enabled on a discrete gpu.
                //  This is a massive performance footgun and likely not what you wanted.
                usage = usage |
                    wgpu::BufferUsages::MAP_READ |
                    wgpu::BufferUsages::MAP_WRITE;
            }

            self.buffers.insert(key.clone(), wgpu_device.create_buffer(
                &wgpu::BufferDescriptor {
                    label: Some(format!("storage_buffer_{}", key).as_str()),
                    size: size as wgpu::BufferAddress,
                    usage,
                    mapped_at_creation: false,
                }
            ));
        }
    }

    /// Queues the points in `points_range` within the `point_buffer` for upload onto the GPU device
    /// and sets the bind group together with its layout.
    /// The actual upload will occur once work is submitted to the GPU.
    ///
    /// Padding is inserted as necessary. Because pasture only supports the `std430` layout for
    /// storage buffers, this currently only affects 3 component vectors which require an additional
    /// element as padding. Scalar values can be tightly packed.
    ///
    /// # Arguments
    /// * `point_buffer` - contains the point data that should be uploaded. While this could be an
    ///                    interleaved buffer, you should only use per-attribute buffers for now
    ///                    because currently there's only an option to download data into an
    ///                    per-attribute buffer.
    /// * `points_range` - which points to upload.
    /// * `buffer_infos` - tells us how the point attributes are defined on the shader side.
    /// * `wgpu_device`  - can be obtained from [Device](gpu::Device).
    /// * `wgpu_queue`   - can be obtained from [Device](gpu::Device).
    ///
    /// # Panics
    /// If no memory or not enough memory has been allocated previously via
    /// [malloc()](GpuPointBufferPerAttribute::malloc), this method will panic.
    pub fn upload(
        &mut self,
        point_buffer: &dyn PointBuffer,
        points_range: std::ops::Range<usize>,
        buffer_infos: &[BufferInfoPerAttribute],
        wgpu_device: &mut wgpu::Device,
        wgpu_queue: &wgpu::Queue)
    {
        let len = points_range.len();

        for info in buffer_infos {
            // Allocate enough space and load the points into the vector
            let bytes_per_element = info.attribute.datatype().size() as usize;
            let mut bytes_to_write: Vec<u8> = vec![0; len * bytes_per_element];
            point_buffer.get_raw_attribute_range(points_range.start..points_range.end, info.attribute, &mut *bytes_to_write);

            // Change Vec<u8> to &[u8] and align bytes
            let mut unused_for_per_attrib: usize = 0;
            let bytes_to_write: &[u8] = &*bytes_to_write;
            let bytes_to_write = &self.align_slice(bytes_to_write, info.attribute.datatype(), &mut unused_for_per_attrib)[..];

            // Schedule write to GPU memory, starting from correct offset
            let mut offset: usize = 0;
            self.calc_size(bytes_per_element * points_range.start, info.attribute.datatype(), &mut offset);

            let gpu_buffer = self.buffers.get(info.attribute.name()).unwrap();
            wgpu_queue.write_buffer(gpu_buffer, offset as wgpu::BufferAddress, bytes_to_write);
        }
    }

    /// Writes the contents of the GPU buffer into `point_buffer`, which is in per-attribute format,
    /// within the `points_range` range.
    /// NOTE: Not supported on the webgl webgpu backend at the moment due
    /// to its limitations (can't create the buffer with map_read | map_write usage)
    pub async fn download_into_per_attribute(
        &self,
        point_buffer: &mut dyn PerAttributePointBufferMut<'_>,
        points_range: std::ops::Range<usize>,
        buffer_infos: &Vec<BufferInfoPerAttribute<'_>>,
        wgpu_device: &wgpu::Device)
    {
        for info in buffer_infos {
            let gpu_buffer = self.buffers.get(info.attribute.name()).unwrap();

            let gpu_buffer_slice = gpu_buffer.slice(..);
            let mapped_future = gpu_buffer_slice.map_async(wgpu::MapMode::Read);
            wgpu_device.poll(wgpu::Maintain::Wait); // TODO: "Should be called in event loop or other thread ..."

            if let Ok(()) = mapped_future.await {
                let mapped_view = gpu_buffer_slice.get_mapped_range();
                let result_as_bytes = mapped_view.to_vec();

                let range = points_range.start..points_range.end;
                match info.attribute.datatype() {
                    PointAttributeDataType::Bool => {
                        let result: Vec<bool> = result_as_bytes
                            .chunks_exact(4)
                            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()) != 0)
                            .collect();

                        let attrib = point_buffer.get_attribute_range_mut::<bool>(range, info.attribute);
                        for i in 0..attrib.len() {
                            attrib[i] = result[i];
                        }
                    },
                    PointAttributeDataType::U8 => {
                        let result: Vec<u8> = result_as_bytes
                            .chunks_exact(4)
                            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()) as u8)
                            .collect();

                        let attrib = point_buffer.get_attribute_range_mut::<u8>(range, info.attribute);
                        for i in 0..attrib.len() {
                            attrib[i] = result[i];
                        }
                    },
                    PointAttributeDataType::I8 => {
                        let result: Vec<i8> = result_as_bytes
                            .chunks_exact(4)
                            .map(|b| i32::from_ne_bytes(b.try_into().unwrap()) as i8)
                            .collect();

                        let attrib = point_buffer.get_attribute_range_mut::<i8>(range, info.attribute);
                        for i in 0..attrib.len() {
                            attrib[i] = result[i];
                        }
                    },
                    PointAttributeDataType::U16 => {
                        let result: Vec<u16> = result_as_bytes
                            .chunks_exact(4)
                            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()) as u16)
                            .collect();

                        let attrib = point_buffer.get_attribute_range_mut::<u16>(range, info.attribute);
                        for i in 0..attrib.len() {
                            attrib[i] = result[i];
                        }
                    },
                    PointAttributeDataType::I16 => {
                        let result: Vec<i16> = result_as_bytes
                            .chunks_exact(4)
                            .map(|b| i32::from_ne_bytes(b.try_into().unwrap()) as i16)
                            .collect();

                        let attrib = point_buffer.get_attribute_range_mut::<i16>(range, info.attribute);
                        for i in 0..attrib.len() {
                            attrib[i] = result[i];
                        }
                    },
                    PointAttributeDataType::U32 => {
                        let result: Vec<u32> = result_as_bytes
                            .chunks_exact(4)
                            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
                            .collect();

                        let attrib = point_buffer.get_attribute_range_mut::<u32>(range, info.attribute);
                        for i in 0..attrib.len() {
                            attrib[i] = result[i];
                        }
                    },
                    PointAttributeDataType::I32 => {
                        let result: Vec<i32> = result_as_bytes
                            .chunks_exact(4)
                            .map(|b| i32::from_ne_bytes(b.try_into().unwrap()))
                            .collect();

                        let attrib = point_buffer.get_attribute_range_mut::<i32>(range, info.attribute);
                        for i in 0..attrib.len() {
                            attrib[i] = result[i];
                        }
                    },
                    PointAttributeDataType::U64 => {},
                    PointAttributeDataType::I64 => {},
                    PointAttributeDataType::F32 => {
                        let result: Vec<f32> = result_as_bytes
                            .chunks_exact(4)
                            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                            .collect();

                        let attrib = point_buffer.get_attribute_range_mut::<f32>(range, info.attribute);
                        for i in 0..attrib.len() {
                            attrib[i] = result[i];
                        }
                    },
                    PointAttributeDataType::F64 => {
                        let result: Vec<f64> = result_as_bytes
                            .chunks_exact(8)
                            .map(|b| f64::from_ne_bytes(b.try_into().unwrap()))
                            .collect();

                        let attrib = point_buffer.get_attribute_range_mut::<f64>(range, info.attribute);
                        for i in 0..attrib.len() {
                            attrib[i] = result[i];
                        }
                    },
                    PointAttributeDataType::Vec3u8 => {
                        let result: Vec<u8> = result_as_bytes
                            .chunks_exact(4)
                            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()) as u8)
                            .collect();

                        let attrib = point_buffer.get_attribute_range_mut::<Vector3<u8>>(range, info.attribute);
                        for i in 0..attrib.len() {
                            attrib[i].x = result[i * 4 + 0];
                            attrib[i].y = result[i * 4 + 1];
                            attrib[i].z = result[i * 4 + 2];
                        }
                    },
                    PointAttributeDataType::Vec4u8 => {
                        let result: Vec<u8> = result_as_bytes
                            .chunks_exact(4)
                            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()) as u8)
                            .collect();

                        let attrib = point_buffer.get_attribute_range_mut::<Vector4<u8>>(range, info.attribute);
                        for i in 0..attrib.len() {
                            attrib[i].x = result[i * 4 + 0];
                            attrib[i].y = result[i * 4 + 1];
                            attrib[i].z = result[i * 4 + 2];
                            attrib[i].w = result[i * 4 + 3];
                        }
                    },
                    PointAttributeDataType::Vec3u16 => {
                        let result: Vec<u16> = result_as_bytes
                            .chunks_exact(4)
                            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()) as u16)
                            .collect();

                        let attrib = point_buffer.get_attribute_range_mut::<Vector3<u16>>(range, info.attribute);
                        for i in 0..attrib.len() {
                            attrib[i].x = result[i * 4 + 0];
                            attrib[i].y = result[i * 4 + 1];
                            attrib[i].z = result[i * 4 + 2];
                        }
                    },
                    PointAttributeDataType::Vec3f32 => {
                        let result: Vec<f32> = result_as_bytes
                            .chunks_exact(4)
                            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                            .collect();

                        let attrib = point_buffer.get_attribute_range_mut::<Vector3<f32>>(range, info.attribute);
                        for i in 0..attrib.len() {
                            attrib[i].x = result[i * 4 + 0];
                            attrib[i].y = result[i * 4 + 1];
                            attrib[i].z = result[i * 4 + 2];
                        }
                    },
                    PointAttributeDataType::Vec3f64 => {
                        let result: Vec<f64> = result_as_bytes
                            .chunks_exact(8)
                            .map(|b| f64::from_ne_bytes(b.try_into().unwrap()))
                            .collect();

                        let attrib = point_buffer.get_attribute_range_mut::<Vector3<f64>>(range, info.attribute);
                        for i in 0..attrib.len() {
                            attrib[i].x = result[i * 4 + 0];
                            attrib[i].y = result[i * 4 + 1];
                            attrib[i].z = result[i * 4 + 2];
                        }
                    },
                };

                // Drop all mapped views before unmapping buffer
                drop(mapped_view);
                gpu_buffer.unmap();
            }
        }
    }

    pub fn create_bind_group(&mut self, wgpu_device: &mut wgpu::Device) {
        let mut group_layout_entries: Vec<wgpu::BindGroupLayoutEntry> = vec![];
        let mut group_entries: Vec<wgpu::BindGroupEntry> = vec![];

        for key in self.buffer_keys.as_slice() {
            let binding = *self.buffer_bindings.get(key.name()).unwrap();

            group_layout_entries.push(
                wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            );

            group_entries.push(
                wgpu::BindGroupEntry {
                    binding,
                    resource: self.buffers.get(key.name()).unwrap().as_entire_binding(),
                }
            );
        }

        let bind_group_layout = wgpu_device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("storage_bind_group_layout"),
                entries: &group_layout_entries,
            }
        );

        let bind_group = wgpu_device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("storage_bind_group"),
                layout: &bind_group_layout,
                entries: &group_entries,
            }
        );

        self.bind_group_layout = Some(bind_group_layout);
        self.bind_group = Some(bind_group);
    }
}
