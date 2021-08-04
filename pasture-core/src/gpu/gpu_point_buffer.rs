use crate::layout::PointAttributeDataType;
use bytemuck::__core::convert::TryInto;
use crate::containers::PointBuffer;
use crate::gpu::{BufferInfoInterleaved, BufferInfoPerAttribute};
use std::collections::HashMap;

pub trait GpuPointBuffer {
    fn bytes_per_element(&self, datatype: PointAttributeDataType) -> u32 {
        let num_bytes = match datatype {
            PointAttributeDataType::U8 => { 1 }
            PointAttributeDataType::I8 => { 1 }
            PointAttributeDataType::U16 => { 2 }
            PointAttributeDataType::I16 => { 2 }
            PointAttributeDataType::U32 => { 4 }
            PointAttributeDataType::I32 => { 4 }
            PointAttributeDataType::U64 => { 8 }
            PointAttributeDataType::I64 => { 8 }
            PointAttributeDataType::F32 => { 4 }
            PointAttributeDataType::F64 => { 8 }
            PointAttributeDataType::Bool => { 1 }
            PointAttributeDataType::Vec3u8 => { 3 }
            PointAttributeDataType::Vec3u16 => { 6 }
            PointAttributeDataType::Vec3f32 => { 12 }
            PointAttributeDataType::Vec3f64 => { 24 }
        };

        num_bytes
    }

    fn alignment_per_element(&self, datatype: PointAttributeDataType) -> usize {
        let alignment = match datatype {
            PointAttributeDataType::U8 => { 4 }
            PointAttributeDataType::I8 => { 4 }
            PointAttributeDataType::U16 => { 4 }
            PointAttributeDataType::I16 => { 4 }
            PointAttributeDataType::U32 => { 4 }
            PointAttributeDataType::I32 => { 4 }
            PointAttributeDataType::U64 => { 8 }    // TODO does not exist
            PointAttributeDataType::I64 => { 8 }    // TODO does not exist
            PointAttributeDataType::F32 => { 4 }
            PointAttributeDataType::F64 => { 8 }
            PointAttributeDataType::Bool => { 4 }
            PointAttributeDataType::Vec3u8 => { 16 }
            PointAttributeDataType::Vec3u16 => { 16 }
            PointAttributeDataType::Vec3f32 => { 16 }
            PointAttributeDataType::Vec3f64 => { 32 }
        };

        alignment
    }

    fn align_slice(&self, slice: &[u8], datatype: PointAttributeDataType, offset: &mut usize) -> Vec<u8> {
        let mut ret_bytes: Vec<u8> = Vec::new();

        let num_bytes = slice.len();

        println!("Num bytes = {}", num_bytes);
        println!("{}: {}", datatype, offset);

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
                let stride = self.bytes_per_element(datatype) as usize;
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
            PointAttributeDataType::Vec3u8 => {
                // Treating as Vec4u32
                let one_as_bytes = 1_u32.to_ne_bytes();

                // Each entry is 8 bits, ie. 1 byte -> each Vec3 has 3 bytes
                let stride = self.bytes_per_element(datatype) as usize;
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
                let stride = self.bytes_per_element(datatype) as usize;   // = 6
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
                let stride = self.bytes_per_element(datatype) as usize;   // = 24
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
                let stride = self.bytes_per_element(datatype) as usize;   // = 24
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
                let stride = self.bytes_per_element(datatype) as usize;
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
                let stride = self.bytes_per_element(datatype) as usize;
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
            PointAttributeDataType::Vec3u16 => {
                // Each entry is 16 bits, ie. 2 bytes -> each Vec3 has 3*2 = 6 bytes
                let stride = self.bytes_per_element(datatype) as usize;   // = 6
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
                let stride = self.bytes_per_element(datatype) as usize;   // = 24
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
                let stride = self.bytes_per_element(datatype) as usize;   // = 24
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

pub struct GpuPointBufferInterleaved {
    pub bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub bind_group: Option<wgpu::BindGroup>,

    buffer: Option<wgpu::Buffer>,
    buffer_size: Option<wgpu::BufferAddress>,
    buffer_binding: Option<u32>,
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
        }
    }

    pub fn malloc(&mut self, num_points: u64, buffer_info: &BufferInfoInterleaved, wgpu_device: &mut wgpu::Device) {
        let mut offset: usize = 0;
        for _ in 0..num_points {
            for attrib in buffer_info.attributes {
                let num_bytes = self.bytes_per_element(attrib.datatype()) as usize;
                self.calc_size(num_bytes, attrib.datatype(), &mut offset);
            }
        }

        let size = offset as wgpu::BufferAddress;
        println!("Calculated size: {}", size);
        self.buffer_size = Some(size);

        self.buffer_binding = Some(buffer_info.binding);

        self.buffer = Some(wgpu_device.create_buffer(
            &wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsage::STORAGE |
                    wgpu::BufferUsage::MAP_READ |
                    wgpu::BufferUsage::MAP_WRITE |
                    wgpu::BufferUsage::COPY_SRC |
                    wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false
            }
        ));
    }

    // TODO: check if points_range valid etc.
    pub fn upload(
        &mut self,
        point_buffer: &dyn PointBuffer,
        points_range: std::ops::Range<usize>,
        buffer_info: &BufferInfoInterleaved,
        wgpu_device: &mut wgpu::Device,
        wgpu_queue: &wgpu::Queue
    ) {
        let pt_rng = &points_range;

        // Get bytes in interleaved format
        let mut offset: usize = 0;
        let mut bytes_to_write: Vec<u8> = vec![];
        for i in pt_rng.start..pt_rng.end {
            // buffer.get_raw_point(i, &mut bytes_to_write[i..(i + bytes_per_point)]);
            for attrib in buffer_info.attributes {
                // Store the bytes for the current attribute in 'bytes_for_attrib'
                let num_bytes = self.bytes_per_element(attrib.datatype()) as usize;

                let mut bytes_for_attrib: Vec<u8> = vec![0; num_bytes];
                point_buffer.get_raw_attribute(i, attrib, &mut *bytes_for_attrib);

                // Align each attribute TODO: may require custom align method for struct
                let bytes_for_attrib: &[u8] = &*bytes_for_attrib;
                let mut bytes_for_attrib = self.align_slice(bytes_for_attrib, attrib.datatype(), &mut offset);

                bytes_to_write.append(&mut bytes_for_attrib);
            }

            println!("Current size: {}", bytes_to_write.len());
        }

        // Change Vec<u8> to &[u8]
        let bytes_to_write: &[u8] = &*bytes_to_write;

        // Schedule write to GPU memory
        let mut offset: usize = 0;
        for _ in 0..pt_rng.start { // Note: inclusive range
            for attrib in buffer_info.attributes {
                let num_bytes = self.bytes_per_element(attrib.datatype()) as usize;
                self.calc_size(num_bytes, attrib.datatype(), &mut offset);
            }
        }
        offset += 8;    // TODO: have to include padding for the last element
        println!("New offset: {}", offset);

        let gpu_buffer = self.buffer.as_ref().unwrap();
        wgpu_queue.write_buffer(&gpu_buffer, offset as wgpu::BufferAddress, bytes_to_write);
        // wgpu_queue.write_buffer(&gpu_buffer, 0, bytes_to_write);

        self.create_bind_group(wgpu_device);
    }

    // TODO: provide range? (But cannot be 'points_range', we only see bytes...)
    pub async fn download(&self, wgpu_device: &wgpu::Device) -> Vec<Vec<u8>> {
        let mut output_bytes: Vec<Vec<u8>> = Vec::new();

        let gpu_buffer = self.buffer.as_ref().unwrap();

        let result_buffer_slice = gpu_buffer.slice(..);
        let map = result_buffer_slice.map_async(wgpu::MapMode::Read);
        wgpu_device.poll(wgpu::Maintain::Wait); // TODO: "Should be called in event loop or other thread ..."

        // TODO: how to know the data type of the current buffer?
        if let Ok(()) = map.await {
            let result_as_bytes = result_buffer_slice.get_mapped_range();
            &output_bytes.push(result_as_bytes.to_vec());

            // Drop all mapped views before unmapping buffer
            drop(result_as_bytes);
            gpu_buffer.unmap();
        }

        output_bytes
    }

    fn create_bind_group(&mut self, wgpu_device: &mut wgpu::Device) {
        let bind_group_layout = wgpu_device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: self.buffer_binding.unwrap(),
                        visibility: wgpu::ShaderStage::COMPUTE,
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
                label: None,
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
}

pub struct GpuPointBufferPerAttribute {
    pub bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub bind_group: Option<wgpu::BindGroup>,

    // String: name of the attribute, eg. "POSITION_3D"
    // Consider (String, PointAttributeDataType)?, eg. ("POSITION_3D", Vec3f64)
    buffers: HashMap<String, wgpu::Buffer>,
    buffer_sizes: HashMap<String, wgpu::BufferAddress>,
    buffer_bindings: HashMap<String, u32>,
    buffer_keys: Vec<String>,   // For now need order (because download code in device_compute depends on it)
}

impl GpuPointBuffer for GpuPointBufferPerAttribute {}

impl GpuPointBufferPerAttribute {
    pub fn new() -> GpuPointBufferPerAttribute {
        GpuPointBufferPerAttribute {
            bind_group_layout: None,
            bind_group: None,
            buffers: HashMap::new(),
            buffer_sizes: HashMap::new(),
            buffer_bindings: HashMap::new(),
            buffer_keys: vec![]
        }
    }

    pub fn malloc(&mut self, num_points: u64, buffer_infos: &Vec<BufferInfoPerAttribute>, wgpu_device: &mut wgpu::Device) {
        for info in buffer_infos {
            let size = (num_points as usize) * self.alignment_per_element(info.attribute.datatype());
            println!("Calculated size ({}): {}", info.attribute, size);

            let key = String::from(info.attribute.name());
            self.buffer_keys.push(key.clone());
            self.buffer_sizes.insert(key.clone(), size as wgpu::BufferAddress);
            self.buffer_bindings.insert(key.clone(), info.binding);
            self.buffers.insert(key.clone(), wgpu_device.create_buffer(
                &wgpu::BufferDescriptor {
                    label: None,
                    size: size as wgpu::BufferAddress,
                    usage: wgpu::BufferUsage::STORAGE |
                        wgpu::BufferUsage::MAP_READ |
                        wgpu::BufferUsage::MAP_WRITE |
                        wgpu::BufferUsage::COPY_SRC |
                        wgpu::BufferUsage::COPY_DST,
                    mapped_at_creation: false,
                }
            ));
        }
    }

    pub fn upload(
        &mut self,
        point_buffer: &dyn PointBuffer,
        points_range: std::ops::Range<usize>,
        buffer_infos: &Vec<BufferInfoPerAttribute>,
        wgpu_device: &mut wgpu::Device,
        wgpu_queue: &wgpu::Queue
    ) {
        let len = points_range.len();

        for info in buffer_infos {
            // Allocate enough space and load the points into the vector
            let bytes_per_element = self.bytes_per_element(info.attribute.datatype()) as usize;
            let mut bytes_to_write: Vec<u8> = vec![0; len * bytes_per_element];
            point_buffer.get_raw_attribute_range(points_range.start..points_range.end, info.attribute, &mut *bytes_to_write);

            // Change Vec<u8> to &[u8] and align bytes
            let mut unused_for_per_attrib: usize = 0;
            let bytes_to_write: &[u8] = &*bytes_to_write;
            let bytes_to_write = &self.align_slice(bytes_to_write, info.attribute.datatype(), &mut unused_for_per_attrib)[..];
            println!("--- Offset = {}", unused_for_per_attrib);

            // Schedule write to GPU memory, starting from correct offset
            let mut offset: usize = 0;
            self.calc_size(bytes_per_element * points_range.start, info.attribute.datatype(), &mut offset);
            println!("*** Offset = {}", offset);

            let gpu_buffer = self.buffers.get(info.attribute.name()).unwrap();
            wgpu_queue.write_buffer(gpu_buffer, offset as wgpu::BufferAddress, bytes_to_write);
        }

        self.create_bind_group(wgpu_device);
    }

    pub async fn download(&self, wgpu_device: &wgpu::Device) -> Vec<Vec<u8>> {
        let mut output_bytes: Vec<Vec<u8>> = Vec::new();

        for key in self.buffer_keys.as_slice() {
            let gpu_buffer = self.buffers.get(key).unwrap();

            let result_buffer_slice = gpu_buffer.slice(..);
            let result_buffer_future = result_buffer_slice.map_async(wgpu::MapMode::Read);
            wgpu_device.poll(wgpu::Maintain::Wait); // TODO: "Should be called in event loop or other thread ..."

            // TODO: how to know the data type of the current buffer?
            if let Ok(()) = result_buffer_future.await {
                let result_as_bytes = result_buffer_slice.get_mapped_range();
                &output_bytes.push(result_as_bytes.to_vec());

                // Drop all mapped views before unmapping buffer
                drop(result_as_bytes);
                gpu_buffer.unmap();
            }
        };

        output_bytes
    }

    fn create_bind_group(&mut self, wgpu_device: &mut wgpu::Device) {
        let mut group_layout_entries: Vec<wgpu::BindGroupLayoutEntry> = vec![];
        let mut group_entries: Vec<wgpu::BindGroupEntry> = vec![];

        for key in self.buffer_keys.as_slice() {
            println!("Key = {}", key);
            let binding = *self.buffer_bindings.get(key).unwrap();

            group_layout_entries.push(
                wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStage::COMPUTE,
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
                    resource: self.buffers.get(key.as_str()).unwrap().as_entire_binding(),
                }
            );
        }

        let bind_group_layout = wgpu_device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &group_layout_entries,
            }
        );

        let bind_group = wgpu_device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &group_entries,
            }
        );

        self.bind_group_layout = Some(bind_group_layout);
        self.bind_group = Some(bind_group);
    }
}
