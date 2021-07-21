use wgpu::util::{DeviceExt, BufferInitDescriptor};
use wgpu::BufferDescriptor;
use crate::layout::{PointAttributeDataType};
use crate::layout;
use crate::containers::{PointBuffer};
use bytemuck::__core::convert::TryInto;

pub struct Device {
    // Private fields
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,

    upload_buffers: Vec<wgpu::Buffer>,
    download_buffers: Vec<wgpu::Buffer>,
    buffer_sizes: Vec<wgpu::BufferAddress>,
    buffer_bindings: Vec<u32>,

    cs_module: Option<wgpu::ShaderModule>,
    bind_group: Option<wgpu::BindGroup>,
    compute_pipeline: Option<wgpu::ComputePipeline>,
}

impl Device {
    /// Create a device with default options:
    /// - Low power GPU
    /// - Primary backend for wgpu to use [Vulkan, Metal, Dx12, Browser]
    pub async fn default() -> Device {
        Device::new(DeviceOptions::default()).await
    }

    /// Create a device respecting the desired [DeviceOptions]
    pub async fn new(device_options: DeviceOptions) -> Device {
        // == Create an instance from the desired backend =========================================

        let backend_bits = match device_options.device_backend {
            DeviceBackend::Primary => { wgpu::BackendBit::PRIMARY }
            DeviceBackend::Secondary => { wgpu::BackendBit::SECONDARY }
            DeviceBackend::Vulkan => { wgpu::BackendBit::VULKAN }
            DeviceBackend::Metal => { wgpu::BackendBit::METAL }
            DeviceBackend::Dx12 => { wgpu::BackendBit::DX12 }
            DeviceBackend::Dx11 => { wgpu::BackendBit::DX11 }
            DeviceBackend::OpenGL => { wgpu::BackendBit::GL }
            DeviceBackend::Browser => { wgpu::BackendBit::BROWSER_WEBGPU }
        };

        let instance = wgpu::Instance::new(backend_bits);

        // == Create an adapter with the desired power preference =================================

        let power_pref = if matches!(device_options.device_power, DevicePower::Low) {
            wgpu::PowerPreference::LowPower
        }
        else {
            wgpu::PowerPreference::HighPerformance
        };

        // The adapter gives us a handle to the actual device.
        // We can query some GPU information, such as the device name, its type (discrete vs integrated)
        // or the backend that is being used.
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: power_pref,
                compatible_surface: None
            }
        ).await.unwrap();

        // == Create a device and a queue from the given adapter ==================================

        let features = if device_options.use_adapter_features {
            adapter.features()
        }
        else {
            wgpu::Features::default()
        };

        let limits = if device_options.use_adapter_limits {
            adapter.limits()
        }
        else {
            // Some important ones that may be worth altering:
            //  - max_storage_buffers_per_shader_stage (defaults to just 4)
            //  - max_uniform_buffers_per_shader_stage (defaults to 12, which seems fine)
            //  - ...
            wgpu::Limits::default()
        };

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features,
                limits,
            },
            None,
        ).await.unwrap();

        // == Initially empty buffers =============================================================

        let upload_buffers: Vec<wgpu::Buffer> = Vec::new();
        let download_buffers: Vec<wgpu::Buffer> = Vec::new();
        let buffer_sizes: Vec<wgpu::BufferAddress> = Vec::new();
        let buffer_bindings: Vec<u32> = Vec::new();

        let cs_module = Option::None;
        let bind_group = Option::None;
        let compute_pipeline = Option::None;

        Device {
            adapter,
            device,
            queue,
            upload_buffers,
            download_buffers,
            buffer_sizes,
            buffer_bindings,
            cs_module,
            bind_group,
            compute_pipeline,
        }
    }

    /// Prints name, type, backend, PCI and vendor PCI id of the device.
    pub fn print_device_info(&self) {
        let info = self.adapter.get_info();

        println!("== Device Information ========");
        println!("Name: {}", info.name);
        println!("Type: {:?}", info.device_type);
        println!("Backend: {:?}", info.backend);
        println!("PCI id: {}", info.device);
        println!("Vendor PCI id: {}\n", info.vendor);
    }

    // TODO: are these feature/limit prints useful?
    pub fn print_default_features(&self) {
        println!("{:?}", wgpu::Features::default());
    }

    pub fn print_adapter_features(&self) {
        println!("{:?}", self.adapter.features());
    }

    pub fn print_active_features(&self) {
        println!("{:?}", self.device.features());
    }

    pub fn print_default_limits(&self) {
        println!("{:?}", wgpu::Limits::default());
    }

    pub fn print_adapter_limits(&self) {
        println!("{:?}", self.adapter.limits());
    }

    pub fn print_active_limits(&self) {
        println!("{:?}", self.device.limits());
    }

    /// Associates the given `PointBuffer` with GPU buffers w.r.t. the layouts defined in `Vec<BufferInfo>`.
    pub fn upload(&mut self, buffer: &mut dyn PointBuffer, buffer_infos: Vec<BufferInfo>) {
        let len = buffer.len();

        for info in buffer_infos {
            let mut offset: usize = 0;

            let num_bytes = self.bytes_per_element(info.attribute.datatype()) as usize;
            let mut bytes_to_write: Vec<u8> = vec![0; len * num_bytes];

            buffer.get_raw_attribute_range(0..len, info.attribute, &mut *bytes_to_write);

            // Change Vec<u8> to &[u8]
            let bytes_to_write: &[u8] = &*bytes_to_write;
            let bytes_to_write = &self.align_slice(bytes_to_write, info.attribute.datatype(), &mut offset)[..];

            let size_in_bytes = bytes_to_write.len() as wgpu::BufferAddress;
            self.buffer_sizes.push(size_in_bytes);

            self.upload_buffers.push(self.device.create_buffer_init(
                &BufferInitDescriptor {
                    label: None,
                    contents: bytes_to_write,
                    usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC
                }
            ));

            self.download_buffers.push(self.device.create_buffer(
                &BufferDescriptor {
                    label: None,
                    size: size_in_bytes,
                    usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
                    mapped_at_creation: false
                }
            ));

            self.buffer_bindings.push(info.binding);
        }
    }

    // Upload in interleaved format TODO: buffer_info, Vec<> or single?
    pub fn upload_interleaved(&mut self, buffer: &mut dyn PointBuffer, buffer_info: BufferInfoInterleaved) {
        let len = buffer.len();

        let layout = buffer.point_layout();
        layout.size_of_point_entry();

        // TODO: needed?
        let mut struct_alignment = 0;
        for attrib in buffer_info.attributes {
            let alignment = self.alignment_per_element(attrib.datatype()) as usize;
            if alignment > struct_alignment {
                struct_alignment = alignment;
            }
        }

        println!("Struct alignment: {}", struct_alignment);

        // Get bytes in interleaved format
        let mut offset: usize = 0;
        let mut bytes_to_write: Vec<u8> = vec![];
        for i in 0..len {
            // buffer.get_raw_point(i, &mut bytes_to_write[i..(i + bytes_per_point)]);
            for attrib in buffer_info.attributes {
                // Store the bytes for the current attribute in 'bytes_for_attrib'
                let num_bytes = self.bytes_per_element(attrib.datatype()) as usize;
                let mut bytes_for_attrib: Vec<u8> = vec![0; num_bytes];
                buffer.get_raw_attribute(i, attrib, &mut *bytes_for_attrib);

                // Align each attribute TODO: may require custom align method for struct
                let bytes_for_attrib: &[u8] = &*bytes_for_attrib;
                let mut bytes_for_attrib = self.align_slice(bytes_for_attrib, attrib.datatype(), &mut offset);

                bytes_to_write.append(&mut bytes_for_attrib);
            }

            println!("Current size: {}", bytes_to_write.len());
        }

        // Change Vec<u8> to &[u8]
        let bytes_to_write: &[u8] = &*bytes_to_write;

        let size_in_bytes = bytes_to_write.len() as wgpu::BufferAddress;
        println!("Size = {}", size_in_bytes);
        self.buffer_sizes.push(size_in_bytes);

        self.upload_buffers.push(self.device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytes_to_write,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC
            }
        ));

        self.download_buffers.push(self.device.create_buffer(
            &BufferDescriptor {
                label: None,
                size: size_in_bytes,
                usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false
            }
        ));

        self.buffer_bindings.push(buffer_info.binding);
    }

    // Given a PointAttributeDataType, returns the number of bytes an element with such type would need
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

    fn alignment_per_element(&self, datatype: PointAttributeDataType) -> u32 {
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

    // Given a slice of bytes and the corresponding data type of those bytes,
    // will ensure the bytes match the std430 layout of GLSL.
    //
    // In particular:
    //  - Unsigned integer types with less than 32 bits will be zero extended to 32 bits
    //  - Signed integer types with less than 32 bits will be sign extended to 32 bits
    //  - Booleans will be zero extended to 32 bits
    //  - 32 bit signed or unsigned integer types will be taken as is
    //  - 32 bit and 64 bit floating point types will be taken as is
    //  - Vec3 will be treated as Vec4 with w-coordinate set to 1
    //  - Above extension rules apply to the elements of vectors
    //
    // Will panic if data type is a 64-bit integer.
    //
    // TODO: Consider whether to support such sign/zero extension or just forbid types that need them.
    fn align_slice(&self, slice: &[u8], datatype: PointAttributeDataType, offset: &mut usize) -> Vec<u8> {
        let mut ret_bytes: Vec<u8> = Vec::new();

        let num_bytes = slice.len();

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
                    *offset += current.len();
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

    /// Downloads contents of GPU buffers
    // TODO: Currently returns vector of bytes per buffer... instead return altered point buffer.
    pub async fn download(&self) -> Vec<Vec<u8>> {
        let mut output_bytes: Vec<Vec<u8>> = Vec::new();

        for i in 0..self.download_buffers.len() {
            let download = self.download_buffers.get(i).unwrap();

            let result_buffer_slice = download.slice(..);
            let result_buffer_future = result_buffer_slice.map_async(wgpu::MapMode::Read);
            self.device.poll(wgpu::Maintain::Wait); // TODO: "Should be called in event loop or other thread ..."

            // TODO: how to know the data type of the current buffer?
            if let Ok(()) = result_buffer_future.await {
                let result_as_bytes = result_buffer_slice.get_mapped_range();
                &output_bytes.push(result_as_bytes.to_vec());

                // Drop all mapped views before unmapping buffer
                drop(result_as_bytes);
                download.unmap();
            }
        };

        output_bytes
    }

    /// Compiles the given compute shader source code and constructs a compute pipeline for it.
    pub fn set_compute_shader(&mut self, compute_shader_src: &str) {
        self.cs_module = self.compile_and_create_compute_module(compute_shader_src);

        let (bind_group, pipeline)
            = self.create_compute_pipeline(self.cs_module.as_ref().unwrap());

        self.bind_group = Some(bind_group);
        self.compute_pipeline = Some(pipeline);
    }

    fn compile_and_create_compute_module(&self, compute_shader_src: &str) -> Option<wgpu::ShaderModule> {
        // WebGPU wants its shaders pre-compiled in binary SPIR-V format.
        // So we'll take the source code of our compute shader and compile it
        // with the help of the shaderc crate.
        let mut compiler = shaderc::Compiler::new().unwrap();
        let cs_spirv = compiler
            .compile_into_spirv(
                compute_shader_src,
                shaderc::ShaderKind::Compute,
                "Compute shader",
                "main",
                None,
            )
            .unwrap();
        let cs_data = wgpu::util::make_spirv(cs_spirv.as_binary_u8());

        // Now with the binary data we can create and return our ShaderModule,
        // which will be executed on the GPU within our compute pipeline.
        Some(
            self.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                source: cs_data,
                flags: wgpu::ShaderFlags::default(),
            })
        )
    }

    fn create_compute_pipeline(&self, cs_module: &wgpu::ShaderModule) -> (wgpu::BindGroup, wgpu::ComputePipeline) {
        // Setup bind groups
        let mut group_layout_entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::new();
        let mut group_entries: Vec<wgpu::BindGroupEntry> = Vec::new();

        // TODO: just assumes that all layouts are COMPUTE + rw STORAGE + ...
        for i in 0..self.buffer_bindings.len() {
            let b = self.buffer_bindings[i];

            group_layout_entries.push(
                wgpu::BindGroupLayoutEntry {
                    binding: b,
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
                    binding: b,
                    resource: self.upload_buffers.get(i).unwrap().as_entire_binding(),
                }
            );
        }

        let bind_group_layout = self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &group_layout_entries,
            }
        );

        let bind_group = self.device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &group_entries,
            }
        );

        // Setup pipeline
        let compute_pipeline_layout = self.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            }
        );

        let compute_pipeline = self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&compute_pipeline_layout),
                module: &cs_module,
                entry_point: "main",
            }
        );

        (bind_group, compute_pipeline)
    }

    /// Launches compute work groups; `x`, `y`, `z` many in their respective dimensions.
    /// To launch a 1D or 2D work group, set the unwanted dimension to 1.
    /// (Work groups in GLSL are the same thing as blocks in CUDA. The equivalent of CUDA threads
    ///  in GLSL are called invocations. These are defined in the shaders themselves.)
    pub fn compute(&mut self, x: u32, y: u32, z: u32) {
        // Use a CommandEncoder to batch all commands that you wish to send to the GPU to execute.
        // The resulting CommandBuffer can then be submitted to the GPU via a Queue.
        // Signal the end of the batch with CommandEncoder#finish().
        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            // The compute pass will start ("dispatch") our compute shader.
            let mut compute_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            compute_pass.set_pipeline(self.compute_pipeline.as_ref().unwrap());
            compute_pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
            compute_pass.insert_debug_marker("Pasture Compute Debug");
            compute_pass.dispatch(x, y, z);
        }

        // Copy buffers
        {
            for i in 0..self.upload_buffers.len() {
                let upload = self.upload_buffers.get(i).unwrap();
                let download = self.download_buffers.get(i).unwrap();
                let size = self.buffer_sizes.get(i).unwrap();

                encoder.copy_buffer_to_buffer(upload, 0, download, 0, *size);
            }
        }

        // Submit to queue
        self.queue.submit(Some(encoder.finish()));
    }
}

// == Helper types ===============================================================================

/// Defines the desired capabilities of a device that is to be retrieved.
// TODO: be more flexible about features and limits
pub struct DeviceOptions {
    pub device_power: DevicePower,
    pub device_backend: DeviceBackend,
    pub use_adapter_features: bool,
    pub use_adapter_limits: bool,
}

impl Default for DeviceOptions {
    fn default() -> Self {
        Self {
            device_power: DevicePower::Low,
            device_backend: DeviceBackend::Primary,
            use_adapter_features: false,
            use_adapter_limits: false,
        }
    }
}

pub enum DevicePower {
    /// Usually an integrated GPU
    Low = 0,
    /// Usually a discrete GPU
    High = 1,
}

impl Default for DevicePower {
    /// Default is [DevicePower::Low]
    fn default() -> Self { Self::Low }
}

pub enum DeviceBackend {
    /// Primary backends for wgpu: Vulkan, Metal, Dx12, Browser
    Primary,
    /// Secondary backends for wgpu: OpenGL, Dx11
    Secondary,
    Vulkan,
    Metal,
    Dx12,
    Dx11,
    OpenGL,
    Browser,
}

impl Default for DeviceBackend {
    /// Default is [DeviceBackend::Primary]
    fn default() -> Self { Self::Primary }
}

/// Associates a point buffer attribute with a binding defined in a (compute) shader.
// TODO: consider usage, size, mapped_at_creation, type (SSBO vs UBO), etc.
pub struct BufferInfo<'a> {
    pub attribute: &'a layout::PointAttributeDefinition,
    pub binding: u32,
}

pub struct BufferInfoInterleaved<'a> {
    pub attributes: &'a [layout::PointAttributeDefinition],
    pub binding: u32,
}