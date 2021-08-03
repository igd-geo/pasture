use crate::layout;

pub struct Device<'a> {
    pub wgpu_device: wgpu::Device,
    pub wgpu_queue: wgpu::Queue,

    // Private fields
    adapter: wgpu::Adapter,

    cs_module: Option<wgpu::ShaderModule>,
    bind_group_layouts: Vec<&'a wgpu::BindGroupLayout>,
    bind_groups: Vec<&'a wgpu::BindGroup>,
    compute_pipeline: Option<wgpu::ComputePipeline>,
}

impl<'a> Device<'a> {
    /// Create a device with default options:
    /// - Low power GPU
    /// - Primary backend for wgpu to use [Vulkan, Metal, Dx12, Browser]
    pub async fn default() -> Device<'a> {
        Device::new(DeviceOptions::default()).await
    }

    /// Create a device respecting the desired [DeviceOptions]
    pub async fn new(device_options: DeviceOptions) -> Device<'a> {
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

        let (wgpu_device, wgpu_queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features,
                limits,
            },
            None,
        ).await.unwrap();

        // == Initially empty buffers =============================================================

        let cs_module = Option::None;
        let compute_pipeline = Option::None;

        let bind_group_layouts: Vec<&'a wgpu::BindGroupLayout> = Vec::new();
        let bind_groups: Vec<&'a wgpu::BindGroup> = Vec::new();

        Device {
            adapter,
            wgpu_device,
            wgpu_queue,
            cs_module,
            bind_group_layouts,
            bind_groups,
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
        println!("{:?}", self.wgpu_device.features());
    }

    pub fn print_default_limits(&self) {
        println!("{:?}", wgpu::Limits::default());
    }

    pub fn print_adapter_limits(&self) {
        println!("{:?}", self.adapter.limits());
    }

    pub fn print_active_limits(&self) {
        println!("{:?}", self.wgpu_device.limits());
    }

    pub fn add_bind_group(&mut self, bind_group_layout: &'a wgpu::BindGroupLayout, bind_group: &'a wgpu::BindGroup) {
        self.bind_group_layouts.push(bind_group_layout);
        self.bind_groups.push(bind_group);
    }

    // TODO: support WGSL shaders?
    pub fn set_compute_shader(&mut self, compute_shader_src: &str) {
        self.cs_module = self.compile_and_create_compute_module(compute_shader_src);

        let pipeline = self.create_compute_pipeline(self.cs_module.as_ref().unwrap());

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
            self.wgpu_device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                source: cs_data,
                flags: wgpu::ShaderFlags::default(),
            })
        )
    }

    fn create_compute_pipeline(&self, cs_module: &wgpu::ShaderModule) -> wgpu::ComputePipeline {
        let mut t: Vec<&wgpu::BindGroupLayout> = vec![];
        for layout in self.bind_group_layouts.as_slice() {
            t.push(&*layout);
        }

        let compute_pipeline_layout = self.wgpu_device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: None,
                // bind_group_layouts: self.bind_group_layouts.as_slice(),
                bind_group_layouts: t.as_slice(),
                push_constant_ranges: &[],
            }
        );

        let compute_pipeline = self.wgpu_device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&compute_pipeline_layout),
                module: &cs_module,
                entry_point: "main",
            }
        );

        compute_pipeline
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
            self.wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut t: Vec<&wgpu::BindGroup> = vec![];
        for group in self.bind_groups.as_slice() {
            t.push(&*group);
        }

        {
            // The compute pass will start ("dispatch") our compute shader.
            let mut compute_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            compute_pass.set_pipeline(self.compute_pipeline.as_ref().unwrap());
            for i in 0..self.bind_groups.len() {
                // compute_pass.set_bind_group(i as u32, self.bind_groups.as_slice()[i], &[]);
                compute_pass.set_bind_group(i as u32, t.as_slice()[i], &[]);
            }
            compute_pass.insert_debug_marker("Pasture Compute Debug");
            compute_pass.dispatch(x, y, z);
        }

        // Submit to queue
        self.wgpu_queue.submit(Some(encoder.finish()));
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
pub struct BufferInfoPerAttribute<'a> {
    pub attribute: &'a layout::PointAttributeDefinition,
    pub binding: u32,
}

pub struct BufferInfoInterleaved<'a> {
    pub attributes: &'a [layout::PointAttributeDefinition],
    pub binding: u32,
}