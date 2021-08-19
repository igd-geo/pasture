use crate::layout;
use wgpu::util::DeviceExt;
use std::collections::BTreeMap;
use std::ops::BitAnd;

/// The base structure used to get access to the GPU. In addition it handles things like
/// shader compilation and the actual dispatch of work to the GPU.
pub struct Device<'a> {
    /// A handle to `wgpu`'s [Device](wgpu::Device). Can be used to create buffers, bind groups, etc.
    pub wgpu_device: wgpu::Device,

    /// A handle to `wgpu`'s [Queue](wgpu::Queue). Can be used to write to buffers, submit work, etc.
    pub wgpu_queue: wgpu::Queue,

    // Private fields
    adapter: wgpu::Adapter,
    cs_module: Option<wgpu::ShaderModule>,
    bind_group_data: BTreeMap<u32, BindGroupPair<'a>>,
    compute_pipeline: Option<wgpu::ComputePipeline>,
}

impl<'a> Device<'a> {
    /// Create a device with default options:
    /// - Low power GPU
    /// - `Vulkan` as backend
    /// - `wgpu`'s [MAPPABLE_PRIMARY_BUFFERS](wgpu::Features::MAPPABLE_PRIMARY_BUFFERS) feature enabled
    /// - Minimal limits
    pub async fn default() -> Result<Device<'a>, wgpu::RequestDeviceError> {
        Device::new(DeviceOptions::default()).await
    }

    /// Create and return a device respecting the desired [DeviceOptions].
    ///
    /// # Arguments
    /// * `device_options` - specifies the capabilities the device should have.
    ///
    /// # Errors
    /// If no device on the physical system can match the requested capabilities, an error
    /// is returned.
    ///
    /// Note that the device must have support for `wgpu`'s
    /// [MAPPABLE_PRIMARY_BUFFERS](wgpu::Features::MAPPABLE_PRIMARY_BUFFERS) feature, otherwise
    /// an error is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use pasture_core::gpu;
    ///
    /// let device = gpu::Device::new(
    ///     gpu::DeviceOptions {
    ///         device_power: gpu::DevicePower::High,
    ///         device_backend: gpu::DeviceBackend::Vulkan,
    ///         use_adapter_features: true,
    ///         use_adapter_limits: true,
    ///     }
    /// ).await;
    ///
    /// let mut device = match device {
    ///     Ok(d) => d ,
    ///     Err(_) => {
    ///         println!("Failed to request device. Aborting.");
    ///         return;
    ///     }
    /// };
    /// ```
    pub async fn new(device_options: DeviceOptions) -> Result<Device<'a>, wgpu::RequestDeviceError> {
        // == Create an instance from the desired backend =========================================

        let backend_bits = match device_options.device_backend {
            // DeviceBackend::Primary => { wgpu::BackendBit::PRIMARY }
            // DeviceBackend::Secondary => { wgpu::BackendBit::SECONDARY }
            DeviceBackend::Vulkan => { wgpu::BackendBit::VULKAN }
            // DeviceBackend::Metal => { wgpu::BackendBit::METAL }
            // DeviceBackend::Dx12 => { wgpu::BackendBit::DX12 }
            // DeviceBackend::Dx11 => { wgpu::BackendBit::DX11 }
            // DeviceBackend::OpenGL => { wgpu::BackendBit::GL }
            // DeviceBackend::Browser => { wgpu::BackendBit::BROWSER_WEBGPU }
        };

        let instance = wgpu::Instance::new(backend_bits);

        // == Create an adapter with the desired power preference =================================

        let power_pref = match device_options.device_power {
            DevicePower::Low => wgpu::PowerPreference::LowPower,
            DevicePower::High => wgpu::PowerPreference::HighPerformance,
        };

        // The adapter gives us a handle to the actual device.
        // We can query some GPU information, such as the device name, its type (discrete vs integrated)
        // or the backend that is being used.
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: power_pref,
                compatible_surface: None
            }
        ).await;

        let adapter = match adapter {
            Some(a) => a,
            None => return Result::Err(wgpu::RequestDeviceError),
        };

        // == Create a device and a queue from the given adapter ==================================

        if !adapter.features().contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS) {
            return Result::Err(wgpu::RequestDeviceError);
        }

        let features = match device_options.use_adapter_features {
            true => adapter.features().bitand(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
            false => wgpu::Features::MAPPABLE_PRIMARY_BUFFERS,
        };

        let limits = match device_options.use_adapter_limits {
            true => adapter.limits(),
            false => wgpu::Limits::default(),
        };

        let (wgpu_device, wgpu_queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("wgpu_device_and_queue"),
                features,
                limits,
            },
            None,
        ).await?;

        // == Other fields =========================================================================

        let cs_module = Option::None;
        let compute_pipeline = Option::None;

        let bind_group_data = BTreeMap::new();

        Ok(Device {
            adapter,
            wgpu_device,
            wgpu_queue,
            cs_module,
            bind_group_data,
            compute_pipeline,
        })
    }

    /// Displays name, type, backend, PCI id and vendor PCI id of the device.
    pub fn print_device_info(&self) {
        let info = self.adapter.get_info();

        println!("== Device Information ========");
        println!("Name: {}", info.name);
        println!("Type: {:?}", info.device_type);
        println!("Backend: {:?}", info.backend);
        println!("PCI id: {}", info.device);
        println!("Vendor PCI id: {}\n", info.vendor);
    }

    /// Displays the features that the physical GPU is able to support.
    pub fn print_adapter_features(&self) {
        println!("Features supported by the adapter: {:?}", self.adapter.features());
    }

    /// Displays the features that are currently active.
    pub fn print_active_features(&self) {
        println!("Currently active features: {:?}", self.wgpu_device.features());
    }

    /// Displays the default limits that are likely supported by all devices.
    pub fn print_default_limits(&self) {
        println!("Default limits: {:?}", wgpu::Limits::default());
    }

    /// Displays the best limits the physical GPU can support.
    pub fn print_adapter_limits(&self) {
        println!("\"Best\" limits supported by the adapter: {:?}", self.adapter.limits());
    }

    /// Displays the limits that are currently active.
    pub fn print_active_limits(&self) {
        println!("Currently active limits: {:?}", self.wgpu_device.limits());
    }

    /// Creates a UBO from `uniform_as_bytes` and returns a bind group together with a layout
    /// for it at the given `binding`.
    ///
    /// # Arguments
    /// * `uniform_as_bytes` - the uniform's content as bytes. Make sure it's correctly aligned
    ///                        according to the `std140` layout rules.
    /// * `binding` - the binding at which the uniform buffer object is set in the shader.
    pub fn create_uniform_bind_group(&self, uniform_as_bytes: &[u8], binding: u32) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
        // TODO: separate buffer from bind group -> should probably become part of Device state
        let uniform_buffer = self.wgpu_device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("uniform_buffer"),
                contents: uniform_as_bytes,
                usage: wgpu::BufferUsage::UNIFORM,
            }
        );

        let uniform_bind_group_layout = self.wgpu_device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("uniform_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None
                        },
                        count: None
                    }
                ],
            }
        );

        let uniform_bind_group = self.wgpu_device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("uniform_bind_group"),
                layout: &uniform_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
            }
        );

        (uniform_bind_group_layout, uniform_bind_group)
    }

    /// Associate a bind group and its layout with a given set on the shader side.
    /// Eg. if on the shader we have a buffer with `layout(std430, set=2, binding=0)`,
    /// then the passed in `index` should equal 2.
    pub fn set_bind_group(&mut self, index: u32, bind_group_layout: &'a wgpu::BindGroupLayout, bind_group: &'a wgpu::BindGroup) {
        let bind_group_pair = BindGroupPair {
            bind_group_layout,
            bind_group,
        };

        self.bind_group_data.insert(index, bind_group_pair);
    }

    /// Sets up a compute pipeline with the passed in WGSL shader source code.
    pub fn set_compute_shader_wgsl(&mut self, wgsl_compute_shader_src: &str) {
        self.cs_module = Some(self.wgpu_device.create_shader_module(
            &wgpu::ShaderModuleDescriptor {
                label: Some("wgsl_computer_shader_module"),
                source: wgpu::ShaderSource::Wgsl(wgsl_compute_shader_src.into()),
                flags: self.get_shader_flags(),
            }
        ));

        let pipeline = self.create_compute_pipeline(self.cs_module.as_ref().unwrap());

        self.compute_pipeline = Some(pipeline);
    }

    /// Compiles the passed in GLSL shader source code into Spir-V and sets up a compute pipeline.
    pub fn set_compute_shader_glsl(&mut self, compute_shader_src: &str) {
        self.cs_module = self.compile_glsl_and_create_compute_module(compute_shader_src);

        let pipeline = self.create_compute_pipeline(self.cs_module.as_ref().unwrap());

        self.compute_pipeline = Some(pipeline);
    }

    fn compile_glsl_and_create_compute_module(&self, compute_shader_src: &str) -> Option<wgpu::ShaderModule> {
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
                label: Some("glsl_compute_shader_module"),
                source: cs_data,
                // No flags for .glsl for now (SHADER_FLOAT_64 not parsed to Spir-V)
                // TODO: issue is known, should be fixed by wgpu in the near future
                //  see https://github.com/gfx-rs/naga/pull/1209
                flags: wgpu::ShaderFlags::default(),
            })
        )
    }

    fn get_shader_flags(&self) -> wgpu::ShaderFlags {
        // Taken from the wgpu examples, eg.
        // https://github.com/gfx-rs/wgpu-rs/blob/master/examples/water/main.rs
        let mut flags = wgpu::ShaderFlags::VALIDATION;
        match self.adapter.get_info().backend {
            wgpu::Backend::Vulkan | wgpu::Backend::Metal | wgpu::Backend::Gl => {
                flags |= wgpu::ShaderFlags::EXPERIMENTAL_TRANSLATION;
            },
            _ => {},
        }

        flags
    }

    fn create_compute_pipeline(&self, cs_module: &wgpu::ShaderModule) -> wgpu::ComputePipeline {
        let layouts = self.bind_group_data
            .values()
            .map(|pair| pair.bind_group_layout)
            .collect::<Vec<&'a wgpu::BindGroupLayout>>();

        let compute_pipeline_layout = self.wgpu_device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("compute_pipeline_layout"),
                bind_group_layouts: layouts.as_slice(),
                push_constant_ranges: &[],
            }
        );

        let compute_pipeline = self.wgpu_device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("compute_pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &cs_module,
                entry_point: "main",
            }
        );

        compute_pipeline
    }

    /// Launches compute work groups; `x`, `y`, `z` many in their respective dimensions.
    /// To launch a 1D or 2D work group, set the unwanted dimension to 1.
    /// Assumes that shaders and bind groups have been sets.
    ///
    /// # Panics
    /// Will panic if no shader is set.
    pub fn compute(&mut self, x: u32, y: u32, z: u32) {
        // Use a CommandEncoder to batch all commands that you wish to send to the GPU to execute.
        // The resulting CommandBuffer can then be submitted to the GPU via a Queue.
        // Signal the end of the batch with CommandEncoder#finish().
        let mut encoder =
            self.wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("command_encoder") });

        {
            // The compute pass will start ("dispatch") our compute shader.
            let mut compute_pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("compute_pass")
                }
            );
            compute_pass.set_pipeline(self.compute_pipeline.as_ref().unwrap());

            for (i, bind_group_pair) in self.bind_group_data.values().enumerate() {
                compute_pass.set_bind_group(i as u32, bind_group_pair.bind_group, &[]);
            }

            compute_pass.insert_debug_marker("Pasture Compute Debug");
            compute_pass.dispatch(x, y, z);
        }

        // Submit to queue
        self.wgpu_queue.submit(Some(encoder.finish()));
    }
}

// == Helper types ===============================================================================

// TODO: it may be beneficial to be more flexible about features and limits.
/// Defines the desired capabilities of a device that is to be retrieved.
pub struct DeviceOptions {
    pub device_power: DevicePower,
    pub device_backend: DeviceBackend,
    pub use_adapter_features: bool,
    pub use_adapter_limits: bool,
}

impl Default for DeviceOptions {
    /// Default uses a low power GPU with Vulkan backend and default features and limits.
    fn default() -> Self {
        Self {
            device_power: DevicePower::Low,
            device_backend: DeviceBackend::Vulkan,
            use_adapter_features: false,
            use_adapter_limits: false,
        }
    }
}

/// Controls which kind of GPU should be retrieved.
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

/// Currently only `Vulkan` is supported, because it is the only backend that allows 64-bit floats
/// on the shader side.
/// In the future, support for other backends such as `DirectX12` and `Metal` may be added.
pub enum DeviceBackend {
    // /// Primary backends for wgpu: Vulkan, Metal, Dx12, Browser
    // Primary,
    // /// Secondary backends for wgpu: OpenGL, Dx11
    // Secondary,
    Vulkan,
    // Metal,
    // Dx12,
    // Dx11,
    // OpenGL,
    // Browser,
}

impl Default for DeviceBackend {
    /// Default is `Vulkan`
    fn default() -> Self { Self::Vulkan }
}

// TODO: consider usage (readonly vs read/write, shader stages, ...), size, mapped_at_creation, etc.
/// Associates a point buffer attribute with one defined in a shader at the given binding.
///
/// # Examples
///
/// If the attributes in the shader are defined as follows at the given bindings:
/// ```ignore
/// layout(std430, set=0, binding=0) buffer PosBuffer {
///     dvec4 positions[];
/// };
///
/// layout(std430, set=0, binding=1) buffer IntensityBuffer {
///     uint intensities[];
/// };
/// ```
///
/// then the corresponding `BufferInfoPerAttribute` structure should look like this:
/// ```
/// use pasture_core::gpu;
/// use pasture_core::layout::PointAttributeDefinition;
/// use pasture_core::layout::attributes;
///
/// let buffer_infos = vec![
///     gpu::BufferInfoPerAttribute {
///         attribute: &attributes::POSITION_3D,
///         binding: 0,
///     },
///     gpu::BufferInfoPerAttribute {
///         attribute: &attributes::INTENSITY,
///         binding: 1,
///     },
/// ];
/// ```
pub struct BufferInfoPerAttribute<'a> {
    pub attribute: &'a layout::PointAttributeDefinition,
    pub binding: u32,
}

/// Associates interleaved point buffer attributes with a struct in a shader at the given binding.
///
/// # Examples
///
/// If the point structure in the shader is defined as follows, set at binding 0:
/// ```ignore
/// struct PointBuffer {
///     dvec4 position;
///     uint intensity;
/// }
///
/// layout(std430, set=0, binding=0) buffer PointBufferSsbo {
///     PointBuffer pointBuffer[];
/// };
/// ```
///
/// then the corresponding `BufferInfoInterleaved` structure should look like this:
///
/// ```
/// use pasture_core::gpu;
/// use pasture_core::layout::PointAttributeDefinition;
/// use pasture_core::layout::attributes;
///
/// let buffer_info = gpu::BufferInfoInterleaved {
///        // Same order as in shader
///        attributes: &[
///            attributes::POSITION_3D,
///            attributes::INTENSITY,
///        ],
///        binding: 0
/// };
/// ```
pub struct BufferInfoInterleaved<'a> {
    pub attributes: &'a [layout::PointAttributeDefinition],
    pub binding: u32,
}

// Helper struct to have a bind group tightly coupled with its layout.
struct BindGroupPair<'a> {
    bind_group_layout: &'a wgpu::BindGroupLayout,
    bind_group: &'a wgpu::BindGroup,
}