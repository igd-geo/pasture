
pub struct Device {
    // Private fields
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Device {
    // Constructors

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

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default()
            },
            None,
        ).await.unwrap();

        Device {
            adapter,
            device,
            queue
        }
    }

    // Methods
    pub fn print_device_info(&self) {
        let info = self.adapter.get_info();

        println!("== Device Information ========");
        println!("Name: {}", info.name);
        println!("Type: {:?}", info.device_type);
        println!("Backend: {:?}", info.backend);
        println!("PCI id: {}", info.device);
        println!("Vendor PCI id: {}\n", info.vendor);
    }
}

// == Helper types ===============================================================================

pub struct DeviceOptions {
    pub device_power: DevicePower,
    pub device_backend: DeviceBackend,
}

impl Default for DeviceOptions {
    fn default() -> Self {
        Self {
            device_power: DevicePower::Low,
            device_backend: DeviceBackend::Primary,
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