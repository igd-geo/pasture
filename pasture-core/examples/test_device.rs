use pasture_core::gpu;
use pasture_core::gpu::DevicePower;

fn main() {
    futures::executor::block_on(run());
}

async fn run() {
    let device = gpu::Device::default().await;
    device.print_device_info();

    let device = gpu::Device::new(
        gpu::DeviceOptions {
            device_power: gpu::DevicePower::High,
            device_backend: gpu::DeviceBackend::Dx12,
        }
    ).await;
    device.print_device_info();
}