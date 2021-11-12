//! Provides data structures to interact with the GPU. Currently only compute (GPGPU) capabilities
//! are supported.
//!
//! Pasture uses [wgpu](https://crates.io/crates/wgpu), a Rust implementation of the
//! [WebGPU](https://www.w3.org/TR/webgpu/) specification, that also has native backend support
//! for APIs such as `Vulkan`, `DirectX12`, and `Metal`.
//!
//! At the core of everything lies [Device](device::Device), which is responsible for obtaining a handle to the GPU
//! and submitting work to it. It also exposes some `wgpu` structures that could be used by the user
//! for more fine grained control.
//!
//! [GpuPointBufferInterleaved](gpu_point_buffer::GpuPointBufferInterleaved) and
//! [GpuPointBufferPerAttribute](gpu_point_buffer::GpuPointBufferPerAttribute) can be used to store
//! point cloud data in either format on the GPU and retrieve it. They also take care of aligning
//! the data so that your shaders work with correct values.
//! It is important to note that for storage buffers only the `std430` layout is supported.

mod device;
pub use self::device::*;

mod gpu_point_buffer;
pub use self::gpu_point_buffer::*;
