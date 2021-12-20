use pasture_core::{
    containers::{
        InterleavedPointBufferMut, InterleavedVecPointStorage, PointBuffer, PointBufferExt,
    },
    gpu,
    layout::{attributes, PointLayout},
    nalgebra::Vector3,
};
use pasture_derive::PointType;
use wgpu;

#[repr(C)]
#[derive(PointType, Debug)]
struct MyPointType {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_COLOR_RGB)]
    pub icolor: Vector3<u16>,
    #[pasture(attribute = "MyColorF32")]
    pub fcolor: Vector3<f32>,
    #[pasture(attribute = "MyVec3U8")]
    pub byte_vec: Vector3<u8>,
    #[pasture(BUILTIN_CLASSIFICATION)]
    pub classification: u8,
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: u16,
    #[pasture(BUILTIN_SCAN_ANGLE)]
    pub scan_angle: i16,
    #[pasture(BUILTIN_SCAN_DIRECTION_FLAG)]
    pub scan_dir_flag: bool,
    #[pasture(attribute = "MyInt32")]
    pub my_int: i32,
    #[pasture(BUILTIN_WAVEFORM_PACKET_SIZE)]
    pub packet_size: u32,
    #[pasture(BUILTIN_RETURN_POINT_WAVEFORM_LOCATION)]
    pub ret_point_loc: f32,
    #[pasture(BUILTIN_GPS_TIME)]
    pub gps_time: f64,
}

struct Boundary {
    nw_front: f32,
    nw_back: f32,
    sw_front: f32,
    sw_back: f32,
    ne_front: f32,
    ne_back: f32,
    se_front: f32,
    se_back: f32,
}

trait OctreeNode {}

struct OctreeRegion {
    boundary: Boundary,
    points: [Vector3<f64>; 8],
}

struct OctreeLeaf {
    point: Vector3<f64>,
}

pub struct GpuOctree<'a> {
    device: gpu::Device<'a>,
    buffer: Option<wgpu::Buffer>,
    buffer_size: Option<wgpu::BufferAddress>,
    buffer_binding: Option<u32>,
    point_buffer: &'a dyn PointBuffer,
}

impl GpuOctree<'_> {
    pub async fn new<'a>(point_buffer: &'a dyn PointBuffer) -> GpuOctree<'a> {
        let device = gpu::Device::new(gpu::DeviceOptions {
            device_power: gpu::DevicePower::High,
            device_backend: gpu::DeviceBackend::Vulkan,
            use_adapter_features: true,
            use_adapter_limits: true,
        })
        .await;

        GpuOctree {
            device: device.unwrap(),
            buffer: None,
            buffer_size: None,
            buffer_binding: None,
            point_buffer: point_buffer,
        }
    }
    pub fn construct(&mut self, layout: PointLayout) {
        let point_count = self.point_buffer.len();
        let points_in_byte: usize = point_count * 8 * 3;
        let mut raw_points = vec![0 as u8; points_in_byte];
        self.point_buffer.get_raw_attribute_range(
            0..point_count,
            &attributes::POSITION_3D,
            raw_points.as_mut_slice(),
        );

    }
}
