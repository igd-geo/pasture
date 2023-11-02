use pasture_core::layout::PointType;
use pasture_core::nalgebra::Vector3;
use pasture_derive::PointType;

/// This is a custom `PointType` that illustrates usage of the `#[derive(PointType)]` attribute.
/// Any that that wants to implement `PointType` must fulfill the following requirements:
///
/// - It must fulfill the requirements of [`bytemuck::AnyBitPattern`](https://docs.rs/bytemuck/latest/bytemuck/trait.AnyBitPattern.html) and [`bytemuck::NoUninit`](https://docs.rs/bytemuck/latest/bytemuck/trait.NoUninit.html), which generally means that the type will be `#[repr(C)]` and potentially `#[repr(packed)]` to prevent any padding bytes. It also means that the type only supports primitive types and does not support types that have invalid bit patterns (such as `bool` or enums)
/// - Consequently, the type must implement both `Clone` and `Copy`
/// - All its members may only be [Pasture primitive types](pasture_core::layout::PointAttributeDataType)
/// - Each member must contain an attribute `#[pasture(X)]`, where `X` is either one of the builtin attributes explained below, or `attribute = "name"` for a custom attribute named `name`
/// - No two members may share the same attribute name
///
/// # Builtin attributes
///
/// To associate a member of a custom `PointType` with one of the builtin point attributes in Pasture, use the `#[pasture(X)]` attribute, where `X` is one of:
///
/// - `BUILTIN_POSITION_3D` corresponding to the [POSITION_3D](pasture_core::layout::attributes::POSITION_3D) attribute
/// - `BUILTIN_INTENSITY` corresponding to the [INTENSITY](pasture_core::layout::attributes::INTENSITY) attribute
/// - `BUILTIN_RETURN_NUMBER` corresponding to the [RETURN_NUMBER](pasture_core::layout::attributes::RETURN_NUMBER) attribute
/// - `BUILTIN_NUMBER_OF_RETURNS` corresponding to the [NUMBER_OF_RETURNS](pasture_core::layout::attributes::NUMBER_OF_RETURNS) attribute
/// - `BUILTIN_CLASSIFICATION_FLAGS` corresponding to the [CLASSIFICATION_FLAGS](pasture_core::layout::attributes::CLASSIFICATION_FLAGS) attribute
/// - `BUILTIN_SCANNER_CHANNEL` corresponding to the [SCANNER_CHANNEL](pasture_core::layout::attributes::SCANNER_CHANNEL) attribute
/// - `BUILTIN_SCAN_DIRECTION_FLAG` corresponding to the [SCAN_DIRECTION_FLAG](pasture_core::layout::attributes::SCAN_DIRECTION_FLAG) attribute
/// - `BUILTIN_EDGE_OF_FLIGHT_LINE` corresponding to the [EDGE_OF_FLIGHT_LINE](pasture_core::layout::attributes::EDGE_OF_FLIGHT_LINE) attribute
/// - `BUILTIN_CLASSIFICATION` corresponding to the [CLASSIFICATION](pasture_core::layout::attributes::CLASSIFICATION) attribute
/// - `BUILTIN_SCAN_ANGLE_RANK` corresponding to the [SCAN_ANGLE_RANK](pasture_core::layout::attributes::SCAN_ANGLE_RANK) attribute
/// - `BUILTIN_SCAN_ANGLE` corresponding to the [SCAN_ANGLE](pasture_core::layout::attributes::SCAN_ANGLE) attribute
/// - `BUILTIN_USER_DATA` corresponding to the [USER_DATA](pasture_core::layout::attributes::USER_DATA) attribute
/// - `BUILTIN_POINT_SOURCE_ID` corresponding to the [POINT_SOURCE_ID](pasture_core::layout::attributes::POINT_SOURCE_ID) attribute
/// - `BUILTIN_COLOR_RGB` corresponding to the [COLOR_RGB](pasture_core::layout::attributes::COLOR_RGB) attribute
/// - `BUILTIN_GPS_TIME` corresponding to the [GPS_TIME](pasture_core::layout::attributes::GPS_TIME) attribute
/// - `BUILTIN_NIR` corresponding to the [NIR](pasture_core::layout::attributes::NIR) attribute
/// - `BUILTIN_WAVE_PACKET_DESCRIPTOR_INDEX` corresponding to the [WAVE_PACKET_DESCRIPTOR_INDEX](pasture_core::layout::attributes::WAVE_PACKET_DESCRIPTOR_INDEX) attribute
/// - `BUILTIN_WAVEFORM_DATA_OFFSET` corresponding to the [WAVEFORM_DATA_OFFSET](pasture_core::layout::attributes::WAVEFORM_DATA_OFFSET) attribute
/// - `BUILTIN_WAVEFORM_PACKET_SIZE` corresponding to the [WAVEFORM_PACKET_SIZE](pasture_core::layout::attributes::WAVEFORM_PACKET_SIZE) attribute
/// - `BUILTIN_RETURN_POINT_WAVEFORM_LOCATION` corresponding to the [RETURN_POINT_WAVEFORM_LOCATION](pasture_core::layout::attributes::RETURN_POINT_WAVEFORM_LOCATION) attribute
/// - `BUILTIN_WAVEFORM_PARAMETERS` corresponding to the [WAVEFORM_PARAMETERS](pasture_core::layout::attributes::WAVEFORM_PARAMETERS) attribute
///
/// # Custom attributes
///
/// To associate a member of a custom `PointType` with a point attribute with custom `name`, use the `#[pasture(attribute = "name")]` attribute
#[derive(PointType, Clone, Copy, bytemuck::NoUninit, bytemuck::AnyBitPattern)]
#[repr(C, packed)]
struct CustomPointType {
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: u16,
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f32>,
    #[pasture(attribute = "CUSTOM_ATTRIBUTE")]
    pub custom_attribute: f32,
}

fn main() {
    let layout = CustomPointType::layout();
    println!("Derived layout: {}", layout);
}
