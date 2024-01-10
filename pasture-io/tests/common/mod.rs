use bitfield::{bitfield_bitrange, bitfield_fields};
use pasture_core::{
    containers::{BorrowedBuffer, BorrowedBufferExt},
    layout::{PointAttributeDataType, PointAttributeDefinition, PrimitiveType},
    nalgebra::{Vector3, Vector4},
};
use pasture_io::las::{
    LasPointFormat0, LasPointFormat1, LasPointFormat10, LasPointFormat2, LasPointFormat3,
    LasPointFormat4, LasPointFormat5, LasPointFormat6, LasPointFormat7, LasPointFormat8,
    LasPointFormat9,
};
use rand::{prelude::Distribution, Rng};
use static_assertions::const_assert_eq;

const RETURN_NUMBER_REGULAR_BITMASK: u8 = 0b111;
const RETURN_NUMBER_EXTENDED_BITMASK: u8 = 0b1111;
const NUMBER_OF_RETURNS_REGULAR_BITMASK: u8 = 0b111;
const NUMBER_OF_RETURNS_EXTENDED_BITMASK: u8 = 0b1111;
const CLASSIFICATION_FLAGS_BITMASK: u8 = 0b1111;
const SCANNER_CHANNEL_BITMASK: u8 = 0b11;

#[derive(Debug, Copy, Clone, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
#[repr(C)]
pub struct BasicFlags(u8);
bitfield_bitrange! {struct BasicFlags(u8)}

impl BasicFlags {
    bitfield_fields! {
        u8;
        pub return_number, set_return_number: 2, 0;
        pub number_of_returns, set_number_of_returns: 5, 3;
        pub scan_direction_flag, set_scan_direction_flag: 6;
        pub edge_of_flight_line, set_edge_of_flight_line: 7;
    }
}

#[derive(Debug, Copy, Clone, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
#[repr(C)]
pub struct ExtendedFlags(u16);
bitfield_bitrange! {struct ExtendedFlags(u16)}

impl ExtendedFlags {
    bitfield_fields! {
        u16;
        pub return_number, set_return_number: 3, 0;
        pub number_of_returns, set_number_of_returns: 7, 4;
        pub classification_flags, set_classification_flags: 11, 8;
        pub scanner_channel, set_scanner_channel: 13, 12;
        pub scan_direction_flag, set_scan_direction_flag: 14;
        pub edge_of_flight_line, set_edge_of_flight_line: 15;
    }
}

#[derive(Debug, Copy, Clone, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
#[repr(C, packed)]
pub struct RawLASPointFormat5 {
    x: i32,
    y: i32,
    z: i32,
    intensity: u16,
    flags: BasicFlags,
    classification: u8,
    scan_angle_rank: i8,
    user_data: u8,
    point_source_id: u16,
    gps_time: f64,
    red: u16,
    green: u16,
    blue: u16,
    wave_packet_desriptor_index: u8,
    byte_offset_to_waveform_data: u64,
    waveform_packet_size: u32,
    return_point_waveform_location: f32,
    dx: f32,
    dy: f32,
    dz: f32,
}
const_assert_eq!(63, std::mem::size_of::<RawLASPointFormat5>());

#[derive(Debug, Copy, Clone, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
#[repr(C, packed)]
pub struct RawLASPointFormat10 {
    x: i32,
    y: i32,
    z: i32,
    intensity: u16,
    flags: ExtendedFlags,
    classification: u8,
    user_data: u8,
    scan_angle: i16,
    point_source_id: u16,
    gps_time: f64,
    red: u16,
    green: u16,
    blue: u16,
    nir: u16,
    wave_packet_desriptor_index: u8,
    byte_offset_to_waveform_data: u64,
    waveform_packet_size: u32,
    return_point_waveform_location: f32,
    dx: f32,
    dy: f32,
    dz: f32,
}
const_assert_eq!(67, std::mem::size_of::<RawLASPointFormat10>());

/// Distribution for sampling random LAS points
pub struct TestLASPointDistribution;

impl Distribution<LasPointFormat0> for TestLASPointDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> LasPointFormat0 {
        LasPointFormat0 {
            classification: rng.gen(),
            edge_of_flight_line: rng.gen::<u8>() & 1,
            intensity: rng.gen(),
            number_of_returns: rng.gen::<u8>() & NUMBER_OF_RETURNS_REGULAR_BITMASK,
            point_source_id: rng.gen(),
            position: Vector3::new(
                // Generate positions in a range that LAS can represent with default scale of 0.001
                // Also generate the positions only as integer coordinates, so that we can be sure that
                // there will be no precision loss due to i32<->f64 conversion while reading/writing
                rng.gen_range(-1000..1000) as f64,
                rng.gen_range(-1000..1000) as f64,
                rng.gen_range(-1000..1000) as f64,
            ),
            return_number: rng.gen::<u8>() & RETURN_NUMBER_REGULAR_BITMASK,
            scan_angle_rank: rng.gen(),
            scan_direction_flag: rng.gen::<u8>() & 1,
            user_data: rng.gen(),
        }
    }
}

impl Distribution<LasPointFormat1> for TestLASPointDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> LasPointFormat1 {
        LasPointFormat1 {
            classification: rng.gen(),
            edge_of_flight_line: rng.gen::<u8>() & 1,
            intensity: rng.gen(),
            number_of_returns: rng.gen::<u8>() & NUMBER_OF_RETURNS_REGULAR_BITMASK,
            point_source_id: rng.gen(),
            position: Vector3::new(
                // Generate positions in a range that LAS can represent with default scale of 0.001
                // Also generate the positions only as integer coordinates, so that we can be sure that
                // there will be no precision loss due to i32<->f64 conversion while reading/writing
                rng.gen_range(-1000..1000) as f64,
                rng.gen_range(-1000..1000) as f64,
                rng.gen_range(-1000..1000) as f64,
            ),
            return_number: rng.gen::<u8>() & RETURN_NUMBER_REGULAR_BITMASK,
            scan_angle_rank: rng.gen(),
            scan_direction_flag: rng.gen::<u8>() & 1,
            user_data: rng.gen(),
            gps_time: rng.gen(),
        }
    }
}

impl Distribution<LasPointFormat2> for TestLASPointDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> LasPointFormat2 {
        LasPointFormat2 {
            classification: rng.gen(),
            edge_of_flight_line: rng.gen::<u8>() & 1,
            intensity: rng.gen(),
            number_of_returns: rng.gen::<u8>() & NUMBER_OF_RETURNS_REGULAR_BITMASK,
            point_source_id: rng.gen(),
            position: Vector3::new(
                // Generate positions in a range that LAS can represent with default scale of 0.001
                // Also generate the positions only as integer coordinates, so that we can be sure that
                // there will be no precision loss due to i32<->f64 conversion while reading/writing
                rng.gen_range(-1000..1000) as f64,
                rng.gen_range(-1000..1000) as f64,
                rng.gen_range(-1000..1000) as f64,
            ),
            return_number: rng.gen::<u8>() & RETURN_NUMBER_REGULAR_BITMASK,
            scan_angle_rank: rng.gen(),
            scan_direction_flag: rng.gen::<u8>() & 1,
            user_data: rng.gen(),
            color_rgb: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
        }
    }
}

impl Distribution<LasPointFormat3> for TestLASPointDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> LasPointFormat3 {
        LasPointFormat3 {
            classification: rng.gen(),
            edge_of_flight_line: rng.gen::<u8>() & 1,
            intensity: rng.gen(),
            number_of_returns: rng.gen::<u8>() & NUMBER_OF_RETURNS_REGULAR_BITMASK,
            point_source_id: rng.gen(),
            position: Vector3::new(
                // Generate positions in a range that LAS can represent with default scale of 0.001
                // Also generate the positions only as integer coordinates, so that we can be sure that
                // there will be no precision loss due to i32<->f64 conversion while reading/writing
                rng.gen_range(-1000..1000) as f64,
                rng.gen_range(-1000..1000) as f64,
                rng.gen_range(-1000..1000) as f64,
            ),
            return_number: rng.gen::<u8>() & RETURN_NUMBER_REGULAR_BITMASK,
            scan_angle_rank: rng.gen(),
            scan_direction_flag: rng.gen::<u8>() & 1,
            user_data: rng.gen(),
            gps_time: rng.gen(),
            color_rgb: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
        }
    }
}

impl Distribution<LasPointFormat4> for TestLASPointDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> LasPointFormat4 {
        LasPointFormat4 {
            classification: rng.gen(),
            edge_of_flight_line: rng.gen::<u8>() & 1,
            intensity: rng.gen(),
            number_of_returns: rng.gen::<u8>() & NUMBER_OF_RETURNS_REGULAR_BITMASK,
            point_source_id: rng.gen(),
            position: Vector3::new(
                // Generate positions in a range that LAS can represent with default scale of 0.001
                // Also generate the positions only as integer coordinates, so that we can be sure that
                // there will be no precision loss due to i32<->f64 conversion while reading/writing
                rng.gen_range(-1000..1000) as f64,
                rng.gen_range(-1000..1000) as f64,
                rng.gen_range(-1000..1000) as f64,
            ),
            return_number: rng.gen::<u8>() & RETURN_NUMBER_REGULAR_BITMASK,
            scan_angle_rank: rng.gen(),
            scan_direction_flag: rng.gen::<u8>() & 1,
            user_data: rng.gen(),
            byte_offset_to_waveform_data: rng.gen::<u32>() as u64,
            gps_time: rng.gen(),
            return_point_waveform_location: rng.gen(),
            wave_packet_descriptor_index: rng.gen(),
            waveform_packet_size: rng.gen(),
            waveform_parameters: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
        }
    }
}

impl Distribution<LasPointFormat5> for TestLASPointDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> LasPointFormat5 {
        LasPointFormat5 {
            classification: rng.gen(),
            edge_of_flight_line: rng.gen::<u8>() & 1,
            intensity: rng.gen(),
            number_of_returns: rng.gen::<u8>() & NUMBER_OF_RETURNS_REGULAR_BITMASK,
            point_source_id: rng.gen(),
            position: Vector3::new(
                // Generate positions in a range that LAS can represent with default scale of 0.001
                // Also generate the positions only as integer coordinates, so that we can be sure that
                // there will be no precision loss due to i32<->f64 conversion while reading/writing
                rng.gen_range(-1000..1000) as f64,
                rng.gen_range(-1000..1000) as f64,
                rng.gen_range(-1000..1000) as f64,
            ),
            return_number: rng.gen::<u8>() & RETURN_NUMBER_REGULAR_BITMASK,
            scan_angle_rank: rng.gen(),
            scan_direction_flag: rng.gen::<u8>() & 1,
            user_data: rng.gen(),
            byte_offset_to_waveform_data: rng.gen::<u32>() as u64,
            gps_time: rng.gen(),
            return_point_waveform_location: rng.gen(),
            wave_packet_descriptor_index: rng.gen(),
            waveform_packet_size: rng.gen(),
            waveform_parameters: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
            color_rgb: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
        }
    }
}

impl Distribution<LasPointFormat6> for TestLASPointDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> LasPointFormat6 {
        LasPointFormat6 {
            classification: rng.gen(),
            edge_of_flight_line: rng.gen::<u8>() & 1,
            intensity: rng.gen(),
            number_of_returns: rng.gen::<u8>() & NUMBER_OF_RETURNS_EXTENDED_BITMASK,
            point_source_id: rng.gen(),
            position: Vector3::new(
                // Generate positions in a range that LAS can represent with default scale of 0.001
                // Also generate the positions only as integer coordinates, so that we can be sure that
                // there will be no precision loss due to i32<->f64 conversion while reading/writing
                rng.gen_range(-1000..1000) as f64,
                rng.gen_range(-1000..1000) as f64,
                rng.gen_range(-1000..1000) as f64,
            ),
            return_number: rng.gen::<u8>() & RETURN_NUMBER_EXTENDED_BITMASK,
            scan_angle: rng.gen(),
            scan_direction_flag: rng.gen::<u8>() & 1,
            user_data: rng.gen(),
            gps_time: rng.gen(),
            classification_flags: rng.gen::<u8>() & CLASSIFICATION_FLAGS_BITMASK,
            scanner_channel: rng.gen::<u8>() & SCANNER_CHANNEL_BITMASK,
        }
    }
}

impl Distribution<LasPointFormat7> for TestLASPointDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> LasPointFormat7 {
        LasPointFormat7 {
            classification: rng.gen(),
            edge_of_flight_line: rng.gen::<u8>() & 1,
            intensity: rng.gen(),
            number_of_returns: rng.gen::<u8>() & NUMBER_OF_RETURNS_EXTENDED_BITMASK,
            point_source_id: rng.gen(),
            position: Vector3::new(
                // Generate positions in a range that LAS can represent with default scale of 0.001
                // Also generate the positions only as integer coordinates, so that we can be sure that
                // there will be no precision loss due to i32<->f64 conversion while reading/writing
                rng.gen_range(-1000..1000) as f64,
                rng.gen_range(-1000..1000) as f64,
                rng.gen_range(-1000..1000) as f64,
            ),
            return_number: rng.gen::<u8>() & RETURN_NUMBER_EXTENDED_BITMASK,
            scan_angle: rng.gen(),
            scan_direction_flag: rng.gen::<u8>() & 1,
            user_data: rng.gen(),
            gps_time: rng.gen(),
            classification_flags: rng.gen::<u8>() & CLASSIFICATION_FLAGS_BITMASK,
            scanner_channel: rng.gen::<u8>() & SCANNER_CHANNEL_BITMASK,
            color_rgb: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
        }
    }
}

impl Distribution<LasPointFormat8> for TestLASPointDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> LasPointFormat8 {
        LasPointFormat8 {
            classification: rng.gen(),
            edge_of_flight_line: rng.gen::<u8>() & 1,
            intensity: rng.gen(),
            number_of_returns: rng.gen::<u8>() & NUMBER_OF_RETURNS_EXTENDED_BITMASK,
            point_source_id: rng.gen(),
            position: Vector3::new(
                // Generate positions in a range that LAS can represent with default scale of 0.001
                // Also generate the positions only as integer coordinates, so that we can be sure that
                // there will be no precision loss due to i32<->f64 conversion while reading/writing
                rng.gen_range(-1000..1000) as f64,
                rng.gen_range(-1000..1000) as f64,
                rng.gen_range(-1000..1000) as f64,
            ),
            return_number: rng.gen::<u8>() & RETURN_NUMBER_EXTENDED_BITMASK,
            scan_angle: rng.gen(),
            scan_direction_flag: rng.gen::<u8>() & 1,
            user_data: rng.gen(),
            gps_time: rng.gen(),
            classification_flags: rng.gen::<u8>() & CLASSIFICATION_FLAGS_BITMASK,
            scanner_channel: rng.gen::<u8>() & SCANNER_CHANNEL_BITMASK,
            color_rgb: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
            nir: rng.gen(),
        }
    }
}

impl Distribution<LasPointFormat9> for TestLASPointDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> LasPointFormat9 {
        LasPointFormat9 {
            classification: rng.gen(),
            edge_of_flight_line: rng.gen::<u8>() & 1,
            intensity: rng.gen(),
            number_of_returns: rng.gen::<u8>() & NUMBER_OF_RETURNS_EXTENDED_BITMASK,
            point_source_id: rng.gen(),
            position: Vector3::new(
                // Generate positions in a range that LAS can represent with default scale of 0.001
                // Also generate the positions only as integer coordinates, so that we can be sure that
                // there will be no precision loss due to i32<->f64 conversion while reading/writing
                rng.gen_range(-1000..1000) as f64,
                rng.gen_range(-1000..1000) as f64,
                rng.gen_range(-1000..1000) as f64,
            ),
            return_number: rng.gen::<u8>() & RETURN_NUMBER_EXTENDED_BITMASK,
            scan_angle: rng.gen(),
            scan_direction_flag: rng.gen::<u8>() & 1,
            user_data: rng.gen(),
            gps_time: rng.gen(),
            classification_flags: rng.gen::<u8>() & CLASSIFICATION_FLAGS_BITMASK,
            scanner_channel: rng.gen::<u8>() & SCANNER_CHANNEL_BITMASK,
            byte_offset_to_waveform_data: 0,
            return_point_waveform_location: rng.gen(),
            wave_packet_descriptor_index: 0,
            waveform_packet_size: 0,
            waveform_parameters: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
        }
    }
}

impl Distribution<LasPointFormat10> for TestLASPointDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> LasPointFormat10 {
        LasPointFormat10 {
            classification: rng.gen(),
            edge_of_flight_line: rng.gen::<u8>() & 1,
            intensity: rng.gen(),
            number_of_returns: rng.gen::<u8>() & NUMBER_OF_RETURNS_EXTENDED_BITMASK,
            point_source_id: rng.gen(),
            position: Vector3::new(
                // Generate positions in a range that LAS can represent with default scale of 0.001
                // Also generate the positions only as integer coordinates, so that we can be sure that
                // there will be no precision loss due to i32<->f64 conversion while reading/writing
                rng.gen_range(-1000..1000) as f64,
                rng.gen_range(-1000..1000) as f64,
                rng.gen_range(-1000..1000) as f64,
            ),
            return_number: rng.gen::<u8>() & RETURN_NUMBER_EXTENDED_BITMASK,
            scan_angle: rng.gen(),
            scan_direction_flag: rng.gen::<u8>() & 1,
            user_data: rng.gen(),
            gps_time: rng.gen(),
            classification_flags: rng.gen::<u8>() & CLASSIFICATION_FLAGS_BITMASK,
            scanner_channel: rng.gen::<u8>() & SCANNER_CHANNEL_BITMASK,
            byte_offset_to_waveform_data: rng.gen::<u32>() as u64,
            return_point_waveform_location: rng.gen(),
            wave_packet_descriptor_index: rng.gen(),
            waveform_packet_size: rng.gen(),
            waveform_parameters: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
            nir: rng.gen(),
            color_rgb: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
        }
    }
}

fn compare_attribute<
    T: PrimitiveType + PartialEq + std::fmt::Debug,
    B1: for<'a> BorrowedBuffer<'a> + std::fmt::Debug,
    B2: for<'a> BorrowedBuffer<'a> + std::fmt::Debug,
>(
    attribute: &PointAttributeDefinition,
    buffer1: &B1,
    buffer2: &B2,
) {
    let attributes1 = buffer1
        .view_attribute_with_conversion::<T>(attribute)
        .expect("Invalid attribute type");
    let attributes2 = buffer2
        .view_attribute_with_conversion::<T>(attribute)
        .expect("Invalid attribute type");
    assert_eq!(buffer1.len(), buffer2.len());
    for (idx, (a1, a2)) in attributes1
        .into_iter()
        .zip(attributes2.into_iter())
        .enumerate()
    {
        assert_eq!(
            a1,
            a2,
            "Attribute {} at index {idx} is different. Expected {a1:#?} but got {a2:#?}",
            attribute.name(),
        );
    }
}

pub fn compare_attributes_dynamically_typed<
    B1: for<'a> BorrowedBuffer<'a> + std::fmt::Debug,
    B2: for<'a> BorrowedBuffer<'a> + std::fmt::Debug,
>(
    attribute: &PointAttributeDefinition,
    buffer1: &B1,
    buffer2: &B2,
) {
    match attribute.datatype() {
        PointAttributeDataType::U8 => compare_attribute::<u8, _, _>(attribute, buffer1, buffer2),
        PointAttributeDataType::I8 => compare_attribute::<i8, _, _>(attribute, buffer1, buffer2),
        PointAttributeDataType::U16 => compare_attribute::<u16, _, _>(attribute, buffer1, buffer2),
        PointAttributeDataType::I16 => compare_attribute::<i16, _, _>(attribute, buffer1, buffer2),
        PointAttributeDataType::U32 => compare_attribute::<u32, _, _>(attribute, buffer1, buffer2),
        PointAttributeDataType::I32 => compare_attribute::<i32, _, _>(attribute, buffer1, buffer2),
        PointAttributeDataType::U64 => compare_attribute::<u64, _, _>(attribute, buffer1, buffer2),
        PointAttributeDataType::I64 => compare_attribute::<i64, _, _>(attribute, buffer1, buffer2),
        PointAttributeDataType::F32 => compare_attribute::<f32, _, _>(attribute, buffer1, buffer2),
        PointAttributeDataType::F64 => compare_attribute::<f64, _, _>(attribute, buffer1, buffer2),
        PointAttributeDataType::Vec3u8 => {
            compare_attribute::<Vector3<u8>, _, _>(attribute, buffer1, buffer2)
        }
        PointAttributeDataType::Vec3u16 => {
            compare_attribute::<Vector3<u16>, _, _>(attribute, buffer1, buffer2)
        }
        PointAttributeDataType::Vec3f32 => {
            compare_attribute::<Vector3<f32>, _, _>(attribute, buffer1, buffer2)
        }
        PointAttributeDataType::Vec3i32 => {
            compare_attribute::<Vector3<i32>, _, _>(attribute, buffer1, buffer2)
        }
        PointAttributeDataType::Vec3f64 => {
            compare_attribute::<Vector3<f64>, _, _>(attribute, buffer1, buffer2)
        }
        PointAttributeDataType::Vec4u8 => {
            compare_attribute::<Vector4<u8>, _, _>(attribute, buffer1, buffer2)
        }
        other => panic!("Unsupported PointAttributeDataType {}", other),
    }
}
