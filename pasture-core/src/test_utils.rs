use nalgebra::Vector3;
use pasture_derive::PointType;
use rand::prelude::Distribution;

use crate::{
    containers::{BorrowedBuffer, BorrowedBufferExt},
    layout::{PointAttributeDataType, PointAttributeDefinition, PrimitiveType},
};

#[derive(
    PointType, Default, Copy, Clone, PartialEq, Debug, bytemuck::AnyBitPattern, bytemuck::NoUninit,
)]
#[repr(C, packed)]
pub(crate) struct CustomPointTypeSmall {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_CLASSIFICATION)]
    pub classification: u8,
}

#[derive(
    PointType, Default, Copy, Clone, PartialEq, Debug, bytemuck::AnyBitPattern, bytemuck::NoUninit,
)]
#[repr(C, packed)]
pub(crate) struct CustomPointTypeBig {
    #[pasture(BUILTIN_GPS_TIME)]
    pub gps_time: f64,
    #[pasture(BUILTIN_COLOR_RGB)]
    pub color: Vector3<u16>,
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_CLASSIFICATION)]
    pub classification: u8,
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: i16,
}

pub(crate) struct DefaultPointDistribution;

impl Distribution<CustomPointTypeSmall> for DefaultPointDistribution {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> CustomPointTypeSmall {
        CustomPointTypeSmall {
            classification: rng.gen(),
            position: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
        }
    }
}

impl Distribution<CustomPointTypeBig> for DefaultPointDistribution {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> CustomPointTypeBig {
        CustomPointTypeBig {
            classification: rng.gen(),
            position: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
            color: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
            gps_time: rng.gen(),
            intensity: rng.gen(),
        }
    }
}

pub(crate) fn compare_attributes_typed<'a, U: PrimitiveType + std::fmt::Debug + PartialEq>(
    buffer: &'a impl BorrowedBuffer<'a>,
    attribute: &PointAttributeDefinition,
    expected_points: &'a impl BorrowedBuffer<'a>,
) {
    let collected_values = buffer
        .view_attribute::<U>(attribute)
        .into_iter()
        .collect::<Vec<_>>();
    let expected_values = expected_points
        .view_attribute::<U>(attribute)
        .into_iter()
        .collect::<Vec<_>>();
    assert_eq!(expected_values, collected_values);
}

/// Compare the given point attribute using the static type corresponding to the attribute's `PointAttributeDataType`
pub(crate) fn compare_attributes<'a>(
    buffer: &'a impl BorrowedBuffer<'a>,
    attribute: &PointAttributeDefinition,
    expected_points: &'a impl BorrowedBuffer<'a>,
) {
    match attribute.datatype() {
        PointAttributeDataType::F32 => {
            compare_attributes_typed::<f32>(buffer, attribute, expected_points)
        }
        PointAttributeDataType::F64 => {
            compare_attributes_typed::<f64>(buffer, attribute, expected_points)
        }
        PointAttributeDataType::I16 => {
            compare_attributes_typed::<i16>(buffer, attribute, expected_points)
        }
        PointAttributeDataType::I32 => {
            compare_attributes_typed::<i32>(buffer, attribute, expected_points)
        }
        PointAttributeDataType::I64 => {
            compare_attributes_typed::<i64>(buffer, attribute, expected_points)
        }
        PointAttributeDataType::I8 => {
            compare_attributes_typed::<i8>(buffer, attribute, expected_points)
        }
        PointAttributeDataType::U16 => {
            compare_attributes_typed::<u16>(buffer, attribute, expected_points)
        }
        PointAttributeDataType::U32 => {
            compare_attributes_typed::<u32>(buffer, attribute, expected_points)
        }
        PointAttributeDataType::U64 => {
            compare_attributes_typed::<u64>(buffer, attribute, expected_points)
        }
        PointAttributeDataType::U8 => {
            compare_attributes_typed::<u8>(buffer, attribute, expected_points)
        }
        PointAttributeDataType::Vec3f32 => {
            compare_attributes_typed::<Vector3<f32>>(buffer, attribute, expected_points)
        }
        PointAttributeDataType::Vec3f64 => {
            compare_attributes_typed::<Vector3<f64>>(buffer, attribute, expected_points);
        }
        PointAttributeDataType::Vec3i32 => {
            compare_attributes_typed::<Vector3<i32>>(buffer, attribute, expected_points)
        }
        PointAttributeDataType::Vec3u16 => {
            compare_attributes_typed::<Vector3<u16>>(buffer, attribute, expected_points)
        }
        PointAttributeDataType::Vec3u8 => {
            compare_attributes_typed::<Vector3<u8>>(buffer, attribute, expected_points)
        }
        _ => unimplemented!(),
    }
}
