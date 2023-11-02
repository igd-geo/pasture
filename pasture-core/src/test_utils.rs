use nalgebra::Vector3;
use pasture_derive::PointType;
use rand::prelude::Distribution;

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
