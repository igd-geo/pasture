//! Contains types for each of the LAS point formats

use las::{point::ScanDirection, Point};
use pasture_core::{
    nalgebra::Vector3,
};
use pasture_derive::PointType;
use static_assertions::const_assert_eq;
use std::convert::From;

/// Point type for LAS point format 0
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, PartialEq, Default, PointType)]
pub struct LasPointFormat0 {
    #[pasture(BUILTIN_POSITION_3D)] pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)] pub intensity: u16,
    #[pasture(BUILTIN_RETURN_NUMBER)] pub return_number: u8,
    #[pasture(BUILTIN_NUMBER_OF_RETURNS)] pub number_of_returns: u8,
    #[pasture(BUILTIN_SCAN_DIRECTION_FLAG)] pub scan_direction_flag: bool,
    #[pasture(BUILTIN_EDGE_OF_FLIGHT_LINE)] pub edge_of_flight_line: bool,
    #[pasture(BUILTIN_CLASSIFICATION)] pub classification: u8,
    #[pasture(BUILTIN_SCAN_ANGLE_RANK)] pub scan_angle_rank: i8,
    #[pasture(BUILTIN_USER_DATA)] pub user_data: u8,
    #[pasture(BUILTIN_POINT_SOURCE_ID)] pub point_source_id: u16,
}

const_assert_eq!(std::mem::size_of::<LasPointFormat0>(), 35);

impl From<Point> for LasPointFormat0 {
    fn from(las_point: Point) -> Self {
        Self {
            position: Vector3::new(las_point.x, las_point.y, las_point.z),
            intensity: las_point.intensity,
            return_number: las_point.return_number,
            number_of_returns: las_point.number_of_returns,
            scan_direction_flag: match las_point.scan_direction {
                ScanDirection::RightToLeft => false,
                ScanDirection::LeftToRight => true,
            },
            edge_of_flight_line: las_point.is_edge_of_flight_line,
            classification: las_point.classification.into(),
            scan_angle_rank: las_point.scan_angle as i8,
            user_data: las_point.user_data,
            point_source_id: las_point.point_source_id,
        }
    }
}

/// Point type for LAS point format 1
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, PartialEq, Default, PointType)]
pub struct LasPointFormat1 {
    #[pasture(BUILTIN_POSITION_3D)] pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)] pub intensity: u16,
    #[pasture(BUILTIN_RETURN_NUMBER)] pub return_number: u8,
    #[pasture(BUILTIN_NUMBER_OF_RETURNS)] pub number_of_returns: u8,
    #[pasture(BUILTIN_SCAN_DIRECTION_FLAG)] pub scan_direction_flag: bool,
    #[pasture(BUILTIN_EDGE_OF_FLIGHT_LINE)] pub edge_of_flight_line: bool,
    #[pasture(BUILTIN_CLASSIFICATION)] pub classification: u8,
    #[pasture(BUILTIN_SCAN_ANGLE_RANK)] pub scan_angle_rank: i8,
    #[pasture(BUILTIN_USER_DATA)] pub user_data: u8,
    #[pasture(BUILTIN_POINT_SOURCE_ID)] pub point_source_id: u16,
    #[pasture(BUILTIN_GPS_TIME)] pub gps_time: f64,
}

const_assert_eq!(std::mem::size_of::<LasPointFormat1>(), 43);

impl From<Point> for LasPointFormat1 {
    fn from(las_point: Point) -> Self {
        Self {
            position: Vector3::new(las_point.x, las_point.y, las_point.z),
            intensity: las_point.intensity,
            return_number: las_point.return_number,
            number_of_returns: las_point.number_of_returns,
            scan_direction_flag: match las_point.scan_direction {
                ScanDirection::RightToLeft => false,
                ScanDirection::LeftToRight => true,
            },
            edge_of_flight_line: las_point.is_edge_of_flight_line,
            classification: las_point.classification.into(),
            scan_angle_rank: las_point.scan_angle as i8,
            user_data: las_point.user_data,
            point_source_id: las_point.point_source_id,
            gps_time: las_point.gps_time.unwrap_or(0.0),
        }
    }
}

/// Point type for LAS point format 2
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, PartialEq, Default, PointType)]
pub struct LasPointFormat2 {
    #[pasture(BUILTIN_POSITION_3D)] pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)] pub intensity: u16,
    #[pasture(BUILTIN_RETURN_NUMBER)] pub return_number: u8,
    #[pasture(BUILTIN_NUMBER_OF_RETURNS)] pub number_of_returns: u8,
    #[pasture(BUILTIN_SCAN_DIRECTION_FLAG)] pub scan_direction_flag: bool,
    #[pasture(BUILTIN_EDGE_OF_FLIGHT_LINE)] pub edge_of_flight_line: bool,
    #[pasture(BUILTIN_CLASSIFICATION)] pub classification: u8,
    #[pasture(BUILTIN_SCAN_ANGLE_RANK)] pub scan_angle_rank: i8,
    #[pasture(BUILTIN_USER_DATA)] pub user_data: u8,
    #[pasture(BUILTIN_POINT_SOURCE_ID)] pub point_source_id: u16,
    #[pasture(BUILTIN_COLOR_RGB)] pub color_rgb: Vector3<u16>,
}

const_assert_eq!(std::mem::size_of::<LasPointFormat2>(), 41);

impl From<Point> for LasPointFormat2 {
    fn from(las_point: Point) -> Self {
        let color = las_point
            .color
            .expect("Conversion from las::Point to LASPoint_Format2: color was None");
        Self {
            position: Vector3::new(las_point.x, las_point.y, las_point.z),
            intensity: las_point.intensity,
            return_number: las_point.return_number,
            number_of_returns: las_point.number_of_returns,
            scan_direction_flag: match las_point.scan_direction {
                ScanDirection::RightToLeft => false,
                ScanDirection::LeftToRight => true,
            },
            edge_of_flight_line: las_point.is_edge_of_flight_line,
            classification: las_point.classification.into(),
            scan_angle_rank: las_point.scan_angle as i8,
            user_data: las_point.user_data,
            point_source_id: las_point.point_source_id,
            color_rgb: Vector3::new(color.red, color.green, color.blue),
        }
    }
}

/// Point type for LAS point format 3
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, PartialEq, Default, PointType)]
pub struct LasPointFormat3 {
    #[pasture(BUILTIN_POSITION_3D)] pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)] pub intensity: u16,
    #[pasture(BUILTIN_RETURN_NUMBER)] pub return_number: u8,
    #[pasture(BUILTIN_NUMBER_OF_RETURNS)] pub number_of_returns: u8,
    #[pasture(BUILTIN_SCAN_DIRECTION_FLAG)] pub scan_direction_flag: bool,
    #[pasture(BUILTIN_EDGE_OF_FLIGHT_LINE)] pub edge_of_flight_line: bool,
    #[pasture(BUILTIN_CLASSIFICATION)] pub classification: u8,
    #[pasture(BUILTIN_SCAN_ANGLE_RANK)] pub scan_angle_rank: i8,
    #[pasture(BUILTIN_USER_DATA)] pub user_data: u8,
    #[pasture(BUILTIN_POINT_SOURCE_ID)] pub point_source_id: u16,
    #[pasture(BUILTIN_GPS_TIME)] pub gps_time: f64,
    #[pasture(BUILTIN_COLOR_RGB)] pub color_rgb: Vector3<u16>,
}

const_assert_eq!(std::mem::size_of::<LasPointFormat3>(), 49);

impl From<Point> for LasPointFormat3 {
    fn from(las_point: Point) -> Self {
        let color = las_point
            .color
            .expect("Conversion from las::Point to LASPoint_Format2: color was None");
        Self {
            position: Vector3::new(las_point.x, las_point.y, las_point.z),
            intensity: las_point.intensity,
            return_number: las_point.return_number,
            number_of_returns: las_point.number_of_returns,
            scan_direction_flag: match las_point.scan_direction {
                ScanDirection::RightToLeft => false,
                ScanDirection::LeftToRight => true,
            },
            edge_of_flight_line: las_point.is_edge_of_flight_line,
            classification: las_point.classification.into(),
            scan_angle_rank: las_point.scan_angle as i8,
            user_data: las_point.user_data,
            point_source_id: las_point.point_source_id,
            gps_time: las_point.gps_time.unwrap_or(0.0),
            color_rgb: Vector3::new(color.red, color.green, color.blue),
        }
    }
}

/// Point type for LAS point format 4
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, PartialEq, Default, PointType)]
pub struct LasPointFormat4 {
    #[pasture(BUILTIN_POSITION_3D)] pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)] pub intensity: u16,
    #[pasture(BUILTIN_RETURN_NUMBER)] pub return_number: u8,
    #[pasture(BUILTIN_NUMBER_OF_RETURNS)] pub number_of_returns: u8,
    #[pasture(BUILTIN_SCAN_DIRECTION_FLAG)] pub scan_direction_flag: bool,
    #[pasture(BUILTIN_EDGE_OF_FLIGHT_LINE)] pub edge_of_flight_line: bool,
    #[pasture(BUILTIN_CLASSIFICATION)] pub classification: u8,
    #[pasture(BUILTIN_SCAN_ANGLE_RANK)] pub scan_angle_rank: i8,
    #[pasture(BUILTIN_USER_DATA)] pub user_data: u8,
    #[pasture(BUILTIN_POINT_SOURCE_ID)] pub point_source_id: u16,
    #[pasture(BUILTIN_GPS_TIME)] pub gps_time: f64,
    #[pasture(BUILTIN_WAVE_PACKET_DESCRIPTOR_INDEX)] pub wave_packet_descriptor_index: u8,
    #[pasture(BUILTIN_WAVEFORM_DATA_OFFSET)] pub byte_offset_to_waveform_data: u64,
    #[pasture(BUILTIN_WAVEFORM_PACKET_SIZE)] pub waveform_packet_size: u32,
    #[pasture(BUILTIN_RETURN_POINT_WAVEFORM_LOCATION)] pub return_point_waveform_location: f32,
    #[pasture(BUILTIN_WAVEFORM_PARAMETERS)] pub waveform_parameters: Vector3<f32>,
}

const_assert_eq!(std::mem::size_of::<LasPointFormat4>(), 72);

impl From<Point> for LasPointFormat4 {
    fn from(las_point: Point) -> Self {
        let waveform = las_point.waveform.expect("LasPoint has no waveform data");
        Self {
            position: Vector3::new(las_point.x, las_point.y, las_point.z),
            intensity: las_point.intensity,
            return_number: las_point.return_number,
            number_of_returns: las_point.number_of_returns,
            scan_direction_flag: match las_point.scan_direction {
                ScanDirection::RightToLeft => false,
                ScanDirection::LeftToRight => true,
            },
            edge_of_flight_line: las_point.is_edge_of_flight_line,
            classification: las_point.classification.into(),
            scan_angle_rank: las_point.scan_angle as i8,
            user_data: las_point.user_data,
            point_source_id: las_point.point_source_id,
            gps_time: las_point.gps_time.unwrap_or(0.0),
            wave_packet_descriptor_index: waveform.wave_packet_descriptor_index,
            byte_offset_to_waveform_data: waveform.byte_offset_to_waveform_data,
            waveform_packet_size: waveform.waveform_packet_size_in_bytes,
            return_point_waveform_location: waveform.return_point_waveform_location,
            waveform_parameters: Vector3::new(waveform.x_t, waveform.y_t, waveform.z_t),
        }
    }
}

/// Point type for LAS point format 5
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, PartialEq, Default, PointType)]
pub struct LasPointFormat5 {
    #[pasture(BUILTIN_POSITION_3D)] pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)] pub intensity: u16,
    #[pasture(BUILTIN_RETURN_NUMBER)] pub return_number: u8,
    #[pasture(BUILTIN_NUMBER_OF_RETURNS)] pub number_of_returns: u8,
    #[pasture(BUILTIN_SCAN_DIRECTION_FLAG)] pub scan_direction_flag: bool,
    #[pasture(BUILTIN_EDGE_OF_FLIGHT_LINE)] pub edge_of_flight_line: bool,
    #[pasture(BUILTIN_CLASSIFICATION)] pub classification: u8,
    #[pasture(BUILTIN_SCAN_ANGLE_RANK)] pub scan_angle_rank: i8,
    #[pasture(BUILTIN_USER_DATA)] pub user_data: u8,
    #[pasture(BUILTIN_POINT_SOURCE_ID)] pub point_source_id: u16,
    #[pasture(BUILTIN_GPS_TIME)] pub gps_time: f64,
    #[pasture(BUILTIN_COLOR_RGB)] pub color_rgb: Vector3<u16>,
    #[pasture(BUILTIN_WAVE_PACKET_DESCRIPTOR_INDEX)] pub wave_packet_descriptor_index: u8,
    #[pasture(BUILTIN_WAVEFORM_DATA_OFFSET)] pub byte_offset_to_waveform_data: u64,
    #[pasture(BUILTIN_WAVEFORM_PACKET_SIZE)] pub waveform_packet_size: u32,
    #[pasture(BUILTIN_RETURN_POINT_WAVEFORM_LOCATION)] pub return_point_waveform_location: f32,
    #[pasture(BUILTIN_WAVEFORM_PARAMETERS)] pub waveform_parameters: Vector3<f32>,
}

const_assert_eq!(std::mem::size_of::<LasPointFormat5>(), 78);

impl From<Point> for LasPointFormat5 {
    fn from(las_point: Point) -> Self {
        let color = las_point.color.expect("LasPoint has no color data");
        let waveform = las_point.waveform.expect("LasPoint has no waveform data");
        Self {
            position: Vector3::new(las_point.x, las_point.y, las_point.z),
            intensity: las_point.intensity,
            return_number: las_point.return_number,
            number_of_returns: las_point.number_of_returns,
            scan_direction_flag: match las_point.scan_direction {
                ScanDirection::RightToLeft => false,
                ScanDirection::LeftToRight => true,
            },
            edge_of_flight_line: las_point.is_edge_of_flight_line,
            classification: las_point.classification.into(),
            scan_angle_rank: las_point.scan_angle as i8,
            user_data: las_point.user_data,
            point_source_id: las_point.point_source_id,
            gps_time: las_point.gps_time.unwrap_or(0.0),
            color_rgb: Vector3::new(color.red, color.green, color.blue),
            wave_packet_descriptor_index: waveform.wave_packet_descriptor_index,
            byte_offset_to_waveform_data: waveform.byte_offset_to_waveform_data,
            waveform_packet_size: waveform.waveform_packet_size_in_bytes,
            return_point_waveform_location: waveform.return_point_waveform_location,
            waveform_parameters: Vector3::new(waveform.x_t, waveform.y_t, waveform.z_t),
        }
    }
}

/// Point type for LAS point format 6
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, PartialEq, Default, PointType)]
pub struct LasPointFormat6 {
    #[pasture(BUILTIN_POSITION_3D)] pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)] pub intensity: u16,
    #[pasture(BUILTIN_RETURN_NUMBER)] pub return_number: u8,
    #[pasture(BUILTIN_NUMBER_OF_RETURNS)] pub number_of_returns: u8,
    #[pasture(BUILTIN_CLASSIFICATION_FLAGS)] pub classification_flags: u8,
    #[pasture(BUILTIN_SCANNER_CHANNEL)] pub scanner_channel: u8,
    #[pasture(BUILTIN_SCAN_DIRECTION_FLAG)] pub scan_direction_flag: bool,
    #[pasture(BUILTIN_EDGE_OF_FLIGHT_LINE)] pub edge_of_flight_line: bool,
    #[pasture(BUILTIN_CLASSIFICATION)] pub classification: u8,
    #[pasture(BUILTIN_USER_DATA)] pub user_data: u8,
    #[pasture(BUILTIN_SCAN_ANGLE)] pub scan_angle: i16,
    #[pasture(BUILTIN_POINT_SOURCE_ID)] pub point_source_id: u16,
    #[pasture(BUILTIN_GPS_TIME)] pub gps_time: f64,
}

const_assert_eq!(std::mem::size_of::<LasPointFormat6>(), 46);

/// Point type for LAS point format 7
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, PartialEq, Default, PointType)]
pub struct LasPointFormat7 {
    #[pasture(BUILTIN_POSITION_3D)] pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)] pub intensity: u16,
    #[pasture(BUILTIN_RETURN_NUMBER)] pub return_number: u8,
    #[pasture(BUILTIN_NUMBER_OF_RETURNS)] pub number_of_returns: u8,
    #[pasture(BUILTIN_CLASSIFICATION_FLAGS)] pub classification_flags: u8,
    #[pasture(BUILTIN_SCANNER_CHANNEL)] pub scanner_channel: u8,
    #[pasture(BUILTIN_SCAN_DIRECTION_FLAG)] pub scan_direction_flag: bool,
    #[pasture(BUILTIN_EDGE_OF_FLIGHT_LINE)] pub edge_of_flight_line: bool,
    #[pasture(BUILTIN_CLASSIFICATION)] pub classification: u8,
    #[pasture(BUILTIN_USER_DATA)] pub user_data: u8,
    #[pasture(BUILTIN_SCAN_ANGLE)] pub scan_angle: i16,
    #[pasture(BUILTIN_POINT_SOURCE_ID)] pub point_source_id: u16,
    #[pasture(BUILTIN_GPS_TIME)] pub gps_time: f64,
    #[pasture(BUILTIN_COLOR_RGB)] pub color_rgb: Vector3<u16>,
}

const_assert_eq!(std::mem::size_of::<LasPointFormat7>(), 52);

/// Point type for LAS point format 8
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, PartialEq, Default, PointType)]
pub struct LasPointFormat8 {
    #[pasture(BUILTIN_POSITION_3D)] pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)] pub intensity: u16,
    #[pasture(BUILTIN_RETURN_NUMBER)] pub return_number: u8,
    #[pasture(BUILTIN_NUMBER_OF_RETURNS)] pub number_of_returns: u8,
    #[pasture(BUILTIN_CLASSIFICATION_FLAGS)] pub classification_flags: u8,
    #[pasture(BUILTIN_SCANNER_CHANNEL)] pub scanner_channel: u8,
    #[pasture(BUILTIN_SCAN_DIRECTION_FLAG)] pub scan_direction_flag: bool,
    #[pasture(BUILTIN_EDGE_OF_FLIGHT_LINE)] pub edge_of_flight_line: bool,
    #[pasture(BUILTIN_CLASSIFICATION)] pub classification: u8,
    #[pasture(BUILTIN_USER_DATA)] pub user_data: u8,
    #[pasture(BUILTIN_SCAN_ANGLE)] pub scan_angle: i16,
    #[pasture(BUILTIN_POINT_SOURCE_ID)] pub point_source_id: u16,
    #[pasture(BUILTIN_GPS_TIME)] pub gps_time: f64,
    #[pasture(BUILTIN_COLOR_RGB)] pub color_rgb: Vector3<u16>,
    #[pasture(BUILTIN_NIR)] pub nir: u16,
}

const_assert_eq!(std::mem::size_of::<LasPointFormat8>(), 54);

/// Point type for LAS point format 9
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, PartialEq, Default, PointType)]
pub struct LasPointFormat9 {
    #[pasture(BUILTIN_POSITION_3D)] pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)] pub intensity: u16,
    #[pasture(BUILTIN_RETURN_NUMBER)] pub return_number: u8,
    #[pasture(BUILTIN_NUMBER_OF_RETURNS)] pub number_of_returns: u8,
    #[pasture(BUILTIN_CLASSIFICATION_FLAGS)] pub classification_flags: u8,
    #[pasture(BUILTIN_SCANNER_CHANNEL)] pub scanner_channel: u8,
    #[pasture(BUILTIN_SCAN_DIRECTION_FLAG)] pub scan_direction_flag: bool,
    #[pasture(BUILTIN_EDGE_OF_FLIGHT_LINE)] pub edge_of_flight_line: bool,
    #[pasture(BUILTIN_CLASSIFICATION)] pub classification: u8,
    #[pasture(BUILTIN_USER_DATA)] pub user_data: u8,
    #[pasture(BUILTIN_SCAN_ANGLE)] pub scan_angle: i16,
    #[pasture(BUILTIN_POINT_SOURCE_ID)] pub point_source_id: u16,
    #[pasture(BUILTIN_GPS_TIME)] pub gps_time: f64,
    #[pasture(BUILTIN_WAVE_PACKET_DESCRIPTOR_INDEX)] pub wave_packet_descriptor_index: u8,
    #[pasture(BUILTIN_WAVEFORM_DATA_OFFSET)] pub byte_offset_to_waveform_data: u64,
    #[pasture(BUILTIN_WAVEFORM_PACKET_SIZE)] pub waveform_packet_size: u32,
    #[pasture(BUILTIN_RETURN_POINT_WAVEFORM_LOCATION)] pub return_point_waveform_location: f32,
    #[pasture(BUILTIN_WAVEFORM_PARAMETERS)] pub waveform_parameters: Vector3<f32>,
}

const_assert_eq!(std::mem::size_of::<LasPointFormat9>(), 75);

/// Point type for LAS point format 10
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, PartialEq, Default, PointType)]
pub struct LasPointFormat10 {
    #[pasture(BUILTIN_POSITION_3D)] pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)] pub intensity: u16,
    #[pasture(BUILTIN_RETURN_NUMBER)] pub return_number: u8,
    #[pasture(BUILTIN_NUMBER_OF_RETURNS)] pub number_of_returns: u8,
    #[pasture(BUILTIN_CLASSIFICATION_FLAGS)] pub classification_flags: u8,
    #[pasture(BUILTIN_SCANNER_CHANNEL)] pub scanner_channel: u8,
    #[pasture(BUILTIN_SCAN_DIRECTION_FLAG)] pub scan_direction_flag: bool,
    #[pasture(BUILTIN_EDGE_OF_FLIGHT_LINE)] pub edge_of_flight_line: bool,
    #[pasture(BUILTIN_CLASSIFICATION)] pub classification: u8,
    #[pasture(BUILTIN_USER_DATA)] pub user_data: u8,
    #[pasture(BUILTIN_SCAN_ANGLE)] pub scan_angle: i16,
    #[pasture(BUILTIN_POINT_SOURCE_ID)] pub point_source_id: u16,
    #[pasture(BUILTIN_GPS_TIME)] pub gps_time: f64,
    #[pasture(BUILTIN_COLOR_RGB)] pub color_rgb: Vector3<u16>,
    #[pasture(BUILTIN_NIR)] pub nir: u16,
    #[pasture(BUILTIN_WAVE_PACKET_DESCRIPTOR_INDEX)] pub wave_packet_descriptor_index: u8,
    #[pasture(BUILTIN_WAVEFORM_DATA_OFFSET)] pub byte_offset_to_waveform_data: u64,
    #[pasture(BUILTIN_WAVEFORM_PACKET_SIZE)] pub waveform_packet_size: u32,
    #[pasture(BUILTIN_RETURN_POINT_WAVEFORM_LOCATION)] pub return_point_waveform_location: f32,
    #[pasture(BUILTIN_WAVEFORM_PARAMETERS)] pub waveform_parameters: Vector3<f32>,
}

const_assert_eq!(std::mem::size_of::<LasPointFormat10>(), 83);

#[derive(Debug, Copy, Clone)]
pub(crate) struct BitAttributesRegular {
    pub return_number: u8,
    pub number_of_returns: u8,
    pub scan_direction_flag: u8,
    pub edge_of_flight_line: u8,
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct BitAttributesExtended {
    pub return_number: u8,
    pub number_of_returns: u8,
    pub classification_flags: u8,
    pub scanner_channel: u8,
    pub scan_direction_flag: u8,
    pub edge_of_flight_line: u8,
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum BitAttributes {
    Regular(BitAttributesRegular),
    Extended(BitAttributesExtended),
}

impl BitAttributes {
    pub fn return_number(&self) -> u8 {
        match &self {
            Self::Regular(attr) => attr.return_number,
            Self::Extended(attr) => attr.return_number,
        }
    }

    pub fn number_of_returns(&self) -> u8 {
        match &self {
            Self::Regular(attr) => attr.number_of_returns,
            Self::Extended(attr) => attr.number_of_returns,
        }
    }

    pub fn classification_flags_or_default(&self) -> u8 {
        match &self {
            Self::Regular(_) => Default::default(),
            Self::Extended(attr) => attr.classification_flags,
        }
    }

    pub fn scanner_channel_or_default(&self) -> u8 {
        match &self {
            Self::Regular(_) => Default::default(),
            Self::Extended(attr) => attr.scanner_channel,
        }
    }

    pub fn scan_direction_flag(&self) -> u8 {
        match &self {
            Self::Regular(attr) => attr.scan_direction_flag,
            Self::Extended(attr) => attr.scan_direction_flag,
        }
    }

    pub fn edge_of_flight_line(&self) -> u8 {
        match &self {
            Self::Regular(attr) => attr.edge_of_flight_line,
            Self::Extended(attr) => attr.edge_of_flight_line,
        }
    }
}
