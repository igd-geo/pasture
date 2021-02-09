//! Contains types for each of the LAS point formats

use las::{point::ScanDirection, Point};
use pasture_core::{
    layout::{attributes, PointAttributeDataType, PointLayout, PointType},
    nalgebra::Vector3,
};
use static_assertions::const_assert_eq;
use std::convert::From;

/// Point type for LAS point format 0
#[repr(packed)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub(crate) struct LasPointFormat0 {
    pub position: Vector3<f64>,
    pub intensity: u16,
    pub return_number: u8,
    pub number_of_returns: u8,
    pub scan_direction_flag: bool,
    pub edge_of_flight_line: bool,
    pub classification: u8,
    pub scan_angle_rank: i8,
    pub user_data: u8,
    pub point_source_id: u16,
}

const_assert_eq!(std::mem::size_of::<LasPointFormat0>(), 35);

// TODO Replace these manual impls of PointType with the custom #[derive(PointType)] macro

impl PointType for LasPointFormat0 {
    fn layout() -> PointLayout {
        PointLayout::from_attributes(&[
            attributes::POSITION_3D,
            attributes::INTENSITY,
            attributes::RETURN_NUMBER,
            attributes::NUMBER_OF_RETURNS,
            attributes::SCAN_DIRECTION_FLAG,
            attributes::EDGE_OF_FLIGHT_LINE,
            attributes::CLASSIFICATION,
            attributes::SCAN_ANGLE_RANK,
            attributes::USER_DATA,
            attributes::POINT_SOURCE_ID,
        ])
    }
}

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
#[repr(packed)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub(crate) struct LasPointFormat1 {
    pub position: Vector3<f64>,
    pub intensity: u16,
    pub return_number: u8,
    pub number_of_returns: u8,
    pub scan_direction_flag: bool,
    pub edge_of_flight_line: bool,
    pub classification: u8,
    pub scan_angle_rank: i8,
    pub user_data: u8,
    pub point_source_id: u16,
    pub gps_time: f64,
}

const_assert_eq!(std::mem::size_of::<LasPointFormat1>(), 43);

impl PointType for LasPointFormat1 {
    fn layout() -> PointLayout {
        PointLayout::from_attributes(&[
            attributes::POSITION_3D,
            attributes::INTENSITY,
            attributes::RETURN_NUMBER,
            attributes::NUMBER_OF_RETURNS,
            attributes::SCAN_DIRECTION_FLAG,
            attributes::EDGE_OF_FLIGHT_LINE,
            attributes::CLASSIFICATION,
            attributes::SCAN_ANGLE_RANK,
            attributes::USER_DATA,
            attributes::POINT_SOURCE_ID,
            attributes::GPS_TIME,
        ])
    }
}

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
#[repr(packed)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub(crate) struct LasPointFormat2 {
    pub position: Vector3<f64>,
    pub intensity: u16,
    pub return_number: u8,
    pub number_of_returns: u8,
    pub scan_direction_flag: bool,
    pub edge_of_flight_line: bool,
    pub classification: u8,
    pub scan_angle_rank: i8,
    pub user_data: u8,
    pub point_source_id: u16,
    pub color_rgb: Vector3<u16>,
}

const_assert_eq!(std::mem::size_of::<LasPointFormat2>(), 41);

impl PointType for LasPointFormat2 {
    fn layout() -> PointLayout {
        PointLayout::from_attributes(&[
            attributes::POSITION_3D,
            attributes::INTENSITY,
            attributes::RETURN_NUMBER,
            attributes::NUMBER_OF_RETURNS,
            attributes::SCAN_DIRECTION_FLAG,
            attributes::EDGE_OF_FLIGHT_LINE,
            attributes::CLASSIFICATION,
            attributes::SCAN_ANGLE_RANK,
            attributes::USER_DATA,
            attributes::POINT_SOURCE_ID,
            attributes::COLOR_RGB,
        ])
    }
}

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
#[repr(packed)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub(crate) struct LasPointFormat3 {
    pub position: Vector3<f64>,
    pub intensity: u16,
    pub return_number: u8,
    pub number_of_returns: u8,
    pub scan_direction_flag: bool,
    pub edge_of_flight_line: bool,
    pub classification: u8,
    pub scan_angle_rank: i8,
    pub user_data: u8,
    pub point_source_id: u16,
    pub gps_time: f64,
    pub color_rgb: Vector3<u16>,
}

const_assert_eq!(std::mem::size_of::<LasPointFormat3>(), 49);

impl PointType for LasPointFormat3 {
    fn layout() -> PointLayout {
        PointLayout::from_attributes(&[
            attributes::POSITION_3D,
            attributes::INTENSITY,
            attributes::RETURN_NUMBER,
            attributes::NUMBER_OF_RETURNS,
            attributes::SCAN_DIRECTION_FLAG,
            attributes::EDGE_OF_FLIGHT_LINE,
            attributes::CLASSIFICATION,
            attributes::SCAN_ANGLE_RANK,
            attributes::USER_DATA,
            attributes::POINT_SOURCE_ID,
            attributes::GPS_TIME,
            attributes::COLOR_RGB,
        ])
    }
}

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
#[repr(packed)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub(crate) struct LasPointFormat4 {
    pub position: Vector3<f64>,
    pub intensity: u16,
    pub return_number: u8,
    pub number_of_returns: u8,
    pub scan_direction_flag: bool,
    pub edge_of_flight_line: bool,
    pub classification: u8,
    pub scan_angle_rank: i8,
    pub user_data: u8,
    pub point_source_id: u16,
    pub gps_time: f64,
    pub wave_packet_descriptor_index: u8,
    pub byte_offset_to_waveform_data: u64,
    pub waveform_packet_size: u32,
    pub return_point_waveform_location: f32,
    pub waveform_parameters: Vector3<f32>,
}

const_assert_eq!(std::mem::size_of::<LasPointFormat4>(), 72);

impl PointType for LasPointFormat4 {
    fn layout() -> PointLayout {
        PointLayout::from_attributes(&[
            attributes::POSITION_3D,
            attributes::INTENSITY,
            attributes::RETURN_NUMBER,
            attributes::NUMBER_OF_RETURNS,
            attributes::SCAN_DIRECTION_FLAG,
            attributes::EDGE_OF_FLIGHT_LINE,
            attributes::CLASSIFICATION,
            attributes::SCAN_ANGLE_RANK,
            attributes::USER_DATA,
            attributes::POINT_SOURCE_ID,
            attributes::GPS_TIME,
            attributes::WAVE_PACKET_DESCRIPTOR_INDEX,
            attributes::WAVEFORM_DATA_OFFSET,
            attributes::WAVEFORM_PACKET_SIZE,
            attributes::RETURN_POINT_WAVEFORM_LOCATION,
            attributes::WAVEFORM_PARAMETERS,
        ])
    }
}

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
#[repr(packed)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub(crate) struct LasPointFormat5 {
    pub position: Vector3<f64>,
    pub intensity: u16,
    pub return_number: u8,
    pub number_of_returns: u8,
    pub scan_direction_flag: bool,
    pub edge_of_flight_line: bool,
    pub classification: u8,
    pub scan_angle_rank: i8,
    pub user_data: u8,
    pub point_source_id: u16,
    pub gps_time: f64,
    pub color_rgb: Vector3<u16>,
    pub wave_packet_descriptor_index: u8,
    pub byte_offset_to_waveform_data: u64,
    pub waveform_packet_size: u32,
    pub return_point_waveform_location: f32,
    pub waveform_parameters: Vector3<f32>,
}

const_assert_eq!(std::mem::size_of::<LasPointFormat5>(), 78);

impl PointType for LasPointFormat5 {
    fn layout() -> PointLayout {
        PointLayout::from_attributes(&[
            attributes::POSITION_3D,
            attributes::INTENSITY,
            attributes::RETURN_NUMBER,
            attributes::NUMBER_OF_RETURNS,
            attributes::SCAN_DIRECTION_FLAG,
            attributes::EDGE_OF_FLIGHT_LINE,
            attributes::CLASSIFICATION,
            attributes::SCAN_ANGLE_RANK,
            attributes::USER_DATA,
            attributes::POINT_SOURCE_ID,
            attributes::GPS_TIME,
            attributes::COLOR_RGB,
            attributes::WAVE_PACKET_DESCRIPTOR_INDEX,
            attributes::WAVEFORM_DATA_OFFSET,
            attributes::WAVEFORM_PACKET_SIZE,
            attributes::RETURN_POINT_WAVEFORM_LOCATION,
            attributes::WAVEFORM_PARAMETERS,
        ])
    }
}

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
#[repr(packed)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub(crate) struct LasPointFormat6 {
    pub position: Vector3<f64>,
    pub intensity: u16,
    pub return_number: u8,
    pub number_of_returns: u8,
    pub classification_flags: u8,
    pub scanner_channel: u8,
    pub scan_direction_flag: bool,
    pub edge_of_flight_line: bool,
    pub classification: u8,
    pub user_data: u8,
    pub scan_angle: i16,
    pub point_source_id: u16,
    pub gps_time: f64,
}

impl PointType for LasPointFormat6 {
    fn layout() -> PointLayout {
        PointLayout::from_attributes(&[
            attributes::POSITION_3D,
            attributes::INTENSITY,
            attributes::RETURN_NUMBER,
            attributes::NUMBER_OF_RETURNS,
            attributes::CLASSIFICATION_FLAGS,
            attributes::SCANNER_CHANNEL,
            attributes::SCAN_DIRECTION_FLAG,
            attributes::EDGE_OF_FLIGHT_LINE,
            attributes::CLASSIFICATION,
            attributes::USER_DATA,
            attributes::SCAN_ANGLE_RANK.with_custom_datatype(PointAttributeDataType::I16),
            attributes::POINT_SOURCE_ID,
            attributes::GPS_TIME,
        ])
    }
}

/// Point type for LAS point format 7
#[repr(packed)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub(crate) struct LasPointFormat7 {
    pub position: Vector3<f64>,
    pub intensity: u16,
    pub return_number: u8,
    pub number_of_returns: u8,
    pub classification_flags: u8,
    pub scanner_channel: u8,
    pub scan_direction_flag: bool,
    pub edge_of_flight_line: bool,
    pub classification: u8,
    pub user_data: u8,
    pub scan_angle: i16,
    pub point_source_id: u16,
    pub gps_time: f64,
    pub color_rgb: Vector3<u16>,
}

impl PointType for LasPointFormat7 {
    fn layout() -> PointLayout {
        PointLayout::from_attributes(&[
            attributes::POSITION_3D,
            attributes::INTENSITY,
            attributes::RETURN_NUMBER,
            attributes::NUMBER_OF_RETURNS,
            attributes::CLASSIFICATION_FLAGS,
            attributes::SCANNER_CHANNEL,
            attributes::SCAN_DIRECTION_FLAG,
            attributes::EDGE_OF_FLIGHT_LINE,
            attributes::CLASSIFICATION,
            attributes::USER_DATA,
            attributes::SCAN_ANGLE_RANK.with_custom_datatype(PointAttributeDataType::I16),
            attributes::POINT_SOURCE_ID,
            attributes::GPS_TIME,
            attributes::COLOR_RGB,
        ])
    }
}

/// Point type for LAS point format 8
#[repr(packed)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub(crate) struct LasPointFormat8 {
    pub position: Vector3<f64>,
    pub intensity: u16,
    pub return_number: u8,
    pub number_of_returns: u8,
    pub classification_flags: u8,
    pub scanner_channel: u8,
    pub scan_direction_flag: bool,
    pub edge_of_flight_line: bool,
    pub classification: u8,
    pub user_data: u8,
    pub scan_angle: i16,
    pub point_source_id: u16,
    pub gps_time: f64,
    pub color_rgb: Vector3<u16>,
    pub nir: u16,
}

impl PointType for LasPointFormat8 {
    fn layout() -> PointLayout {
        PointLayout::from_attributes(&[
            attributes::POSITION_3D,
            attributes::INTENSITY,
            attributes::RETURN_NUMBER,
            attributes::NUMBER_OF_RETURNS,
            attributes::CLASSIFICATION_FLAGS,
            attributes::SCANNER_CHANNEL,
            attributes::SCAN_DIRECTION_FLAG,
            attributes::EDGE_OF_FLIGHT_LINE,
            attributes::CLASSIFICATION,
            attributes::USER_DATA,
            attributes::SCAN_ANGLE_RANK.with_custom_datatype(PointAttributeDataType::I16),
            attributes::POINT_SOURCE_ID,
            attributes::GPS_TIME,
            attributes::COLOR_RGB,
            attributes::NIR,
        ])
    }
}

/// Point type for LAS point format 9
#[repr(packed)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub(crate) struct LasPointFormat9 {
    pub position: Vector3<f64>,
    pub intensity: u16,
    pub return_number: u8,
    pub number_of_returns: u8,
    pub classification_flags: u8,
    pub scanner_channel: u8,
    pub scan_direction_flag: bool,
    pub edge_of_flight_line: bool,
    pub classification: u8,
    pub user_data: u8,
    pub scan_angle: i16,
    pub point_source_id: u16,
    pub gps_time: f64,
    pub wave_packet_descriptor_index: u8,
    pub byte_offset_to_waveform_data: u64,
    pub waveform_packet_size: u32,
    pub return_point_waveform_location: f32,
    pub waveform_parameters: Vector3<f32>,
}

impl PointType for LasPointFormat9 {
    fn layout() -> PointLayout {
        PointLayout::from_attributes(&[
            attributes::POSITION_3D,
            attributes::INTENSITY,
            attributes::RETURN_NUMBER,
            attributes::NUMBER_OF_RETURNS,
            attributes::CLASSIFICATION_FLAGS,
            attributes::SCANNER_CHANNEL,
            attributes::SCAN_DIRECTION_FLAG,
            attributes::EDGE_OF_FLIGHT_LINE,
            attributes::CLASSIFICATION,
            attributes::USER_DATA,
            attributes::SCAN_ANGLE_RANK.with_custom_datatype(PointAttributeDataType::I16),
            attributes::POINT_SOURCE_ID,
            attributes::GPS_TIME,
            attributes::WAVE_PACKET_DESCRIPTOR_INDEX,
            attributes::WAVEFORM_DATA_OFFSET,
            attributes::WAVEFORM_PACKET_SIZE,
            attributes::RETURN_POINT_WAVEFORM_LOCATION,
            attributes::WAVEFORM_PARAMETERS,
        ])
    }
}

/// Point type for LAS point format 10
#[repr(packed)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub(crate) struct LasPointFormat10 {
    pub position: Vector3<f64>,
    pub intensity: u16,
    pub return_number: u8,
    pub number_of_returns: u8,
    pub classification_flags: u8,
    pub scanner_channel: u8,
    pub scan_direction_flag: bool,
    pub edge_of_flight_line: bool,
    pub classification: u8,
    pub user_data: u8,
    pub scan_angle: i16,
    pub point_source_id: u16,
    pub gps_time: f64,
    pub color_rgb: Vector3<u16>,
    pub nir: u16,
    pub wave_packet_descriptor_index: u8,
    pub byte_offset_to_waveform_data: u64,
    pub waveform_packet_size: u32,
    pub return_point_waveform_location: f32,
    pub waveform_parameters: Vector3<f32>,
}

impl PointType for LasPointFormat10 {
    fn layout() -> PointLayout {
        PointLayout::from_attributes(&[
            attributes::POSITION_3D,
            attributes::INTENSITY,
            attributes::RETURN_NUMBER,
            attributes::NUMBER_OF_RETURNS,
            attributes::CLASSIFICATION_FLAGS,
            attributes::SCANNER_CHANNEL,
            attributes::SCAN_DIRECTION_FLAG,
            attributes::EDGE_OF_FLIGHT_LINE,
            attributes::CLASSIFICATION,
            attributes::USER_DATA,
            attributes::SCAN_ANGLE_RANK.with_custom_datatype(PointAttributeDataType::I16),
            attributes::POINT_SOURCE_ID,
            attributes::GPS_TIME,
            attributes::COLOR_RGB,
            attributes::NIR,
            attributes::WAVE_PACKET_DESCRIPTOR_INDEX,
            attributes::WAVEFORM_DATA_OFFSET,
            attributes::WAVEFORM_PACKET_SIZE,
            attributes::RETURN_POINT_WAVEFORM_LOCATION,
            attributes::WAVEFORM_PARAMETERS,
        ])
    }
}

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
