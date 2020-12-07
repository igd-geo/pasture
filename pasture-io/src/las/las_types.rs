//! Contains types for each of the LAS point formats

use las::{point::ScanDirection, Point};
use pasture_core::{
    layout::{attributes, PointLayout, PointType},
    nalgebra::Vector3,
};
use std::convert::From;
use static_assertions::const_assert_eq;

/// Point type for LAS point format 0
#[repr(packed)]
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
            gps_time: las_point
                .gps_time
                .expect("Conversion from las::Point to LASPoint_Format1: gps_time was None"),
        }
    }
}

/// Point type for LAS point format 2
#[repr(packed)]
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
            gps_time: las_point
                .gps_time
                .expect("Conversion from las::Point to LASPoint_Format1: gps_time was None"),
            color_rgb: Vector3::new(color.red, color.green, color.blue),
        }
    }
}

/// Point type for LAS point format 4
#[repr(packed)]
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
            gps_time: las_point
                .gps_time
                .expect("Conversion from las::Point to LASPoint_Format1: gps_time was None"),
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
            gps_time: las_point
                .gps_time
                .expect("Conversion from las::Point to LASPoint_Format1: gps_time was None"),
            color_rgb: Vector3::new(color.red, color.green, color.blue),
            wave_packet_descriptor_index: waveform.wave_packet_descriptor_index,
            byte_offset_to_waveform_data: waveform.byte_offset_to_waveform_data,
            waveform_packet_size: waveform.waveform_packet_size_in_bytes,
            return_point_waveform_location: waveform.return_point_waveform_location,
            waveform_parameters: Vector3::new(waveform.x_t, waveform.y_t, waveform.z_t),
        }
    }
}

// TODO Other formats
