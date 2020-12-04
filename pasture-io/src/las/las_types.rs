//! Contains types for each of the LAS point formats

use las::{point::ScanDirection, Point};
use pasture_core::{
    layout::{attributes, PointLayout, PointType},
    nalgebra::Vector3,
};
use std::convert::From;

/// Point type for LAS point format 0
pub(crate) struct LASPoint_Format0 {
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

// TODO Replace these manual impls of PointType with the custom #[derive(PointType)] macro

impl PointType for LASPoint_Format0 {
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

impl From<Point> for LASPoint_Format0 {
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
pub(crate) struct LASPoint_Format1 {
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

impl PointType for LASPoint_Format1 {
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

impl From<Point> for LASPoint_Format1 {
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
pub(crate) struct LASPoint_Format2 {
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

impl PointType for LASPoint_Format2 {
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

impl From<Point> for LASPoint_Format2 {
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
pub(crate) struct LASPoint_Format3 {
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

impl PointType for LASPoint_Format3 {
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

impl From<Point> for LASPoint_Format3 {
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

// TODO Other formats
