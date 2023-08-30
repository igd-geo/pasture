use anyhow::Result;
use pasture_core::{
    containers::HashMapBuffer,
    layout::{attributes, PointLayout},
    nalgebra::Vector3,
};
use std::path::PathBuf;

/// Returns the resource/test/folder
pub(crate) fn get_test_file_path(filename: &str) -> PathBuf {
    let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    test_file_path.push(format!("resources/test/{}", filename));
    test_file_path
}

pub(crate) fn test_data_buffer() -> Result<HashMapBuffer> {
    let layout = PointLayout::from_attributes(&[
        attributes::POSITION_3D,
        attributes::INTENSITY,
        attributes::RETURN_NUMBER,
        attributes::NUMBER_OF_RETURNS,
        attributes::SCAN_DIRECTION_FLAG,
        attributes::EDGE_OF_FLIGHT_LINE,
        attributes::USER_DATA,
        attributes::CLASSIFICATION,
        attributes::SCAN_ANGLE_RANK,
        attributes::POINT_SOURCE_ID,
        attributes::GPS_TIME,
        attributes::COLOR_RGB,
        attributes::NIR,
    ]);
    let mut buffer = HashMapBuffer::with_capacity(10, layout);
    let mut pusher = buffer.begin_push_attributes();
    pusher.push_attribute_range(&attributes::POSITION_3D, test_data_positions().as_slice());
    pusher.push_attribute_range(&attributes::INTENSITY, test_data_intensities().as_slice());
    pusher.push_attribute_range(
        &attributes::RETURN_NUMBER,
        test_data_return_numbers().as_slice(),
    );
    pusher.push_attribute_range(
        &attributes::NUMBER_OF_RETURNS,
        &test_data_number_of_returns().as_slice(),
    );
    pusher.push_attribute_range(
        &attributes::SCAN_DIRECTION_FLAG,
        test_data_scan_direction_flags().as_slice(),
    );
    pusher.push_attribute_range(
        &attributes::EDGE_OF_FLIGHT_LINE,
        test_data_edge_of_flight_lines().as_slice(),
    );
    pusher.push_attribute_range(
        &attributes::CLASSIFICATION,
        test_data_classifications().as_slice(),
    );
    pusher.push_attribute_range(
        &attributes::SCAN_ANGLE_RANK,
        test_data_scan_angle_ranks().as_slice(),
    );
    pusher.push_attribute_range(
        &attributes::POINT_SOURCE_ID,
        test_data_point_source_ids().as_slice(),
    );
    pusher.push_attribute_range(&attributes::USER_DATA, test_data_user_data().as_slice());
    pusher.push_attribute_range(&attributes::GPS_TIME, test_data_gps_times().as_slice());
    pusher.push_attribute_range(&attributes::COLOR_RGB, test_data_colors().as_slice());
    pusher.push_attribute_range(&attributes::NIR, test_data_nirs().as_slice());
    pusher.done();

    Ok(buffer)
}

pub(crate) fn test_data_positions() -> Vec<Vector3<f64>> {
    vec![
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(1.0, 1.0, 1.0),
        Vector3::new(2.0, 2.0, 2.0),
        Vector3::new(3.0, 3.0, 3.0),
        Vector3::new(4.0, 4.0, 4.0),
        Vector3::new(5.0, 5.0, 5.0),
        Vector3::new(6.0, 6.0, 6.0),
        Vector3::new(7.0, 7.0, 7.0),
        Vector3::new(8.0, 8.0, 8.0),
        Vector3::new(9.0, 9.0, 9.0),
    ]
}

pub(crate) fn test_data_intensities() -> Vec<u16> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

pub(crate) fn test_data_return_numbers() -> Vec<u8> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

pub(crate) fn test_data_number_of_returns() -> Vec<u8> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

pub(crate) fn test_data_scan_direction_flags() -> Vec<u8> {
    vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

pub(crate) fn test_data_edge_of_flight_lines() -> Vec<u8> {
    vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

pub(crate) fn test_data_classifications() -> Vec<u8> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

pub(crate) fn test_data_scan_angle_ranks() -> Vec<i8> {
    vec![0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
}

pub(crate) fn test_data_user_data() -> Vec<u8> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

pub(crate) fn test_data_point_source_ids() -> Vec<u16> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

pub(crate) fn test_data_gps_times() -> Vec<f64> {
    vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
}

pub(crate) fn test_data_colors() -> Vec<Vector3<u16>> {
    vec![
        Vector3::new(0, 0, 0),
        Vector3::new(1, 1, 1),
        Vector3::new(2, 2, 2),
        Vector3::new(3, 3, 3),
        Vector3::new(4, 4, 4),
        Vector3::new(5, 5, 5),
        Vector3::new(6, 6, 6),
        Vector3::new(7, 7, 7),
        Vector3::new(8, 8, 8),
        Vector3::new(9, 9, 9),
    ]
}

pub(crate) fn test_data_nirs() -> Vec<u16> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}
