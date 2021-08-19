use std::path::PathBuf;

use pasture_core::nalgebra::Vector3;

/// Returns the resource/test/folder
pub(crate) fn get_test_file_path(filename: &str) -> PathBuf {
    let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    test_file_path.push(format!("resources/test/{}", filename));
    test_file_path
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

pub(crate) fn test_data_scan_direction_flags() -> Vec<bool> {
    vec![
        false, true, false, true, false, true, false, true, false, true,
    ]
}

pub(crate) fn test_data_edge_of_flight_lines() -> Vec<bool> {
    vec![
        false, true, false, true, false, true, false, true, false, true,
    ]
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
