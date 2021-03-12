use std::{ops::Range, path::PathBuf};

use anyhow::Result;
use las_rs::point::Format;
use pasture_core::{
    containers::PointBuffer,
    containers::{PerAttributeVecPointStorage, PointBufferExt},
    layout::attributes,
    math::AABB,
    nalgebra::{Point3, Vector3},
};

use super::point_layout_from_las_point_format;

/// Returns the path to a LAS test file with the given `format`
pub(crate) fn get_test_las_path(format: u8) -> PathBuf {
    let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    test_file_path.push(format!("resources/test/10_points_format_{}.las", format));
    test_file_path
}

/// Returns the path to a LAZ test file with the given `format`
pub(crate) fn get_test_laz_path(format: u8) -> PathBuf {
    let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    test_file_path.push(format!("resources/test/10_points_format_{}.laz", format));
    test_file_path
}

pub(crate) fn format_has_gps_times(format: u8) -> bool {
    match format {
        1 => true,
        3..=10 => true,
        _ => false,
    }
}

pub(crate) fn format_has_colors(format: u8) -> bool {
    match format {
        2..=3 => true,
        5 => true,
        7..=8 => true,
        10 => true,
        _ => false,
    }
}

pub(crate) fn format_has_nir(format: u8) -> bool {
    match format {
        8 => true,
        10 => true,
        _ => false,
    }
}

pub(crate) fn format_has_wavepacket(format: u8) -> bool {
    match format {
        4..=5 => true,
        9..=10 => true,
        _ => false,
    }
}

fn format_is_extended(format: u8) -> bool {
    format >= 6
}

pub(crate) fn test_data_point_count() -> usize {
    10
}

pub(crate) fn test_data_bounds() -> AABB<f64> {
    AABB::from_min_max_unchecked(Point3::new(0.0, 0.0, 0.0), Point3::new(9.0, 9.0, 9.0))
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
    vec![
        0,
        255,
        2 * 255,
        3 * 255,
        4 * 255,
        5 * 255,
        6 * 255,
        7 * 255,
        8 * 255,
        9 * 255,
    ]
}

pub(crate) fn test_data_return_numbers() -> Vec<u8> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 0, 1]
}

pub(crate) fn test_data_return_numbers_extended() -> Vec<u8> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

pub(crate) fn test_data_number_of_returns() -> Vec<u8> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 0, 1]
}

pub(crate) fn test_data_number_of_returns_extended() -> Vec<u8> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

fn test_data_classification_flags() -> Vec<u8> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

fn test_data_scanner_channels() -> Vec<u8> {
    vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
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
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

pub(crate) fn test_data_scan_angles_extended() -> Vec<i16> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

pub(crate) fn test_data_user_data() -> Vec<u8> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

pub(crate) fn test_data_point_source_ids() -> Vec<u16> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

pub(crate) fn test_data_gps_times() -> Vec<f64> {
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
}

pub(crate) fn test_data_colors() -> Vec<Vector3<u16>> {
    vec![
        Vector3::new(0, 1 << 4, 2 << 8),
        Vector3::new(1, 2 << 4, 3 << 8),
        Vector3::new(2, 3 << 4, 4 << 8),
        Vector3::new(3, 4 << 4, 5 << 8),
        Vector3::new(4, 5 << 4, 6 << 8),
        Vector3::new(5, 6 << 4, 7 << 8),
        Vector3::new(6, 7 << 4, 8 << 8),
        Vector3::new(7, 8 << 4, 9 << 8),
        Vector3::new(8, 9 << 4, 10 << 8),
        Vector3::new(9, 10 << 4, 11 << 8),
    ]
}

pub(crate) fn test_data_nirs() -> Vec<u16> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

pub(crate) fn test_data_wavepacket_index() -> Vec<u8> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

pub(crate) fn test_data_wavepacket_offset() -> Vec<u64> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

pub(crate) fn test_data_wavepacket_size() -> Vec<u32> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

pub(crate) fn test_data_wavepacket_location() -> Vec<f32> {
    vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
}

pub(crate) fn test_data_wavepacket_parameters() -> Vec<Vector3<f32>> {
    vec![
        Vector3::new(1.0, 2.0, 3.0),
        Vector3::new(2.0, 3.0, 4.0),
        Vector3::new(3.0, 4.0, 5.0),
        Vector3::new(4.0, 5.0, 6.0),
        Vector3::new(5.0, 6.0, 7.0),
        Vector3::new(6.0, 7.0, 8.0),
        Vector3::new(7.0, 8.0, 9.0),
        Vector3::new(8.0, 9.0, 10.0),
        Vector3::new(9.0, 10.0, 11.0),
        Vector3::new(10.0, 11.0, 12.0),
    ]
}

pub(crate) fn compare_to_reference_data_range(
    points: &dyn PointBuffer,
    point_format: u8,
    range: Range<usize>,
) {
    let positions = points
        .iter_attribute::<Vector3<f64>>(&attributes::POSITION_3D)
        .collect::<Vec<_>>();
    assert_eq!(
        &test_data_positions()[range.clone()],
        positions,
        "Positions do not match"
    );

    let intensities = points
        .iter_attribute::<u16>(&attributes::INTENSITY)
        .collect::<Vec<_>>();
    assert_eq!(
        &test_data_intensities()[range.clone()],
        intensities,
        "Intensities do not match"
    );

    let return_numbers = points
        .iter_attribute::<u8>(&attributes::RETURN_NUMBER)
        .collect::<Vec<_>>();
    let expected_return_numbers = if format_is_extended(point_format) {
        test_data_return_numbers_extended()
    } else {
        test_data_return_numbers()
    };
    assert_eq!(
        &expected_return_numbers[range.clone()],
        return_numbers,
        "Return numbers do not match"
    );

    let number_of_returns = points
        .iter_attribute::<u8>(&attributes::NUMBER_OF_RETURNS)
        .collect::<Vec<_>>();
    let expected_number_of_returns = if format_is_extended(point_format) {
        test_data_number_of_returns_extended()
    } else {
        test_data_number_of_returns()
    };
    assert_eq!(
        &expected_number_of_returns[range.clone()],
        number_of_returns,
        "Number of returns do not match"
    );

    if format_is_extended(point_format) {
        let classification_flags = points
            .iter_attribute::<u8>(&attributes::CLASSIFICATION_FLAGS)
            .collect::<Vec<_>>();
        assert_eq!(
            &test_data_classification_flags()[range.clone()],
            classification_flags,
            "Classification flags do not match"
        );

        let scanner_channels = points
            .iter_attribute::<u8>(&attributes::SCANNER_CHANNEL)
            .collect::<Vec<_>>();
        assert_eq!(
            &test_data_scanner_channels()[range.clone()],
            scanner_channels,
            "Scanner channels do not match"
        );
    }

    let scan_direction_flags = points
        .iter_attribute::<bool>(&attributes::SCAN_DIRECTION_FLAG)
        .collect::<Vec<_>>();
    assert_eq!(
        &test_data_scan_direction_flags()[range.clone()],
        scan_direction_flags,
        "Scan direction flags do not match"
    );

    let eof = points
        .iter_attribute::<bool>(&attributes::EDGE_OF_FLIGHT_LINE)
        .collect::<Vec<_>>();
    assert_eq!(
        &test_data_edge_of_flight_lines()[range.clone()],
        eof,
        "Edge of flight lines do not match"
    );

    let classifications = points
        .iter_attribute::<u8>(&attributes::CLASSIFICATION)
        .collect::<Vec<_>>();
    assert_eq!(
        &test_data_classifications()[range.clone()],
        classifications,
        "Classifications do not match"
    );

    if point_format < 6 {
        let scan_angle_ranks = points
            .iter_attribute::<i8>(&attributes::SCAN_ANGLE_RANK)
            .collect::<Vec<_>>();
        assert_eq!(
            &test_data_scan_angle_ranks()[range.clone()],
            scan_angle_ranks,
            "Scan angle ranks do not match"
        );
    } else {
        let scan_angles = points
            .iter_attribute::<i16>(&attributes::SCAN_ANGLE)
            .collect::<Vec<_>>();
        assert_eq!(
            &test_data_scan_angles_extended()[range.clone()],
            scan_angles,
            "Scan angles do not match"
        );
    }

    let user_data = points
        .iter_attribute::<u8>(&attributes::USER_DATA)
        .collect::<Vec<_>>();
    assert_eq!(
        &test_data_user_data()[range.clone()],
        user_data,
        "User data do not match"
    );

    let point_source_ids = points
        .iter_attribute::<u16>(&attributes::POINT_SOURCE_ID)
        .collect::<Vec<_>>();
    assert_eq!(
        &test_data_point_source_ids()[range.clone()],
        point_source_ids,
        "Point source IDs do not match"
    );

    if format_has_gps_times(point_format) {
        let gps_times = points
            .iter_attribute::<f64>(&attributes::GPS_TIME)
            .collect::<Vec<_>>();
        assert_eq!(
            &test_data_gps_times()[range.clone()],
            gps_times,
            "GPS times do not match"
        );
    }

    if format_has_colors(point_format) {
        let colors = points
            .iter_attribute::<Vector3<u16>>(&attributes::COLOR_RGB)
            .collect::<Vec<_>>();
        assert_eq!(
            &test_data_colors()[range.clone()],
            colors,
            "Colors do not match"
        );
    }

    if format_has_nir(point_format) {
        let nirs = points
            .iter_attribute::<u16>(&attributes::NIR)
            .collect::<Vec<_>>();
        assert_eq!(
            &test_data_nirs()[range.clone()],
            nirs,
            "NIR values do not match"
        );
    }

    if format_has_wavepacket(point_format) {
        let wp_indices = points
            .iter_attribute::<u8>(&attributes::WAVE_PACKET_DESCRIPTOR_INDEX)
            .collect::<Vec<_>>();
        assert_eq!(
            &test_data_wavepacket_index()[range.clone()],
            wp_indices,
            "Wave Packet Descriptor Indices do not match"
        );

        let wp_offsets = points
            .iter_attribute::<u64>(&attributes::WAVEFORM_DATA_OFFSET)
            .collect::<Vec<_>>();
        assert_eq!(
            &test_data_wavepacket_offset()[range.clone()],
            wp_offsets,
            "Waveform data offsets do not match"
        );

        let wp_sizes = points
            .iter_attribute::<u32>(&attributes::WAVEFORM_PACKET_SIZE)
            .collect::<Vec<_>>();
        assert_eq!(
            &test_data_wavepacket_size()[range.clone()],
            wp_sizes,
            "Waveform packet sizes do not match"
        );

        let wp_return_points = points
            .iter_attribute::<f32>(&attributes::RETURN_POINT_WAVEFORM_LOCATION)
            .collect::<Vec<_>>();
        assert_eq!(
            &test_data_wavepacket_location()[range.clone()],
            wp_return_points,
            "WAveform return point locations do not match"
        );

        let wp_parameters = points
            .iter_attribute::<Vector3<f32>>(&attributes::WAVEFORM_PARAMETERS)
            .collect::<Vec<_>>();
        assert_eq!(
            &test_data_wavepacket_parameters()[range.clone()],
            wp_parameters,
            "Waveform parameters do not match"
        );
    }
}

/// Compare the `points` in the given `point_format` to the reference data for the format
pub(crate) fn compare_to_reference_data(points: &dyn PointBuffer, point_format: u8) {
    compare_to_reference_data_range(points, point_format, 0..test_data_point_count());
}

pub(crate) fn get_test_points_in_las_format(point_format: u8) -> Result<Box<dyn PointBuffer>> {
    let format = Format::new(point_format)?;
    let layout = point_layout_from_las_point_format(&format)?;
    let mut buffer = PerAttributeVecPointStorage::with_capacity(10, layout);
    let mut pusher = buffer.begin_push_attributes();
    pusher.push_attribute_range(&attributes::POSITION_3D, test_data_positions().as_slice());
    pusher.push_attribute_range(&attributes::INTENSITY, test_data_intensities().as_slice());

    if format.is_extended {
        pusher.push_attribute_range(
            &attributes::RETURN_NUMBER,
            test_data_return_numbers_extended().as_slice(),
        );
        pusher.push_attribute_range(
            &attributes::NUMBER_OF_RETURNS,
            test_data_number_of_returns_extended().as_slice(),
        );
        pusher.push_attribute_range(
            &attributes::CLASSIFICATION_FLAGS,
            test_data_classification_flags().as_slice(),
        );
        pusher.push_attribute_range(
            &attributes::SCANNER_CHANNEL,
            test_data_scanner_channels().as_slice(),
        );
    } else {
        pusher.push_attribute_range(
            &attributes::RETURN_NUMBER,
            test_data_return_numbers().as_slice(),
        );
        pusher.push_attribute_range(
            &attributes::NUMBER_OF_RETURNS,
            test_data_number_of_returns().as_slice(),
        );
    }

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

    if format.is_extended {
        pusher.push_attribute_range(&attributes::USER_DATA, test_data_user_data().as_slice());
        pusher.push_attribute_range(
            &attributes::SCAN_ANGLE,
            test_data_scan_angles_extended().as_slice(),
        );
    } else {
        pusher.push_attribute_range(
            &attributes::SCAN_ANGLE_RANK,
            test_data_scan_angle_ranks().as_slice(),
        );
        pusher.push_attribute_range(&attributes::USER_DATA, test_data_user_data().as_slice());
    }

    pusher.push_attribute_range(
        &attributes::POINT_SOURCE_ID,
        test_data_point_source_ids().as_slice(),
    );

    if format.has_gps_time {
        pusher.push_attribute_range(&attributes::GPS_TIME, test_data_gps_times().as_slice());
    }

    if format.has_color {
        pusher.push_attribute_range(&attributes::COLOR_RGB, test_data_colors().as_slice());
    }

    if format.has_nir {
        pusher.push_attribute_range(&attributes::NIR, test_data_nirs().as_slice());
    }

    if format.has_waveform {
        pusher.push_attribute_range(
            &attributes::WAVE_PACKET_DESCRIPTOR_INDEX,
            test_data_wavepacket_index().as_slice(),
        );
        pusher.push_attribute_range(
            &attributes::WAVEFORM_DATA_OFFSET,
            test_data_wavepacket_offset().as_slice(),
        );
        pusher.push_attribute_range(
            &attributes::WAVEFORM_PACKET_SIZE,
            test_data_wavepacket_size().as_slice(),
        );
        pusher.push_attribute_range(
            &attributes::RETURN_POINT_WAVEFORM_LOCATION,
            test_data_wavepacket_location().as_slice(),
        );
        pusher.push_attribute_range(
            &attributes::WAVEFORM_PARAMETERS,
            test_data_wavepacket_parameters().as_slice(),
        );
    }

    pusher.done();

    Ok(Box::new(buffer))
}

pub(crate) fn _epsilon_compare_vec3f32(expected: &Vector3<f32>, actual: &Vector3<f32>) -> bool {
    const EPSILON: f32 = 1e-5;
    let dx = (expected.x - actual.x).abs();
    let dy = (expected.y - actual.y).abs();
    let dz = (expected.z - actual.z).abs();
    dx <= EPSILON && dy <= EPSILON && dz <= EPSILON
}

pub(crate) fn epsilon_compare_vec3f64(expected: &Vector3<f64>, actual: &Vector3<f64>) -> bool {
    const EPSILON: f64 = 1e-7;
    let dx = (expected.x - actual.x).abs();
    let dy = (expected.y - actual.y).abs();
    let dz = (expected.z - actual.z).abs();
    dx <= EPSILON && dy <= EPSILON && dz <= EPSILON
}

pub(crate) fn epsilon_compare_point3f64(expected: &Point3<f64>, actual: &Point3<f64>) -> bool {
    const EPSILON: f64 = 1e-7;
    let dx = (expected.x - actual.x).abs();
    let dy = (expected.y - actual.y).abs();
    let dz = (expected.z - actual.z).abs();
    dx <= EPSILON && dy <= EPSILON && dz <= EPSILON
}
