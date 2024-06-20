use std::{borrow::Cow, ops::Range, path::PathBuf};

use anyhow::Result;
use las_rs::point::Format;
use pasture_core::{
    containers::{
        BorrowedBuffer, BorrowedBufferExt, BorrowedMutBufferExt, HashMapBuffer, OwningBuffer,
    },
    layout::{attributes, FieldAlignment, PointAttributeDataType, PointAttributeDefinition},
    math::AABB,
    nalgebra::{Point3, Vector3},
};

use super::point_layout_from_las_point_format;

//use super::point_layout_from_las_point_format;

/// Returns the path to a LAS test file with the given `format`
pub(crate) fn get_test_las_path(format: u8) -> PathBuf {
    let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    test_file_path.push(format!("resources/test/10_points_format_{}.las", format));
    test_file_path
}

/// Returns the path to a LAS test file with the given `format` and extra bytes
pub(crate) fn get_test_las_path_with_extra_bytes(format: u8) -> PathBuf {
    let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    test_file_path.push(format!(
        "resources/test/10_points_with_extra_bytes_format_{}.las",
        format
    ));
    test_file_path
}

/// Returns the path to a LAZ test file with the given `format`
pub(crate) fn get_test_laz_path(format: u8) -> PathBuf {
    let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    test_file_path.push(format!("resources/test/10_points_format_{}.laz", format));
    test_file_path
}

pub(crate) const fn test_data_point_count() -> usize {
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

pub(crate) fn test_data_extra_bytes_unsigned() -> Vec<u32> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

pub(crate) fn compare_to_reference_data_range<'a, B: BorrowedBuffer<'a>>(
    points: &'a B,
    point_format: Format,
    range: Range<usize>,
) {
    let positions = points
        .view_attribute::<Vector3<f64>>(&attributes::POSITION_3D)
        .into_iter()
        .collect::<Vec<_>>();
    assert_eq!(
        &test_data_positions()[range.clone()],
        positions,
        "Positions do not match"
    );

    let intensities = points
        .view_attribute::<u16>(&attributes::INTENSITY)
        .into_iter()
        .collect::<Vec<_>>();
    assert_eq!(
        &test_data_intensities()[range.clone()],
        intensities,
        "Intensities do not match"
    );

    let return_numbers = points
        .view_attribute::<u8>(&attributes::RETURN_NUMBER)
        .into_iter()
        .collect::<Vec<_>>();
    let expected_return_numbers = if point_format.is_extended {
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
        .view_attribute::<u8>(&attributes::NUMBER_OF_RETURNS)
        .into_iter()
        .collect::<Vec<_>>();
    let expected_number_of_returns = if point_format.is_extended {
        test_data_number_of_returns_extended()
    } else {
        test_data_number_of_returns()
    };
    assert_eq!(
        &expected_number_of_returns[range.clone()],
        number_of_returns,
        "Number of returns do not match"
    );

    if point_format.is_extended {
        let classification_flags = points
            .view_attribute::<u8>(&attributes::CLASSIFICATION_FLAGS)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(
            &test_data_classification_flags()[range.clone()],
            classification_flags,
            "Classification flags do not match"
        );

        let scanner_channels = points
            .view_attribute::<u8>(&attributes::SCANNER_CHANNEL)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(
            &test_data_scanner_channels()[range.clone()],
            scanner_channels,
            "Scanner channels do not match"
        );
    }

    let scan_direction_flags = points
        .view_attribute::<u8>(&attributes::SCAN_DIRECTION_FLAG)
        .into_iter()
        .collect::<Vec<_>>();
    assert_eq!(
        &test_data_scan_direction_flags()[range.clone()],
        scan_direction_flags,
        "Scan direction flags do not match"
    );

    let eof = points
        .view_attribute::<u8>(&attributes::EDGE_OF_FLIGHT_LINE)
        .into_iter()
        .collect::<Vec<_>>();
    assert_eq!(
        &test_data_edge_of_flight_lines()[range.clone()],
        eof,
        "Edge of flight lines do not match"
    );

    let classifications = points
        .view_attribute::<u8>(&attributes::CLASSIFICATION)
        .into_iter()
        .collect::<Vec<_>>();
    assert_eq!(
        &test_data_classifications()[range.clone()],
        classifications,
        "Classifications do not match"
    );

    if !point_format.is_extended {
        let scan_angle_ranks = points
            .view_attribute::<i8>(&attributes::SCAN_ANGLE_RANK)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(
            &test_data_scan_angle_ranks()[range.clone()],
            scan_angle_ranks,
            "Scan angle ranks do not match"
        );
    } else {
        let scan_angles = points
            .view_attribute::<i16>(&attributes::SCAN_ANGLE)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(
            &test_data_scan_angles_extended()[range.clone()],
            scan_angles,
            "Scan angles do not match"
        );
    }

    let user_data = points
        .view_attribute::<u8>(&attributes::USER_DATA)
        .into_iter()
        .collect::<Vec<_>>();
    assert_eq!(
        &test_data_user_data()[range.clone()],
        user_data,
        "User data do not match"
    );

    let point_source_ids = points
        .view_attribute::<u16>(&attributes::POINT_SOURCE_ID)
        .into_iter()
        .collect::<Vec<_>>();
    assert_eq!(
        &test_data_point_source_ids()[range.clone()],
        point_source_ids,
        "Point source IDs do not match"
    );

    if point_format.has_gps_time {
        let gps_times = points
            .view_attribute::<f64>(&attributes::GPS_TIME)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(
            &test_data_gps_times()[range.clone()],
            gps_times,
            "GPS times do not match"
        );
    }

    if point_format.has_color {
        let colors = points
            .view_attribute::<Vector3<u16>>(&attributes::COLOR_RGB)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(
            &test_data_colors()[range.clone()],
            colors,
            "Colors do not match"
        );
    }

    if point_format.has_nir {
        let nirs = points
            .view_attribute::<u16>(&attributes::NIR)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(
            &test_data_nirs()[range.clone()],
            nirs,
            "NIR values do not match"
        );
    }

    if point_format.has_waveform {
        let wp_indices = points
            .view_attribute::<u8>(&attributes::WAVE_PACKET_DESCRIPTOR_INDEX)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(
            &test_data_wavepacket_index()[range.clone()],
            wp_indices,
            "Wave Packet Descriptor Indices do not match"
        );

        let wp_offsets = points
            .view_attribute::<u64>(&attributes::WAVEFORM_DATA_OFFSET)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(
            &test_data_wavepacket_offset()[range.clone()],
            wp_offsets,
            "Waveform data offsets do not match"
        );

        let wp_sizes = points
            .view_attribute::<u32>(&attributes::WAVEFORM_PACKET_SIZE)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(
            &test_data_wavepacket_size()[range.clone()],
            wp_sizes,
            "Waveform packet sizes do not match"
        );

        let wp_return_points = points
            .view_attribute::<f32>(&attributes::RETURN_POINT_WAVEFORM_LOCATION)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(
            &test_data_wavepacket_location()[range.clone()],
            wp_return_points,
            "Waveform return point locations do not match"
        );

        let wp_parameters = points
            .view_attribute::<Vector3<f32>>(&attributes::WAVEFORM_PARAMETERS)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(
            &test_data_wavepacket_parameters()[range.clone()],
            wp_parameters,
            "Waveform parameters do not match"
        );
    }

    if point_format.extra_bytes > 0 {
        let extra_bytes = points
            .view_attribute::<u32>(&DEFAULT_EXTRA_BYTES_ATTRIBUTE)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(
            &test_data_extra_bytes_unsigned()[range.clone()],
            extra_bytes,
            "Extra bytes do not match"
        );
    }
}

/// Compare the `points` in the given `point_format` to the reference data for the format
pub(crate) fn compare_to_reference_data<'a, B: BorrowedBuffer<'a>>(
    points: &'a B,
    point_format: Format,
) {
    compare_to_reference_data_range(points, point_format, 0..test_data_point_count());
}

pub(crate) const DEFAULT_EXTRA_BYTES_ATTRIBUTE: PointAttributeDefinition =
    PointAttributeDefinition::custom(Cow::Borrowed("TestExtraBytes"), PointAttributeDataType::U32);

pub(crate) fn get_test_points_in_las_format(
    point_format: u8,
    with_extra_bytes: bool,
) -> Result<HashMapBuffer> {
    let format = Format::new(point_format)?;
    let mut layout = point_layout_from_las_point_format(&format, false)?;
    if with_extra_bytes {
        layout.add_attribute(DEFAULT_EXTRA_BYTES_ATTRIBUTE, FieldAlignment::Packed(1));
    }

    let mut buffer = HashMapBuffer::with_capacity(10, layout);
    buffer.resize(10);
    for (idx, value) in test_data_positions().iter().copied().enumerate() {
        buffer
            .view_attribute_mut(&attributes::POSITION_3D)
            .set_at(idx, value);
    }
    for (idx, value) in test_data_intensities().iter().copied().enumerate() {
        buffer
            .view_attribute_mut(&attributes::INTENSITY)
            .set_at(idx, value);
    }

    if format.is_extended {
        for (idx, value) in test_data_return_numbers().iter().copied().enumerate() {
            buffer
                .view_attribute_mut(&attributes::RETURN_NUMBER)
                .set_at(idx, value);
        }
        for (idx, value) in test_data_number_of_returns().iter().copied().enumerate() {
            buffer
                .view_attribute_mut(&attributes::NUMBER_OF_RETURNS)
                .set_at(idx, value);
        }
        for (idx, value) in test_data_classification_flags().iter().copied().enumerate() {
            buffer
                .view_attribute_mut(&attributes::CLASSIFICATION_FLAGS)
                .set_at(idx, value);
        }
        for (idx, value) in test_data_scanner_channels().iter().copied().enumerate() {
            buffer
                .view_attribute_mut(&attributes::SCANNER_CHANNEL)
                .set_at(idx, value);
        }
    } else {
        for (idx, value) in test_data_return_numbers().iter().copied().enumerate() {
            buffer
                .view_attribute_mut(&attributes::RETURN_NUMBER)
                .set_at(idx, value);
        }
        for (idx, value) in test_data_number_of_returns().iter().copied().enumerate() {
            buffer
                .view_attribute_mut(&attributes::NUMBER_OF_RETURNS)
                .set_at(idx, value);
        }
    }

    for (idx, value) in test_data_scan_direction_flags().iter().copied().enumerate() {
        buffer
            .view_attribute_mut(&attributes::SCAN_DIRECTION_FLAG)
            .set_at(idx, value);
    }
    for (idx, value) in test_data_edge_of_flight_lines().iter().copied().enumerate() {
        buffer
            .view_attribute_mut(&attributes::EDGE_OF_FLIGHT_LINE)
            .set_at(idx, value);
    }
    for (idx, value) in test_data_classifications().iter().copied().enumerate() {
        buffer
            .view_attribute_mut(&attributes::CLASSIFICATION)
            .set_at(idx, value);
    }

    if format.is_extended {
        for (idx, value) in test_data_user_data().iter().copied().enumerate() {
            buffer
                .view_attribute_mut(&attributes::USER_DATA)
                .set_at(idx, value);
        }
        for (idx, value) in test_data_scan_angles_extended().iter().copied().enumerate() {
            buffer
                .view_attribute_mut(&attributes::SCAN_ANGLE)
                .set_at(idx, value);
        }
    } else {
        for (idx, value) in test_data_scan_angle_ranks().iter().copied().enumerate() {
            buffer
                .view_attribute_mut(&attributes::SCAN_ANGLE_RANK)
                .set_at(idx, value);
        }
        for (idx, value) in test_data_user_data().iter().copied().enumerate() {
            buffer
                .view_attribute_mut(&attributes::USER_DATA)
                .set_at(idx, value);
        }
    }

    for (idx, value) in test_data_point_source_ids().iter().copied().enumerate() {
        buffer
            .view_attribute_mut(&attributes::POINT_SOURCE_ID)
            .set_at(idx, value);
    }

    if format.has_gps_time {
        for (idx, value) in test_data_gps_times().iter().copied().enumerate() {
            buffer
                .view_attribute_mut(&attributes::GPS_TIME)
                .set_at(idx, value);
        }
    }

    if format.has_color {
        for (idx, value) in test_data_colors().iter().copied().enumerate() {
            buffer
                .view_attribute_mut(&attributes::COLOR_RGB)
                .set_at(idx, value);
        }
    }

    if format.has_nir {
        for (idx, value) in test_data_nirs().iter().copied().enumerate() {
            buffer
                .view_attribute_mut(&attributes::NIR)
                .set_at(idx, value);
        }
    }

    if format.has_waveform {
        for (idx, value) in test_data_wavepacket_index().iter().copied().enumerate() {
            buffer
                .view_attribute_mut(&attributes::WAVE_PACKET_DESCRIPTOR_INDEX)
                .set_at(idx, value);
        }
        for (idx, value) in test_data_wavepacket_offset().iter().copied().enumerate() {
            buffer
                .view_attribute_mut(&attributes::WAVEFORM_DATA_OFFSET)
                .set_at(idx, value);
        }
        for (idx, value) in test_data_wavepacket_size().iter().copied().enumerate() {
            buffer
                .view_attribute_mut(&attributes::WAVEFORM_PACKET_SIZE)
                .set_at(idx, value);
        }
        for (idx, value) in test_data_wavepacket_location().iter().copied().enumerate() {
            buffer
                .view_attribute_mut(&attributes::RETURN_POINT_WAVEFORM_LOCATION)
                .set_at(idx, value);
        }
        for (idx, value) in test_data_wavepacket_parameters()
            .iter()
            .copied()
            .enumerate()
        {
            buffer
                .view_attribute_mut(&attributes::WAVEFORM_PARAMETERS)
                .set_at(idx, value);
        }
    }

    if with_extra_bytes {
        for (idx, value) in test_data_extra_bytes_unsigned().iter().copied().enumerate() {
            buffer
                .view_attribute_mut(&DEFAULT_EXTRA_BYTES_ATTRIBUTE)
                .set_at(idx, value);
        }
    }

    Ok(buffer)
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
