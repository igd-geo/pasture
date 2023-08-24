use std::{
    convert::TryInto,
    fs::File,
    io::{BufWriter, Write},
};

use anyhow::{Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use las::{point::Format, raw::header::LargeFile, Builder, Header, Transform, Version};
use pasture_core::nalgebra::Vector3;
use pasture_io::las::{
    write_las_bit_attributes, BitAttributes, BitAttributesExtended, BitAttributesRegular,
    ExtraBytesDataType, ExtraBytesEntryBuilder, ExtraBytesVlr,
};

fn position_to_local_space(
    position: Vector3<f64>,
    transforms: &pasture_io::las_rs::Vector<Transform>,
) -> Result<Vector3<i32>> {
    let local_x = (((position.x - transforms.x.offset) / transforms.x.scale) as u64)
        .try_into()
        .context("Position overflows i32")?;
    let local_y = (((position.y - transforms.y.offset) / transforms.y.scale) as u64)
        .try_into()
        .context("Position overflows i32")?;
    let local_z = (((position.z - transforms.z.offset) / transforms.z.scale) as u64)
        .try_into()
        .context("Position overflows i32")?;
    Ok(Vector3::new(local_x, local_y, local_z))
}

fn test_data_positions() -> Vec<Vector3<f64>> {
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

fn test_data_intensities() -> Vec<u16> {
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

fn test_data_return_numbers() -> Vec<u8> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 0, 1]
}

fn test_data_return_numbers_extended() -> Vec<u8> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

fn test_data_number_of_returns() -> Vec<u8> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 0, 1]
}

fn test_data_number_of_returns_extended() -> Vec<u8> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

fn test_data_classification_flags() -> Vec<u8> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

fn test_data_scanner_channels() -> Vec<u8> {
    vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
}

fn test_data_scan_direction_flags() -> Vec<bool> {
    vec![
        false, true, false, true, false, true, false, true, false, true,
    ]
}

fn test_data_edge_of_flight_lines() -> Vec<bool> {
    vec![
        false, true, false, true, false, true, false, true, false, true,
    ]
}

fn test_data_classifications() -> Vec<u8> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

fn test_data_scan_angle_ranks() -> Vec<i8> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

fn test_data_scan_angles_extended() -> Vec<i16> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

fn test_data_user_data() -> Vec<u8> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

fn test_data_point_source_ids() -> Vec<u16> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

fn test_data_gps_times() -> Vec<f64> {
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
}

fn test_data_colors() -> Vec<Vector3<u16>> {
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

fn test_data_nirs() -> Vec<u16> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

fn test_data_wavepacket_index() -> Vec<u8> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

fn test_data_wavepacket_offset() -> Vec<u64> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

fn test_data_wavepacket_size() -> Vec<u32> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

fn test_data_wavepacket_location() -> Vec<f32> {
    vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
}

fn test_data_wavepacket_parameters() -> Vec<Vector3<f32>> {
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

fn test_data_extra_bytes_unsigned() -> Vec<u32> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

fn write_test_point<W: Write>(writer: &mut W, index: usize, las_header: &Header) -> Result<()> {
    let position = test_data_positions()[index];
    let local_position = position_to_local_space(position, las_header.transforms())?;
    writer.write_i32::<LittleEndian>(local_position.x)?;
    writer.write_i32::<LittleEndian>(local_position.y)?;
    writer.write_i32::<LittleEndian>(local_position.z)?;

    writer.write_u16::<LittleEndian>(test_data_intensities()[index])?;

    if las_header.point_format().is_extended {
        let bit_attributes = BitAttributesExtended {
            classification_flags: test_data_classification_flags()[index],
            edge_of_flight_line: test_data_edge_of_flight_lines()[index] as u8,
            number_of_returns: test_data_number_of_returns_extended()[index],
            return_number: test_data_return_numbers_extended()[index],
            scan_direction_flag: test_data_scan_direction_flags()[index] as u8,
            scanner_channel: test_data_scanner_channels()[index],
        };
        write_las_bit_attributes(BitAttributes::Extended(bit_attributes), writer)?;
    } else {
        let bit_attributes = BitAttributesRegular {
            edge_of_flight_line: test_data_edge_of_flight_lines()[index] as u8,
            number_of_returns: test_data_number_of_returns()[index],
            return_number: test_data_return_numbers()[index],
            scan_direction_flag: test_data_scan_direction_flags()[index] as u8,
        };
        write_las_bit_attributes(BitAttributes::Regular(bit_attributes), writer)?;
    }

    writer.write_u8(test_data_classifications()[index])?;

    if las_header.point_format().is_extended {
        writer.write_u8(test_data_user_data()[index])?;
        writer.write_i16::<LittleEndian>(test_data_scan_angles_extended()[index])?;
    } else {
        writer.write_i8(test_data_scan_angle_ranks()[index])?;
        writer.write_u8(test_data_user_data()[index])?;
    }

    writer.write_u16::<LittleEndian>(test_data_point_source_ids()[index])?;

    if las_header.point_format().has_gps_time {
        writer.write_f64::<LittleEndian>(test_data_gps_times()[index])?;
    }
    if las_header.point_format().has_color {
        let test_color = test_data_colors()[index];
        writer.write_u16::<LittleEndian>(test_color.x)?;
        writer.write_u16::<LittleEndian>(test_color.y)?;
        writer.write_u16::<LittleEndian>(test_color.z)?;
    }
    if las_header.point_format().has_nir {
        writer.write_u16::<LittleEndian>(test_data_nirs()[index])?;
    }
    if las_header.point_format().has_waveform {
        writer.write_u8(test_data_wavepacket_index()[index])?;
        writer.write_u64::<LittleEndian>(test_data_wavepacket_offset()[index])?;
        writer.write_u32::<LittleEndian>(test_data_wavepacket_size()[index])?;
        writer.write_f32::<LittleEndian>(test_data_wavepacket_location()[index])?;

        let wave_params = test_data_wavepacket_parameters()[index];
        writer.write_f32::<LittleEndian>(wave_params.x)?;
        writer.write_f32::<LittleEndian>(wave_params.y)?;
        writer.write_f32::<LittleEndian>(wave_params.z)?;
    }

    if las_header.point_format().extra_bytes > 0 {
        writer.write_u32::<LittleEndian>(test_data_extra_bytes_unsigned()[index])?;
    }

    Ok(())
}

fn main() -> Result<()> {
    for format_id in 0..11 {
        let mut header_builder = Builder::default();
        header_builder.version = Version::new(1, 4);
        header_builder.has_synthetic_return_numbers = true;
        header_builder.point_format = Format::new(format_id)?;
        header_builder.point_format.extra_bytes = 4;

        let extra_bytes_vlr: ExtraBytesVlr =
            IntoIterator::into_iter([ExtraBytesEntryBuilder::new(
                ExtraBytesDataType::U32,
                "TestExtraBytes".into(),
                "Extra bytes for testing".into(),
            )
            .build()])
            .collect();
        header_builder.vlrs.push((&extra_bytes_vlr).try_into()?);

        let header = header_builder.into_header()?;
        let mut raw_header = header.clone().into_raw()?;
        raw_header.number_of_point_records = 10;
        raw_header.large_file = Some(LargeFile {
            number_of_point_records: 10,
            number_of_points_by_return: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        });
        raw_header.number_of_points_by_return = [2, 2, 1, 1, 1];
        raw_header.min_x = 0.0;
        raw_header.min_y = 0.0;
        raw_header.min_z = 0.0;
        raw_header.max_x = 9.0;
        raw_header.max_y = 9.0;
        raw_header.max_z = 9.0;

        let mut writer = BufWriter::new(File::create(format!(
            "10_points_with_extra_bytes_format_{}.las",
            format_id
        ))?);

        raw_header.write_to(&mut writer)?;
        for vlr in header.vlrs() {
            let raw_vlr = vlr.clone().into_raw(false)?;
            raw_vlr.write_to(&mut writer)?;
        }

        for point_index in 0..test_data_positions().len() {
            write_test_point(&mut writer, point_index, &header)?;
        }
    }

    Ok(())
}
