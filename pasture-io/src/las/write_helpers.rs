use std::{convert::TryInto, io::Write};

use anyhow::Result;
use byteorder::{LittleEndian, WriteBytesExt};
use pasture_core::nalgebra::Vector3;

use super::BitAttributes;

/// Writes the given world space position as a LAS position to the given `writer`
pub(crate) fn write_position_as_las_position<T: Write>(
    world_space_position: &Vector3<f64>,
    las_header: &las::raw::Header,
    mut writer: T,
) -> Result<()> {
    let local_x : i32 = (((world_space_position.x - las_header.x_offset) / las_header.x_scale_factor) as i64).try_into().expect("write_position_as_las_position: Position is out of bounds given the current LAS offset and scale!");
    let local_y : i32 = (((world_space_position.y - las_header.y_offset) / las_header.y_scale_factor) as i64).try_into().expect("write_position_as_las_position: Position is out of bounds given the current LAS offset and scale!");
    let local_z : i32 = (((world_space_position.z - las_header.z_offset) / las_header.z_scale_factor) as i64).try_into().expect("write_position_as_las_position: Position is out of bounds given the current LAS offset and scale!");
    writer.write_i32::<LittleEndian>(local_x)?;
    writer.write_i32::<LittleEndian>(local_y)?;
    writer.write_i32::<LittleEndian>(local_z)?;

    Ok(())
}

/// Writes the given `BitAttributes` in LAS format to the given `writer`
pub(crate) fn write_las_bit_attributes<T: Write>(
    bit_attributes: BitAttributes,
    mut writer: T,
) -> Result<()> {
    match bit_attributes {
        BitAttributes::Regular(attributes) => {
            let mask = (attributes.return_number & 0b111)
                | (attributes.number_of_returns & 0b111) << 3
                | (attributes.scan_direction_flag & 0b1) << 6
                | (attributes.edge_of_flight_line & 0b1) << 7;
            writer.write_u8(mask)?;
        }
        BitAttributes::Extended(attributes) => {
            let low_mask =
                (attributes.return_number & 0b1111) | (attributes.number_of_returns & 0b1111) << 4;
            let high_mask = (attributes.classification_flags & 0b1111)
                | (attributes.scanner_channel & 0b11) << 4
                | (attributes.scan_direction_flag & 0b1) << 6
                | (attributes.edge_of_flight_line & 0b1) << 7;
            writer.write_u8(low_mask)?;
            writer.write_u8(high_mask)?;
        }
    }

    Ok(())
}
