use anyhow::{anyhow, Result};
use las::point::Format;
use pasture_core::{
    layout::attributes,
    layout::{
        attributes::{
            CLASSIFICATION, COLOR_RGB, GPS_TIME, INTENSITY, NIR, POINT_SOURCE_ID, POSITION_3D,
            RETURN_POINT_WAVEFORM_LOCATION, SCAN_ANGLE, SCAN_ANGLE_RANK, USER_DATA,
            WAVEFORM_DATA_OFFSET, WAVEFORM_PACKET_SIZE, WAVEFORM_PARAMETERS,
            WAVE_PACKET_DESCRIPTOR_INDEX,
        },
        FieldAlignment, PointAttributeDataType, PointAttributeDefinition, PointLayout, PointType,
    },
};

use super::{
    LASMetadata, LasPointFormat0, LasPointFormat1, LasPointFormat10, LasPointFormat2,
    LasPointFormat3, LasPointFormat4, LasPointFormat5, LasPointFormat6, LasPointFormat7,
    LasPointFormat8, LasPointFormat9,
};

/// Returns the offset to the first extra byte in the given LAS point format. Returns `None` if the format
/// has no extra bytes
pub fn offset_to_extra_bytes(format: Format) -> Option<usize> {
    if format.extra_bytes == 0 {
        None
    } else {
        Some((format.len() - format.extra_bytes) as usize)
    }
}

/// LAS flags for the basic (0-5) point record types. For internal use only!
pub(crate) const ATTRIBUTE_BASIC_FLAGS: PointAttributeDefinition =
    PointAttributeDefinition::custom("LAS_BASIC_FLAGS", PointAttributeDataType::U8);
/// LAS flags for extended formats. For internal use only!
pub(crate) const ATTRIBUTE_EXTENDED_FLAGS: PointAttributeDefinition =
    PointAttributeDefinition::custom("LAS_EXTENDED_FLAGS", PointAttributeDataType::U16);

/// Returns the default `PointLayout` for the given LAS point format. If `exact_binary_representation` is true, the
/// layout mirrors the binary layout of the point records in the LAS format, as defined by the [LAS specification](http://www.asprs.org/wp-content/uploads/2019/03/LAS_1_4_r14.pdf).
/// Mirroring in this context means that the binary layout exactly matches that of the specification. This is fast,
/// as it allows parsing through `memcpy`, but it is not very usable, as this means positions are in local space instead
/// of world space, and the bit fields are packed instead of represented as separate bytes
///
/// # Errors
///
/// Returns an error if `format` is an invalid LAS point format, or if the format contains extra bytes.
pub fn point_layout_from_las_point_format(
    format: &Format,
    exact_binary_representation: bool,
) -> Result<PointLayout> {
    let format_number = format.to_u8()?;

    if exact_binary_representation {
        // Build the layout by hand!
        let mut layout = PointLayout::default();
        layout.add_attribute(
            POSITION_3D.with_custom_datatype(PointAttributeDataType::Vec3i32),
            FieldAlignment::Packed(1),
        );
        layout.add_attribute(INTENSITY, FieldAlignment::Packed(1));
        if format.is_extended {
            layout.add_attribute(ATTRIBUTE_EXTENDED_FLAGS, FieldAlignment::Packed(1));
        } else {
            layout.add_attribute(ATTRIBUTE_BASIC_FLAGS, FieldAlignment::Packed(1));
        }
        layout.add_attribute(CLASSIFICATION, FieldAlignment::Packed(1));
        if format.is_extended {
            layout.add_attribute(USER_DATA, FieldAlignment::Packed(1));
            layout.add_attribute(SCAN_ANGLE, FieldAlignment::Packed(1));
        } else {
            layout.add_attribute(SCAN_ANGLE_RANK, FieldAlignment::Packed(1));
            layout.add_attribute(USER_DATA, FieldAlignment::Packed(1));
        }
        layout.add_attribute(POINT_SOURCE_ID, FieldAlignment::Packed(1));

        // GPS time, then colors, then NIR, then waveform
        if format.has_gps_time {
            layout.add_attribute(GPS_TIME, FieldAlignment::Packed(1));
        }
        if format.has_color {
            layout.add_attribute(COLOR_RGB, FieldAlignment::Packed(1));
        }
        if format.has_nir {
            layout.add_attribute(NIR, FieldAlignment::Packed(1));
        }
        if format.has_waveform {
            layout.add_attribute(WAVE_PACKET_DESCRIPTOR_INDEX, FieldAlignment::Packed(1));
            layout.add_attribute(WAVEFORM_DATA_OFFSET, FieldAlignment::Packed(1));
            layout.add_attribute(WAVEFORM_PACKET_SIZE, FieldAlignment::Packed(1));
            layout.add_attribute(RETURN_POINT_WAVEFORM_LOCATION, FieldAlignment::Packed(1));
            layout.add_attribute(WAVEFORM_PARAMETERS, FieldAlignment::Packed(1));
        }

        Ok(layout)
    } else {
        match format_number {
            0 => Ok(LasPointFormat0::layout()),
            1 => Ok(LasPointFormat1::layout()),
            2 => Ok(LasPointFormat2::layout()),
            3 => Ok(LasPointFormat3::layout()),
            4 => Ok(LasPointFormat4::layout()),
            5 => Ok(LasPointFormat5::layout()),
            6 => Ok(LasPointFormat6::layout()),
            7 => Ok(LasPointFormat7::layout()),
            8 => Ok(LasPointFormat8::layout()),
            9 => Ok(LasPointFormat9::layout()),
            10 => Ok(LasPointFormat10::layout()),
            _ => Err(anyhow!("Unsupported LAS point format {}", format_number)),
        }
    }
}

/// Returns a matching `PointLayout` for the given `LASMetadata`. This function is similar to `point_layout_from_format`, but
/// also supports extra bytes if the given `LASMetadata` contains an Extra Bytes VLR. If it does not, but the point format in
/// the `LASMetadata` indicates that extra bytes are present, the extra bytes will be included in the `PointLayout` as raw bytes
///
/// # Errors
///
/// Returns an error if `format` is an invalid LAS point format, or if the format contains extra bytes.
pub fn point_layout_from_las_metadata(
    las_metadata: &LASMetadata,
    exact_binary_representation: bool,
) -> Result<PointLayout> {
    let format = las_metadata.point_format();
    let mut base_layout = point_layout_from_las_point_format(&format, exact_binary_representation)?;
    if format.extra_bytes == 0 {
        return Ok(base_layout);
    }

    let extra_byte_attributes = las_metadata
        .extra_bytes_vlr()
        .map(|vlr| {
            vlr.entries()
                .iter()
                .map(|entry| entry.get_point_attribute())
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?
        .unwrap_or_default();
    let num_described_bytes = extra_byte_attributes
        .iter()
        .map(|attribute| attribute.size() as usize)
        .sum::<usize>();

    // Add the extra bytes attributes with a 1-byte alignment, because the base LAS point types are all tightly packed
    // Currently, the RawLASReader and RawLAZReader both rely on this fact when reading chunks in the default layout

    // TODO It is debatable if there is much gain with using packed alignment as the default, both for the extra bytes
    //      as well as the LAS types in `las_types.rs` in general. In the end unaligned I/O might be slower, and we can't
    //      even memcpy directly because we still use Vector3<f64> as the default type for positions, even though LAS
    //      uses Vector3<i32>... So it might be worthwhile to change this once we have support for reading positions in
    //      Vector3<i32> using the LASReader
    for extra_byte_attribute in extra_byte_attributes {
        base_layout.add_attribute(extra_byte_attribute, FieldAlignment::Packed(1));
    }

    let num_undescribed_bytes = format.extra_bytes as usize - num_described_bytes;
    if num_undescribed_bytes > 0 {
        // Add a PointAttributeDefinition describing a raw byte array for all undescribed extra bytes
        base_layout.add_attribute(
            PointAttributeDefinition::custom(
                "UndescribedExtraBytes",
                PointAttributeDataType::ByteArray(num_described_bytes as u64),
            ),
            FieldAlignment::Packed(1),
        );
    }

    Ok(base_layout)
}

/// Returns the best matching LAS point format for the given `PointLayout`. This method tries to match as many attributes
/// as possible in the given `PointLayout` to attributes that are supported by the LAS format (v1.4) natively. Attributes
/// that do not have a corresponding LAS attribute are ignored. If no matching attributes are found, LAS point format 0 is
/// returned, as it is the most basic format.
/// ```
/// # use pasture_io::las::*;
/// # use pasture_core::layout::*;
///
/// let layout_a = PointLayout::from_attributes(&[attributes::POSITION_3D]);
/// let las_format_a = las_point_format_from_point_layout(&layout_a);
/// assert_eq!(las_format_a, las::point::Format::new(0).unwrap());
///
/// let layout_b = PointLayout::from_attributes(&[attributes::POSITION_3D, attributes::GPS_TIME]);
/// let las_format_b = las_point_format_from_point_layout(&layout_b);
/// assert_eq!(las_format_b, las::point::Format::new(1).unwrap());
/// ```
pub fn las_point_format_from_point_layout(point_layout: &PointLayout) -> Format {
    let has_gps_time = point_layout.has_attribute_with_name(attributes::GPS_TIME.name());
    let has_colors = point_layout.has_attribute_with_name(attributes::COLOR_RGB.name());
    let has_any_waveform_attribute = point_layout
        .has_attribute_with_name(attributes::WAVE_PACKET_DESCRIPTOR_INDEX.name())
        || point_layout.has_attribute_with_name(attributes::WAVEFORM_DATA_OFFSET.name())
        || point_layout.has_attribute_with_name(attributes::WAVEFORM_PACKET_SIZE.name())
        || point_layout.has_attribute_with_name(attributes::RETURN_POINT_WAVEFORM_LOCATION.name())
        || point_layout.has_attribute_with_name(attributes::WAVEFORM_PARAMETERS.name());
    let has_nir = point_layout.has_attribute_with_name(attributes::NIR.name());

    let mut format = Format::new(0).unwrap();
    format.has_color = has_colors;
    format.has_gps_time = has_gps_time;
    format.has_nir = has_nir;
    format.has_waveform = has_any_waveform_attribute;

    if has_nir || has_any_waveform_attribute {
        format.is_extended = true;
    }

    format
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_layout_from_las_format_with_exact_layout() -> Result<()> {
        let expected_sizes_per_format = [20, 28, 26, 34, 57, 63, 30, 36, 38, 59, 67];
        for (format_number, expected_size) in expected_sizes_per_format.iter().enumerate() {
            let format = Format::new(format_number as u8)?;
            let layout = point_layout_from_las_point_format(&format, true)?;
            assert_eq!(layout.size_of_point_entry(), *expected_size);
        }
        Ok(())
    }
}
