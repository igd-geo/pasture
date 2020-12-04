use anyhow::Result;
use las::point::Format;
use pasture_core::layout::{attributes, PointAttributeDataType, PointLayout};

/// Returns the default `PointLayout` for the given LAS point format. This layout mirrors the binary layout
/// of the point records in the LAS format, as defined by the [LAS specification](http://www.asprs.org/wp-content/uploads/2019/03/LAS_1_4_r14.pdf).
/// The notable exception are the X, Y, and Z fields, which are combined into a single `POSITION_3D` attribute
/// with f64 datatype, instead of three separate fields with i32 datatype. The reason for this is usability:
/// Positions more often than not are treated as a single semantic unit instead of individual components. The
/// shift from i32 to f64 is the result of how LAS stores positions internally (normalized to integer values within
/// the bounding box of the file). The true positions are reconstructed from the internal representation automatically
/// as f64 values.
///
/// # Errors
///
/// Returns an error if `format` is an invalid LAS point format
pub fn point_layout_from_las_point_format(format: &Format) -> Result<PointLayout> {
    // Check if format is valid
    format.to_u8()?;

    // We set the data types explicitly here, even if they are equal to the default values, simply because
    // default values may change and this is easier to understand

    let mut layout = PointLayout::new();
    // LAS has X, Y, and Z as 32-bit integer values, however the default will almost always be 64-bit float values. If the raw 32-bit
    // integer values are desired, this can be set with a flag in the `LASReader` type
    layout.add_attribute(
        attributes::POSITION_3D.with_custom_datatype(PointAttributeDataType::Vec3f64),
    );
    layout.add_attribute(attributes::INTENSITY.with_custom_datatype(PointAttributeDataType::U16));

    // TODO Promoting all the bit fields to full u8 values increases memory usage. Maybe combine the attributes into one?
    layout
        .add_attribute(attributes::RETURN_NUMBER.with_custom_datatype(PointAttributeDataType::U8));
    layout.add_attribute(
        attributes::NUMBER_OF_RETURNS.with_custom_datatype(PointAttributeDataType::U8),
    );
    layout.add_attribute(
        attributes::SCAN_DIRECTION_FLAG.with_custom_datatype(PointAttributeDataType::Bool),
    );
    layout.add_attribute(
        attributes::EDGE_OF_FLIGHT_LINE.with_custom_datatype(PointAttributeDataType::Bool),
    );

    layout
        .add_attribute(attributes::CLASSIFICATION.with_custom_datatype(PointAttributeDataType::U8));

    if format.is_extended {
        layout.add_attribute(
            attributes::SCAN_ANGLE_RANK.with_custom_datatype(PointAttributeDataType::I16),
        );
    } else {
        layout.add_attribute(
            attributes::SCAN_ANGLE_RANK.with_custom_datatype(PointAttributeDataType::I8),
        );
    }

    layout.add_attribute(attributes::USER_DATA.with_custom_datatype(PointAttributeDataType::U8));
    layout.add_attribute(
        attributes::POINT_SOURCE_ID.with_custom_datatype(PointAttributeDataType::U16),
    );

    if format.has_gps_time {
        layout
            .add_attribute(attributes::GPS_TIME.with_custom_datatype(PointAttributeDataType::F64));
    }

    if format.has_color {
        layout.add_attribute(
            attributes::COLOR_RGB.with_custom_datatype(PointAttributeDataType::Vec3u16),
        );
    }

    if format.has_nir {
        layout.add_attribute(attributes::NIR.with_custom_datatype(PointAttributeDataType::U16));
    }

    // TODO waveform data

    Ok(layout)
}
