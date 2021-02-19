use anyhow::{anyhow, Result};
use las::point::Format;
use pasture_core::{
    layout::attributes,
    layout::{PointLayout, PointType},
};

use super::{
    LasPointFormat0, LasPointFormat1, LasPointFormat10, LasPointFormat2, LasPointFormat3,
    LasPointFormat4, LasPointFormat5, LasPointFormat6, LasPointFormat7, LasPointFormat8,
    LasPointFormat9,
};

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
    let format_number = format.to_u8()?;

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
    // TODO Explicit support for extended size formats (6-10)

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

    if has_nir {
        format.is_extended = true;
    }

    format
}
