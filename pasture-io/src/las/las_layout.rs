use anyhow::{anyhow, Result};
use las::point::Format;
use pasture_core::layout::{PointLayout, PointType};

use super::{LasPointFormat0, LasPointFormat1, LasPointFormat2, LasPointFormat3, LasPointFormat4, LasPointFormat5};

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
    let format_number = format.to_u8()?;

    match format_number {
        0 => Ok(LasPointFormat0::layout()),
        1 => Ok(LasPointFormat1::layout()),
        2 => Ok(LasPointFormat2::layout()),
        3 => Ok(LasPointFormat3::layout()),
        4 => Ok(LasPointFormat4::layout()),
        5 => Ok(LasPointFormat5::layout()),
        _ => Err(anyhow!("Unsupported LAS point format {}", format_number)),
    }
}
