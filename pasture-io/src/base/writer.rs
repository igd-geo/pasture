use anyhow::Result;
use pasture_core::{containers::PointBuffer, layout::PointLayout};

/// Base trait for all types that support writing point data
pub trait PointWriter {
    /// Write the points in the given `PointBuffer` to the associated `PointWriter`.
    fn write(&mut self, points: &dyn PointBuffer) -> Result<()>;

    /// Returns the default `PointLayout` of the associated `PointWriter`
    fn get_default_point_layout(&self) -> &PointLayout;
}
