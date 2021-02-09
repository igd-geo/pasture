use anyhow::Result;
use pasture_core::{containers::PointBuffer, layout::PointLayout};

/// Base trait for all types that support writing point data
pub trait PointWriter {
    /// Write the points in the given `PointBuffer` to the associated `PointWriter`.
    fn write(&mut self, points: &dyn PointBuffer) -> Result<()>;
    /// Flush this `PointWriter`, ensuring that all points are written to their destination and that all required
    /// metadata is written as well
    fn flush(&mut self) -> Result<()>;

    /// Returns the default `PointLayout` of the associated `PointWriter`
    fn get_default_point_layout(&self) -> &PointLayout;
}
