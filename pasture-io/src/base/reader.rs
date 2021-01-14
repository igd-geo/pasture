use anyhow::Result;
use pasture_core::containers::{PointBuffer, PointBufferWriteable};
use pasture_core::layout::PointLayout;
use pasture_core::meta::Metadata;

/// Base trait for all types that support reading point data
pub trait PointReader {
    /// Read `count` points from this `PointReader`. Returns an opaque `PointBuffer` type filled with
    /// the read points in the default `PointLayout` of this `PointReader`.
    fn read(&mut self, count: usize) -> Result<Box<dyn PointBuffer>>;
    /// Read `count` points from this `PointReader` into the given `PointBuffer`. Uses the `PointLayout`
    /// of the given `PointBuffer` for reading. If no conversion from the default `PointLayout` to this
    /// new layout are possible, an error is returned. On success, returns the number of points that
    /// were read.
    fn read_into(
        &mut self,
        point_buffer: &mut dyn PointBufferWriteable,
        count: usize,
    ) -> Result<usize>;

    /// Returns the `Metadata` of the associated `PointReader`
    fn get_metadata(&self) -> &dyn Metadata;
    /// Returns the default `PointLayout` of the associated `PointReader`
    fn get_default_point_layout(&self) -> &PointLayout;
}
