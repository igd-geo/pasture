use anyhow::Result;
use pasture_core::containers::{MakeBufferFromLayout, OwningBuffer};
use pasture_core::layout::PointLayout;
use pasture_core::meta::Metadata;

/// Base trait for all types that support reading point data
pub trait PointReader {
    /// Read `count` points from this `PointReader` into the given `PointBuffer`. Uses the `PointLayout`
    /// of the given `PointBuffer` for reading. If no conversion from the default `PointLayout` to this
    /// new layout are possible, an error is returned. On success, returns the number of points that
    /// were read.
    fn read_into<'a, 'b, B: OwningBuffer<'a>>(
        &mut self,
        point_buffer: &'b mut B,
        count: usize,
    ) -> Result<usize>
    where
        'a: 'b;

    fn read<'a, B: OwningBuffer<'a> + MakeBufferFromLayout<'a> + 'a>(
        &mut self,
        count: usize,
    ) -> Result<B> {
        let mut buffer = B::new_from_layout(self.get_default_point_layout().clone());
        self.read_into(&mut buffer, count)?;
        Ok(buffer)
    }

    /// Returns the `Metadata` of the associated `PointReader`
    fn get_metadata(&self) -> &dyn Metadata;
    /// Returns the default `PointLayout` of the associated `PointReader`
    fn get_default_point_layout(&self) -> &PointLayout;
}
