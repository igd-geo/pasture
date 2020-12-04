use anyhow::Result;
use std::io::SeekFrom;

/// Base trait for all readers and writers that support seeking to a specific point in their
/// underlying stream. This trait is similar to [std::io::Seek](std::io::Seek) but instead
/// of seeking to a specific byte offset, it allows seeking to a specific point.
pub trait SeekToPoint {
    /// Seek to the point at the given `position` in the underlying stream.
    ///
    /// If the seek operation completed successfully, this method returns the new point position
    /// from the start of the underlying stream.
    fn seek_point(&mut self, position: SeekFrom) -> Result<usize>;
    /// Returns the index of the current point in the underlying stream. This is equivalent to
    /// calling `seek_point(SeekFrom::Current(0))`.
    fn point_index(&mut self) -> Result<usize> {
        self.seek_point(SeekFrom::Current(0))
    }
    /// Returns the total number of points in the underlying stream.
    fn point_count(&mut self) -> Result<usize> {
        let current_pos = self.point_index()? as u64;
        let len = self.seek_point(SeekFrom::End(0))?;
        self.seek_point(SeekFrom::Start(current_pos))?;
        Ok(len)
    }
}
