use anyhow::{Context, Result};
use pasture_core::containers::BorrowedBuffer;
use pasture_core::containers::MakeBufferFromLayout;
use pasture_core::containers::OwningBuffer;

mod reader;
use std::path::Path;

pub use self::reader::*;

mod writer;
pub use self::writer::*;

mod seek;
pub use self::seek::*;

mod io_factory;
pub use self::io_factory::*;

/// Try to read all points in the given point cloud file. This function uses the default `IOFactory` to determine the
/// file type from the file extension of `path`. If this succeeds, an appropriate reader is created and all points are
/// read into an implementation-defined `PointBuffer` type. If you want to use a specific type of `PointBuffer`, use
/// `read_all_into` instead!
pub fn read_all<'a, B: MakeBufferFromLayout<'a> + OwningBuffer<'a> + 'a, P: AsRef<Path>>(
    path: P,
) -> Result<B> {
    let mut reader = GenericPointReader::open_file(path.as_ref()).context(format!(
        "Could not create appropriate reader for point cloud file {}",
        path.as_ref().display()
    ))?;
    // While not pretty, it is allowed to call `read_into` with a value larger than the total number of points
    // By falling back to `usize::MAX` for unknown point counts, we guarantee that we always read the whole
    // point cloud
    let num_points = reader.point_count().unwrap_or(usize::MAX);
    reader.read::<B>(num_points)
}

/// Try to read all points in the given point cloud file into the given `buffer`. All points are appended to the end of
/// the `buffer`. Otherwise behaves exactly like `read_all`.
pub fn read_all_into<'a, B: OwningBuffer<'a>, P: AsRef<Path>>(
    buffer: &'a mut B,
    path: P,
) -> Result<usize> {
    let mut reader = GenericPointReader::open_file(path.as_ref()).context(format!(
        "Could not create appropriate reader for point cloud file {}",
        path.as_ref().display()
    ))?;
    let num_points = reader.point_count().context(format!(
        "Could not determine number of points in point cloud file {}",
        path.as_ref().display()
    ))?;
    reader.read_into(buffer, num_points)
}

/// Writes all points in the given `buffer` into the file at `path`
pub fn write_all<'a, B: BorrowedBuffer<'a>, P: AsRef<Path>>(buffer: &'a B, path: P) -> Result<()> {
    todo!()

    // let io_factory = IOFactory::default();
    // let mut writer = io_factory
    //     .make_writer(path.as_ref(), buffer.point_layout())
    //     .context(format!(
    //         "Could not create appropriate writer for point cloud file {}",
    //         path.as_ref().display()
    //     ))?;
    // writer.write(buffer).context("Failed to write points")?;
    // writer.flush()
}
