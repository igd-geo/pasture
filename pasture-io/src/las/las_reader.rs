use std::{
    fs::File,
    io::{BufReader, Read, Seek},
};
use std::{io::SeekFrom, path::Path};

use anyhow::{anyhow, Result};

use crate::base::{PointReader, SeekToPoint};
use pasture_core::{containers::PointBufferWriteable, layout::PointLayout, meta::Metadata};

use super::{LASReaderBase, RawLASReader, RawLAZReader};

fn path_is_compressed_las_file<P: AsRef<Path>>(path: P) -> Result<bool> {
    path.as_ref()
        .extension()
        .map(|extension| extension == "laz")
        .ok_or(anyhow!(
            "Could not determine file extension of file {}",
            path.as_ref().display()
        ))
}

trait AnyLASReader: PointReader + SeekToPoint + LASReaderBase {}

impl<T: PointReader + SeekToPoint + LASReaderBase> AnyLASReader for T {}

/// `PointReader` implementation for LAS/LAZ files
pub struct LASReader<'a> {
    raw_reader: Box<dyn AnyLASReader + 'a>,
}

impl<'a> LASReader<'a> {
    // TODO LAS files store 32-bit integer coordinates in local space internally, but we almost always want
    // floating point values in world space instead. The conversion from integer to float in world space will
    // have to happen automatically during reading, and the default datatype of attribute POSITION_3D will be
    // Vector3<f64>. However, we should make it possible to switch this default conversion off for cases where
    // we want to read just the raw i32 data!

    /// Creates a new `LASReader` by opening the file at the given `path`. Tries to determine whether
    /// the file is compressed from the file extension (i.e. files with extension `.laz` are assumed to be
    /// compressed).
    ///
    /// # Errors
    ///
    /// If `path` does not exist, cannot be opened or does not point to a valid LAS/LAZ file, an error is returned.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let is_compressed = path_is_compressed_las_file(path.as_ref())?;
        let file = BufReader::new(File::open(path)?);
        Self::from_read(file, is_compressed)
    }

    /// Creates a new `LASReader` from the given `read`. This method has to know whether
    /// the `read` points to a compressed LAZ file or a regular LAS file.
    ///
    /// # Errors
    ///
    /// If the given `Read` does not represent a valid LAS/LAZ file, an error is returned.
    pub fn from_read<R: Read + Seek + Send + 'a>(read: R, is_compressed: bool) -> Result<Self> {
        let raw_reader: Box<dyn AnyLASReader> = if is_compressed {
            Box::new(RawLAZReader::from_read(read)?)
        } else {
            Box::new(RawLASReader::from_read(read)?)
        };
        Ok(Self {
            raw_reader: raw_reader,
        })
    }

    pub fn remaining_points(&mut self) -> usize {
        self.raw_reader.remaining_points()
    }
}

impl<'a> PointReader for LASReader<'a> {
    fn read(&mut self, count: usize) -> Result<Box<dyn pasture_core::containers::PointBuffer>> {
        self.raw_reader.read(count)
    }

    fn read_into(
        &mut self,
        point_buffer: &mut dyn PointBufferWriteable,
        count: usize,
    ) -> Result<usize> {
        self.raw_reader.read_into(point_buffer, count)
    }

    fn get_metadata(&self) -> &dyn Metadata {
        self.raw_reader.get_metadata()
    }

    fn get_default_point_layout(&self) -> &PointLayout {
        self.raw_reader.get_default_point_layout()
    }
}

impl<'a> SeekToPoint for LASReader<'a> {
    fn seek_point(&mut self, position: SeekFrom) -> Result<usize> {
        self.raw_reader.seek_point(position)
    }
}
