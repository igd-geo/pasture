#![allow(clippy::large_enum_variant)]

use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use anyhow::{anyhow, Context, Result};
use pasture_core::{containers::BorrowedMutBuffer, layout::PointLayout};

// use crate::las::{LASReader, LASWriter};

use crate::{
    las::{LASReader, LASWriter},
    tiles3d::{PntsReader, PntsWriter},
};

use super::{PointReader, PointWriter, SeekToPoint};

#[derive(Debug)]
enum SupportedFileExtensions {
    Las,
    Tiles3D,
}

/// Returns a lookup value for the file extension of the given file path
fn get_extension_lookup(path: &Path) -> Result<SupportedFileExtensions> {
    let extension = path.extension().ok_or_else(|| {
        anyhow!(
            "File extension could not be determined from path {}",
            path.display()
        )
    })?;
    let extension_str = extension.to_str().ok_or_else(|| {
        anyhow!(
            "File extension of path {} is no valid Unicode string",
            path.display()
        )
    })?;
    match extension_str.to_lowercase().as_str() {
        "las" | "laz" => Ok(SupportedFileExtensions::Las),
        "pnts" => Ok(SupportedFileExtensions::Tiles3D),
        other => Err(anyhow!("Unsupported file extension {other}")),
    }
}

pub enum GenericPointReader {
    LAS(LASReader<'static, BufReader<File>>),
    Tiles3D(PntsReader<BufReader<File>>),
}

impl GenericPointReader {
    pub fn open_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let extension = get_extension_lookup(path.as_ref())?;
        match extension {
            SupportedFileExtensions::Las => {
                let reader = LASReader::from_path(path, false)?;
                Ok(Self::LAS(reader))
            }
            SupportedFileExtensions::Tiles3D => {
                let reader = PntsReader::from_path(path)?;
                Ok(Self::Tiles3D(reader))
            }
        }
    }

    /// Returns the total number of points in the underlying point cloud file. Returns `None` if the number of
    /// points is unknown (e.g. for ASCII files which don't have header information)
    pub fn point_count(&self) -> Option<usize> {
        match self {
            GenericPointReader::LAS(reader) => reader.get_metadata().number_of_points(),
            GenericPointReader::Tiles3D(reader) => reader.get_metadata().number_of_points(),
        }
    }
}

impl PointReader for GenericPointReader {
    fn read_into<'a, 'b, B: BorrowedMutBuffer<'a>>(
        &mut self,
        point_buffer: &'b mut B,
        count: usize,
    ) -> Result<usize>
    where
        'a: 'b,
    {
        match self {
            GenericPointReader::LAS(reader) => reader.read_into(point_buffer, count),
            GenericPointReader::Tiles3D(reader) => reader.read_into(point_buffer, count),
        }
    }

    fn get_metadata(&self) -> &dyn pasture_core::meta::Metadata {
        match self {
            GenericPointReader::LAS(reader) => reader.get_metadata(),
            GenericPointReader::Tiles3D(reader) => reader.get_metadata(),
        }
    }

    fn get_default_point_layout(&self) -> &PointLayout {
        match self {
            GenericPointReader::LAS(reader) => reader.get_default_point_layout(),
            GenericPointReader::Tiles3D(reader) => reader.get_default_point_layout(),
        }
    }
}

impl SeekToPoint for GenericPointReader {
    fn seek_point(&mut self, position: std::io::SeekFrom) -> Result<usize> {
        match self {
            GenericPointReader::LAS(reader) => reader.seek_point(position),
            GenericPointReader::Tiles3D(reader) => reader.seek_point(position),
        }
    }
}

pub enum GenericPointWriter {
    LAS(LASWriter<BufWriter<File>>),
    Tiles3D(PntsWriter<BufWriter<File>>),
}

impl GenericPointWriter {
    pub fn open_file<P: AsRef<Path>>(path: P, point_layout: &PointLayout) -> Result<Self> {
        let extension = get_extension_lookup(path.as_ref())?;
        match extension {
            SupportedFileExtensions::Las => {
                let writer = LASWriter::from_path_and_point_layout(path, point_layout)?;
                Ok(Self::LAS(writer))
            }
            SupportedFileExtensions::Tiles3D => {
                let file = BufWriter::new(File::create(path.as_ref()).context(format!(
                    "Could not open file {} for writing",
                    path.as_ref().display()
                ))?);
                let writer = PntsWriter::from_write_and_layout(file, point_layout.clone());
                Ok(Self::Tiles3D(writer))
            }
        }
    }
}

impl PointWriter for GenericPointWriter {
    fn write<'a, B: pasture_core::containers::BorrowedBuffer<'a>>(
        &mut self,
        points: &'a B,
    ) -> Result<()> {
        match self {
            GenericPointWriter::LAS(writer) => writer.write(points),
            GenericPointWriter::Tiles3D(writer) => writer.write(points),
        }
    }

    fn flush(&mut self) -> Result<()> {
        match self {
            GenericPointWriter::LAS(writer) => writer.flush(),
            GenericPointWriter::Tiles3D(writer) => writer.flush(),
        }
    }

    fn get_default_point_layout(&self) -> &PointLayout {
        match self {
            GenericPointWriter::LAS(writer) => writer.get_default_point_layout(),
            GenericPointWriter::Tiles3D(writer) => writer.get_default_point_layout(),
        }
    }
}