use std::{fs::File, io::BufWriter, path::Path};

use anyhow::Result;
use pasture_core::{containers::PointBuffer, layout::PointLayout};

use crate::base::PointWriter;

use super::{AsciiFormat, PointWriterFormatting, RawAsciiWriter};



/// `PointWriter` implementation for Ascii files
pub struct AsciiWriter {
    raw_writer: Box<dyn PointWriterFormatting>,
}

impl AsciiWriter {
    pub fn from_path<P: AsRef<Path>>(path: P, format: &str) -> Result<Self> {
        let file = BufWriter::new(File::create(path)?);
        Self::from_write(file, format)
    }
    pub fn from_write<T: std::io::Write + std::io::Seek + 'static>(write: T, format: &str) -> Result<Self> {
        let raw_writer: Box<dyn PointWriterFormatting> =
            Box::new(RawAsciiWriter::from_write(write, format)?);
        Ok(Self{
            raw_writer
        })
    }


}

impl PointWriter for AsciiWriter {
    fn write(&mut self, points: &dyn PointBuffer) -> Result<()> {
        self.raw_writer.write(points)
    }

    fn flush(&mut self) -> Result<()> {
        self.raw_writer.flush()
    }

    fn get_default_point_layout(&self) -> &PointLayout {
        self.raw_writer.get_default_point_layout()
    }
}


impl AsciiFormat for AsciiWriter {
    fn set_delimiter(&mut self, delimiter: &str) {
        self.raw_writer.set_delimiter(delimiter);
    }

    fn set_precision(&mut self, precision: usize) {
        self.raw_writer.set_precision(precision);
    }
}

