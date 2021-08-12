use std::{fs::File, io::BufWriter, path::Path};

use anyhow::Result;
use pasture_core::{containers::PointBuffer, layout::PointLayout};

use crate::base::PointWriter;

use super::{AsciiFormat, PointWriterFormatting, RawAsciiWriter};



/// `PointWriterFormatting` implementation for Ascii files
pub struct AsciiWriter {
    raw_writer: Box<dyn PointWriterFormatting>,
}

impl AsciiWriter {
    /// Creates a new `AsciiWriter` by opening the file at the given `path`.
    /// The `format` string slice coordinates the interpretation of each column.
    /// This functions just wraps a `BufWriter` around a `File` and uses [`AsciiWriter::from_write`].
    /// For more information see [`AsciiWriter::from_write`].
    ///
    /// # Examples
    /// ```no_run
    /// use std::path::Path;
    /// use anyhow::Result;
    /// use pasture_io::ascii::AsciiWriter;
    /// fn main() -> Result<()> {
    ///     let path = Path::new("output.txt");
    ///     let writer = AsciiWriter::from_path(path, "xyzie")?;
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// If `path` cannot be created or overwritten, an error is returned.
    ///
    /// If `format` contains unrecoginzed literals, an error is returned.
    pub fn from_path<P: AsRef<Path>>(path: P, format: &str) -> Result<Self> {
        let file = BufWriter::new(File::create(path)?);
        Self::from_write(file, format)
    }
    /// Creates a new `AsciiWriter` from the given `write`.
    /// The `format` string slice coordinates the interpretation of each column.
    /// The following literals can be interpreted from `AsciiWriter`:
    /// - s → skip this column  
    /// - x → x coordinate  
    /// - y → y coordinate  
    /// - z → z coordinate  
    /// - i → intensity  
    /// - r → ReturnNumber  
    /// - n → number of returns of given pulse  
    /// - c → classification  
    /// - t → gps time  
    /// - u → user data  
    /// - p → point source ID  
    /// - R → red channel of RGB color  
    /// - G → green channel of RGB color  
    /// - B → blue channel of RGB color  
    /// - e → edge of flight line flag  
    /// - d → direction of scan flag  
    /// - a → scan angle rank
    /// - I → NIR channel  
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io::{BufWriter};
    /// use std::fs::File;
    /// use anyhow::Result;
    /// use pasture_io::ascii::AsciiWriter;
    /// fn main() -> Result<()> {
    ///     let write = BufWriter::new(File::create("output.txt")?);
    ///     let writer = AsciiWriter::from_write(write, "xyzi")?;
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// If the given `Write` cannot write, an error is returned.
    ///
    /// If `format` contains unrecoginzed literals, an error is returned.
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

