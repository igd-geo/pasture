use std::{
    fs::File,
    io::{BufWriter, Seek, Write},
    path::Path,
};

use anyhow::Result;
use pasture_core::{containers::BorrowedBuffer, layout::PointLayout};

use crate::base::PointWriter;

use super::{AsciiFormat, RawAsciiWriter};

/// `PointWriterFormatting` implementation for Ascii files
pub struct AsciiWriter<T: Write + Seek> {
    raw_writer: RawAsciiWriter<T>,
}

impl AsciiWriter<BufWriter<File>> {
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
}

impl<T: Write + Seek> AsciiWriter<T> {
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
    pub fn from_write(write: T, format: &str) -> Result<Self> {
        Ok(Self {
            raw_writer: RawAsciiWriter::from_write(write, format)?,
        })
    }
}

impl<T: Write + Seek> PointWriter for AsciiWriter<T> {
    fn write<'a, B: BorrowedBuffer<'a>>(&mut self, points: &'a B) -> Result<()> {
        self.raw_writer.write(points)
    }

    fn flush(&mut self) -> Result<()> {
        self.raw_writer.flush()
    }

    fn get_default_point_layout(&self) -> &PointLayout {
        self.raw_writer.get_default_point_layout()
    }
}

impl<T: Write + Seek> AsciiFormat for AsciiWriter<T> {
    fn set_delimiter(&mut self, delimiter: &str) {
        self.raw_writer.set_delimiter(delimiter);
    }

    fn set_precision(&mut self, precision: usize) {
        self.raw_writer.set_precision(precision);
    }
}
