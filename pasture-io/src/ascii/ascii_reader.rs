use anyhow::Result;
use pasture_core::meta::Metadata;
use pasture_core::{containers::PointBufferWriteable, layout::PointLayout};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use crate::ascii::RawAsciiReader;
use crate::base::PointReader;

/// `PointReader` implementation for ascii files

pub struct AsciiReader<'a> {
    raw_reader: Box<dyn PointReader + 'a>,
}
impl<'a> AsciiReader<'a> {
    /// Creates a new `AsciiReader` by opening the file at the given `path`.
    /// The `delimiter` string slice is the column seperation pattern.
    /// The `format` string slice coordinates the interpretation of each column.
    /// This functions just wraps a `BufReader` around a `File` and uses [`AsciiReader::from_read`].
    /// For more information see [`AsciiReader::from_read`].
    ///
    /// # Examples
    /// ```no_run
    /// use std::path::Path;
    /// use anyhow::Result;
    /// use pasture_io::ascii:AsciiReader;
    /// fn main() -> Result<()> {
    ///     let path = Path::new("foo.txt");
    ///     let reader = AsciiReader::from_path(path, "xyzie", ", ")?;
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// If `path` does not exist, cannot be opened or does not point to a valid file, an error is returned.
    ///
    /// If `format` contains unrecoginzed literals, an error is returned.
    pub fn from_path<P: AsRef<Path>>(path: P, format: &str, delimiter: &str) -> Result<Self> {
        let file = BufReader::new(File::open(path)?);
        Self::from_read(file, format, delimiter)
    }

    /// Creates a new `AsciiReader` from the given `read`.
    /// The `delimiter` string slice is the column seperation pattern.
    /// The `format` string slice coordinates the interpretation of each column.
    /// The following literals can be interpreted from `AsciiReader`:
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
    /// For the following data an AsciiReader should be created. The first three columns correspondens
    /// to the position attribute. The 4th column is the intensity attribute.
    /// and the 5th is the edge_of_line_flag.
    ///
    /// ```no_run
    /// use std::io::{BufReader};
    /// use anyhow::Result;
    /// use pasture_io::ascii::AsciiReader;
    /// fn main() -> Result<()> {
    ///     let data = "0.0, 1.0, 2.0, 11, 0\n1.0, -2.0, 2.0, 22, 0".as_bytes();
    ///     let read = BufReader::new(data);
    ///     let reader = AsciiReader::from_read(read, "xyzie", ", ")?;
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// If the given `Read` does not represent a valid file, an error is returned.
    ///
    /// If `format` contains unrecoginzed literals, an error is returned.
    pub fn from_read<R: Read + Send + BufRead + 'a>(
        read: R,
        format: &str,
        delimiter: &str,
    ) -> Result<Self> {
        let raw_reader: Box<dyn PointReader> =
            Box::new(RawAsciiReader::from_read(read, format, delimiter)?);
        Ok(Self {
            raw_reader: raw_reader,
        })
    }

    pub fn print_format_literals() {
        println!(
            "The following literals can be interpreted from this AsciiReader:
            s - skip this number
            x - x coordinate
            y - y coordinate
            z - z coordinate
            i - intensity
            n - number of returns of given pulse
            r - ReturnNumber
            c - classification
            t - gps time
            u - user data
            p - point source ID
            R - red channel of RGB color
            G - green channel of RGB color
            B - blue channel of RGB color
            I - NIR channel
            e - edge of flight line flag
            d - direction of scan flag
            a - scan angle rank"
        );
    }
}

impl<'a> PointReader for AsciiReader<'a> {
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
