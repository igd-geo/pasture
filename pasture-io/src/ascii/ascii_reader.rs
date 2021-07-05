use anyhow::Result;
use pasture_core::meta::Metadata;
use pasture_core::{containers::PointBufferWriteable, layout::PointLayout};
use std::io::{BufRead, BufReader};
use std::io::{Read, Seek};
use std::path::Path;
use std::fs::File;

use crate::base::PointReader;
use crate::ascii::RawAsciiReader;


/// `PointReader` implementation for ascii files
pub struct AsciiReader<'a>{
    raw_reader: Box<dyn PointReader + 'a>,
    
}
impl<'a> AsciiReader<'a> {
    pub fn from_read<R: Read + Seek + Send + BufRead + 'a>(read: R, format: &str, delimiter: &str) -> Result<Self> {
        let raw_reader: Box<dyn PointReader> = Box::new(RawAsciiReader::from_read(read, format, delimiter)?);
        Ok(Self {
            raw_reader: raw_reader,
        })
    }
    pub fn from_path<P: AsRef<Path>>(path: P, format: &str, delimiter: &str) -> Result<Self> {
        let file = BufReader::new(File::open(path)?);
        Self::from_read(file, format, delimiter)
    }

    pub fn print_format_literals(){
        println!("The following literals can be interpreted from this AsciiReader:
            s - skip this number
            x - x coordinate
            y - y coordinate
            z - z coordinate
            i - intensity
            n - number of returns of given pulse,
            r - ReturnNumber,
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
            a - scan angle rank");
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