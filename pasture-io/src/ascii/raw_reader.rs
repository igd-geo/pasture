use anyhow::{bail, Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use itertools::{EitherOrBoth::*, Itertools};
use pasture_core::layout::attributes;
use pasture_core::meta::Metadata;
use pasture_core::{containers::PointBufferWriteable, layout::PointLayout};
use pasture_core::{
    containers::{InterleavedVecPointStorage, PointBuffer},
    layout::PointAttributeDefinition,
};
use std::io::BufRead;
use std::io::{Read, Seek};
use std::str::FromStr;

use super::AsciiMetadata;
use pasture_core::containers::UntypedPoint;
use crate::base::PointReader;

pub(crate) struct RawAsciiReader<T: Read + Seek + BufRead> {
    reader: T,
    metadata: AsciiMetadata,
    delimiter: String,
    point_layout: PointLayout,
    parse_layout: Vec<PointDataTypes>,
}

impl<T: Read + Seek + BufRead> RawAsciiReader<T> {
    pub fn from_read(read: T, format: &str, delimiter: &str) -> Result<Self> {
        let parse_layout = Self::get_parse_layout(format)?;
        let layout = Self::get_point_layout_from_parse_layout(&parse_layout);
        let metadata = AsciiMetadata::new();

        Ok(Self {
            reader: read,
            metadata: metadata,
            delimiter: delimiter.to_string(),
            point_layout: layout,
            parse_layout: parse_layout,
        })
    }

    fn get_point_layout_from_parse_layout(parse_layout: &[PointDataTypes]) -> PointLayout {
        let mut attributes = Vec::<PointAttributeDefinition>::new();
        parse_layout
            .iter()
            .filter(|data_type| !matches!(data_type, PointDataTypes::Skip))
            .map(|data_type| match data_type {
                PointDataTypes::CoordinateX
                | PointDataTypes::CoordinateY
                | PointDataTypes::CoordinateZ => attributes::POSITION_3D,
                PointDataTypes::Intensity => attributes::INTENSITY,
                PointDataTypes::ReturnNumber => attributes::RETURN_NUMBER,
                PointDataTypes::NumberOfReturns => attributes::NUMBER_OF_RETURNS,
                PointDataTypes::Classification => attributes::CLASSIFICATION,
                PointDataTypes::UserData => attributes::USER_DATA,
                PointDataTypes::ColorR | PointDataTypes::ColorG | PointDataTypes::ColorB => {
                    attributes::COLOR_RGB
                }
                PointDataTypes::GpsTime => attributes::GPS_TIME,
                PointDataTypes::PointSourceID => attributes::POINT_SOURCE_ID,
                PointDataTypes::EdgeOfFlightLine => attributes::EDGE_OF_FLIGHT_LINE,
                PointDataTypes::ScanDirectionFlag => attributes::SCAN_DIRECTION_FLAG,
                PointDataTypes::ScanAngleRank => attributes::SCAN_ANGLE_RANK,
                PointDataTypes::NIR => attributes::NIR,
                PointDataTypes::Skip => panic!("Skip should be filtered"),
            })
            .for_each(|attribute| attributes.push_unique(attribute));
        PointLayout::from_attributes(&attributes)
    }
    fn get_point<'layout>(
        line: &str,
        delimiter: &str,
        parse_layout: &[PointDataTypes],
        layout: &'layout PointLayout,
    ) -> Result<UntypedPoint<'layout>> {
        let mut point = UntypedPoint::new(layout);
        for pair in line.split(delimiter).zip_longest(parse_layout) {
            match pair {
                Both(value_str, data_type) => match data_type {
                    PointDataTypes::CoordinateX => {
                        Self::parse_to_point_f64(
                            &mut point,
                            &attributes::POSITION_3D,
                            0,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'x'))?;
                    }
                    PointDataTypes::CoordinateY => {
                        Self::parse_to_point_f64(
                            &mut point,
                            &attributes::POSITION_3D,
                            std::mem::size_of::<f64>() as u64,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'y'))?;
                    }
                    PointDataTypes::CoordinateZ => {
                        Self::parse_to_point_f64(
                            &mut point,
                            &attributes::POSITION_3D,
                            2 * std::mem::size_of::<f64>() as u64,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'z'))?;
                    }
                    PointDataTypes::Intensity => {
                        Self::parse_to_point_u16(&mut point, &attributes::INTENSITY, 0, value_str)
                            .with_context(|| generate_parse_error(data_type, 'i'))?;
                    }
                    PointDataTypes::ReturnNumber => {
                        Self::parse_to_point_u8(
                            &mut point,
                            &attributes::RETURN_NUMBER,
                            0,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'r'))?;
                    }
                    PointDataTypes::NumberOfReturns => {
                        Self::parse_to_point_u8(
                            &mut point,
                            &attributes::NUMBER_OF_RETURNS,
                            0,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'n'))?;
                    }
                    PointDataTypes::Classification => {
                        Self::parse_to_point_u8(
                            &mut point,
                            &attributes::CLASSIFICATION,
                            0,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'c'))?;
                    }
                    PointDataTypes::UserData => {
                        Self::parse_to_point_u8(&mut point, &attributes::USER_DATA, 0, value_str)
                            .with_context(|| generate_parse_error(data_type, 'u'))?;
                    }
                    PointDataTypes::ColorR => {
                        Self::parse_to_point_u16(&mut point, &attributes::COLOR_RGB, 0, value_str)
                            .with_context(|| generate_parse_error(data_type, 'R'))?;
                    }
                    PointDataTypes::ColorG => {
                        Self::parse_to_point_u16(
                            &mut point,
                            &attributes::COLOR_RGB,
                            std::mem::size_of::<u16>() as u64,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'G'))?;
                    }
                    PointDataTypes::ColorB => {
                        Self::parse_to_point_u16(
                            &mut point,
                            &attributes::COLOR_RGB,
                            2 * std::mem::size_of::<u16>() as u64,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'B'))?;
                    }
                    PointDataTypes::GpsTime => {
                        Self::parse_to_point_f64(&mut point, &attributes::GPS_TIME, 0, value_str)
                            .with_context(|| generate_parse_error(data_type, 't'))?;
                    }
                    PointDataTypes::PointSourceID => {
                        Self::parse_to_point_u16(
                            &mut point,
                            &attributes::POINT_SOURCE_ID,
                            0,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'p'))?;
                    }
                    PointDataTypes::EdgeOfFlightLine => {
                        Self::parse_to_point_bool(
                            &mut point,
                            &attributes::EDGE_OF_FLIGHT_LINE,
                            0,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'e'))?;
                    }
                    PointDataTypes::ScanDirectionFlag => {
                        Self::parse_to_point_bool(
                            &mut point,
                            &attributes::SCAN_DIRECTION_FLAG,
                            0,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'd'))?;
                    }
                    PointDataTypes::ScanAngleRank => {
                        Self::parse_to_point_i8(
                            &mut point,
                            &attributes::SCAN_ANGLE_RANK,
                            0,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'a'))?;
                    }
                    PointDataTypes::NIR => {
                        Self::parse_to_point_u16(&mut point, &attributes::NIR, 0, value_str)
                            .with_context(|| generate_parse_error(data_type, 'I'))?;
                    }
                    PointDataTypes::Skip => {}
                },
                Left(_) => continue,
                Right(_) => {
                    bail!("Input format string expected more items in the line. Found End-of-Line.")
                }
            }
        }

        Ok(point)
    }

    fn parse_to_point_i8(
        point: &mut UntypedPoint,
        attribute: &PointAttributeDefinition,
        offset: u64,
        value_str: &str,
    ) -> Result<()> {
        let data = Self::parse_string::<i8>(value_str)?;
        let attribute_offset = point.get_offset_from_attribute(attribute);
        match attribute_offset {
            Some(attribute_offset) => {
                let mut cursor = point.get_buffer_cursor();
                cursor.set_position(attribute_offset + offset);
                cursor.write_i8(data)?;
            }
            None => (),
        }
        Ok(())
    }

    fn parse_to_point_bool(
        point: &mut UntypedPoint,
        attribute: &PointAttributeDefinition,
        offset: u64,
        value_str: &str,
    ) -> Result<()> {
        let data = Self::parse_string::<u8>(value_str)?;
        if !(data == 0 || data == 1) {
            bail!("ParseError expected bool found '{}'.", data);
        }
        let attribute_offset = point.get_offset_from_attribute(attribute);
        match attribute_offset {
            Some(attribute_offset) => {
                let mut cursor = point.get_buffer_cursor();
                cursor.set_position(attribute_offset + offset);
                cursor.write_u8(data)?;
            }
            None => (),
        }
        Ok(())
    }

    fn parse_to_point_u8(
        point: &mut UntypedPoint,
        attribute: &PointAttributeDefinition,
        offset: u64,
        value_str: &str,
    ) -> Result<()> {
        let data = Self::parse_string::<u8>(value_str)?;
        let attribute_offset = point.get_offset_from_attribute(attribute);
        match attribute_offset {
            Some(attribute_offset) => {
                let mut cursor = point.get_buffer_cursor();
                cursor.set_position(attribute_offset + offset);
                cursor.write_u8(data)?;
            }
            None => (),
        }
        Ok(())
    }

    fn parse_to_point_u16(
        point: &mut UntypedPoint,
        attribute: &PointAttributeDefinition,
        offset: u64,
        value_str: &str,
    ) -> Result<()> {
        let data = Self::parse_string::<u16>(value_str)?;
        let attribute_offset = point.get_offset_from_attribute(attribute);
        match attribute_offset {
            Some(attribute_offset) => {
                let mut cursor = point.get_buffer_cursor();
                cursor.set_position(attribute_offset + offset);
                cursor.write_u16::<LittleEndian>(data)?;
            }
            None => (),
        }
        Ok(())
    }

    fn parse_to_point_f64(
        point: &mut UntypedPoint,
        attribute: &PointAttributeDefinition,
        offset: u64,
        value_str: &str,
    ) -> Result<()> {
        let data = Self::parse_string::<f64>(value_str)?;
        let attribute_offset = point.get_offset_from_attribute(attribute);
        match attribute_offset {
            Some(attribute_offset) => {
                let mut cursor = point.get_buffer_cursor();
                cursor.set_position(attribute_offset + offset);
                cursor.write_f64::<LittleEndian>(data)?;
            }
            None => (),
        }
        Ok(())
    }

    fn parse_string<V: FromStr>(value_str: &str) -> Result<V, anyhow::Error> {
        value_str.parse::<V>()
            .map_err(|_| anyhow::anyhow!("ParseError expected {} found '{}'.", std::any::type_name::<V>(), value_str))
    }

    //from LAStools
    //s - skip this number
    //x - x coordinate
    //y - y coordinate
    //z - z coordinate
    //i - intensity
    //n - number of returns of given pulse,
    //r - ReturnNumber,
    //c - classification
    //t - gps time
    //u - user data
    //p - point source ID
    //R - red channel of RGB color
    //G - green channel of RGB color
    //B - blue channel of RGB color
    //I - NIR channel
    //e - edge of flight line flag
    //d - direction of scan flag
    //a - scan angle rank
    fn get_parse_layout(format: &str) -> Result<Vec<PointDataTypes>> {
        let mut parse_layout = Vec::<PointDataTypes>::new();
        for character in format.chars() {
            match character {
                's' => parse_layout.push(PointDataTypes::Skip),
                'x' => parse_layout.push(PointDataTypes::CoordinateX),
                'y' => parse_layout.push(PointDataTypes::CoordinateY),
                'z' => parse_layout.push(PointDataTypes::CoordinateZ),
                'i' => parse_layout.push(PointDataTypes::Intensity),
                'n' => parse_layout.push(PointDataTypes::NumberOfReturns),
                'r' => parse_layout.push(PointDataTypes::ReturnNumber),
                'c' => parse_layout.push(PointDataTypes::Classification),
                't' => parse_layout.push(PointDataTypes::GpsTime),
                'u' => parse_layout.push(PointDataTypes::UserData),
                'p' => parse_layout.push(PointDataTypes::PointSourceID),
                'R' => parse_layout.push(PointDataTypes::ColorR),
                'G' => parse_layout.push(PointDataTypes::ColorG),
                'B' => parse_layout.push(PointDataTypes::ColorB),
                'I' => parse_layout.push(PointDataTypes::NIR),
                'a' => parse_layout.push(PointDataTypes::ScanAngleRank),
                'e' => parse_layout.push(PointDataTypes::EdgeOfFlightLine),
                'd' => parse_layout.push(PointDataTypes::ScanDirectionFlag),
                _ => {
                    bail!(
                        "FormatError can't interpret format literal '{}' in format string '{}'.",
                        character,
                        format
                    );
                }
            }
        }
        Ok(parse_layout)
    }
}

// This enum maps the different entrys on an ascii file to later map these entries to the corresponding attribute.
#[derive(Debug)]
enum PointDataTypes {
    Skip,
    CoordinateX,
    CoordinateY,
    CoordinateZ,       //Vec3f64
    Intensity,         //U16
    ReturnNumber,      //U8
    NumberOfReturns,   //U8
    Classification,    //U8
    UserData,          //U8
    ColorR,            //U16
    ColorG,            //U16
    ColorB,            //U16
    GpsTime,           //F64
    PointSourceID,     // U16
    EdgeOfFlightLine,  //bool
    ScanDirectionFlag, //bool
    ScanAngleRank,     //I8
    NIR,               //U16
}

impl std::fmt::Display for PointDataTypes {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
fn generate_parse_error(datatype: &PointDataTypes, character: char) -> String {
    format!(
        "ParseError at parsing {} for format literal '{}'.",
        datatype.to_string(),
        character
    )
}

// To insert only points that dont exist in the vector already.
// Only requires the PartialEq trait.
trait VecPushUnique<T: PartialEq>: Sized {
    fn push_unique(&mut self, item: T);
}

impl<T: PartialEq> VecPushUnique<T> for Vec<T> {
    fn push_unique(&mut self, item: T) {
        if !self.contains(&item) {
            self.push(item);
        }
    }
}

impl<T: Read + Seek + BufRead> PointReader for RawAsciiReader<T> {
    fn read(&mut self, count: usize) -> Result<Box<dyn PointBuffer>> {
        let mut buffer =
            InterleavedVecPointStorage::with_capacity(count, self.point_layout.clone());
        self.read_into(&mut buffer, count)?;
        Ok(Box::new(buffer))
    }
    fn read_into(
        &mut self,
        point_buffer: &mut dyn PointBufferWriteable,
        count: usize,
    ) -> Result<usize> {
        let layout = point_buffer.point_layout().clone();
        //read line by line
        for (index, line) in (&mut self.reader).lines().take(count).enumerate() {
            let line = line?;
            //convert to a untypedpoint
            let point = Self::get_point(&line, &self.delimiter, &self.parse_layout, &layout)
                .with_context(|| format!("ReadError in line {}.", index))?;
            //put it in the buffer
            point_buffer.push(&point.get_interlieved_point_view());
        }
        Ok(count)
    }
    fn get_default_point_layout(&self) -> &PointLayout {
        &self.point_layout
    }
    fn get_metadata(&self) -> &dyn Metadata {
        &self.metadata
    }
}

// Ascii Tests
//  - Reading
//      - `read` has to be correct.
//      - `read_into` has to be correct for a buffer with the same layout.
//      - `read_into` has to be correct for a buffer with a different layout.
//  - Errors
//      - Unrecognized format literal
//      - To many format literals
//      - Float parsing error
//      - Integer parsing error
//      - Bool parsing error
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ascii::{get_test_file_path, test_data_positions};
    use anyhow::Result;
    use nalgebra::Vector3;
    use pasture_core::containers::PointBufferExt;
    use pasture_core::layout::{attributes, PointType};
    use pasture_derive::PointType;
    use std::{fs::File, io::BufReader};

    #[test]
    fn test_read() -> Result<()> {
        let path = get_test_file_path("10_points_ascii.txt");
        let reader = BufReader::new(File::open(path)?);
        let mut ascii_reader = RawAsciiReader::from_read(reader, "xyz", ", ")?;
        let buffer = ascii_reader.read(10)?;
        let interleaved_buffer = buffer
            .as_interleaved()
            .ok_or_else(|| anyhow::anyhow!("DowncastError"))?;

        //Check 3d Positions
        let position3d_expected = test_data_positions();
        for (index, position) in interleaved_buffer
            .iter_attribute::<Vector3<f64>>(&attributes::POSITION_3D)
            .enumerate()
        {
            assert_eq!(position, position3d_expected[index]);
        }

        Ok(())
    }

    #[repr(C)]
    #[derive(PointType, Debug)]
    struct TestPointSame {
        #[pasture(BUILTIN_POSITION_3D)]
        pub position: Vector3<f64>,
        #[pasture(BUILTIN_INTENSITY)]
        pub intensity: u16,
        #[pasture(BUILTIN_EDGE_OF_FLIGHT_LINE)]
        pub edge_of_flight_line: bool,
        #[pasture(BUILTIN_COLOR_RGB)]
        pub color_rgb: Vector3<u16>,
    }
    #[test]
    fn test_read_into_same_layout() -> Result<()> {
        let path = get_test_file_path("10_points_ascii.txt");
        let reader = BufReader::new(File::open(path)?);
        let mut ascii_reader = RawAsciiReader::from_read(reader, "xyzieRGB", ", ")?;
        let mut buffer = InterleavedVecPointStorage::new(TestPointSame::layout());
        ascii_reader.read_into(&mut buffer, 10)?;

        //Check 3d Positions
        let position3d_expected = test_data_positions();
        for (index, point) in buffer.iter_point::<TestPointSame>().enumerate() {
            assert_eq!(point.position, position3d_expected[index]);
        }
        Ok(())
    }

    #[repr(C)]
    #[derive(PointType, Debug)]
    struct TestPointDifferent {
        #[pasture(BUILTIN_POSITION_3D)]
        pub position: Vector3<f64>,
        #[pasture(BUILTIN_USER_DATA)]
        pub intensity: u16,
        #[pasture(BUILTIN_CLASSIFICATION)]
        pub classification: u16,
    }

    #[test]
    fn test_read_into_different_layout() -> Result<()> {
        let path = get_test_file_path("10_points_ascii.txt");
        let reader = BufReader::new(File::open(path)?);
        let mut ascii_reader = RawAsciiReader::from_read(reader, "xyzieRGB", ", ")?;
        let mut buffer = InterleavedVecPointStorage::new(TestPointDifferent::layout());
        ascii_reader.read_into(&mut buffer, 10)?;

        //Check 3d Positions
        let position3d_expected = test_data_positions();
        for (index, point) in buffer.iter_point::<TestPointDifferent>().enumerate() {
            assert_eq!(point.position, position3d_expected[index]);
        }
        Ok(())
    }

    #[test]
    fn test_error_format_unrecognized_literal() -> Result<()> {
        let path = get_test_file_path("10_points_ascii.txt");
        let reader = BufReader::new(File::open(path)?);
        let ascii_reader = RawAsciiReader::from_read(reader, "xyzQ", ", ");
        assert!(ascii_reader.is_err());
        assert!(format!("{:?}", ascii_reader.err().unwrap())
            .contains("FormatError can't interpret format literal"));
        Ok(())
    }

    #[test]
    fn test_error_format_to_many_literal() -> Result<()> {
        let path = get_test_file_path("10_points_ascii.txt");
        let reader = BufReader::new(File::open(path)?);
        let ascii_reader = RawAsciiReader::from_read(reader, "ssssssssx", ", ");
        let buffer = ascii_reader?.read(10);
        assert!(buffer.is_err());
        assert!(format!("{:?}", buffer.err().unwrap())
            .contains("Input format string expected more items in the line"));
        Ok(())
    }

    #[test]
    fn test_error_parse_error_integer() -> Result<()> {
        let path = get_test_file_path("10_points_ascii_parsing_errors.txt");
        let reader = BufReader::new(File::open(path)?);
        let ascii_reader = RawAsciiReader::from_read(reader, "sssi", ", ");
        let buffer = ascii_reader?.read(10);
        assert!(buffer.is_err());
        assert!(format!("{:?}", buffer.err().unwrap())
            .contains("ParseError at parsing Intensity for format literal 'i'"));
        Ok(())
    }
    #[test]
    fn test_error_parse_error_bool() -> Result<()> {
        let path = get_test_file_path("10_points_ascii_parsing_errors.txt");
        let reader = BufReader::new(File::open(path)?);
        let ascii_reader = RawAsciiReader::from_read(reader, "sssse", ", ");
        let buffer = ascii_reader?.read(10);
        assert!(buffer.is_err());
        println!("{:?}", buffer.err().unwrap());
        //assert!(format!("{:?}", buffer.err().unwrap()).contains("ParseError expected bool found"));
        Ok(())
    }
    #[test]
    fn test_error_parse_error_float() -> Result<()> {
        let path = get_test_file_path("10_points_ascii_parsing_errors.txt");
        let reader = BufReader::new(File::open(path)?);
        let ascii_reader = RawAsciiReader::from_read(reader, "x", ", ");
        let buffer = ascii_reader?.read(10);
        assert!(buffer.is_err());
        assert!(format!("{:?}", buffer.err().unwrap())
            .contains("ParseError at parsing CoordinateX for format literal 'x'"));
        Ok(())
    }
}
