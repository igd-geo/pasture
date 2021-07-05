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
use std::collections::HashSet;
use std::io::{BufRead, Read};
use std::iter::FromIterator;
use std::str::FromStr;

use super::AsciiMetadata;
use crate::base::PointReader;
use pasture_core::containers::UntypedPoint;

pub(crate) struct RawAsciiReader<T: Read + BufRead> {
    reader: T,
    metadata: AsciiMetadata,
    delimiter: String,
    point_layout: PointLayout,
    parse_layout: Vec<PointDataTypes>,
}

impl<T: Read + BufRead> RawAsciiReader<T> {
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
        let hashset = parse_layout
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
            .collect::<HashSet<_>>();
        PointLayout::from_attributes(&Vec::from_iter(hashset))
    }

    fn parse_point(
        point: &mut UntypedPoint,
        line: &str,
        delimiter: &str,
        parse_layout: &[PointDataTypes],
    ) -> Result<()> {
        for pair in line.split(delimiter).zip_longest(parse_layout) {
            match pair {
                Both(value_str, data_type) => match data_type {
                    PointDataTypes::CoordinateX => {
                        Self::parse_to_point_f64(point, &attributes::POSITION_3D, 0, value_str)
                            .with_context(|| generate_parse_error(data_type, 'x'))?;
                    }
                    PointDataTypes::CoordinateY => {
                        Self::parse_to_point_f64(
                            point,
                            &attributes::POSITION_3D,
                            std::mem::size_of::<f64>() as u64,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'y'))?;
                    }
                    PointDataTypes::CoordinateZ => {
                        Self::parse_to_point_f64(
                            point,
                            &attributes::POSITION_3D,
                            2 * std::mem::size_of::<f64>() as u64,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'z'))?;
                    }
                    PointDataTypes::Intensity => {
                        Self::parse_to_point_u16(point, &attributes::INTENSITY, 0, value_str)
                            .with_context(|| generate_parse_error(data_type, 'i'))?;
                    }
                    PointDataTypes::ReturnNumber => {
                        Self::parse_to_point_u8(point, &attributes::RETURN_NUMBER, 0, value_str)
                            .with_context(|| generate_parse_error(data_type, 'r'))?;
                    }
                    PointDataTypes::NumberOfReturns => {
                        Self::parse_to_point_u8(
                            point,
                            &attributes::NUMBER_OF_RETURNS,
                            0,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'n'))?;
                    }
                    PointDataTypes::Classification => {
                        Self::parse_to_point_u8(point, &attributes::CLASSIFICATION, 0, value_str)
                            .with_context(|| generate_parse_error(data_type, 'c'))?;
                    }
                    PointDataTypes::UserData => {
                        Self::parse_to_point_u8(point, &attributes::USER_DATA, 0, value_str)
                            .with_context(|| generate_parse_error(data_type, 'u'))?;
                    }
                    PointDataTypes::ColorR => {
                        Self::parse_to_point_u16(point, &attributes::COLOR_RGB, 0, value_str)
                            .with_context(|| generate_parse_error(data_type, 'R'))?;
                    }
                    PointDataTypes::ColorG => {
                        Self::parse_to_point_u16(
                            point,
                            &attributes::COLOR_RGB,
                            std::mem::size_of::<u16>() as u64,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'G'))?;
                    }
                    PointDataTypes::ColorB => {
                        Self::parse_to_point_u16(
                            point,
                            &attributes::COLOR_RGB,
                            2 * std::mem::size_of::<u16>() as u64,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'B'))?;
                    }
                    PointDataTypes::GpsTime => {
                        Self::parse_to_point_f64(point, &attributes::GPS_TIME, 0, value_str)
                            .with_context(|| generate_parse_error(data_type, 't'))?;
                    }
                    PointDataTypes::PointSourceID => {
                        Self::parse_to_point_u16(point, &attributes::POINT_SOURCE_ID, 0, value_str)
                            .with_context(|| generate_parse_error(data_type, 'p'))?;
                    }
                    PointDataTypes::EdgeOfFlightLine => {
                        Self::parse_to_point_bool(
                            point,
                            &attributes::EDGE_OF_FLIGHT_LINE,
                            0,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'e'))?;
                    }
                    PointDataTypes::ScanDirectionFlag => {
                        Self::parse_to_point_bool(
                            point,
                            &attributes::SCAN_DIRECTION_FLAG,
                            0,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'd'))?;
                    }
                    PointDataTypes::ScanAngleRank => {
                        Self::parse_to_point_i8(point, &attributes::SCAN_ANGLE_RANK, 0, value_str)
                            .with_context(|| generate_parse_error(data_type, 'a'))?;
                    }
                    PointDataTypes::NIR => {
                        Self::parse_to_point_u16(point, &attributes::NIR, 0, value_str)
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
        Ok(())
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
        value_str.parse::<V>().map_err(|_| {
            anyhow::anyhow!(
                "ParseError expected {} found '{}'.",
                std::any::type_name::<V>(),
                value_str
            )
        })
    }

    //from LAStools
    //s - skip this number
    //x - x coordinate
    //y - y coordinate
    //z - z coordinate
    //i - intensity
    //r - ReturnNumber,
    //n - number of returns of given pulse,
    //c - classification
    //u - user data
    //R - red channel of RGB color
    //G - green channel of RGB color
    //B - blue channel of RGB color
    //t - gps time
    //p - point source ID
    //e - edge of flight line flag
    //d - direction of scan flag
    //a - scan angle rank
    //I - NIR channel
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

impl<T: Read + BufRead> PointReader for RawAsciiReader<T> {
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
        let mut temp_point = UntypedPoint::new(&layout);
        //read line by line
        for (index, line) in (&mut self.reader).lines().take(count).enumerate() {
            let line = line?;
            //parse the line in an untypedpoint
            Self::parse_point(&mut temp_point, &line, &self.delimiter, &self.parse_layout)
                .with_context(|| format!("ReadError in line {}.", index))?;
            //put it in the buffer
            point_buffer.push(&temp_point.get_interlieved_point_view());
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
    use crate::ascii::{
        get_test_file_path, test_data_classifications, test_data_colors,
        test_data_edge_of_flight_lines, test_data_gps_times, test_data_intensities, test_data_nirs,
        test_data_number_of_returns, test_data_point_source_ids, test_data_positions,
        test_data_return_numbers, test_data_scan_angle_ranks, test_data_scan_direction_flags,
        test_data_user_data,
    };
    use anyhow::Result;
    use pasture_core::containers::PointBufferExt;
    use pasture_core::layout::{attributes, PointType};
    use pasture_core::nalgebra::Vector3;
    use pasture_derive::PointType;
    use std::{fs::File, io::BufReader};

    #[test]
    fn test_read() -> Result<()> {
        let path = get_test_file_path("10_points_ascii_all_attributes.txt");
        let reader = BufReader::new(File::open(path)?);
        let mut ascii_reader = RawAsciiReader::from_read(reader, "xyzirncuRGBtpedaI", ", ")?;
        let buffer = ascii_reader.read(10)?;
        let interleaved_buffer = buffer
            .as_interleaved()
            .ok_or_else(|| anyhow::anyhow!("DowncastError"))?;

        let positions = interleaved_buffer
            .iter_attribute::<Vector3<f64>>(&attributes::POSITION_3D)
            .collect::<Vec<_>>();
        assert_eq!(test_data_positions(), positions);

        let intensities = interleaved_buffer
            .iter_attribute::<u16>(&attributes::INTENSITY)
            .collect::<Vec<_>>();
        assert_eq!(test_data_intensities(), intensities);

        let return_numbers = interleaved_buffer
            .iter_attribute::<u8>(&attributes::RETURN_NUMBER)
            .collect::<Vec<_>>();
        assert_eq!(test_data_return_numbers(), return_numbers);

        let number_of_returns = interleaved_buffer
            .iter_attribute::<u8>(&attributes::NUMBER_OF_RETURNS)
            .collect::<Vec<_>>();
        assert_eq!(test_data_number_of_returns(), number_of_returns);

        let classifications = interleaved_buffer
            .iter_attribute::<u8>(&attributes::CLASSIFICATION)
            .collect::<Vec<_>>();
        assert_eq!(test_data_classifications(), classifications);

        let user_datas = interleaved_buffer
            .iter_attribute::<u8>(&attributes::USER_DATA)
            .collect::<Vec<_>>();
        assert_eq!(test_data_user_data(), user_datas);

        let colors = interleaved_buffer
            .iter_attribute::<Vector3<u16>>(&attributes::COLOR_RGB)
            .collect::<Vec<_>>();
        assert_eq!(test_data_colors(), colors);

        let gps_times = interleaved_buffer
            .iter_attribute::<f64>(&attributes::GPS_TIME)
            .collect::<Vec<_>>();
        assert_eq!(test_data_gps_times(), gps_times);

        let point_source_ids = interleaved_buffer
            .iter_attribute::<u16>(&attributes::POINT_SOURCE_ID)
            .collect::<Vec<_>>();
        assert_eq!(test_data_point_source_ids(), point_source_ids);

        let edge_of_flight_lines = interleaved_buffer
            .iter_attribute::<bool>(&attributes::EDGE_OF_FLIGHT_LINE)
            .collect::<Vec<_>>();
        assert_eq!(test_data_edge_of_flight_lines(), edge_of_flight_lines);

        let scan_direction_flags = interleaved_buffer
            .iter_attribute::<bool>(&attributes::SCAN_DIRECTION_FLAG)
            .collect::<Vec<_>>();
        assert_eq!(test_data_scan_direction_flags(), scan_direction_flags);

        let scan_angle_ranks = interleaved_buffer
            .iter_attribute::<i8>(&attributes::SCAN_ANGLE_RANK)
            .collect::<Vec<_>>();
        assert_eq!(test_data_scan_angle_ranks(), scan_angle_ranks);

        let nirs = interleaved_buffer
            .iter_attribute::<u16>(&attributes::NIR)
            .collect::<Vec<_>>();
        assert_eq!(test_data_nirs(), nirs);

        Ok(())
    }

    #[repr(C)]
    #[derive(PointType, Debug)]
    struct TestPointAll {
        #[pasture(BUILTIN_POSITION_3D)]
        pub position: Vector3<f64>,
        #[pasture(BUILTIN_INTENSITY)]
        pub intensity: u16,
        #[pasture(BUILTIN_RETURN_NUMBER)]
        pub return_number: u8,
        #[pasture(BUILTIN_NUMBER_OF_RETURNS)]
        pub number_of_returns: u8,
        #[pasture(BUILTIN_CLASSIFICATION)]
        pub classification: u8,
        #[pasture(BUILTIN_USER_DATA)]
        pub user_data: u8,
        #[pasture(BUILTIN_COLOR_RGB)]
        pub color_rgb: Vector3<u16>,
        #[pasture(BUILTIN_GPS_TIME)]
        pub gps_time: f64,
        #[pasture(BUILTIN_POINT_SOURCE_ID)]
        pub point_source_id: u16,
        #[pasture(BUILTIN_EDGE_OF_FLIGHT_LINE)]
        pub edge_of_flight_line: bool,
        #[pasture(BUILTIN_SCAN_DIRECTION_FLAG)]
        pub scan_direction_flag: bool,
        #[pasture(BUILTIN_SCAN_ANGLE_RANK)]
        pub scan_angle_rank: i8,
        #[pasture(BUILTIN_NIR)]
        pub nir: u16,
    }

    #[test]
    fn test_read_into_same_layout() -> Result<()> {
        let path = get_test_file_path("10_points_ascii_all_attributes.txt");
        let reader = BufReader::new(File::open(path)?);
        let mut ascii_reader = RawAsciiReader::from_read(reader, "xyzirncuRGBtpedaI", ", ")?;
        let mut buffer = InterleavedVecPointStorage::new(TestPointAll::layout());
        ascii_reader.read_into(&mut buffer, 10)?;

        
        let positions = buffer
        .iter_attribute::<Vector3<f64>>(&attributes::POSITION_3D)
        .collect::<Vec<_>>();
        assert_eq!(test_data_positions(), positions);

        let intensities = buffer
            .iter_attribute::<u16>(&attributes::INTENSITY)
            .collect::<Vec<_>>();
        assert_eq!(test_data_intensities(), intensities);

        let return_numbers = buffer
            .iter_attribute::<u8>(&attributes::RETURN_NUMBER)
            .collect::<Vec<_>>();
        assert_eq!(test_data_return_numbers(), return_numbers);

        let number_of_returns = buffer
            .iter_attribute::<u8>(&attributes::NUMBER_OF_RETURNS)
            .collect::<Vec<_>>();
        assert_eq!(test_data_number_of_returns(), number_of_returns);

        let classifications = buffer
            .iter_attribute::<u8>(&attributes::CLASSIFICATION)
            .collect::<Vec<_>>();
        assert_eq!(test_data_classifications(), classifications);

        let user_datas = buffer
            .iter_attribute::<u8>(&attributes::USER_DATA)
            .collect::<Vec<_>>();
        assert_eq!(test_data_user_data(), user_datas);

        let colors = buffer
            .iter_attribute::<Vector3<u16>>(&attributes::COLOR_RGB)
            .collect::<Vec<_>>();
        assert_eq!(test_data_colors(), colors);

        let gps_times = buffer
            .iter_attribute::<f64>(&attributes::GPS_TIME)
            .collect::<Vec<_>>();
        assert_eq!(test_data_gps_times(), gps_times);

        let point_source_ids = buffer
            .iter_attribute::<u16>(&attributes::POINT_SOURCE_ID)
            .collect::<Vec<_>>();
        assert_eq!(test_data_point_source_ids(), point_source_ids);

        let edge_of_flight_lines = buffer
            .iter_attribute::<bool>(&attributes::EDGE_OF_FLIGHT_LINE)
            .collect::<Vec<_>>();
        assert_eq!(test_data_edge_of_flight_lines(), edge_of_flight_lines);

        let scan_direction_flags = buffer
            .iter_attribute::<bool>(&attributes::SCAN_DIRECTION_FLAG)
            .collect::<Vec<_>>();
        assert_eq!(test_data_scan_direction_flags(), scan_direction_flags);

        let scan_angle_ranks = buffer
            .iter_attribute::<i8>(&attributes::SCAN_ANGLE_RANK)
            .collect::<Vec<_>>();
        assert_eq!(test_data_scan_angle_ranks(), scan_angle_ranks);

        let nirs = buffer
            .iter_attribute::<u16>(&attributes::NIR)
            .collect::<Vec<_>>();
        assert_eq!(test_data_nirs(), nirs);
        Ok(())
    }

    #[repr(C)]
    #[derive(PointType, Debug)]
    struct TestPointDifferent {
        #[pasture(BUILTIN_POSITION_3D)]
        pub position: Vector3<f64>,
        #[pasture(BUILTIN_USER_DATA)]
        pub user_data: u16,
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

        let position_expected = test_data_positions();
        for (index, point) in buffer.iter_point::<TestPointDifferent>().enumerate() {
            assert_eq!(point.position, position_expected[index]);
            assert_eq!(point.classification, 0);
            assert_eq!(point.user_data, 0);
        }
        Ok(())
    }

    #[test]
    #[should_panic(expected = "FormatError can't interpret format literal")]
    fn test_error_format_unrecognized_literal() {
        let path = get_test_file_path("10_points_ascii.txt");
        let reader = BufReader::new(File::open(path).unwrap());
        RawAsciiReader::from_read(reader, "xyzQ", ", ").unwrap();
    }

    #[test]
    #[should_panic(expected = "Input format string expected more items in the line")]
    fn test_error_format_to_many_literal() {
        let path = get_test_file_path("10_points_ascii.txt");
        let reader = BufReader::new(File::open(path).unwrap());
        let ascii_reader = RawAsciiReader::from_read(reader, "ssssssssx", ", ");
        ascii_reader.unwrap().read(10).unwrap();
    }

    #[test]
    #[should_panic(expected = "ParseError at parsing Intensity for format literal 'i'")]
    fn test_error_parse_error_integer() {
        let path = get_test_file_path("10_points_ascii_parsing_errors.txt");
        let reader = BufReader::new(File::open(path).unwrap());
        let ascii_reader = RawAsciiReader::from_read(reader, "sssi", ", ");
        ascii_reader.unwrap().read(10).unwrap();
    }
    #[test]
    #[should_panic(expected = "ParseError expected bool found")]
    fn test_error_parse_error_bool() {
        let path = get_test_file_path("10_points_ascii_parsing_errors.txt");
        let reader = BufReader::new(File::open(path).unwrap());
        let ascii_reader = RawAsciiReader::from_read(reader, "sssse", ", ");
        ascii_reader.unwrap().read(10).unwrap();
    }

    #[test]
    #[should_panic(expected = "ParseError at parsing CoordinateX for format literal 'x'")]
    fn test_error_parse_error_float() {
        let path = get_test_file_path("10_points_ascii_parsing_errors.txt");
        let reader = BufReader::new(File::open(path).unwrap());
        let ascii_reader = RawAsciiReader::from_read(reader, "x", ", ");
        ascii_reader.unwrap().read(10).unwrap();
    }
}
