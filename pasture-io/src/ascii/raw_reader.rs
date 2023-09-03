use anyhow::{bail, Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use itertools::{EitherOrBoth::*, Itertools};
use pasture_core::layout::attributes;
use pasture_core::layout::PointAttributeDefinition;
use pasture_core::layout::PointLayout;
use pasture_core::meta::Metadata;
use std::collections::HashSet;
use std::io::{BufRead, Read};
use std::iter::FromIterator;
use std::str::FromStr;

use super::AsciiMetadata;
use super::PointDataType;
use crate::base::PointReader;
use pasture_core::containers::{UntypedPoint, UntypedPointBuffer};

pub(crate) struct RawAsciiReader<T: Read + BufRead> {
    reader: T,
    metadata: AsciiMetadata,
    delimiter: String,
    point_layout: PointLayout,
    parse_layout: Vec<PointDataType>,
}
impl<T: Read + BufRead> RawAsciiReader<T> {
    pub fn from_read(read: T, format: &str, delimiter: &str) -> Result<Self> {
        let parse_layout = PointDataType::get_parse_layout(format)?;
        let layout = Self::get_point_layout_from_parse_layout(&parse_layout);
        let metadata = AsciiMetadata;

        Ok(Self {
            reader: read,
            metadata,
            delimiter: delimiter.to_string(),
            point_layout: layout,
            parse_layout,
        })
    }

    fn get_point_layout_from_parse_layout(parse_layout: &[PointDataType]) -> PointLayout {
        let hashset = parse_layout
            .iter()
            .filter(|data_type| !matches!(data_type, PointDataType::Skip))
            .map(|data_type| match data_type {
                PointDataType::CoordinateX
                | PointDataType::CoordinateY
                | PointDataType::CoordinateZ => attributes::POSITION_3D,
                PointDataType::Intensity => attributes::INTENSITY,
                PointDataType::ReturnNumber => attributes::RETURN_NUMBER,
                PointDataType::NumberOfReturns => attributes::NUMBER_OF_RETURNS,
                PointDataType::Classification => attributes::CLASSIFICATION,
                PointDataType::UserData => attributes::USER_DATA,
                PointDataType::ColorR | PointDataType::ColorG | PointDataType::ColorB => {
                    attributes::COLOR_RGB
                }
                PointDataType::GpsTime => attributes::GPS_TIME,
                PointDataType::PointSourceID => attributes::POINT_SOURCE_ID,
                PointDataType::EdgeOfFlightLine => attributes::EDGE_OF_FLIGHT_LINE,
                PointDataType::ScanDirectionFlag => attributes::SCAN_DIRECTION_FLAG,
                PointDataType::ScanAngleRank => attributes::SCAN_ANGLE_RANK,
                PointDataType::Nir => attributes::NIR,
                PointDataType::Skip => panic!("Skip should be filtered"),
            })
            .collect::<HashSet<_>>();
        PointLayout::from_attributes(&Vec::from_iter(hashset))
    }

    fn parse_point(
        point: &mut UntypedPointBuffer,
        line: &str,
        delimiter: &str,
        parse_layout: &[PointDataType],
    ) -> Result<()> {
        for pair in line.split(delimiter).zip_longest(parse_layout) {
            match pair {
                Both(value_str, data_type) => match data_type {
                    PointDataType::CoordinateX => {
                        Self::parse_to_point_f64(point, &attributes::POSITION_3D, 0, value_str)
                            .with_context(|| generate_parse_error(data_type, 'x'))?;
                    }
                    PointDataType::CoordinateY => {
                        Self::parse_to_point_f64(
                            point,
                            &attributes::POSITION_3D,
                            std::mem::size_of::<f64>() as u64,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'y'))?;
                    }
                    PointDataType::CoordinateZ => {
                        Self::parse_to_point_f64(
                            point,
                            &attributes::POSITION_3D,
                            2 * std::mem::size_of::<f64>() as u64,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'z'))?;
                    }
                    PointDataType::Intensity => {
                        Self::parse_to_point_u16(point, &attributes::INTENSITY, 0, value_str)
                            .with_context(|| generate_parse_error(data_type, 'i'))?;
                    }
                    PointDataType::ReturnNumber => {
                        Self::parse_to_point_u8(point, &attributes::RETURN_NUMBER, 0, value_str)
                            .with_context(|| generate_parse_error(data_type, 'r'))?;
                    }
                    PointDataType::NumberOfReturns => {
                        Self::parse_to_point_u8(
                            point,
                            &attributes::NUMBER_OF_RETURNS,
                            0,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'n'))?;
                    }
                    PointDataType::Classification => {
                        Self::parse_to_point_u8(point, &attributes::CLASSIFICATION, 0, value_str)
                            .with_context(|| generate_parse_error(data_type, 'c'))?;
                    }
                    PointDataType::UserData => {
                        Self::parse_to_point_u8(point, &attributes::USER_DATA, 0, value_str)
                            .with_context(|| generate_parse_error(data_type, 'u'))?;
                    }
                    PointDataType::ColorR => {
                        Self::parse_to_point_u16(point, &attributes::COLOR_RGB, 0, value_str)
                            .with_context(|| generate_parse_error(data_type, 'R'))?;
                    }
                    PointDataType::ColorG => {
                        Self::parse_to_point_u16(
                            point,
                            &attributes::COLOR_RGB,
                            std::mem::size_of::<u16>() as u64,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'G'))?;
                    }
                    PointDataType::ColorB => {
                        Self::parse_to_point_u16(
                            point,
                            &attributes::COLOR_RGB,
                            2 * std::mem::size_of::<u16>() as u64,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'B'))?;
                    }
                    PointDataType::GpsTime => {
                        Self::parse_to_point_f64(point, &attributes::GPS_TIME, 0, value_str)
                            .with_context(|| generate_parse_error(data_type, 't'))?;
                    }
                    PointDataType::PointSourceID => {
                        Self::parse_to_point_u16(point, &attributes::POINT_SOURCE_ID, 0, value_str)
                            .with_context(|| generate_parse_error(data_type, 'p'))?;
                    }
                    PointDataType::EdgeOfFlightLine => {
                        Self::parse_to_point_bool(
                            point,
                            &attributes::EDGE_OF_FLIGHT_LINE,
                            0,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'e'))?;
                    }
                    PointDataType::ScanDirectionFlag => {
                        Self::parse_to_point_bool(
                            point,
                            &attributes::SCAN_DIRECTION_FLAG,
                            0,
                            value_str,
                        )
                        .with_context(|| generate_parse_error(data_type, 'd'))?;
                    }
                    PointDataType::ScanAngleRank => {
                        Self::parse_to_point_i8(point, &attributes::SCAN_ANGLE_RANK, 0, value_str)
                            .with_context(|| generate_parse_error(data_type, 'a'))?;
                    }
                    PointDataType::Nir => {
                        Self::parse_to_point_u16(point, &attributes::NIR, 0, value_str)
                            .with_context(|| generate_parse_error(data_type, 'I'))?;
                    }
                    PointDataType::Skip => {}
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
        point: &mut UntypedPointBuffer,
        attribute: &PointAttributeDefinition,
        offset: u64,
        value_str: &str,
    ) -> Result<()> {
        let data = Self::parse_string::<i8>(value_str)?;
        let attribute_offset = point.get_layout().offset_of(attribute);
        if let Some(attribute_offset) = attribute_offset {
            let mut cursor = point.get_cursor();
            cursor.set_position(attribute_offset + offset);
            cursor.write_i8(data)?;
        }
        Ok(())
    }

    fn parse_to_point_bool(
        point: &mut UntypedPointBuffer,
        attribute: &PointAttributeDefinition,
        offset: u64,
        value_str: &str,
    ) -> Result<()> {
        let data = Self::parse_string::<u8>(value_str)?;
        if !(data == 0 || data == 1) {
            bail!("ParseError expected bool found '{}'.", data);
        }
        let attribute_offset = point.get_layout().offset_of(attribute);
        if let Some(attribute_offset) = attribute_offset {
            let mut cursor = point.get_cursor();
            cursor.set_position(attribute_offset + offset);
            cursor.write_u8(data)?;
        }
        Ok(())
    }

    fn parse_to_point_u8(
        point: &mut UntypedPointBuffer,
        attribute: &PointAttributeDefinition,
        offset: u64,
        value_str: &str,
    ) -> Result<()> {
        let data = Self::parse_string::<u8>(value_str)?;
        let attribute_offset = point.get_layout().offset_of(attribute);
        if let Some(attribute_offset) = attribute_offset {
            let mut cursor = point.get_cursor();
            cursor.set_position(attribute_offset + offset);
            cursor.write_u8(data)?;
        }
        Ok(())
    }

    fn parse_to_point_u16(
        point: &mut UntypedPointBuffer,
        attribute: &PointAttributeDefinition,
        offset: u64,
        value_str: &str,
    ) -> Result<()> {
        let data = Self::parse_string::<u16>(value_str)?;
        let attribute_offset = point.get_layout().offset_of(attribute);
        if let Some(attribute_offset) = attribute_offset {
            let mut cursor = point.get_cursor();
            cursor.set_position(attribute_offset + offset);
            cursor.write_u16::<LittleEndian>(data)?;
        }
        Ok(())
    }

    fn parse_to_point_f64(
        point: &mut UntypedPointBuffer,
        attribute: &PointAttributeDefinition,
        offset: u64,
        value_str: &str,
    ) -> Result<()> {
        let data = Self::parse_string::<f64>(value_str)?;
        let attribute_offset = point.get_layout().offset_of(attribute);
        if let Some(attribute_offset) = attribute_offset {
            let mut cursor = point.get_cursor();
            cursor.set_position(attribute_offset + offset);
            cursor.write_f64::<LittleEndian>(data)?;
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
}

fn generate_parse_error(datatype: &PointDataType, character: char) -> String {
    format!(
        "ParseError at parsing {} for format literal '{}'.",
        datatype, character
    )
}

impl<T: Read + BufRead> PointReader for RawAsciiReader<T> {
    fn read_into<'a, 'b, B: pasture_core::containers::OwningBuffer<'a>>(
        &mut self,
        point_buffer: &'b mut B,
        count: usize,
    ) -> Result<usize>
    where
        'a: 'b,
    {
        let layout = point_buffer.point_layout().clone();
        let mut temp_point = UntypedPointBuffer::new(&layout);
        //read line by line
        for (index, line) in (&mut self.reader).lines().take(count).enumerate() {
            let line = line?;
            //parse the line in an untypedpoint
            Self::parse_point(&mut temp_point, &line, &self.delimiter, &self.parse_layout)
                .with_context(|| format!("ReadError in line {}.", index))?;
            //put it in the buffer
            unsafe {
                point_buffer.push_points(temp_point.get_buffer());
            }
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
    use pasture_core::containers::{BorrowedBuffer, MakeBufferFromLayout, VectorBuffer};
    use pasture_core::layout::{attributes, PointType};
    use pasture_core::nalgebra::Vector3;
    use pasture_derive::PointType;
    use std::{fs::File, io::BufReader};

    #[test]
    fn test_read() -> Result<()> {
        let path = get_test_file_path("10_points_ascii_all_attributes.txt");
        let reader = BufReader::new(File::open(path)?);
        let mut ascii_reader = RawAsciiReader::from_read(reader, "xyzirncuRGBtpedaI", ", ")?;
        let interleaved_buffer = ascii_reader.read::<VectorBuffer>(10)?;

        let positions = interleaved_buffer
            .view_attribute::<Vector3<f64>>(&attributes::POSITION_3D)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_positions(), positions);

        let intensities = interleaved_buffer
            .view_attribute::<u16>(&attributes::INTENSITY)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_intensities(), intensities);

        let return_numbers = interleaved_buffer
            .view_attribute::<u8>(&attributes::RETURN_NUMBER)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_return_numbers(), return_numbers);

        let number_of_returns = interleaved_buffer
            .view_attribute::<u8>(&attributes::NUMBER_OF_RETURNS)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_number_of_returns(), number_of_returns);

        let classifications = interleaved_buffer
            .view_attribute::<u8>(&attributes::CLASSIFICATION)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_classifications(), classifications);

        let user_datas = interleaved_buffer
            .view_attribute::<u8>(&attributes::USER_DATA)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_user_data(), user_datas);

        let colors = interleaved_buffer
            .view_attribute::<Vector3<u16>>(&attributes::COLOR_RGB)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_colors(), colors);

        let gps_times = interleaved_buffer
            .view_attribute::<f64>(&attributes::GPS_TIME)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_gps_times(), gps_times);

        let point_source_ids = interleaved_buffer
            .view_attribute::<u16>(&attributes::POINT_SOURCE_ID)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_point_source_ids(), point_source_ids);

        let edge_of_flight_lines = interleaved_buffer
            .view_attribute::<u8>(&attributes::EDGE_OF_FLIGHT_LINE)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_edge_of_flight_lines(), edge_of_flight_lines);

        let scan_direction_flags = interleaved_buffer
            .view_attribute::<u8>(&attributes::SCAN_DIRECTION_FLAG)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_scan_direction_flags(), scan_direction_flags);

        let scan_angle_ranks = interleaved_buffer
            .view_attribute::<i8>(&attributes::SCAN_ANGLE_RANK)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_scan_angle_ranks(), scan_angle_ranks);

        let nirs = interleaved_buffer
            .view_attribute::<u16>(&attributes::NIR)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_nirs(), nirs);

        Ok(())
    }

    #[repr(C, packed)]
    #[derive(PointType, Debug, Copy, Clone, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
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
        pub edge_of_flight_line: u8,
        #[pasture(BUILTIN_SCAN_DIRECTION_FLAG)]
        pub scan_direction_flag: u8,
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
        let mut buffer = VectorBuffer::new_from_layout(TestPointAll::layout());
        ascii_reader.read_into(&mut buffer, 10)?;

        let positions = buffer
            .view_attribute::<Vector3<f64>>(&attributes::POSITION_3D)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_positions(), positions);

        let intensities = buffer
            .view_attribute::<u16>(&attributes::INTENSITY)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_intensities(), intensities);

        let return_numbers = buffer
            .view_attribute::<u8>(&attributes::RETURN_NUMBER)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_return_numbers(), return_numbers);

        let number_of_returns = buffer
            .view_attribute::<u8>(&attributes::NUMBER_OF_RETURNS)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_number_of_returns(), number_of_returns);

        let classifications = buffer
            .view_attribute::<u8>(&attributes::CLASSIFICATION)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_classifications(), classifications);

        let user_datas = buffer
            .view_attribute::<u8>(&attributes::USER_DATA)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_user_data(), user_datas);

        let colors = buffer
            .view_attribute::<Vector3<u16>>(&attributes::COLOR_RGB)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_colors(), colors);

        let gps_times = buffer
            .view_attribute::<f64>(&attributes::GPS_TIME)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_gps_times(), gps_times);

        let point_source_ids = buffer
            .view_attribute::<u16>(&attributes::POINT_SOURCE_ID)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_point_source_ids(), point_source_ids);

        let edge_of_flight_lines = buffer
            .view_attribute::<u8>(&attributes::EDGE_OF_FLIGHT_LINE)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_edge_of_flight_lines(), edge_of_flight_lines);

        let scan_direction_flags = buffer
            .view_attribute::<u8>(&attributes::SCAN_DIRECTION_FLAG)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_scan_direction_flags(), scan_direction_flags);

        let scan_angle_ranks = buffer
            .view_attribute::<i8>(&attributes::SCAN_ANGLE_RANK)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_scan_angle_ranks(), scan_angle_ranks);

        let nirs = buffer
            .view_attribute::<u16>(&attributes::NIR)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(test_data_nirs(), nirs);
        Ok(())
    }

    #[repr(C, packed)]
    #[derive(PointType, Debug, Copy, Clone, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
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
        let mut buffer = VectorBuffer::new_from_layout(TestPointDifferent::layout());
        ascii_reader.read_into(&mut buffer, 10)?;

        let position_expected = test_data_positions();
        for (index, point) in buffer.view::<TestPointDifferent>().into_iter().enumerate() {
            let position = point.position;
            let classification = point.classification;
            let user_data = point.user_data;
            assert_eq!(position, position_expected[index]);
            assert_eq!(classification, 0);
            assert_eq!(user_data, 0);
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
        ascii_reader.unwrap().read::<VectorBuffer>(10).unwrap();
    }

    #[test]
    #[should_panic(expected = "ParseError at parsing Intensity for format literal 'i'")]
    fn test_error_parse_error_integer() {
        let path = get_test_file_path("10_points_ascii_parsing_errors.txt");
        let reader = BufReader::new(File::open(path).unwrap());
        let ascii_reader = RawAsciiReader::from_read(reader, "sssi", ", ");
        ascii_reader.unwrap().read::<VectorBuffer>(10).unwrap();
    }
    #[test]
    #[should_panic(expected = "ParseError expected bool found")]
    fn test_error_parse_error_bool() {
        let path = get_test_file_path("10_points_ascii_parsing_errors.txt");
        let reader = BufReader::new(File::open(path).unwrap());
        let ascii_reader = RawAsciiReader::from_read(reader, "sssse", ", ");
        ascii_reader.unwrap().read::<VectorBuffer>(10).unwrap();
    }

    #[test]
    #[should_panic(expected = "ParseError at parsing CoordinateX for format literal 'x'")]
    fn test_error_parse_error_float() {
        let path = get_test_file_path("10_points_ascii_parsing_errors.txt");
        let reader = BufReader::new(File::open(path).unwrap());
        let ascii_reader = RawAsciiReader::from_read(reader, "x", ", ");
        ascii_reader.unwrap().read::<VectorBuffer>(10).unwrap();
    }
}
