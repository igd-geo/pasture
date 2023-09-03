use super::PointDataType;
use crate::base::PointWriter;
use anyhow::{Context, Result};
use pasture_core::containers::{BorrowedBuffer, UntypedPoint, UntypedPointSlice};
use pasture_core::layout::{attributes, PointLayout};
use pasture_core::nalgebra::Vector3;
// combined trait to handle the PointWriter trait aswell as the AsciiFormat trait
pub trait PointWriterFormatting: PointWriter + AsciiFormat {}
pub trait AsciiFormat {
    fn set_delimiter(&mut self, delimiter: &str);
    fn set_precision(&mut self, precision: usize);
}
pub(crate) struct RawAsciiWriter<T: std::io::Write + std::io::Seek> {
    writer: T,
    delimiter: String,
    precision: usize,
    parse_layout: Vec<PointDataType>,
    default_layout: PointLayout,
}

impl<T: std::io::Write + std::io::Seek> RawAsciiWriter<T> {
    pub fn from_write(write: T, format: &str) -> Result<Self> {
        Ok(Self {
            writer: write,
            delimiter: String::from(", "),
            precision: 5,
            parse_layout: PointDataType::get_parse_layout(format)?,
            default_layout: PointLayout::default(),
        })
    }
}

impl<T: std::io::Write + std::io::Seek> AsciiFormat for RawAsciiWriter<T> {
    fn set_delimiter(&mut self, delimiter: &str) {
        self.delimiter = String::from(delimiter);
    }
    fn set_precision(&mut self, precision: usize) {
        self.precision = precision;
    }
}
impl<T: std::io::Write + std::io::Seek> PointWriterFormatting for RawAsciiWriter<T> {}

impl<T: std::io::Write + std::io::Seek> PointWriter for RawAsciiWriter<T> {
    fn write<'a, B: BorrowedBuffer<'a>>(&mut self, points: &'a B) -> anyhow::Result<()> {
        //let point = UntypedPointBuffer::new(&self.default_layout);
        let buffer_layout = points.point_layout();

        // Similar to RawLASReader, write points in chunks of a fixed size to prevent overhead of
        // repeated virtual calls to 'dyn PointBuffer'

        let size_of_single_point = buffer_layout.size_of_point_entry() as usize;
        let num_points_in_chunk = 50_000;
        let num_chunks = (points.len() + (num_points_in_chunk - 1)) / num_points_in_chunk;
        let mut chunk_buffer: Vec<u8> = vec![0; num_points_in_chunk * size_of_single_point];

        for chunk_index in 0..num_chunks {
            let points_in_cur_chunk = std::cmp::min(
                num_points_in_chunk,
                points.len() - (chunk_index * num_points_in_chunk),
            );
            let start_point_index = chunk_index * num_points_in_chunk;
            points.get_point_range(
                start_point_index..(start_point_index + points_in_cur_chunk),
                &mut chunk_buffer[..points_in_cur_chunk * size_of_single_point],
            );

            //Iterate over each point
            for point_index_in_chunk in 0..points_in_cur_chunk {
                let start = point_index_in_chunk * size_of_single_point;
                let end = start + size_of_single_point;
                let point = UntypedPointSlice::new(buffer_layout, &mut chunk_buffer[start..end]);
                //write point
                for (index, format_literal) in self.parse_layout.iter().enumerate() {
                    match format_literal {
                        PointDataType::Skip => {}
                        PointDataType::CoordinateX => {
                            let pos =
                                point.get_attribute::<Vector3<f64>>(&attributes::POSITION_3D)?;
                            self.writer.write_all(
                                trim_unnecessary_tailing_zeros(&format!(
                                    "{:.1$}",
                                    pos.x, self.precision
                                ))
                                .as_bytes(),
                            )?;
                        }
                        PointDataType::CoordinateY => {
                            let pos =
                                point.get_attribute::<Vector3<f64>>(&attributes::POSITION_3D)?;
                            self.writer.write_all(
                                trim_unnecessary_tailing_zeros(&format!(
                                    "{:.1$}",
                                    pos.y, self.precision
                                ))
                                .as_bytes(),
                            )?;
                        }
                        PointDataType::CoordinateZ => {
                            let pos =
                                point.get_attribute::<Vector3<f64>>(&attributes::POSITION_3D)?;
                            self.writer.write_all(
                                trim_unnecessary_tailing_zeros(&format!(
                                    "{:.1$}",
                                    pos.z, self.precision
                                ))
                                .as_bytes(),
                            )?;
                        }
                        PointDataType::Intensity => {
                            let intensity = point.get_attribute::<u64>(&attributes::INTENSITY)?;
                            self.writer.write_all(intensity.to_string().as_bytes())?;
                        }
                        PointDataType::ReturnNumber => {
                            let return_number =
                                point.get_attribute::<u64>(&attributes::RETURN_NUMBER)?;
                            self.writer
                                .write_all(return_number.to_string().as_bytes())?;
                        }
                        PointDataType::NumberOfReturns => {
                            let number_of_returns =
                                point.get_attribute::<u64>(&attributes::NUMBER_OF_RETURNS)?;
                            self.writer
                                .write_all(number_of_returns.to_string().as_bytes())?;
                        }
                        PointDataType::Classification => {
                            let classification =
                                point.get_attribute::<u64>(&attributes::RETURN_NUMBER)?;
                            self.writer
                                .write_all(classification.to_string().as_bytes())?;
                        }
                        PointDataType::UserData => {
                            let classification =
                                point.get_attribute::<u64>(&attributes::RETURN_NUMBER)?;
                            self.writer
                                .write_all(classification.to_string().as_bytes())?;
                        }
                        PointDataType::ColorR => {
                            let color =
                                point.get_attribute::<Vector3<u16>>(&attributes::COLOR_RGB)?;
                            self.writer.write_all(color[0].to_string().as_bytes())?;
                        }
                        PointDataType::ColorG => {
                            let color =
                                point.get_attribute::<Vector3<u16>>(&attributes::COLOR_RGB)?;
                            self.writer.write_all(color[1].to_string().as_bytes())?;
                        }
                        PointDataType::ColorB => {
                            let color =
                                point.get_attribute::<Vector3<u16>>(&attributes::COLOR_RGB)?;
                            self.writer.write_all(color[2].to_string().as_bytes())?;
                        }
                        PointDataType::GpsTime => {
                            let gps_time = point.get_attribute::<f64>(&attributes::GPS_TIME)?;
                            self.writer.write_all(
                                trim_unnecessary_tailing_zeros(&format!(
                                    "{:.1$}",
                                    gps_time, self.precision
                                ))
                                .as_bytes(),
                            )?;
                        }
                        PointDataType::PointSourceID => {
                            let point_source_id =
                                point.get_attribute::<u64>(&attributes::POINT_SOURCE_ID)?;
                            self.writer
                                .write_all(point_source_id.to_string().as_bytes())?;
                        }
                        PointDataType::EdgeOfFlightLine => {
                            let edge_of_flight_line =
                                point.get_attribute::<u8>(&attributes::EDGE_OF_FLIGHT_LINE)?;
                            self.writer.write_all(
                                (if edge_of_flight_line > 0 { "1" } else { "0" }).as_bytes(),
                            )?;
                        }
                        PointDataType::ScanDirectionFlag => {
                            let scan_direction_flag =
                                point.get_attribute::<u8>(&attributes::SCAN_DIRECTION_FLAG)?;
                            self.writer.write_all(
                                (if scan_direction_flag > 0 { "1" } else { "0" }).as_bytes(),
                            )?;
                        }
                        PointDataType::ScanAngleRank => {
                            let scan_angle_rank =
                                point.get_attribute::<i64>(&attributes::SCAN_ANGLE_RANK)?;
                            self.writer
                                .write_all(scan_angle_rank.to_string().as_bytes())?;
                        }
                        PointDataType::Nir => {
                            let nir = point.get_attribute::<u64>(&attributes::NIR)?;
                            self.writer.write_all(nir.to_string().as_bytes())?;
                        }
                    }
                    if index != self.parse_layout.len() - 1 {
                        self.writer.write_all(self.delimiter.as_bytes())?;
                    }
                }
                self.writer.write_all(b"\n")?;
            }
        }
        Ok(())
    }

    fn flush(&mut self) -> anyhow::Result<()> {
        self.writer.flush().context("Flush failed")
    }

    fn get_default_point_layout(&self) -> &PointLayout {
        &self.default_layout
    }
}

fn trim_unnecessary_tailing_zeros(slice: &str) -> &str {
    let start = 0;
    let mut end = slice.len();
    while slice[start..end].ends_with('0') && !slice[start..end].ends_with(".0") {
        end -= 1;
    }
    &slice[start..end]
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{BufRead, BufReader, BufWriter},
    };

    use crate::ascii::{get_test_file_path, test_data_buffer};

    use super::*;
    use anyhow::Result;
    use itertools::Itertools;
    use pasture_core::containers::{
        MakeBufferFromLayout, OwningBuffer, UntypedPointBuffer, VectorBuffer,
    };
    use scopeguard::defer;

    #[test]
    fn test_write() -> Result<()> {
        // create point buffer with one point
        let layout =
            PointLayout::from_attributes(&[attributes::POSITION_3D, attributes::INTENSITY]);
        let mut buffer = VectorBuffer::new_from_layout(layout.clone());
        let mut point = UntypedPointBuffer::new(&layout);
        point.set_attribute(
            &attributes::POSITION_3D,
            &Vector3::<f32>::new(1.1, 2.2, 3.3),
        )?;
        point.set_attribute(&attributes::INTENSITY, &32_u16)?;
        unsafe {
            buffer.push_points(point.get_buffer());
        }
        let out_path = "./test_ascii_writer.txt";
        defer! {
            std::fs::remove_file(out_path).expect("Could not remove test file");
        }
        let mut writer =
            RawAsciiWriter::from_write(BufWriter::new(File::create(out_path)?), "ixyz")?;
        writer.write(&buffer)?;
        Ok(())
    }
    #[test]
    fn test_write_all_attribute() -> Result<()> {
        let out_path = "./test_ascii_writer_attributes.txt";
        defer! {
             std::fs::remove_file(out_path).expect("Could not remove test file");
        }
        let test_data = test_data_buffer()?;
        {
            let mut writer = RawAsciiWriter::from_write(
                BufWriter::new(File::create(out_path)?),
                "xyzirncuRGBtpedaI",
            )?;
            writer.write(&test_data)?;
            writer.flush()?;
        }
        //Check result file
        let result_file = BufReader::new(File::open(out_path)?);
        let reference_file = BufReader::new(File::open(get_test_file_path(
            "10_points_ascii_all_attributes.txt",
        ))?);
        for (line_first_file, line_second_file) in
            result_file.lines().zip_eq(reference_file.lines())
        {
            assert_eq!(line_first_file?, line_second_file?);
        }
        Ok(())
    }

    #[test]
    #[should_panic(expected = "FormatError can't interpret format literal")]
    fn test_error_format_unrecognized_literal() {
        let path = "./test_ascii_writer_format_error.txt";
        defer! {
             std::fs::remove_file(path).expect("Could not remove test file");
        }
        let writer = BufWriter::new(File::create(path).unwrap());
        RawAsciiWriter::from_write(writer, "xyzQ").unwrap();
    }

    #[test]
    #[should_panic(expected = "Cannot find attribute.")]
    fn test_attribute_not_found_error() {
        // create point buffer with one point
        let layout =
            PointLayout::from_attributes(&[attributes::POSITION_3D, attributes::INTENSITY]);
        let mut buffer = VectorBuffer::new_from_layout(layout.clone());
        let mut point = UntypedPointBuffer::new(&layout);
        point
            .set_attribute(&attributes::INTENSITY, &32_u16)
            .unwrap();
        unsafe {
            buffer.push_points(point.get_buffer());
        }
        let out_path = "./test_ascii_writer_attribute_error.txt";
        defer! {
            std::fs::remove_file(out_path).expect("Could not remove test file");
        }
        let mut writer =
            RawAsciiWriter::from_write(BufWriter::new(File::create(out_path).unwrap()), "e")
                .unwrap();
        writer.write(&buffer).unwrap();
    }
}
