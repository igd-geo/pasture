use std::{fs::File, io::BufWriter, path::Path};

use anyhow::{anyhow, Result};
use las::{raw::point::Waveform, Color, Write, Writer};
use pasture_core::{
    containers::PointBuffer,
    layout::{attributes, PointLayout},
    nalgebra::Vector3,
    util::view_raw_bytes_mut,
};

use crate::base::PointWriter;

use super::{point_layout_from_las_point_format, LASMetadata};

/// `PointWriter` implementation for LAS/LAZ files
pub struct LASWriter<T: std::io::Write + std::io::Seek + std::fmt::Debug + Send + 'static> {
    writer: Writer<T>,
    metadata: LASMetadata,
    layout: PointLayout,
    current_point_index: usize,
}

impl<T: std::io::Write + std::io::Seek + std::fmt::Debug + Send + 'static> LASWriter<T> {
    /// Creates a new 'LASWriter` from the given path and LAS header
    pub fn from_path_and_header<P: AsRef<Path>>(
        path: P,
        header: las::Header,
    ) -> Result<LASWriter<BufWriter<File>>> {
        let writer = BufWriter::new(File::create(path)?);
        LASWriter::<BufWriter<File>>::from_writer_and_header(writer, header)
    }

    /// Creates a new 'LASWriter` from the given writer and LAS header
    pub fn from_writer_and_header(writer: T, header: las::Header) -> Result<Self> {
        let metadata: LASMetadata = header.clone().into();
        let point_layout = point_layout_from_las_point_format(header.point_format())?;
        let las_writer = Writer::new(writer, header)?;
        Ok(Self {
            writer: las_writer,
            metadata,
            layout: point_layout,
            current_point_index: 0,
        })
    }
}

impl<T: std::io::Write + std::io::Seek + std::fmt::Debug + Send + 'static> PointWriter
    for LASWriter<T>
{
    fn write(&mut self, points: &dyn PointBuffer) -> Result<()> {
        // TODO Support conversions of attributes with the same name, but different datatypes
        let supported_attributes = points
            .point_layout()
            .attributes()
            .filter(|&attribute| {
                if let Some(self_attribute) = self.layout.get_attribute_by_name(attribute.name()) {
                    self_attribute.datatype() == attribute.datatype()
                } else {
                    false
                }
            })
            .collect::<Vec<_>>();

        if supported_attributes.is_empty() {
            return Err(anyhow!("LASWriter::write: No attributes in the given PointBuffer are supported by LASWriter!"));
        }

        let has_positions = self.layout.has_attribute(attributes::POSITION_3D.name())
            && points
                .point_layout()
                .has_attribute(attributes::POSITION_3D.name());

        let has_intensities = self.layout.has_attribute(attributes::INTENSITY.name())
            && points
                .point_layout()
                .has_attribute(attributes::INTENSITY.name());

        let has_return_numbers = self.layout.has_attribute(attributes::RETURN_NUMBER.name())
            && points
                .point_layout()
                .has_attribute(attributes::RETURN_NUMBER.name());

        let has_number_of_returns = self
            .layout
            .has_attribute(attributes::NUMBER_OF_RETURNS.name())
            && points
                .point_layout()
                .has_attribute(attributes::NUMBER_OF_RETURNS.name());

        let has_scan_direction_flags = self
            .layout
            .has_attribute(attributes::SCAN_DIRECTION_FLAG.name())
            && points
                .point_layout()
                .has_attribute(attributes::SCAN_DIRECTION_FLAG.name());

        let has_edge_of_flight_lines = self
            .layout
            .has_attribute(attributes::EDGE_OF_FLIGHT_LINE.name())
            && points
                .point_layout()
                .has_attribute(attributes::EDGE_OF_FLIGHT_LINE.name());

        let has_classifications = self.layout.has_attribute(attributes::CLASSIFICATION.name())
            && points
                .point_layout()
                .has_attribute(attributes::CLASSIFICATION.name());

        let has_scan_angle_ranks = self
            .layout
            .has_attribute(attributes::SCAN_ANGLE_RANK.name())
            && points
                .point_layout()
                .has_attribute(attributes::SCAN_ANGLE_RANK.name());

        let has_user_data = self.layout.has_attribute(attributes::USER_DATA.name())
            && points
                .point_layout()
                .has_attribute(attributes::USER_DATA.name());

        let has_point_source_ids = self
            .layout
            .has_attribute(attributes::POINT_SOURCE_ID.name())
            && points
                .point_layout()
                .has_attribute(attributes::POINT_SOURCE_ID.name());

        let self_requires_gps_times = self.layout.has_attribute(attributes::GPS_TIME.name());
        let has_gps_times = self_requires_gps_times
            && points
                .point_layout()
                .has_attribute(attributes::GPS_TIME.name());

        let self_requires_colors = self.layout.has_attribute(attributes::COLOR_RGB.name());
        let has_colors = self_requires_colors
            && points
                .point_layout()
                .has_attribute(attributes::COLOR_RGB.name());

        let self_requires_waveform = self
            .layout
            .has_attribute(attributes::WAVE_PACKET_DESCRIPTOR_INDEX.name());
        let has_waveform_packet_descriptor_index = self_requires_waveform
            && points
                .point_layout()
                .has_attribute(attributes::WAVE_PACKET_DESCRIPTOR_INDEX.name());
        let has_waveform_data_offset = self_requires_waveform
            && points
                .point_layout()
                .has_attribute(attributes::WAVEFORM_DATA_OFFSET.name());
        let has_waveform_packet_size = self_requires_waveform
            && points
                .point_layout()
                .has_attribute(attributes::WAVEFORM_PACKET_SIZE.name());
        let has_return_point_waveform_location = self_requires_waveform
            && points
                .point_layout()
                .has_attribute(attributes::RETURN_POINT_WAVEFORM_LOCATION.name());
        let has_waveform_parameters = self_requires_waveform
            && points
                .point_layout()
                .has_attribute(attributes::WAVEFORM_PARAMETERS.name());

        // TODO other attributes of extended formats (6-10)
        // TODO This seems quite inefficient and it doesn't even support format conversion. Maybe we can enforce
        // that the passed in buffer has a format that exactly matches the format of this writer. If the user wants
        // to write different formats, we could then enforce a buffer conversion into the target format. Maybe this
        // is cleaner?

        for point_idx in 0..points.len() {
            //TODO las-rs has a bug in its writer implementation where writing a default point fails in all but format 0
            let mut las_point: las::Point = Default::default();

            if has_positions {
                let mut pos: Vector3<f64> = Default::default();
                let pos_bytes = unsafe { view_raw_bytes_mut(&mut pos) };
                points.get_attribute_by_copy(point_idx, &attributes::POSITION_3D, pos_bytes);
                las_point.x = pos.x;
                las_point.y = pos.y;
                las_point.z = pos.z;
            }

            if has_intensities {
                let mut intensity: u16 = Default::default();
                let intensity_bytes = unsafe { view_raw_bytes_mut(&mut intensity) };
                points.get_attribute_by_copy(point_idx, &attributes::INTENSITY, intensity_bytes);
                las_point.intensity = intensity;
            }

            if has_return_numbers {
                let mut return_number: u8 = Default::default();
                let return_number_bytes = unsafe { view_raw_bytes_mut(&mut return_number) };
                points.get_attribute_by_copy(
                    point_idx,
                    &attributes::RETURN_NUMBER,
                    return_number_bytes,
                );
                las_point.return_number = return_number;
            }

            if has_number_of_returns {
                let mut number_of_returns: u8 = Default::default();
                let number_of_returns_bytes = unsafe { view_raw_bytes_mut(&mut number_of_returns) };
                points.get_attribute_by_copy(
                    point_idx,
                    &attributes::NUMBER_OF_RETURNS,
                    number_of_returns_bytes,
                );
                las_point.number_of_returns = number_of_returns;
            }

            if has_scan_direction_flags {
                let mut scan_direction_flag: bool = Default::default();
                let scan_direction_flag_bytes =
                    unsafe { view_raw_bytes_mut(&mut scan_direction_flag) };
                points.get_attribute_by_copy(
                    point_idx,
                    &attributes::SCAN_DIRECTION_FLAG,
                    scan_direction_flag_bytes,
                );
                las_point.scan_direction = if scan_direction_flag {
                    las::point::ScanDirection::LeftToRight
                } else {
                    las::point::ScanDirection::RightToLeft
                };
            }

            if has_edge_of_flight_lines {
                let mut eof: bool = Default::default();
                let eof_bytes = unsafe { view_raw_bytes_mut(&mut eof) };
                points.get_attribute_by_copy(
                    point_idx,
                    &attributes::EDGE_OF_FLIGHT_LINE,
                    eof_bytes,
                );
                las_point.is_edge_of_flight_line = eof;
            }

            if has_classifications {
                let mut classification: u8 = Default::default();
                let classification_bytes = unsafe { view_raw_bytes_mut(&mut classification) };
                points.get_attribute_by_copy(
                    point_idx,
                    &attributes::CLASSIFICATION,
                    classification_bytes,
                );
                las_point.classification = las::point::Classification::new(classification)?;
            }

            if has_scan_angle_ranks {
                let mut scan_angle_rank: i8 = Default::default();
                let scan_angle_rank_bytes = unsafe { view_raw_bytes_mut(&mut scan_angle_rank) };
                points.get_attribute_by_copy(
                    point_idx,
                    &attributes::SCAN_ANGLE_RANK,
                    scan_angle_rank_bytes,
                );
                las_point.scan_angle = scan_angle_rank as f32;
            }

            if has_user_data {
                let mut user_data: u8 = Default::default();
                let user_data_bytes = unsafe { view_raw_bytes_mut(&mut user_data) };
                points.get_attribute_by_copy(point_idx, &attributes::USER_DATA, user_data_bytes);
                las_point.user_data = user_data;
            }

            if has_point_source_ids {
                let mut source_id: u16 = Default::default();
                let source_id_bytes = unsafe { view_raw_bytes_mut(&mut source_id) };
                points.get_attribute_by_copy(
                    point_idx,
                    &attributes::POINT_SOURCE_ID,
                    source_id_bytes,
                );
                las_point.point_source_id = source_id;
            }

            if has_gps_times {
                let mut gps_time: f64 = Default::default();
                let gps_time_bytes = unsafe { view_raw_bytes_mut(&mut gps_time) };
                points.get_attribute_by_copy(point_idx, &attributes::GPS_TIME, gps_time_bytes);
                las_point.gps_time = Some(gps_time);
            } else if self_requires_gps_times {
                las_point.gps_time = Some(0.0);
            }

            if has_colors {
                let mut color: Vector3<u16> = Default::default();
                let color_bytes = unsafe { view_raw_bytes_mut(&mut color) };
                points.get_attribute_by_copy(point_idx, &attributes::COLOR_RGB, color_bytes);
                las_point.color = Some(Color::new(color.x, color.y, color.z));
            } else if self_requires_colors {
                las_point.color = Some(Color::new(0, 0, 0));
            }

            if self_requires_waveform {
                let mut waveform: Waveform = Default::default();

                if has_waveform_packet_descriptor_index {
                    let index_bytes =
                        unsafe { view_raw_bytes_mut(&mut waveform.wave_packet_descriptor_index) };
                    points.get_attribute_by_copy(
                        point_idx,
                        &attributes::WAVE_PACKET_DESCRIPTOR_INDEX,
                        index_bytes,
                    );
                }

                if has_waveform_data_offset {
                    let offset_bytes =
                        unsafe { view_raw_bytes_mut(&mut waveform.byte_offset_to_waveform_data) };
                    points.get_attribute_by_copy(
                        point_idx,
                        &attributes::WAVEFORM_DATA_OFFSET,
                        offset_bytes,
                    );
                }

                if has_waveform_packet_size {
                    let packet_size_bytes =
                        unsafe { view_raw_bytes_mut(&mut waveform.waveform_packet_size_in_bytes) };
                    points.get_attribute_by_copy(
                        point_idx,
                        &attributes::WAVEFORM_PACKET_SIZE,
                        packet_size_bytes,
                    );
                }

                if has_return_point_waveform_location {
                    let return_bytes =
                        unsafe { view_raw_bytes_mut(&mut waveform.return_point_waveform_location) };
                    points.get_attribute_by_copy(
                        point_idx,
                        &attributes::RETURN_POINT_WAVEFORM_LOCATION,
                        return_bytes,
                    );
                }

                if has_waveform_parameters {
                    let mut parameters: Vector3<f32> = Default::default();
                    let parameters_bytes = unsafe { view_raw_bytes_mut(&mut parameters) };
                    points.get_attribute_by_copy(
                        point_idx,
                        &attributes::WAVEFORM_PARAMETERS,
                        parameters_bytes,
                    );
                    waveform.x_t = parameters.x;
                    waveform.y_t = parameters.y;
                    waveform.z_t = parameters.z;
                }

                las_point.waveform = Some(waveform);
            }

            // TODO Extract the rest of the attributes

            self.writer.write(las_point)?;
        }

        Ok(())
    }

    fn get_default_point_layout(&self) -> &PointLayout {
        &self.layout
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use las::{point::Format, Builder};
    use pasture_core::{
        containers::points, containers::InterleavedVecPointStorage, layout::PointType,
    };
    use scopeguard::defer;

    use crate::{
        base::PointReader,
        las::{
            LASReader, LasPointFormat0, LasPointFormat1, LasPointFormat2, LasPointFormat3,
            LasPointFormat4, LasPointFormat5,
        },
    };

    use super::*;

    #[repr(packed)]
    #[derive(Debug, Clone, Copy)]
    struct TestPoint {
        pub position: Vector3<f64>,
        pub color: Vector3<u16>,
    }

    impl PointType for TestPoint {
        fn layout() -> PointLayout {
            PointLayout::from_attributes(&[attributes::POSITION_3D, attributes::COLOR_RGB])
        }
    }

    fn get_test_points_custom_format() -> Vec<TestPoint> {
        vec![
            TestPoint {
                position: Vector3::new(1.0, 2.0, 3.0),
                color: Vector3::new(128, 129, 130),
            },
            TestPoint {
                position: Vector3::new(4.0, 5.0, 6.0),
                color: Vector3::new(1024, 1025, 1026),
            },
        ]
    }

    fn get_test_points_las_format_0() -> Vec<LasPointFormat0> {
        vec![
            LasPointFormat0 {
                classification: 1,
                edge_of_flight_line: false,
                intensity: 1,
                number_of_returns: 1,
                point_source_id: 1,
                position: Vector3::new(1.0, 1.0, 1.0),
                return_number: 1,
                scan_angle_rank: 1,
                scan_direction_flag: false,
                user_data: 1,
            },
            LasPointFormat0 {
                classification: 2,
                edge_of_flight_line: true,
                intensity: 2,
                number_of_returns: 2,
                point_source_id: 2,
                position: Vector3::new(2.0, 2.0, 2.0),
                return_number: 2,
                scan_angle_rank: 2,
                scan_direction_flag: true,
                user_data: 2,
            },
        ]
    }

    fn get_test_points_las_format_1() -> Vec<LasPointFormat1> {
        vec![
            LasPointFormat1 {
                classification: 1,
                edge_of_flight_line: false,
                intensity: 1,
                number_of_returns: 1,
                point_source_id: 1,
                position: Vector3::new(1.0, 1.0, 1.0),
                return_number: 1,
                scan_angle_rank: 1,
                scan_direction_flag: false,
                user_data: 1,
                gps_time: 1234.0,
            },
            LasPointFormat1 {
                classification: 2,
                edge_of_flight_line: true,
                intensity: 2,
                number_of_returns: 2,
                point_source_id: 2,
                position: Vector3::new(2.0, 2.0, 2.0),
                return_number: 2,
                scan_angle_rank: 2,
                scan_direction_flag: true,
                user_data: 2,
                gps_time: 5678.0,
            },
        ]
    }

    fn get_test_points_las_format_2() -> Vec<LasPointFormat2> {
        vec![
            LasPointFormat2 {
                classification: 1,
                edge_of_flight_line: false,
                intensity: 1,
                number_of_returns: 1,
                point_source_id: 1,
                position: Vector3::new(1.0, 1.0, 1.0),
                return_number: 1,
                scan_angle_rank: 1,
                scan_direction_flag: false,
                user_data: 1,
                color_rgb: Vector3::new(128, 129, 130),
            },
            LasPointFormat2 {
                classification: 2,
                edge_of_flight_line: true,
                intensity: 2,
                number_of_returns: 2,
                point_source_id: 2,
                position: Vector3::new(2.0, 2.0, 2.0),
                return_number: 2,
                scan_angle_rank: 2,
                scan_direction_flag: true,
                user_data: 2,
                color_rgb: Vector3::new(1024, 1025, 1026),
            },
        ]
    }

    fn get_test_points_las_format_3() -> Vec<LasPointFormat3> {
        vec![
            LasPointFormat3 {
                classification: 1,
                edge_of_flight_line: false,
                intensity: 1,
                number_of_returns: 1,
                point_source_id: 1,
                position: Vector3::new(1.0, 1.0, 1.0),
                return_number: 1,
                scan_angle_rank: 1,
                scan_direction_flag: false,
                user_data: 1,
                color_rgb: Vector3::new(128, 129, 130),
                gps_time: 1234.0,
            },
            LasPointFormat3 {
                classification: 2,
                edge_of_flight_line: true,
                intensity: 2,
                number_of_returns: 2,
                point_source_id: 2,
                position: Vector3::new(2.0, 2.0, 2.0),
                return_number: 2,
                scan_angle_rank: 2,
                scan_direction_flag: true,
                user_data: 2,
                color_rgb: Vector3::new(1024, 1025, 1026),
                gps_time: 5678.0,
            },
        ]
    }

    fn get_test_points_las_format_4() -> Vec<LasPointFormat4> {
        vec![
            LasPointFormat4 {
                classification: 1,
                edge_of_flight_line: false,
                intensity: 1,
                number_of_returns: 1,
                point_source_id: 1,
                position: Vector3::new(1.0, 1.0, 1.0),
                return_number: 1,
                scan_angle_rank: 1,
                scan_direction_flag: false,
                user_data: 1,
                gps_time: 1234.0,
                byte_offset_to_waveform_data: 10,
                return_point_waveform_location: 20.0,
                wave_packet_descriptor_index: 30,
                waveform_packet_size: 40,
                waveform_parameters: Vector3::new(1.0, 2.0, 3.0),
            },
            LasPointFormat4 {
                classification: 2,
                edge_of_flight_line: true,
                intensity: 2,
                number_of_returns: 2,
                point_source_id: 2,
                position: Vector3::new(2.0, 2.0, 2.0),
                return_number: 2,
                scan_angle_rank: 2,
                scan_direction_flag: true,
                user_data: 2,
                gps_time: 5678.0,
                byte_offset_to_waveform_data: 11,
                return_point_waveform_location: 22.0,
                wave_packet_descriptor_index: 33,
                waveform_packet_size: 44,
                waveform_parameters: Vector3::new(4.0, 5.0, 6.0),
            },
        ]
    }

    fn get_test_points_las_format_5() -> Vec<LasPointFormat5> {
        vec![
            LasPointFormat5 {
                classification: 1,
                edge_of_flight_line: false,
                intensity: 1,
                number_of_returns: 1,
                point_source_id: 1,
                position: Vector3::new(1.0, 1.0, 1.0),
                return_number: 1,
                scan_angle_rank: 1,
                scan_direction_flag: false,
                user_data: 1,
                gps_time: 1234.0,
                color_rgb: Vector3::new(128, 129, 130),
                byte_offset_to_waveform_data: 10,
                return_point_waveform_location: 20.0,
                wave_packet_descriptor_index: 30,
                waveform_packet_size: 40,
                waveform_parameters: Vector3::new(1.0, 2.0, 3.0),
            },
            LasPointFormat5 {
                classification: 2,
                edge_of_flight_line: true,
                intensity: 2,
                number_of_returns: 2,
                point_source_id: 2,
                position: Vector3::new(2.0, 2.0, 2.0),
                return_number: 2,
                scan_angle_rank: 2,
                scan_direction_flag: true,
                user_data: 2,
                gps_time: 5678.0,
                color_rgb: Vector3::new(1024, 1025, 1026),
                byte_offset_to_waveform_data: 11,
                return_point_waveform_location: 22.0,
                wave_packet_descriptor_index: 33,
                waveform_packet_size: 44,
                waveform_parameters: Vector3::new(4.0, 5.0, 6.0),
            },
        ]
    }

    fn prepare_point_buffer<T: PointType + Clone>(test_points: &[T]) -> InterleavedVecPointStorage {
        let layout = T::layout();
        let mut source_point_buffer =
            InterleavedVecPointStorage::with_capacity(test_points.len(), layout);

        for point in test_points.iter().cloned() {
            source_point_buffer.push_point(point);
        }

        source_point_buffer
    }

    #[test]
    fn test_write_las_format_0() -> Result<()> {
        let source_points = get_test_points_las_format_0();
        let source_point_buffer = prepare_point_buffer(&source_points);

        //Write, then read, then check for equality

        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("test_write_las_format_0.las");

        defer! {
            std::fs::remove_file(&test_file_path).expect("Removing test file failed!");
        }

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(0)?;

        {
            let mut writer = LASWriter::<BufWriter<File>>::from_path_and_header(
                &test_file_path,
                las_header_builder.into_header().unwrap(),
            )?;
            writer.write(&source_point_buffer)?;
        }

        {
            let mut reader = LASReader::from_path(&test_file_path)?;
            let read_points_buffer = reader.read(source_points.len())?;
            let read_points =
                points::<LasPointFormat0>(read_points_buffer.as_ref()).collect::<Vec<_>>();

            assert_eq!(read_points, source_points);
        }

        Ok(())
    }

    #[test]
    fn test_write_las_format_0_different_layout() -> Result<()> {
        let source_points = get_test_points_custom_format();
        let source_point_buffer = prepare_point_buffer(&source_points);

        //Write, then read, then check for equality

        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("test_write_las_format_0_different_layout.las");

        defer! {
            std::fs::remove_file(&test_file_path).expect("Removing test file failed!");
        }

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(0)?;

        {
            let mut writer = LASWriter::<BufWriter<File>>::from_path_and_header(
                &test_file_path,
                las_header_builder.into_header().unwrap(),
            )?;
            writer.write(&source_point_buffer)?;
        }

        {
            let mut reader = LASReader::from_path(&test_file_path)?;
            let read_points_buffer = reader.read(source_points.len())?;
            let read_points =
                points::<LasPointFormat0>(read_points_buffer.as_ref()).collect::<Vec<_>>();

            for (source, read) in source_points.iter().zip(read_points.iter()) {
                assert_eq!(
                    { source.position },
                    { read.position },
                    "Position of read point is different than of source point!"
                );
                assert_eq!(
                    0, read.classification,
                    "Classification of read point was not zero!"
                );
                assert_eq!(
                    false, read.edge_of_flight_line,
                    "Edge of flight line of read point was not false!"
                );
                assert_eq!(
                    0,
                    { read.intensity },
                    "Intensity of read point was not zero!"
                );
                assert_eq!(
                    0, read.number_of_returns,
                    "Number of returns of read point was not zero!"
                );
                assert_eq!(
                    0,
                    { read.point_source_id },
                    "Point source ID of read point was not zero!"
                );
                assert_eq!(
                    0, read.return_number,
                    "Return number of read point was not zero!"
                );
                assert_eq!(
                    0, read.scan_angle_rank,
                    "Scan angle rank of read point was not zero!"
                );
                assert_eq!(
                    false, read.scan_direction_flag,
                    "Scan direction flag of read point was not false!"
                );
                assert_eq!(0, read.user_data, "User data of read point was not zero!");
            }
        }

        Ok(())
    }

    #[test]
    fn test_write_las_format_1() -> Result<()> {
        let source_points = get_test_points_las_format_1();
        let source_point_buffer = prepare_point_buffer(&source_points);

        //Write, then read, then check for equality

        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("test_write_las_format_1.las");

        defer! {
            std::fs::remove_file(&test_file_path).expect("Removing test file failed!");
        }

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(1)?;

        {
            let mut writer = LASWriter::<BufWriter<File>>::from_path_and_header(
                &test_file_path,
                las_header_builder.into_header().unwrap(),
            )?;
            writer.write(&source_point_buffer)?;
        }

        {
            let mut reader = LASReader::from_path(&test_file_path)?;
            let read_points_buffer = reader.read(source_points.len())?;
            let read_points =
                points::<LasPointFormat1>(read_points_buffer.as_ref()).collect::<Vec<_>>();

            assert_eq!(read_points, source_points);
        }

        Ok(())
    }

    #[test]
    fn test_write_las_format_1_different_layout() -> Result<()> {
        let source_points = get_test_points_custom_format();
        let source_point_buffer = prepare_point_buffer(&source_points);

        //Write, then read, then check for equality

        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("test_write_las_format_1_different_layout.las");

        defer! {
            std::fs::remove_file(&test_file_path).expect("Removing test file failed!");
        }

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(1)?;

        {
            let mut writer = LASWriter::<BufWriter<File>>::from_path_and_header(
                &test_file_path,
                las_header_builder.into_header().unwrap(),
            )?;
            writer.write(&source_point_buffer)?;
        }

        {
            let mut reader = LASReader::from_path(&test_file_path)?;
            let read_points_buffer = reader.read(source_points.len())?;
            let read_points =
                points::<LasPointFormat1>(read_points_buffer.as_ref()).collect::<Vec<_>>();

            for (source, read) in source_points.iter().zip(read_points.iter()) {
                assert_eq!(
                    { source.position },
                    { read.position },
                    "Position of read point is different than of source point!"
                );
                assert_eq!(
                    0, read.classification,
                    "Classification of read point was not zero!"
                );
                assert_eq!(
                    false, read.edge_of_flight_line,
                    "Edge of flight line of read point was not false!"
                );
                assert_eq!(
                    0,
                    { read.intensity },
                    "Intensity of read point was not zero!"
                );
                assert_eq!(
                    0, read.number_of_returns,
                    "Number of returns of read point was not zero!"
                );
                assert_eq!(
                    0,
                    { read.point_source_id },
                    "Point source ID of read point was not zero!"
                );
                assert_eq!(
                    0, read.return_number,
                    "Return number of read point was not zero!"
                );
                assert_eq!(
                    0, read.scan_angle_rank,
                    "Scan angle rank of read point was not zero!"
                );
                assert_eq!(
                    false, read.scan_direction_flag,
                    "Scan direction flag of read point was not false!"
                );
                assert_eq!(0, read.user_data, "User data of read point was not zero!");
                assert_eq!(
                    0.0,
                    { read.gps_time },
                    "GPS time of read point was not zero!"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_write_las_format_2() -> Result<()> {
        let source_points = get_test_points_las_format_2();
        let source_point_buffer = prepare_point_buffer(&source_points);

        //Write, then read, then check for equality

        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("test_write_las_format_2.las");

        defer! {
            std::fs::remove_file(&test_file_path).expect("Removing test file failed!");
        }

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(2)?;

        {
            let mut writer = LASWriter::<BufWriter<File>>::from_path_and_header(
                &test_file_path,
                las_header_builder.into_header().unwrap(),
            )?;
            writer.write(&source_point_buffer)?;
        }

        {
            let mut reader = LASReader::from_path(&test_file_path)?;
            let read_points_buffer = reader.read(source_points.len())?;
            let read_points =
                points::<LasPointFormat2>(read_points_buffer.as_ref()).collect::<Vec<_>>();

            assert_eq!(read_points, source_points);
        }

        Ok(())
    }

    #[test]
    fn test_write_las_format_2_different_layout() -> Result<()> {
        let source_points = get_test_points_custom_format();
        let source_point_buffer = prepare_point_buffer(&source_points);

        //Write, then read, then check for equality

        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("test_write_las_format_2_different_layout.las");

        defer! {
            std::fs::remove_file(&test_file_path).expect("Removing test file failed!");
        }

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(2)?;

        {
            let mut writer = LASWriter::<BufWriter<File>>::from_path_and_header(
                &test_file_path,
                las_header_builder.into_header().unwrap(),
            )?;
            writer.write(&source_point_buffer)?;
        }

        {
            let mut reader = LASReader::from_path(&test_file_path)?;
            let read_points_buffer = reader.read(source_points.len())?;
            let read_points =
                points::<LasPointFormat2>(read_points_buffer.as_ref()).collect::<Vec<_>>();

            for (source, read) in source_points.iter().zip(read_points.iter()) {
                assert_eq!(
                    { source.position },
                    { read.position },
                    "Position of read point is different than of source point!"
                );
                assert_eq!(
                    { source.color },
                    { read.color_rgb },
                    "Color of read point is different than of source point!"
                );
                assert_eq!(
                    0, read.classification,
                    "Classification of read point was not zero!"
                );
                assert_eq!(
                    false, read.edge_of_flight_line,
                    "Edge of flight line of read point was not false!"
                );
                assert_eq!(
                    0,
                    { read.intensity },
                    "Intensity of read point was not zero!"
                );
                assert_eq!(
                    0, read.number_of_returns,
                    "Number of returns of read point was not zero!"
                );
                assert_eq!(
                    0,
                    { read.point_source_id },
                    "Point source ID of read point was not zero!"
                );
                assert_eq!(
                    0, read.return_number,
                    "Return number of read point was not zero!"
                );
                assert_eq!(
                    0, read.scan_angle_rank,
                    "Scan angle rank of read point was not zero!"
                );
                assert_eq!(
                    false, read.scan_direction_flag,
                    "Scan direction flag of read point was not false!"
                );
                assert_eq!(0, read.user_data, "User data of read point was not zero!");
            }
        }

        Ok(())
    }

    #[test]
    fn test_write_las_format_3() -> Result<()> {
        let source_points = get_test_points_las_format_3();
        let source_point_buffer = prepare_point_buffer(&source_points);

        //Write, then read, then check for equality

        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("test_write_las_format_3.las");

        defer! {
            std::fs::remove_file(&test_file_path).expect("Removing test file failed!");
        }

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(3)?;

        {
            let mut writer = LASWriter::<BufWriter<File>>::from_path_and_header(
                &test_file_path,
                las_header_builder.into_header().unwrap(),
            )?;
            writer.write(&source_point_buffer)?;
        }

        {
            let mut reader = LASReader::from_path(&test_file_path)?;
            let read_points_buffer = reader.read(source_points.len())?;
            let read_points =
                points::<LasPointFormat3>(read_points_buffer.as_ref()).collect::<Vec<_>>();

            assert_eq!(read_points, source_points);
        }

        Ok(())
    }

    #[test]
    fn test_write_las_format_3_different_layout() -> Result<()> {
        let source_points = get_test_points_custom_format();
        let source_point_buffer = prepare_point_buffer(&source_points);

        //Write, then read, then check for equality

        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("test_write_las_format_3_different_layout.las");

        defer! {
            std::fs::remove_file(&test_file_path).expect("Removing test file failed!");
        }

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(3)?;

        {
            let mut writer = LASWriter::<BufWriter<File>>::from_path_and_header(
                &test_file_path,
                las_header_builder.into_header().unwrap(),
            )?;
            writer.write(&source_point_buffer)?;
        }

        {
            let mut reader = LASReader::from_path(&test_file_path)?;
            let read_points_buffer = reader.read(source_points.len())?;
            let read_points =
                points::<LasPointFormat3>(read_points_buffer.as_ref()).collect::<Vec<_>>();

            for (source, read) in source_points.iter().zip(read_points.iter()) {
                assert_eq!(
                    { source.position },
                    { read.position },
                    "Position of read point is not equal to position of source point!"
                );
                assert_eq!(
                    { source.color },
                    { read.color_rgb },
                    "Color of read point is not equal to color of source point!"
                );
                assert_eq!(
                    0, read.classification,
                    "Classification of read point was not zero!"
                );
                assert_eq!(
                    false, read.edge_of_flight_line,
                    "Edge of flight line of read point was not false!"
                );
                assert_eq!(
                    0,
                    { read.intensity },
                    "Intensity of read point was not zero!"
                );
                assert_eq!(
                    0, read.number_of_returns,
                    "Number of returns of read point was not zero!"
                );
                assert_eq!(
                    0,
                    { read.point_source_id },
                    "Point source ID of read point was not zero!"
                );
                assert_eq!(
                    0, read.return_number,
                    "Return number of read point was not zero!"
                );
                assert_eq!(
                    0, read.scan_angle_rank,
                    "Scan angle rank of read point was not zero!"
                );
                assert_eq!(
                    false, read.scan_direction_flag,
                    "Scan direction flag of read point was not false!"
                );
                assert_eq!(0, read.user_data, "User data of read point was not zero!");
                assert_eq!(
                    0.0,
                    { read.gps_time },
                    "GPS time of read point was not zero!"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_write_las_format_4() -> Result<()> {
        let source_points = get_test_points_las_format_4();
        let source_point_buffer = prepare_point_buffer(&source_points);

        //Write, then read, then check for equality

        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("test_write_las_format_4.las");

        defer! {
            std::fs::remove_file(&test_file_path).expect("Removing test file failed!");
        }

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(4)?;

        {
            let mut writer = LASWriter::<BufWriter<File>>::from_path_and_header(
                &test_file_path,
                las_header_builder.into_header().unwrap(),
            )?;
            writer.write(&source_point_buffer)?;
        }

        {
            let mut reader = LASReader::from_path(&test_file_path)?;
            let read_points_buffer = reader.read(source_points.len())?;
            let read_points =
                points::<LasPointFormat4>(read_points_buffer.as_ref()).collect::<Vec<_>>();

            assert_eq!(read_points, source_points);
        }

        Ok(())
    }

    #[test]
    fn test_write_las_format_4_different_layout() -> Result<()> {
        let source_points = get_test_points_custom_format();
        let source_point_buffer = prepare_point_buffer(&source_points);

        //Write, then read, then check for equality

        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("test_write_las_format_4_different_layout.las");

        defer! {
            std::fs::remove_file(&test_file_path).expect("Removing test file failed!");
        }

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(4)?;

        {
            let mut writer = LASWriter::<BufWriter<File>>::from_path_and_header(
                &test_file_path,
                las_header_builder.into_header().unwrap(),
            )?;
            writer.write(&source_point_buffer)?;
        }

        {
            let mut reader = LASReader::from_path(&test_file_path)?;
            let read_points_buffer = reader.read(source_points.len())?;
            let read_points =
                points::<LasPointFormat4>(read_points_buffer.as_ref()).collect::<Vec<_>>();

            for (source, read) in source_points.iter().zip(read_points.iter()) {
                assert_eq!(
                    { source.position },
                    { read.position },
                    "Position of read point is different than of source point!"
                );
                assert_eq!(
                    0, read.classification,
                    "Classification of read point was not zero!"
                );
                assert_eq!(
                    false, read.edge_of_flight_line,
                    "Edge of flight line of read point was not false!"
                );
                assert_eq!(
                    0,
                    { read.intensity },
                    "Intensity of read point was not zero!"
                );
                assert_eq!(
                    0, read.number_of_returns,
                    "Number of returns of read point was not zero!"
                );
                assert_eq!(
                    0,
                    { read.point_source_id },
                    "Point source ID of read point was not zero!"
                );
                assert_eq!(
                    0, read.return_number,
                    "Return number of read point was not zero!"
                );
                assert_eq!(
                    0, read.scan_angle_rank,
                    "Scan angle rank of read point was not zero!"
                );
                assert_eq!(
                    false, read.scan_direction_flag,
                    "Scan direction flag of read point was not false!"
                );
                assert_eq!(0, read.user_data, "User data of read point was not zero!");
                assert_eq!(
                    0,
                    { read.byte_offset_to_waveform_data },
                    "Byte offset to waveform data of read point was not zero!"
                );
                assert_eq!(
                    0.0,
                    { read.return_point_waveform_location },
                    "Return point waveform location of read point was not zero!"
                );
                assert_eq!(
                    0, read.wave_packet_descriptor_index,
                    "Wave packet descriptor index of read point was not zero!"
                );
                assert_eq!(
                    0,
                    { read.waveform_packet_size },
                    "Waveform packet size of read point was not zero!"
                );
                assert_eq!(
                    { Vector3::<f32>::new(0.0, 0.0, 0.0) },
                    { read.waveform_parameters },
                    "Waveform parameters of read point were not zero!"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_write_las_format_5() -> Result<()> {
        let source_points = get_test_points_las_format_5();
        let source_point_buffer = prepare_point_buffer(&source_points);

        //Write, then read, then check for equality

        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("test_write_las_format_5.las");

        defer! {
            std::fs::remove_file(&test_file_path).expect("Removing test file failed!");
        }

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(5)?;

        {
            let mut writer = LASWriter::<BufWriter<File>>::from_path_and_header(
                &test_file_path,
                las_header_builder.into_header().unwrap(),
            )?;
            writer.write(&source_point_buffer)?;
        }

        {
            let mut reader = LASReader::from_path(&test_file_path)?;
            let read_points_buffer = reader.read(source_points.len())?;
            let read_points =
                points::<LasPointFormat5>(read_points_buffer.as_ref()).collect::<Vec<_>>();

            assert_eq!(read_points, source_points);
        }

        Ok(())
    }

    #[test]
    fn test_write_las_format_5_different_layout() -> Result<()> {
        let source_points = get_test_points_custom_format();
        let source_point_buffer = prepare_point_buffer(&source_points);

        //Write, then read, then check for equality

        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("test_write_las_format_5_different_layout.las");

        defer! {
            std::fs::remove_file(&test_file_path).expect("Removing test file failed!");
        }

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(5)?;

        {
            let mut writer = LASWriter::<BufWriter<File>>::from_path_and_header(
                &test_file_path,
                las_header_builder.into_header().unwrap(),
            )?;
            writer.write(&source_point_buffer)?;
        }

        {
            let mut reader = LASReader::from_path(&test_file_path)?;
            let read_points_buffer = reader.read(source_points.len())?;
            let read_points =
                points::<LasPointFormat5>(read_points_buffer.as_ref()).collect::<Vec<_>>();

            for (source, read) in source_points.iter().zip(read_points.iter()) {
                assert_eq!(
                    { source.position },
                    { read.position },
                    "Position of read point is different than of source point!"
                );
                assert_eq!(
                    { source.color },
                    { read.color_rgb },
                    "Colors of read point are different from colors in source point!"
                );
                assert_eq!(
                    0, read.classification,
                    "Classification of read point was not zero!"
                );
                assert_eq!(
                    false, read.edge_of_flight_line,
                    "Edge of flight line of read point was not false!"
                );
                assert_eq!(
                    0,
                    { read.intensity },
                    "Intensity of read point was not zero!"
                );
                assert_eq!(
                    0, read.number_of_returns,
                    "Number of returns of read point was not zero!"
                );
                assert_eq!(
                    0,
                    { read.point_source_id },
                    "Point source ID of read point was not zero!"
                );
                assert_eq!(
                    0, read.return_number,
                    "Return number of read point was not zero!"
                );
                assert_eq!(
                    0, read.scan_angle_rank,
                    "Scan angle rank of read point was not zero!"
                );
                assert_eq!(
                    false, read.scan_direction_flag,
                    "Scan direction flag of read point was not false!"
                );
                assert_eq!(0, read.user_data, "User data of read point was not zero!");
                assert_eq!(
                    0.0,
                    { read.gps_time },
                    "GPS time of read point was not zero!"
                );
                assert_eq!(
                    0,
                    { read.byte_offset_to_waveform_data },
                    "Byte offset to waveform data of read point was not zero!"
                );
                assert_eq!(
                    0.0,
                    { read.return_point_waveform_location },
                    "Return point waveform location of read point was not zero!"
                );
                assert_eq!(
                    0, read.wave_packet_descriptor_index,
                    "Wave packet descriptor index of read point was not zero!"
                );
                assert_eq!(
                    0,
                    { read.waveform_packet_size },
                    "Waveform packet size of read point was not zero!"
                );
                assert_eq!(
                    { Vector3::<f32>::new(0.0, 0.0, 0.0) },
                    { read.waveform_parameters },
                    "Waveform parameters of read point were not zero!"
                );
            }
        }

        Ok(())
    }
}
