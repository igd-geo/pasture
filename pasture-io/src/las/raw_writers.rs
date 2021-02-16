use std::{
    collections::HashMap,
    convert::TryInto,
    io::{Cursor, SeekFrom},
};

use anyhow::{anyhow, Result};
use byteorder::{LittleEndian, NativeEndian, ReadBytesExt, WriteBytesExt};
use las_rs::point::Format;
use pasture_core::{containers::PointBuffer, layout::PointLayout, nalgebra::Vector3};

use crate::base::PointWriter;

use super::{
    get_classification_flags_reader, get_classification_reader, get_color_reader,
    get_edge_of_flight_line_reader, get_extended_scan_angle_rank_reader, get_gps_time_reader,
    get_intensity_reader, get_nir_reader, get_number_of_returns_reader, get_point_source_id_reader,
    get_position_reader, get_return_number_reader, get_return_point_waveform_location_reader,
    get_scan_angle_rank_reader, get_scan_direction_flag_reader, get_scanner_channel_reader,
    get_user_data_reader, get_wave_packet_descriptor_index_reader, get_waveform_data_offset_reader,
    get_waveform_packet_size_reader, get_waveform_parameters_reader,
    point_layout_from_las_point_format, BitAttributes, BitAttributesExtended, BitAttributesRegular,
};

pub(crate) struct RawLASWriter<T: std::io::Write + std::io::Seek> {
    writer: T,
    default_layout: PointLayout,
    current_header: las::raw::Header,
    evlrs: Vec<las::raw::Vlr>,
    _point_start_index: u64,
    requires_flush: bool,
}

impl<T: std::io::Write + std::io::Seek> RawLASWriter<T> {
    pub fn from_write_and_header(mut write: T, header: las::Header) -> Result<Self> {
        let default_layout = point_layout_from_las_point_format(header.point_format())?;

        // Sanitize header, i.e. clear point counts and bounds
        // TODO Add flag to prevent recalculating bounds
        let mut raw_header = header.clone().into_raw()?;
        raw_header.number_of_point_records = 0;
        raw_header.number_of_points_by_return = [0; 5];
        if let Some(large_file) = &mut raw_header.large_file {
            large_file.number_of_point_records = 0;
            large_file.number_of_points_by_return = [0; 15];
        }
        raw_header.min_x = std::f64::MAX;
        raw_header.min_y = std::f64::MAX;
        raw_header.min_z = std::f64::MAX;
        raw_header.max_x = std::f64::MIN;
        raw_header.max_y = std::f64::MIN;
        raw_header.max_z = std::f64::MIN;

        if raw_header.x_scale_factor == 0.0
            || raw_header.y_scale_factor == 0.0
            || raw_header.z_scale_factor == 0.0
        {
            return Err(anyhow!("RawLASWriter::from_write_and_header: Scale factors in LAS header must not be zero!"));
        }

        raw_header.write_to(&mut write)?;
        for vlr in header.vlrs().iter() {
            if vlr.has_large_data() {
                panic!("RawLASWriter::from_write_and_header: Header with large VLRs is currently unsupported! Please add any large VLRs to the 'evlrs' parameter of the header!");
            }
            let raw_vlr = vlr.clone().into_raw(false)?;
            raw_vlr.write_to(&mut write)?;
        }

        let point_start_index = write.seek(SeekFrom::Current(0))?;
        assert_eq!(point_start_index, raw_header.offset_to_point_data as u64);

        Ok(Self {
            writer: write,
            default_layout,
            current_header: raw_header,
            evlrs: header
                .evlrs()
                .iter()
                .map(|evlr| evlr.clone().into_raw(true))
                .collect::<Result<Vec<_>, _>>()?,
            _point_start_index: point_start_index,
            requires_flush: true,
        })
    }

    /// Writes the current header to the start of the file
    fn write_header(&mut self) -> Result<()> {
        let current_position = self.writer.seek(SeekFrom::Current(0))?;
        self.writer.seek(SeekFrom::Start(0))?;
        self.current_header.write_to(&mut self.writer)?;
        self.writer.seek(SeekFrom::Start(current_position))?;
        Ok(())
    }

    /// Writes the extended VLRs to the end of the file
    fn write_evlrs(&mut self) -> Result<()> {
        // Assumes that self.writer is at the end of the file!
        for evlr in self.evlrs.iter() {
            evlr.write_to(&mut self.writer)?;
        }
        Ok(())
    }

    fn write_points_default_layout(&mut self, points: &dyn PointBuffer) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }

        // Similar to RawLASReader, write points in chunks of a fixed size to prevent overhead of
        // repeated virtual calls to 'dyn PointBuffer'

        let size_of_single_point = self.default_layout.size_of_point_entry() as usize;
        let num_points_in_chunk = 50_000;
        let num_chunks = (points.len() + (num_points_in_chunk - 1)) / num_points_in_chunk;
        let mut chunk_buffer: Vec<u8> = vec![0; num_points_in_chunk * size_of_single_point];

        let source_format = Format::new(self.current_header.point_data_record_format)?;

        let mut points_by_return: HashMap<u8, u64> = HashMap::new();
        let max_return_number = if self.current_header.large_file.is_some() {
            15
        } else {
            5
        };
        for return_number in 1..=max_return_number {
            points_by_return.insert(return_number, 0);
        }

        for chunk_index in 0..num_chunks {
            let points_in_cur_chunk = std::cmp::min(
                num_points_in_chunk,
                points.len() - (chunk_index * num_points_in_chunk),
            );
            let start_point_index = chunk_index * num_points_in_chunk;
            points.get_points_by_copy(
                start_point_index..(start_point_index + points_in_cur_chunk),
                &mut chunk_buffer,
            );
            let mut point_read = Cursor::new(chunk_buffer);

            // Read all the attributes from the raw memory inside `points` and transform them into the format that LAS expects
            for _ in 0..points_in_cur_chunk {
                let pos_x = point_read.read_f64::<NativeEndian>()?;
                let pos_y = point_read.read_f64::<NativeEndian>()?;
                let pos_z = point_read.read_f64::<NativeEndian>()?;
                self.write_next_position(&Vector3::new(pos_x, pos_y, pos_z))?;

                let intensity = point_read.read_u16::<NativeEndian>()?;
                self.writer.write_u16::<LittleEndian>(intensity)?;

                let bit_attributes = if source_format.is_extended {
                    let return_number = point_read.read_u8()?;
                    points_by_return
                        .get_mut(&return_number)
                        .map(|count| *count += 1);
                    let number_of_returns = point_read.read_u8()?;
                    let classification_flags = point_read.read_u8()?;
                    let scanner_channel = point_read.read_u8()?;
                    let scan_direction_flag = point_read.read_u8()?;
                    let edge_of_flight_line = point_read.read_u8()?;
                    BitAttributes::Extended(BitAttributesExtended {
                        return_number,
                        number_of_returns,
                        classification_flags,
                        scanner_channel,
                        scan_direction_flag,
                        edge_of_flight_line,
                    })
                } else {
                    let return_number = point_read.read_u8()?;
                    points_by_return
                        .get_mut(&return_number)
                        .map(|count| *count += 1);
                    let number_of_returns = point_read.read_u8()?;
                    let scan_direction_flag = point_read.read_u8()?;
                    let edge_of_flight_line = point_read.read_u8()?;
                    BitAttributes::Regular(BitAttributesRegular {
                        return_number,
                        number_of_returns,
                        scan_direction_flag,
                        edge_of_flight_line,
                    })
                };
                self.write_next_bit_attributes(bit_attributes)?;

                let classification = point_read.read_u8()?;
                self.writer.write_u8(classification)?;

                if source_format.is_extended {
                    let user_data = point_read.read_u8()?;
                    let scan_angle = point_read.read_i16::<NativeEndian>()?;

                    self.writer.write_u8(user_data)?;
                    self.writer.write_i16::<LittleEndian>(scan_angle)?;
                } else {
                    let scan_angle = point_read.read_i8()?;
                    let user_data = point_read.read_u8()?;

                    self.writer.write_i8(scan_angle)?;
                    self.writer.write_u8(user_data)?;
                }

                let point_source_id = point_read.read_u16::<NativeEndian>()?;
                self.writer.write_u16::<LittleEndian>(point_source_id)?;

                if source_format.has_gps_time {
                    let gps_time = point_read.read_f64::<NativeEndian>()?;
                    self.writer.write_f64::<LittleEndian>(gps_time)?;
                }

                if source_format.has_color {
                    let r = point_read.read_u16::<NativeEndian>()?;
                    let g = point_read.read_u16::<NativeEndian>()?;
                    let b = point_read.read_u16::<NativeEndian>()?;
                    self.writer.write_u16::<LittleEndian>(r)?;
                    self.writer.write_u16::<LittleEndian>(g)?;
                    self.writer.write_u16::<LittleEndian>(b)?;
                }

                if source_format.has_nir {
                    let nir = point_read.read_u16::<NativeEndian>()?;
                    self.writer.write_u16::<LittleEndian>(nir)?;
                }

                if source_format.has_waveform {
                    let wave_descriptor = point_read.read_u8()?;
                    let wave_data_offset = point_read.read_u64::<NativeEndian>()?;
                    let wave_packet_size = point_read.read_u32::<NativeEndian>()?;
                    let wave_return_point = point_read.read_f32::<NativeEndian>()?;
                    let px = point_read.read_f32::<NativeEndian>()?;
                    let py = point_read.read_f32::<NativeEndian>()?;
                    let pz = point_read.read_f32::<NativeEndian>()?;

                    self.writer.write_u8(wave_descriptor)?;
                    self.writer.write_u64::<LittleEndian>(wave_data_offset)?;
                    self.writer.write_u32::<LittleEndian>(wave_packet_size)?;
                    self.writer.write_f32::<LittleEndian>(wave_return_point)?;
                    self.writer.write_f32::<LittleEndian>(px)?;
                    self.writer.write_f32::<LittleEndian>(py)?;
                    self.writer.write_f32::<LittleEndian>(pz)?;
                }
            }

            chunk_buffer = point_read.into_inner();
        }

        self.update_point_counts(points.len(), &points_by_return);
        self.requires_flush = true;

        Ok(())
    }

    fn write_points_custom_layout(&mut self, points: &dyn PointBuffer) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }

        let size_of_single_point = points.point_layout().size_of_point_entry() as usize;
        let num_points_in_chunk = 50_000;
        let num_chunks = (points.len() + (num_points_in_chunk - 1)) / num_points_in_chunk;
        let mut chunk_buffer: Vec<u8> = vec![0; num_points_in_chunk * size_of_single_point];

        let target_format = Format::new(self.current_header.point_data_record_format)?;

        let mut points_by_return: HashMap<u8, u64> = HashMap::new();
        let max_return_number = if self.current_header.large_file.is_some() {
            15
        } else {
            5
        };
        for return_number in 1..=max_return_number {
            points_by_return.insert(return_number, 0);
        }

        // ########### TODO ############
        // I need a small function for each attribute that does 'read_from_chunk_buffer_or_default', i.e. it tries to read
        // the attribute from the binary data in the chunk_buffer - if the attribute exists - or returns some default value
        // In addition to that, IF the attribute exists in the chunk buffer, this function also has to apply conversion if necessary!
        // Then, the inner loop is just a loop over these functions

        let position_reader = get_position_reader(points.point_layout());
        let intensity_reader = get_intensity_reader(points.point_layout());
        let return_number_reader = get_return_number_reader(points.point_layout());
        let number_of_returns_reader = get_number_of_returns_reader(points.point_layout());
        let classification_flags_reader = if target_format.is_extended {
            Some(get_classification_flags_reader(points.point_layout()))
        } else {
            None
        };
        let scanner_channel_reader = if target_format.is_extended {
            Some(get_scanner_channel_reader(points.point_layout()))
        } else {
            None
        };
        let scan_direction_flag_reader = get_scan_direction_flag_reader(points.point_layout());
        let edge_of_flight_line_reader = get_edge_of_flight_line_reader(points.point_layout());
        let classification_reader = get_classification_reader(points.point_layout());
        let user_data_reader = get_user_data_reader(points.point_layout());
        let scan_angle_reader = if target_format.is_extended {
            None
        } else {
            Some(get_scan_angle_rank_reader(points.point_layout()))
        };
        let extended_scan_angle_reader = if target_format.is_extended {
            Some(get_extended_scan_angle_rank_reader(points.point_layout()))
        } else {
            None
        };
        let point_source_id_reader = get_point_source_id_reader(points.point_layout());
        let gps_time_reader = if target_format.has_gps_time {
            Some(get_gps_time_reader(points.point_layout()))
        } else {
            None
        };
        let color_reader = if target_format.has_color {
            Some(get_color_reader(points.point_layout()))
        } else {
            None
        };
        let nir_reader = if target_format.has_nir {
            Some(get_nir_reader(points.point_layout()))
        } else {
            None
        };
        let wave_packet_descriptor_index_reader = if target_format.has_waveform {
            Some(get_wave_packet_descriptor_index_reader(
                points.point_layout(),
            ))
        } else {
            None
        };
        let waveform_data_offset_reader = if target_format.has_waveform {
            Some(get_waveform_data_offset_reader(points.point_layout()))
        } else {
            None
        };
        let waveform_packet_size_reader = if target_format.has_waveform {
            Some(get_waveform_packet_size_reader(points.point_layout()))
        } else {
            None
        };
        let return_point_waveform_location_reader = if target_format.has_waveform {
            Some(get_return_point_waveform_location_reader(
                points.point_layout(),
            ))
        } else {
            None
        };
        let waveform_parameters_reader = if target_format.has_waveform {
            Some(get_waveform_parameters_reader(points.point_layout()))
        } else {
            None
        };

        for chunk_index in 0..num_chunks {
            let points_in_cur_chunk = std::cmp::min(
                num_points_in_chunk,
                points.len() - (chunk_index * num_points_in_chunk),
            );
            let start_point_index = chunk_index * num_points_in_chunk;
            points.get_points_by_copy(
                start_point_index..(start_point_index + points_in_cur_chunk),
                &mut chunk_buffer[0..(points_in_cur_chunk * size_of_single_point)],
            );
            let mut point_read = Cursor::new(chunk_buffer);

            // Read all the attributes from the raw memory inside `points` and transform them into the format that LAS expects
            for point_index in 0..points_in_cur_chunk {
                let position = position_reader(point_index, &mut point_read)?;
                self.write_next_position(&position)?;

                self.writer
                    .write_u16::<LittleEndian>(intensity_reader(point_index, &mut point_read)?)?;

                let bit_attributes: BitAttributes = if target_format.is_extended {
                    BitAttributes::Extended(BitAttributesExtended {
                        return_number: return_number_reader(point_index, &mut point_read)?,
                        number_of_returns: number_of_returns_reader(point_index, &mut point_read)?,
                        classification_flags: classification_flags_reader.as_ref().unwrap()(
                            point_index,
                            &mut point_read,
                        )?,
                        scanner_channel: scanner_channel_reader.as_ref().unwrap()(
                            point_index,
                            &mut point_read,
                        )?,
                        scan_direction_flag: if scan_direction_flag_reader(
                            point_index,
                            &mut point_read,
                        )? {
                            1
                        } else {
                            0
                        },
                        edge_of_flight_line: if edge_of_flight_line_reader(
                            point_index,
                            &mut point_read,
                        )? {
                            1
                        } else {
                            0
                        },
                    })
                } else {
                    BitAttributes::Regular(BitAttributesRegular {
                        return_number: return_number_reader(point_index, &mut point_read)?,
                        number_of_returns: number_of_returns_reader(point_index, &mut point_read)?,
                        scan_direction_flag: if scan_direction_flag_reader(
                            point_index,
                            &mut point_read,
                        )? {
                            1
                        } else {
                            0
                        },
                        edge_of_flight_line: if edge_of_flight_line_reader(
                            point_index,
                            &mut point_read,
                        )? {
                            1
                        } else {
                            0
                        },
                    })
                };
                self.write_next_bit_attributes(bit_attributes)?;

                self.writer
                    .write_u8(classification_reader(point_index, &mut point_read)?)?;

                if target_format.is_extended {
                    self.writer
                        .write_u8(user_data_reader(point_index, &mut point_read)?)?;
                    self.writer
                        .write_i16::<LittleEndian>(extended_scan_angle_reader.as_ref().unwrap()(
                            point_index,
                            &mut point_read,
                        )?)?;
                } else {
                    self.writer.write_i8(scan_angle_reader.as_ref().unwrap()(
                        point_index,
                        &mut point_read,
                    )?)?;
                    self.writer
                        .write_u8(user_data_reader(point_index, &mut point_read)?)?;
                }

                self.writer
                    .write_u16::<LittleEndian>(point_source_id_reader(
                        point_index,
                        &mut point_read,
                    )?)?;

                if let Some(ref reader) = gps_time_reader {
                    self.writer
                        .write_f64::<LittleEndian>(reader(point_index, &mut point_read)?)?;
                }

                if let Some(ref reader) = color_reader {
                    let color = reader(point_index, &mut point_read)?;
                    self.writer.write_u16::<LittleEndian>(color.x)?;
                    self.writer.write_u16::<LittleEndian>(color.y)?;
                    self.writer.write_u16::<LittleEndian>(color.z)?;
                }

                if let Some(ref reader) = nir_reader {
                    self.writer
                        .write_u16::<LittleEndian>(reader(point_index, &mut point_read)?)?;
                }

                if let Some(ref reader) = wave_packet_descriptor_index_reader {
                    self.writer
                        .write_u8(reader(point_index, &mut point_read)?)?;
                }
                if let Some(ref reader) = waveform_data_offset_reader {
                    self.writer
                        .write_u64::<LittleEndian>(reader(point_index, &mut point_read)?)?;
                }
                if let Some(ref reader) = waveform_packet_size_reader {
                    self.writer
                        .write_u32::<LittleEndian>(reader(point_index, &mut point_read)?)?;
                }
                if let Some(ref reader) = return_point_waveform_location_reader {
                    self.writer
                        .write_f32::<LittleEndian>(reader(point_index, &mut point_read)?)?;
                }
                if let Some(ref reader) = waveform_parameters_reader {
                    let params = reader(point_index, &mut point_read)?;
                    self.writer.write_f32::<LittleEndian>(params.x)?;
                    self.writer.write_f32::<LittleEndian>(params.y)?;
                    self.writer.write_f32::<LittleEndian>(params.z)?;
                }
            }

            chunk_buffer = point_read.into_inner();
        }

        self.update_point_counts(points.len(), &points_by_return);
        self.requires_flush = true;

        Ok(())
    }

    fn write_next_position(&mut self, world_space_position: &Vector3<f64>) -> Result<()> {
        let local_x : i32 = (((world_space_position.x - self.current_header.x_offset) / self.current_header.x_scale_factor) as i64).try_into().expect("RawLASWriter::write_next_position: Position is out of bounds given the current LAS offset and scale!");
        let local_y : i32 = (((world_space_position.y - self.current_header.y_offset) / self.current_header.y_scale_factor) as i64).try_into().expect("RawLASWriter::write_next_position: Position is out of bounds given the current LAS offset and scale!");
        let local_z : i32 = (((world_space_position.z - self.current_header.z_offset) / self.current_header.z_scale_factor) as i64).try_into().expect("RawLASWriter::write_next_position: Position is out of bounds given the current LAS offset and scale!");
        self.writer.write_i32::<LittleEndian>(local_x)?;
        self.writer.write_i32::<LittleEndian>(local_y)?;
        self.writer.write_i32::<LittleEndian>(local_z)?;

        self.update_bounds(world_space_position);

        Ok(())
    }

    fn write_next_bit_attributes(&mut self, bit_attributes: BitAttributes) -> Result<()> {
        match bit_attributes {
            BitAttributes::Regular(attributes) => {
                let mask = (attributes.return_number & 0b111)
                    | (attributes.number_of_returns & 0b111) << 3
                    | (attributes.scan_direction_flag & 0b1) << 6
                    | (attributes.edge_of_flight_line & 0b1) << 7;
                self.writer.write_u8(mask)?;
            }
            BitAttributes::Extended(attributes) => {
                let low_mask = (attributes.return_number & 0b1111)
                    | (attributes.number_of_returns & 0b1111) << 4;
                let high_mask = (attributes.classification_flags & 0b1111)
                    | (attributes.scanner_channel & 0b11) << 4
                    | (attributes.scan_direction_flag & 0b1) << 6
                    | (attributes.edge_of_flight_line & 0b1) << 7;
                self.writer.write_u8(low_mask)?;
                self.writer.write_u8(high_mask)?;
            }
        }

        Ok(())
    }

    /// Update the bounds in the LAS header using the given `new_position`
    fn update_bounds(&mut self, new_position: &Vector3<f64>) {
        if new_position.x < self.current_header.min_x {
            self.current_header.min_x = new_position.x;
        }
        if new_position.y < self.current_header.min_y {
            self.current_header.min_y = new_position.y;
        }
        if new_position.z < self.current_header.min_z {
            self.current_header.min_z = new_position.z;
        }
        if new_position.x > self.current_header.max_x {
            self.current_header.max_x = new_position.x;
        }
        if new_position.y > self.current_header.max_y {
            self.current_header.max_y = new_position.y;
        }
        if new_position.z > self.current_header.max_z {
            self.current_header.max_z = new_position.z;
        }
    }

    fn update_point_counts(
        &mut self,
        additional_points: usize,
        additional_points_by_return: &HashMap<u8, u64>,
    ) {
        if let Some(large_file) = self.current_header.large_file.as_mut() {
            large_file.number_of_point_records += additional_points as u64;
            additional_points_by_return
                .iter()
                .for_each(|(return_number, additional_count)| {
                    large_file.number_of_points_by_return[(return_number - 1) as usize] +=
                        additional_count;
                });
        } else {
            self.current_header.number_of_point_records =
                (self.current_header.number_of_point_records as u64 + additional_points as u64)
                    .try_into()
                    .expect(
                        "RawLASWriter::write: Number of points too large for LAS file version! Consider using LAS version 1.4, which supports 64-bit point counts (set the version field in the LAS header to Version::new(1,4)).",
                    );
            additional_points_by_return
                    .iter()
                    .for_each(|(return_number, additional_count)| {
                        let current_count = self.current_header.number_of_points_by_return[(return_number - 1) as usize];
                        self.current_header.number_of_points_by_return[(return_number - 1) as usize] = (current_count as u64 + additional_count).try_into().expect("RawLASWriter::write: Number of points by return too large for LAS file version! Consider using LAS version 1.4, which supports 64-bit point counts (set the version field in the LAS header to Version::new(1,4)).",)
                    });
        }
    }
}

impl<T: std::io::Write + std::io::Seek> PointWriter for RawLASWriter<T> {
    fn write(&mut self, points: &dyn PointBuffer) -> Result<()> {
        if *points.point_layout() == self.default_layout {
            self.write_points_default_layout(points)
        } else {
            self.write_points_custom_layout(points)
        }
    }

    fn flush(&mut self) -> Result<()> {
        if !self.requires_flush {
            return Ok(());
        }

        let current_index = self.writer.seek(SeekFrom::Current(0))?;
        self.write_header()?;
        self.write_evlrs()?;
        self.writer.seek(SeekFrom::Start(current_index))?;

        self.requires_flush = false;

        Ok(())
    }

    fn get_default_point_layout(&self) -> &PointLayout {
        &self.default_layout
    }
}

impl<T: std::io::Write + std::io::Seek> Drop for RawLASWriter<T> {
    fn drop(&mut self) {
        self.flush()
            .expect("RawLASWriter::drop: Could not flush point data");
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::BufWriter};

    use las_rs::Builder;
    use pasture_core::containers::{points, InterleavedVecPointStorage};
    use pasture_core::layout::PointType;
    use pasture_core::math::AABB;
    use pasture_core::nalgebra::Point3;

    use crate::{
        base::PointReader,
        las::{
            epsilon_compare_point3f64, epsilon_compare_vec3f64, get_test_points_in_las_format,
            test_data_bounds, LASReader, LasPointFormat0, LasPointFormat1, LasPointFormat10,
            LasPointFormat2, LasPointFormat3, LasPointFormat4, LasPointFormat5, LasPointFormat6,
            LasPointFormat7, LasPointFormat8, LasPointFormat9,
        },
    };
    use pasture_derive::PointType;
    use scopeguard::defer;

    use super::*;

    macro_rules! las_write_tests {
        ($name:ident, $format:expr, $point_type:ident) => {
            mod $name {
                use super::*;

                #[test]
                fn test_raw_las_writer() -> Result<()> {
                    let test_data = get_test_points_in_las_format($format)?;

                    let format = Format::new($format)?;
                    let mut header_builder = Builder::from((1, 4));
                    header_builder.point_format = format.clone();

                    let out_path = format!("./test_raw_las_writer_format_{}.las", $format);
                    defer! {
                        std::fs::remove_file(&out_path).expect("Could not remove test file");
                    }
                    {
                        let mut writer = RawLASWriter::from_write_and_header(
                            BufWriter::new(File::create(&out_path)?),
                            header_builder.into_header()?,
                        )?;

                        let expected_format = point_layout_from_las_point_format(&format)?;
                        assert_eq!(expected_format, *writer.get_default_point_layout());

                        writer.write(test_data.as_ref())?;
                    }

                    {
                        let mut reader = LASReader::from_path(&out_path)?;
                        let metadata = reader.get_metadata();
                        assert_eq!(Some(test_data_bounds()), metadata.bounds());
                        assert_eq!(test_data.len(), reader.remaining_points());

                        let read_points = reader.read(test_data.len())?;

                        assert_eq!(read_points.point_layout(), test_data.point_layout());
                        assert_eq!(read_points.len(), test_data.len());

                        let expected_points =
                            points::<$point_type>(test_data.as_ref()).collect::<Vec<_>>();
                        let actual_points =
                            points::<$point_type>(read_points.as_ref()).collect::<Vec<_>>();

                        assert_eq!(expected_points, actual_points);
                    }

                    Ok(())
                }

                #[repr(C)]
                #[derive(PointType)]
                struct CustomPointType {
                    #[pasture(BUILTIN_INTENSITY)]
                    pub intensity: u16,
                    #[pasture(BUILTIN_POSITION_3D)]
                    pub lowp_position: Vector3<f32>,
                    #[pasture(attribute = "Custom")]
                    pub custom_attribute: u32,
                }

                #[test]
                fn test_raw_las_writer_from_different_point_layout() -> Result<()> {
                    // Get test data with a different layout, e.g. some missing attributes, some additional
                    // attributes that are not supported by LAS
                    let test_data = vec![
                        CustomPointType {
                            intensity: 42,
                            lowp_position: Vector3::new(0.1, 0.2, 0.3),
                            custom_attribute: 1337,
                        },
                        CustomPointType {
                            intensity: 43,
                            lowp_position: Vector3::new(0.4, 0.5, 0.6),
                            custom_attribute: 7331,
                        },
                    ];

                    let expected_bounds =
                        AABB::from_min_max(Point3::new(0.1, 0.2, 0.3), Point3::new(0.4, 0.5, 0.6));

                    let mut expected_data =
                        InterleavedVecPointStorage::new(CustomPointType::layout());
                    expected_data.push_points(test_data.as_slice());

                    let format = Format::new($format)?;
                    let mut header_builder = Builder::from((1, 4));
                    header_builder.point_format = format.clone();

                    let out_path =
                        format!("./test_raw_las_writer_different_format_{}.las", $format);

                    defer! {
                        std::fs::remove_file(&out_path).expect("Could not remove test file");
                    }

                    {
                        let mut writer = RawLASWriter::from_write_and_header(
                            BufWriter::new(File::create(&out_path)?),
                            header_builder.into_header()?,
                        )?;

                        writer.write(&expected_data)?;
                    }

                    {
                        let mut reader = LASReader::from_path(&out_path)?;
                        let metadata = reader.get_metadata();
                        assert!(metadata.bounds().is_some());
                        let actual_bounds = metadata.bounds().unwrap();
                        assert!(
                            epsilon_compare_point3f64(expected_bounds.min(), actual_bounds.min()),
                            "Bounds are different! Expected {:?} but was {:?}",
                            expected_bounds,
                            actual_bounds
                        );
                        assert!(
                            epsilon_compare_point3f64(expected_bounds.max(), actual_bounds.max()),
                            "Bounds are different! Expected {:?} but was {:?}",
                            expected_bounds,
                            actual_bounds
                        );

                        assert_eq!(test_data.len(), reader.remaining_points());

                        let read_points = reader.read(test_data.len())?;

                        assert_eq!(read_points.len(), test_data.len());

                        let mut actual_points =
                            points::<$point_type>(read_points.as_ref()).collect::<Vec<_>>();

                        // Expected positions were f32, converted to f64, this might yield rounding errors, so we compare positions separately
                        for (idx, (expected, actual)) in
                            test_data.iter().zip(actual_points.iter()).enumerate()
                        {
                            let expected_pos_lowp = expected.lowp_position;
                            let actual_position = actual.position;
                            let expected_highp = Vector3::new(
                                expected_pos_lowp.x as f64,
                                expected_pos_lowp.y as f64,
                                expected_pos_lowp.z as f64,
                            );
                            assert!(
                                epsilon_compare_vec3f64(&expected_highp, &actual_position),
                                "Position {} is different! Expected {} but was {}",
                                idx,
                                expected_highp,
                                actual_position
                            );
                        }

                        // Zero out positions so that we can compare the other attributes
                        actual_points
                            .iter_mut()
                            .for_each(|point| point.position = Default::default());

                        let expected_points = test_data
                            .iter()
                            .map(|test_point| -> $point_type {
                                let mut default_point: $point_type = Default::default();
                                default_point.intensity = test_point.intensity;
                                default_point
                            })
                            .collect::<Vec<_>>();

                        assert_eq!(expected_points, actual_points);
                    }

                    Ok(())
                }
            }
        };
    }

    las_write_tests!(las_write_0, 0, LasPointFormat0);
    las_write_tests!(las_write_1, 1, LasPointFormat1);
    las_write_tests!(las_write_2, 2, LasPointFormat2);
    las_write_tests!(las_write_3, 3, LasPointFormat3);
    las_write_tests!(las_write_4, 4, LasPointFormat4);
    las_write_tests!(las_write_5, 5, LasPointFormat5);
    las_write_tests!(las_write_6, 6, LasPointFormat6);
    las_write_tests!(las_write_7, 7, LasPointFormat7);
    las_write_tests!(las_write_8, 8, LasPointFormat8);
    las_write_tests!(las_write_9, 9, LasPointFormat9);
    las_write_tests!(las_write_10, 10, LasPointFormat10);
}
