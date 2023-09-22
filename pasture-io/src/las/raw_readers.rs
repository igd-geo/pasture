use std::convert::TryInto;
use std::io::{Read, Seek, SeekFrom};

use anyhow::{anyhow, bail, Context, Result};
use las_rs::Header;
use las_rs::{raw, Builder, Vlr};
use laz::LasZipDecompressor;
use pasture_core::containers::{BorrowedMutBuffer, OwningBuffer, VectorBuffer};
use pasture_core::layout::attributes::{
    CLASSIFICATION_FLAGS, EDGE_OF_FLIGHT_LINE, NUMBER_OF_RETURNS, POSITION_3D, RETURN_NUMBER,
    SCANNER_CHANNEL, SCAN_DIRECTION_FLAG,
};
use pasture_core::layout::conversion::BufferLayoutConverter;
use pasture_core::layout::PointAttributeDataType;
use pasture_core::nalgebra::Vector3;
use pasture_core::{layout::PointLayout, meta::Metadata};

use super::{
    map_laz_err, point_layout_from_las_metadata, LASMetadata, ATTRIBUTE_LOCAL_LAS_POSITION,
};
use crate::base::{PointReader, SeekToPoint};
use crate::las::{ATTRIBUTE_BASIC_FLAGS, ATTRIBUTE_EXTENDED_FLAGS};

/// Is the given VLR the LASzip VLR? Function taken from the `las` crate because it is not exported there
fn is_laszip_vlr(vlr: &Vlr) -> bool {
    vlr.user_id == laz::LazVlr::USER_ID && vlr.record_id == laz::LazVlr::RECORD_ID
}

/// Returns a `BufferLayoutConverter` that performs a conversion from the given raw LAS `PointLayout` into
/// the given `target_layout`
fn get_default_las_converter<'a>(
    raw_las_layout: &'a PointLayout,
    target_layout: &'a PointLayout,
    las_header: &Header,
) -> Result<BufferLayoutConverter<'a>> {
    let mut converter =
        BufferLayoutConverter::for_layouts_with_default(raw_las_layout, target_layout);
    // Add custom conversions depending on the target layout
    if let Some(position_attribute) = target_layout.get_attribute_by_name(POSITION_3D.name()) {
        let transforms = *las_header.transforms();
        match position_attribute.datatype() {
            PointAttributeDataType::Vec3f64 => converter.set_custom_mapping_with_transformation(&ATTRIBUTE_LOCAL_LAS_POSITION, position_attribute.attribute_definition(), move |pos: Vector3<f64>| -> Vector3<f64> {
                Vector3::new(
                    (pos.x * transforms.x.scale) + transforms.x.offset,
                    (pos.y * transforms.y.scale) + transforms.y.offset,
                    (pos.z * transforms.z.scale) + transforms.z.offset,
                )
            }, false),
            PointAttributeDataType::Vec3f32 => converter.set_custom_mapping_with_transformation(&ATTRIBUTE_LOCAL_LAS_POSITION, position_attribute.attribute_definition(), move |pos: Vector3<f32>| -> Vector3<f32> {
                Vector3::new(
                    ((pos.x as f64 * transforms.x.scale) + transforms.x.offset) as f32,
                    ((pos.y as f64 * transforms.y.scale) + transforms.y.offset) as f32,
                    ((pos.z as f64 * transforms.z.scale) + transforms.z.offset) as f32,
                )
            }, false),
            other => bail!("Invalid datatype {other} for POSITION_3D attribute. Only Vec3f64 and Vec3f32 are supported!"),
        }
    }

    // Extract the bit attributes into separate attributes, if the target layout has them!
    if raw_las_layout.has_attribute(&ATTRIBUTE_BASIC_FLAGS) {
        if let Some(return_number_attribute) =
            target_layout.get_attribute_by_name(RETURN_NUMBER.name())
        {
            converter.set_custom_mapping_with_transformation(
                &ATTRIBUTE_BASIC_FLAGS,
                return_number_attribute.attribute_definition(),
                |flags: u8| -> u8 { flags & 0b111 },
                true,
            );
        }

        if let Some(nr_returns_attribute) =
            target_layout.get_attribute_by_name(NUMBER_OF_RETURNS.name())
        {
            converter.set_custom_mapping_with_transformation(
                &ATTRIBUTE_BASIC_FLAGS,
                nr_returns_attribute.attribute_definition(),
                |flags: u8| -> u8 { (flags >> 3) & 0b111 },
                true,
            );
        }

        if let Some(scan_direction_flag_attribute) =
            target_layout.get_attribute_by_name(SCAN_DIRECTION_FLAG.name())
        {
            converter.set_custom_mapping_with_transformation(
                &ATTRIBUTE_BASIC_FLAGS,
                scan_direction_flag_attribute.attribute_definition(),
                |flags: u8| -> u8 { (flags >> 6) & 0b1 },
                true,
            );
        }

        if let Some(eof_attribute) = target_layout.get_attribute_by_name(EDGE_OF_FLIGHT_LINE.name())
        {
            converter.set_custom_mapping_with_transformation(
                &ATTRIBUTE_BASIC_FLAGS,
                eof_attribute.attribute_definition(),
                |flags: u8| -> u8 { (flags >> 7) & 0b1 },
                true,
            );
        }
    } else {
        if let Some(return_number_attribute) =
            target_layout.get_attribute_by_name(RETURN_NUMBER.name())
        {
            converter.set_custom_mapping_with_transformation(
                &ATTRIBUTE_EXTENDED_FLAGS,
                return_number_attribute.attribute_definition(),
                |flags: u16| -> u16 { flags & 0b1111 },
                true,
            );
        }
        if let Some(nr_returns_attribute) =
            target_layout.get_attribute_by_name(NUMBER_OF_RETURNS.name())
        {
            converter.set_custom_mapping_with_transformation(
                &ATTRIBUTE_EXTENDED_FLAGS,
                nr_returns_attribute.attribute_definition(),
                |flags: u16| -> u16 { (flags >> 4) & 0b1111 },
                true,
            );
        }
        if let Some(classification_flags_attribute) =
            target_layout.get_attribute_by_name(CLASSIFICATION_FLAGS.name())
        {
            converter.set_custom_mapping_with_transformation(
                &ATTRIBUTE_EXTENDED_FLAGS,
                classification_flags_attribute.attribute_definition(),
                |flags: u16| -> u16 { (flags >> 8) & 0b1111 },
                true,
            );
        }
        if let Some(scanner_channel_attribute) =
            target_layout.get_attribute_by_name(SCANNER_CHANNEL.name())
        {
            converter.set_custom_mapping_with_transformation(
                &ATTRIBUTE_EXTENDED_FLAGS,
                scanner_channel_attribute.attribute_definition(),
                |flags: u16| -> u16 { (flags >> 12) & 0b11 },
                true,
            );
        }
        if let Some(scan_direction_flag_attribute) =
            target_layout.get_attribute_by_name(SCAN_DIRECTION_FLAG.name())
        {
            converter.set_custom_mapping_with_transformation(
                &ATTRIBUTE_EXTENDED_FLAGS,
                scan_direction_flag_attribute.attribute_definition(),
                |flags: u16| -> u16 { (flags >> 14) & 0b1 },
                true,
            );
        }
        if let Some(eof_attribute) = target_layout.get_attribute_by_name(EDGE_OF_FLIGHT_LINE.name())
        {
            converter.set_custom_mapping_with_transformation(
                &ATTRIBUTE_EXTENDED_FLAGS,
                eof_attribute.attribute_definition(),
                |flags: u16| -> u16 { (flags >> 15) & 0b1 },
                true,
            );
        }
    }

    Ok(converter)
}

pub(crate) trait LASReaderBase {
    /// Returns the remaining number of points in the underyling `LASReaderBase`
    fn remaining_points(&self) -> usize;
    fn header(&self) -> &Header;
}

pub struct RawLASReader<T: Read + Seek> {
    reader: T,
    metadata: LASMetadata,
    layout: PointLayout,
    las_point_records_layout: PointLayout,
    current_point_index: usize,
    offset_to_first_point_in_file: u64,
    size_of_point_in_file: u64,
}

impl<T: Read + Seek> RawLASReader<T> {
    /// Creates a new `RawLASReader` from the given `reader`. If `point_layout_matches_memory_layout` is `true`, the
    /// default `PointLayout` of the `RawLASReader` will exactly match the binary layout of the LAS point records.
    /// Otherwise, a more practical `PointLayout` is used that stores positions as `Vector3<f64>` values in world-space
    /// and stores attributes such as `RETURN_NUMBER`, `NUMBER_OF_RETURNS` etc. as separate values instead of the
    /// packed bitfield values. See [`point_layout_from_las_point_format`] for more information
    pub fn from_read(mut reader: T, point_layout_matches_memory_layout: bool) -> Result<Self> {
        let raw_header = raw::Header::read_from(&mut reader)?;
        let offset_to_first_point_in_file = raw_header.offset_to_point_data as u64;
        let size_of_point_in_file = raw_header.point_data_record_length as u64;

        // Manually read the VLRs
        reader.seek(SeekFrom::Start(raw_header.header_size as u64))?;
        let vlrs = (0..raw_header.number_of_variable_length_records as usize)
            .map(|_| las_rs::raw::Vlr::read_from(&mut reader, false).map(las_rs::Vlr::new))
            .collect::<Result<Vec<_>, _>>()
            .context("Failed to read VLRs")?;

        let mut builder = Builder::new(raw_header).context("Invalid LAS header")?;
        builder.vlrs = vlrs;

        // Even after reading all VLRs, there might be leftover bytes before the start of the actual point
        // data. These bytes have to be read and correctly stored in the LAS header, otherwise conversion
        // of the Header to a raw::Header will be wrong, and the LASMetadata will be wrong as well
        // Note: Took this code pretty much straight from the las::Reader, but as far as I can tell, the
        // LAS 1.4 spec states that the `offset_to_point_data` field has to be updated by software that changes
        // the number of VLRs...
        let position_after_reading_vlrs = reader.stream_position()?;
        if position_after_reading_vlrs < offset_to_first_point_in_file {
            reader
                .by_ref()
                .take(offset_to_first_point_in_file - position_after_reading_vlrs)
                .read_to_end(&mut builder.vlr_padding)?;
        }

        let header = builder.into_header().context("Invalid LAS header")?;

        let metadata: LASMetadata = header
            .clone()
            .try_into()
            .context("Failed to parse LAS header")?;
        let point_layout =
            point_layout_from_las_metadata(&metadata, point_layout_matches_memory_layout)?;
        let matching_memory_layout = point_layout_from_las_metadata(&metadata, true)?;

        reader.seek(SeekFrom::Start(offset_to_first_point_in_file))?;

        Ok(Self {
            reader,
            metadata,
            layout: point_layout,
            las_point_records_layout: matching_memory_layout,
            current_point_index: 0,
            offset_to_first_point_in_file,
            size_of_point_in_file,
        })
    }

    pub fn las_metadata(&self) -> &LASMetadata {
        &self.metadata
    }

    fn read_into_default_layout<'a, 'b, B: BorrowedMutBuffer<'a>>(
        &mut self,
        point_buffer: &'b mut B,
        count: usize,
    ) -> Result<usize>
    where
        'a: 'b,
    {
        let num_points_to_read = usize::min(count, self.remaining_points());
        if num_points_to_read == 0 {
            return Ok(0);
        }

        if let Some(interleaved_buffer) = point_buffer.as_interleaved_mut() {
            let new_point_data = interleaved_buffer.get_point_range_mut(0..num_points_to_read);
            self.reader
                .read_exact(new_point_data)
                .context("Failed to read point records")?;
        } else {
            // Read point data in chunks of ~1MiB size to prevent memory problems for very large files if we were
            // to read all data in a single chunk
            const CHUNK_MEM_SIZE: usize = 1 << 20;
            let num_points_per_chunk = CHUNK_MEM_SIZE / self.size_of_point_in_file as usize;
            let num_chunks = (num_points_to_read + num_points_per_chunk - 1) / num_points_per_chunk;
            let mut read_buffer =
                vec![0; num_points_per_chunk * self.size_of_point_in_file as usize];
            for chunk_idx in 0..num_chunks {
                let bytes_in_chunk = if chunk_idx == num_chunks - 1 {
                    (num_points_to_read - (chunk_idx * num_points_per_chunk))
                        * self.size_of_point_in_file as usize
                } else {
                    read_buffer.len()
                };
                let chunk_bytes = &mut read_buffer[..bytes_in_chunk];
                self.reader
                    .read_exact(chunk_bytes)
                    .context("Failed to read chunk of points")?;
                let first_point_in_chunk = chunk_idx * num_points_per_chunk;
                let chunk_end = ((chunk_idx + 1) * num_points_per_chunk).min(num_points_to_read);
                // Safe because this function (`read_into_default_layout`) is only called if the buffer has the exact
                // binary memory layout of the LAS file
                unsafe {
                    point_buffer.set_point_range(first_point_in_chunk..chunk_end, chunk_bytes);
                }
            }
        }

        self.current_point_index += num_points_to_read;

        Ok(num_points_to_read)
    }

    fn read_into_custom_layout<'a, 'b, B: BorrowedMutBuffer<'a>>(
        &mut self,
        point_buffer: &'b mut B,
        count: usize,
    ) -> Result<usize>
    where
        'a: 'b,
    {
        let num_points_to_read = usize::min(count, self.remaining_points());
        if num_points_to_read == 0 {
            return Ok(0);
        }

        let mut convert_buffer =
            VectorBuffer::with_capacity(num_points_to_read, self.las_point_records_layout.clone());
        convert_buffer.resize(num_points_to_read);
        self.read_into_default_layout(&mut convert_buffer, num_points_to_read)?;

        let target_layout = point_buffer.point_layout().clone();
        let converter = get_default_las_converter(
            &self.las_point_records_layout,
            &target_layout,
            self.metadata.raw_las_header().expect("Missing LAS header"),
        )
        .context("Unsupported conversion")?;
        converter.convert_into(&convert_buffer, point_buffer);

        Ok(num_points_to_read)
    }
}

impl<T: Read + Seek> LASReaderBase for RawLASReader<T> {
    fn remaining_points(&self) -> usize {
        self.metadata.point_count() - self.current_point_index
    }

    fn header(&self) -> &Header {
        self.metadata.raw_las_header().unwrap()
    }
}

impl<T: Read + Seek> PointReader for RawLASReader<T> {
    fn read_into<'a, 'b, B: BorrowedMutBuffer<'a>>(
        &mut self,
        point_buffer: &'b mut B,
        count: usize,
    ) -> Result<usize>
    where
        'a: 'b,
    {
        if point_buffer.len() < count {
            panic!("point_buffer.len() must be >= count");
        }

        if *point_buffer.point_layout() != self.las_point_records_layout {
            self.read_into_custom_layout(point_buffer, count)
        } else {
            self.read_into_default_layout(point_buffer, count)
        }
    }

    fn get_metadata(&self) -> &dyn Metadata {
        &self.metadata
    }

    fn get_default_point_layout(&self) -> &PointLayout {
        &self.layout
    }
}

impl<T: Read + Seek> SeekToPoint for RawLASReader<T> {
    fn seek_point(&mut self, position: SeekFrom) -> Result<usize> {
        let new_position = match position {
            SeekFrom::Start(from_start) => from_start as i64,
            SeekFrom::End(from_end) => self.metadata.point_count() as i64 + from_end,
            SeekFrom::Current(from_current) => self.current_point_index as i64 + from_current,
        };
        if new_position < 0 {
            panic!("RawLASReader::seek_point: It is an error to seek to a point position smaller than zero!");
        }
        let clamped_position =
            std::cmp::min(self.metadata.point_count() as i64, new_position) as usize;

        if self.current_point_index != clamped_position {
            let position_within_file = self.offset_to_first_point_in_file
                + clamped_position as u64 * self.size_of_point_in_file;
            self.reader.seek(SeekFrom::Start(position_within_file))?;
            self.current_point_index = clamped_position;
        }

        Ok(self.current_point_index)
    }
}

pub struct RawLAZReader<'a, T: Read + Seek + Send + 'a> {
    reader: LasZipDecompressor<'a, T>,
    metadata: LASMetadata,
    layout: PointLayout,
    las_point_records_layout: PointLayout,
    current_point_index: usize,
    size_of_point_in_file: u64,
}

impl<'a, T: Read + Seek + Send + 'a> RawLAZReader<'a, T> {
    pub fn from_read(mut read: T, point_layout_matches_memory_layout: bool) -> Result<Self> {
        let raw_header = raw::Header::read_from(&mut read)?;
        let offset_to_first_point_in_file = raw_header.offset_to_point_data as u64;
        let size_of_point_in_file = raw_header.point_data_record_length as u64;
        let number_of_vlrs = raw_header.number_of_variable_length_records;

        let mut header_builder = Builder::new(raw_header)?;
        // Read VLRs
        for _ in 0..number_of_vlrs {
            let vlr = las_rs::raw::Vlr::read_from(&mut read, false).map(Vlr::new)?;
            header_builder.vlrs.push(vlr);
        }
        // TODO Read EVLRs

        // Put padding bytes into header (e.g. from leftover VLRs that have been deleted but not removed from the file)
        let position_after_reading_vlrs = read.stream_position()?;
        if position_after_reading_vlrs < offset_to_first_point_in_file {
            read.by_ref()
                .take(offset_to_first_point_in_file - position_after_reading_vlrs)
                .read_to_end(&mut header_builder.vlr_padding)?;
        }

        let header = header_builder.into_header()?;
        if header.point_format().is_extended && header.point_format().has_waveform {
            return Err(anyhow!(
                "Compressed LAZ files with extended formats 9 and 10 are currently not supported!"
            ));
        }

        let metadata: LASMetadata = header
            .clone()
            .try_into()
            .context("Could not parse LAS header")?;
        let point_layout =
            point_layout_from_las_metadata(&metadata, point_layout_matches_memory_layout)?;
        let matching_memory_layout = point_layout_from_las_metadata(&metadata, true)?;

        read.seek(SeekFrom::Start(offset_to_first_point_in_file))?;

        let laszip_vlr = match header.vlrs().iter().find(|vlr| is_laszip_vlr(vlr)) {
            None => Err(anyhow!(
                "RawLAZReader::new: LAZ variable length record not found in file!"
            )),
            Some(vlr) => {
                let laz_record =
                    laz::las::laszip::LazVlr::from_buffer(&vlr.data).map_err(map_laz_err)?;
                Ok(laz_record)
            }
        }?;
        let reader = LasZipDecompressor::new(read, laszip_vlr).map_err(map_laz_err)?;

        Ok(Self {
            reader,
            metadata,
            layout: point_layout,
            las_point_records_layout: matching_memory_layout,
            current_point_index: 0,
            size_of_point_in_file,
        })
    }

    pub fn las_metadata(&self) -> &LASMetadata {
        &self.metadata
    }

    fn read_into_default_layout<'b, 'c, B: BorrowedMutBuffer<'b>>(
        &mut self,
        point_buffer: &'c mut B,
        count: usize,
    ) -> Result<usize>
    where
        'b: 'c,
    {
        let num_points_to_read = usize::min(count, self.remaining_points());
        if num_points_to_read == 0 {
            return Ok(0);
        }

        if let Some(interleaved_buffer) = point_buffer.as_interleaved_mut() {
            let new_point_data = interleaved_buffer.get_point_range_mut(0..num_points_to_read);
            self.reader
                .decompress_many(new_point_data)
                .context("Failed to read point records")?;
        } else {
            // Read point data in chunks of ~1MiB size to prevent memory problems for very large files if we were
            // to read all data in a single chunk
            const CHUNK_MEM_SIZE: usize = 1 << 20;
            let num_points_per_chunk = CHUNK_MEM_SIZE / self.size_of_point_in_file as usize;
            let num_chunks = (num_points_to_read + num_points_per_chunk - 1) / num_points_per_chunk;
            let mut read_buffer =
                vec![0; num_points_per_chunk * self.size_of_point_in_file as usize];
            for chunk_idx in 0..num_chunks {
                let bytes_in_chunk = if chunk_idx == num_chunks - 1 {
                    (num_points_to_read - (chunk_idx * num_points_per_chunk))
                        * self.size_of_point_in_file as usize
                } else {
                    read_buffer.len()
                };
                let chunk_bytes = &mut read_buffer[..bytes_in_chunk];
                self.reader
                    .decompress_many(chunk_bytes)
                    .context("Failed to read chunk of points")?;
                let first_point_in_chunk = chunk_idx * num_points_per_chunk;
                let chunk_end = ((chunk_idx + 1) * num_points_per_chunk).min(num_points_to_read);
                // Safe because this function (`read_into_default_layout`) is only called if the buffer has the exact
                // binary memory layout of the LAS file
                unsafe {
                    point_buffer.set_point_range(first_point_in_chunk..chunk_end, chunk_bytes);
                }
            }
        }

        self.current_point_index += num_points_to_read;

        Ok(num_points_to_read)
    }

    fn read_into_custom_layout<'b, 'c, B: BorrowedMutBuffer<'b>>(
        &mut self,
        point_buffer: &'c mut B,
        count: usize,
    ) -> Result<usize>
    where
        'b: 'c,
    {
        let num_points_to_read = usize::min(count, self.remaining_points());
        if num_points_to_read == 0 {
            return Ok(0);
        }

        let mut convert_buffer =
            VectorBuffer::with_capacity(num_points_to_read, self.las_point_records_layout.clone());
        convert_buffer.resize(num_points_to_read);
        self.read_into_default_layout(&mut convert_buffer, num_points_to_read)?;

        let target_layout = point_buffer.point_layout().clone();
        let converter = get_default_las_converter(
            &self.las_point_records_layout,
            &target_layout,
            self.metadata.raw_las_header().expect("Missing LAS header"),
        )
        .context("Unsupported conversion")?;
        converter.convert_into(&convert_buffer, point_buffer);

        Ok(num_points_to_read)
    }
}

impl<'a, T: Read + Seek + Send + 'a> LASReaderBase for RawLAZReader<'a, T> {
    fn remaining_points(&self) -> usize {
        self.metadata.point_count() - self.current_point_index
    }

    fn header(&self) -> &Header {
        self.metadata.raw_las_header().unwrap()
    }
}

impl<'a, T: Read + Seek + Send + 'a> PointReader for RawLAZReader<'a, T> {
    fn read_into<'b, 'c, B: BorrowedMutBuffer<'b>>(
        &mut self,
        point_buffer: &'c mut B,
        count: usize,
    ) -> Result<usize>
    where
        'b: 'c,
    {
        if point_buffer.len() < count {
            panic!("point_buffer.len() must be >= count");
        }

        if *point_buffer.point_layout() != self.las_point_records_layout {
            self.read_into_custom_layout(point_buffer, count)
        } else {
            self.read_into_default_layout(point_buffer, count)
        }
    }

    fn get_metadata(&self) -> &dyn Metadata {
        &self.metadata
    }

    fn get_default_point_layout(&self) -> &PointLayout {
        &self.layout
    }
}

impl<'a, T: Read + Seek + Send + 'a> SeekToPoint for RawLAZReader<'a, T> {
    fn seek_point(&mut self, position: SeekFrom) -> Result<usize> {
        let new_position = match position {
            SeekFrom::Start(from_start) => from_start as i64,
            SeekFrom::End(from_end) => self.metadata.point_count() as i64 + from_end,
            SeekFrom::Current(from_current) => self.current_point_index as i64 + from_current,
        };
        if new_position < 0 {
            panic!("RawLAZReader::seek_point: It is an error to seek to a point position smaller than zero!");
        }
        let clamped_position =
            std::cmp::min(self.metadata.point_count() as i64, new_position) as usize;

        if self.current_point_index != clamped_position {
            self.reader.seek(clamped_position as u64)?;
            self.current_point_index = clamped_position;
        }

        Ok(self.current_point_index)
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::BufReader};

    use las_rs::point::Format;
    use pasture_core::containers::BorrowedBuffer;
    use pasture_core::layout::attributes;
    use pasture_core::layout::PointAttributeDataType;
    use pasture_core::nalgebra::Vector3;

    use crate::las::get_test_las_path_with_extra_bytes;
    use crate::las::{
        compare_to_reference_data, compare_to_reference_data_range, get_test_las_path,
        get_test_laz_path, test_data_bounds, test_data_classifications, test_data_colors,
        test_data_point_count, test_data_point_source_ids, test_data_positions,
        test_data_wavepacket_parameters,
    };

    use super::*;

    // LAS:
    // - Check that metadata is correct (num points etc.)
    // - `read` has to be correct
    //  - it has to return a buffer with the expected format
    //  - it has to return the correct points
    // - `read_into` has to be correct for a buffer with the same layout
    // - `read_into` has to be correct for a buffer with a different layout
    //  - all attributes, but different formats
    //  - some attributes missing
    // - `seek` has to be correct
    //  - it finds the correct position (checked by successive read call)
    //  - it deals correctly with out of bounds, forward, backward search

    macro_rules! test_read_with_format {
        ($name:ident, $format:expr, $reader:ident, $get_test_file:ident) => {
            mod $name {
                use super::*;
                use pasture_core::containers::*;
                use std::path::PathBuf;

                fn get_test_file_path() -> PathBuf {
                    $get_test_file($format)
                }

                #[test]
                fn test_raw_las_reader_metadata() -> Result<()> {
                    let read = BufReader::new(File::open(get_test_file_path())?);
                    let mut reader = $reader::from_read(read, false)?;

                    assert_eq!(reader.remaining_points(), test_data_point_count());
                    assert_eq!(reader.point_count()?, test_data_point_count());
                    assert_eq!(reader.point_index()?, 0);

                    let layout = reader.get_default_point_layout();
                    let expected_layout =
                        point_layout_from_las_metadata(reader.las_metadata(), false)?;
                    assert_eq!(expected_layout, *layout);

                    let bounds = reader.get_metadata().bounds();
                    let expected_bounds = test_data_bounds();
                    assert_eq!(Some(expected_bounds), bounds);

                    Ok(())
                }

                #[test]
                fn test_raw_las_reader_read() -> Result<()> {
                    let read = BufReader::new(File::open(get_test_file_path())?);
                    let mut reader = $reader::from_read(read, false)?;
                    let format = Format::new($format)?;

                    let points = reader.read::<VectorBuffer>(10)?;
                    let expected_layout =
                        point_layout_from_las_metadata(reader.las_metadata(), false)?;
                    assert_eq!(*points.point_layout(), expected_layout);
                    compare_to_reference_data(&points, format);

                    assert_eq!(10, reader.point_index()?);
                    assert_eq!(0, reader.remaining_points());

                    Ok(())
                }

                #[test]
                fn test_raw_las_reader_read_into_interleaved() -> Result<()> {
                    let read = BufReader::new(File::open(get_test_file_path())?);
                    let mut reader = $reader::from_read(read, false)?;
                    let format = Format::new($format)?;

                    let layout = point_layout_from_las_metadata(reader.las_metadata(), false)?;
                    let mut buffer = VectorBuffer::new_from_layout(layout);
                    buffer.resize(10);

                    reader.read_into(&mut buffer, 10)?;
                    compare_to_reference_data(&buffer, format);

                    assert_eq!(10, reader.point_index()?);
                    assert_eq!(0, reader.remaining_points());

                    Ok(())
                }

                #[test]
                fn test_raw_las_reader_read_into_interleaved_in_multiple_chunks() -> Result<()> {
                    let read = BufReader::new(File::open(get_test_file_path())?);
                    let mut reader = $reader::from_read(read, false)?;
                    let format = Format::new($format)?;

                    let layout = point_layout_from_las_metadata(reader.las_metadata(), false)?;

                    const POINTS_PER_CHUNK: usize = 5;
                    let mut buffer = VectorBuffer::new_from_layout(layout);
                    buffer.resize(POINTS_PER_CHUNK);

                    const NUM_CHUNKS: usize = test_data_point_count() / POINTS_PER_CHUNK;
                    for chunk in 0..NUM_CHUNKS {
                        reader.read_into(&mut buffer, POINTS_PER_CHUNK)?;
                        compare_to_reference_data_range(
                            &buffer,
                            format,
                            (chunk * POINTS_PER_CHUNK)..((chunk + 1) * POINTS_PER_CHUNK),
                        );
                    }

                    assert_eq!(test_data_point_count(), reader.point_index()?);
                    assert_eq!(0, reader.remaining_points());

                    Ok(())
                }

                #[test]
                fn test_raw_las_reader_read_into_columnar() -> Result<()> {
                    let read = BufReader::new(File::open(get_test_file_path())?);
                    let mut reader = $reader::from_read(read, false)?;
                    let format = Format::new($format)?;

                    let layout = point_layout_from_las_metadata(reader.las_metadata(), false)?;
                    let mut buffer = HashMapBuffer::new_from_layout(layout);
                    buffer.resize(10);

                    reader.read_into(&mut buffer, 10)?;
                    compare_to_reference_data(&buffer, format);

                    assert_eq!(10, reader.point_index()?);
                    assert_eq!(0, reader.remaining_points());

                    Ok(())
                }

                #[test]
                fn test_raw_las_reader_read_into_columnar_in_multiple_chunks() -> Result<()> {
                    let read = BufReader::new(File::open(get_test_file_path())?);
                    let mut reader = $reader::from_read(read, false)?;
                    let format = Format::new($format)?;

                    let layout = point_layout_from_las_metadata(reader.las_metadata(), false)?;

                    const POINTS_PER_CHUNK: usize = 5;
                    let mut buffer = HashMapBuffer::new_from_layout(layout);
                    buffer.resize(POINTS_PER_CHUNK);

                    const NUM_CHUNKS: usize = test_data_point_count() / POINTS_PER_CHUNK;
                    for chunk in 0..NUM_CHUNKS {
                        reader.read_into(&mut buffer, POINTS_PER_CHUNK)?;
                        compare_to_reference_data_range(
                            &buffer,
                            format,
                            (chunk * POINTS_PER_CHUNK)..((chunk + 1) * POINTS_PER_CHUNK),
                        );
                    }

                    assert_eq!(test_data_point_count(), reader.point_index()?);
                    assert_eq!(0, reader.remaining_points());

                    Ok(())
                }

                #[test]
                fn test_raw_las_reader_read_into_different_layout_interleaved() -> Result<()> {
                    let read = BufReader::new(File::open(get_test_file_path())?);
                    let mut reader = $reader::from_read(read, false)?;

                    let format = Format::new($format)?;
                    let layout = PointLayout::from_attributes(&[
                        attributes::POSITION_3D
                            .with_custom_datatype(PointAttributeDataType::Vec3f32),
                        attributes::CLASSIFICATION
                            .with_custom_datatype(PointAttributeDataType::U32),
                        attributes::COLOR_RGB.with_custom_datatype(PointAttributeDataType::Vec3u8),
                        attributes::POINT_SOURCE_ID,
                        attributes::WAVEFORM_PARAMETERS,
                    ]);
                    let mut buffer = VectorBuffer::new_from_layout(layout);
                    buffer.resize(10);
                    reader.read_into(&mut buffer, 10)?;

                    let positions = buffer
                        .view_attribute::<Vector3<f32>>(
                            &attributes::POSITION_3D
                                .with_custom_datatype(PointAttributeDataType::Vec3f32),
                        )
                        .into_iter()
                        .collect::<Vec<_>>();
                    let expected_positions = test_data_positions()
                        .into_iter()
                        .map(|p| Vector3::new(p.x as f32, p.y as f32, p.z as f32))
                        .collect::<Vec<_>>();
                    assert_eq!(expected_positions, positions, "Positions do not match");

                    let classifications = buffer
                        .view_attribute::<u32>(
                            &attributes::CLASSIFICATION
                                .with_custom_datatype(PointAttributeDataType::U32),
                        )
                        .into_iter()
                        .collect::<Vec<_>>();
                    let expected_classifications = test_data_classifications()
                        .into_iter()
                        .map(|c| c as u32)
                        .collect::<Vec<_>>();
                    assert_eq!(
                        expected_classifications, classifications,
                        "Classifications do not match"
                    );

                    let colors = buffer
                        .view_attribute::<Vector3<u8>>(
                            &attributes::COLOR_RGB
                                .with_custom_datatype(PointAttributeDataType::Vec3u8),
                        )
                        .into_iter()
                        .collect::<Vec<_>>();
                    let expected_colors = if format.has_color {
                        test_data_colors()
                            .iter()
                            .map(|c| Vector3::new(c.x as u8, c.y as u8, c.z as u8))
                            .collect::<Vec<_>>()
                    } else {
                        (0..10)
                            .map(|_| -> Vector3<u8> { Default::default() })
                            .collect::<Vec<_>>()
                    };
                    assert_eq!(expected_colors, colors, "Colors do not match");

                    let point_source_ids = buffer
                        .view_attribute::<u16>(&attributes::POINT_SOURCE_ID)
                        .into_iter()
                        .collect::<Vec<_>>();
                    let expected_point_source_ids = test_data_point_source_ids();
                    assert_eq!(
                        expected_point_source_ids, point_source_ids,
                        "Point source IDs do not match"
                    );

                    let waveform_params = buffer
                        .view_attribute::<Vector3<f32>>(&attributes::WAVEFORM_PARAMETERS)
                        .into_iter()
                        .collect::<Vec<_>>();
                    let expected_waveform_params = if format.has_waveform {
                        test_data_wavepacket_parameters()
                    } else {
                        (0..10)
                            .map(|_| -> Vector3<f32> { Default::default() })
                            .collect::<Vec<_>>()
                    };
                    assert_eq!(
                        expected_waveform_params, waveform_params,
                        "Wavepacket parameters do not match"
                    );

                    assert_eq!(10, reader.point_index()?);
                    assert_eq!(0, reader.remaining_points());

                    Ok(())
                }

                #[test]
                fn test_raw_las_reader_read_into_different_layout_interleaved_in_multiple_chunks(
                ) -> Result<()> {
                    let read = BufReader::new(File::open(get_test_file_path())?);
                    let mut reader = $reader::from_read(read, false)?;

                    let format = Format::new($format)?;
                    let layout = PointLayout::from_attributes(&[
                        attributes::POSITION_3D
                            .with_custom_datatype(PointAttributeDataType::Vec3f32),
                        attributes::CLASSIFICATION
                            .with_custom_datatype(PointAttributeDataType::U32),
                        attributes::COLOR_RGB.with_custom_datatype(PointAttributeDataType::Vec3u8),
                        attributes::POINT_SOURCE_ID,
                        attributes::WAVEFORM_PARAMETERS,
                    ]);
                    let mut buffer = VectorBuffer::new_from_layout(layout);
                    buffer.resize(test_data_point_count());

                    const POINTS_PER_CHUNK: usize = 5;
                    const NUM_CHUNKS: usize = test_data_point_count() / POINTS_PER_CHUNK;
                    for chunk_index in 0..NUM_CHUNKS {
                        let chunk_range = (chunk_index * POINTS_PER_CHUNK)
                            ..((chunk_index + 1) * POINTS_PER_CHUNK);
                        reader.read_into(&mut buffer.slice_mut(chunk_range), POINTS_PER_CHUNK)?;
                    }

                    let positions = buffer
                        .view_attribute::<Vector3<f32>>(
                            &attributes::POSITION_3D
                                .with_custom_datatype(PointAttributeDataType::Vec3f32),
                        )
                        .into_iter()
                        .collect::<Vec<_>>();
                    let expected_positions = test_data_positions()
                        .into_iter()
                        .map(|p| Vector3::new(p.x as f32, p.y as f32, p.z as f32))
                        .collect::<Vec<_>>();
                    assert_eq!(expected_positions, positions, "Positions do not match");

                    let classifications = buffer
                        .view_attribute::<u32>(
                            &attributes::CLASSIFICATION
                                .with_custom_datatype(PointAttributeDataType::U32),
                        )
                        .into_iter()
                        .collect::<Vec<_>>();
                    let expected_classifications = test_data_classifications()
                        .into_iter()
                        .map(|c| c as u32)
                        .collect::<Vec<_>>();
                    assert_eq!(
                        expected_classifications, classifications,
                        "Classifications do not match"
                    );

                    let colors = buffer
                        .view_attribute::<Vector3<u8>>(
                            &attributes::COLOR_RGB
                                .with_custom_datatype(PointAttributeDataType::Vec3u8),
                        )
                        .into_iter()
                        .collect::<Vec<_>>();
                    let expected_colors = if format.has_color {
                        test_data_colors()
                            .iter()
                            .map(|c| Vector3::new(c.x as u8, c.y as u8, c.z as u8))
                            .collect::<Vec<_>>()
                    } else {
                        (0..10)
                            .map(|_| -> Vector3<u8> { Default::default() })
                            .collect::<Vec<_>>()
                    };
                    assert_eq!(expected_colors, colors, "Colors do not match");

                    let point_source_ids = buffer
                        .view_attribute::<u16>(&attributes::POINT_SOURCE_ID)
                        .into_iter()
                        .collect::<Vec<_>>();
                    let expected_point_source_ids = test_data_point_source_ids();
                    assert_eq!(
                        expected_point_source_ids, point_source_ids,
                        "Point source IDs do not match"
                    );

                    let waveform_params = buffer
                        .view_attribute::<Vector3<f32>>(&attributes::WAVEFORM_PARAMETERS)
                        .into_iter()
                        .collect::<Vec<_>>();
                    let expected_waveform_params = if format.has_waveform {
                        test_data_wavepacket_parameters()
                    } else {
                        (0..10)
                            .map(|_| -> Vector3<f32> { Default::default() })
                            .collect::<Vec<_>>()
                    };
                    assert_eq!(
                        expected_waveform_params, waveform_params,
                        "Wavepacket parameters do not match"
                    );

                    assert_eq!(10, reader.point_index()?);
                    assert_eq!(0, reader.remaining_points());

                    Ok(())
                }

                #[test]
                fn test_raw_las_reader_seek() -> Result<()> {
                    let read = BufReader::new(File::open(get_test_file_path())?);
                    let mut reader = $reader::from_read(read, false)?;
                    let format = Format::new($format)?;

                    let seek_index: usize = 5;
                    let new_pos = reader.seek_point(SeekFrom::Current(seek_index as i64))?;
                    assert_eq!(seek_index, new_pos);

                    let points = reader.read::<VectorBuffer>((10 - seek_index) as usize)?;
                    assert_eq!(10 - seek_index, points.len());

                    compare_to_reference_data_range(&points, format, seek_index..10);

                    Ok(())
                }

                #[test]
                fn test_raw_las_reader_seek_out_of_bounds() -> Result<()> {
                    let read = BufReader::new(File::open(get_test_file_path())?);
                    let mut reader = $reader::from_read(read, false)?;

                    let seek_index: usize = 23;
                    let new_pos = reader.seek_point(SeekFrom::Current(seek_index as i64))?;
                    assert_eq!(10, new_pos);

                    let points = reader.read::<VectorBuffer>(10)?;
                    assert_eq!(0, points.len());

                    Ok(())
                }

                #[test]
                fn test_raw_las_reader_read_with_extra_bytes() -> Result<()> {
                    let read =
                        BufReader::new(File::open(get_test_las_path_with_extra_bytes($format))?);
                    let mut reader = RawLASReader::from_read(read, false)?;
                    let mut format = Format::new($format)?;
                    format.extra_bytes = 4;

                    let points = reader.read::<VectorBuffer>(10)?;
                    let expected_layout =
                        point_layout_from_las_metadata(reader.las_metadata(), false)?;
                    assert_eq!(*points.point_layout(), expected_layout);
                    compare_to_reference_data(&points, format);

                    assert_eq!(10, reader.point_index()?);
                    assert_eq!(0, reader.remaining_points());

                    Ok(())
                }
            }
        };
    }

    test_read_with_format!(las_format_0, 0, RawLASReader, get_test_las_path);
    test_read_with_format!(las_format_1, 1, RawLASReader, get_test_las_path);
    test_read_with_format!(las_format_2, 2, RawLASReader, get_test_las_path);
    test_read_with_format!(las_format_3, 3, RawLASReader, get_test_las_path);
    test_read_with_format!(las_format_4, 4, RawLASReader, get_test_las_path);
    test_read_with_format!(las_format_5, 5, RawLASReader, get_test_las_path);
    test_read_with_format!(las_format_6, 6, RawLASReader, get_test_las_path);
    test_read_with_format!(las_format_7, 7, RawLASReader, get_test_las_path);
    test_read_with_format!(las_format_8, 8, RawLASReader, get_test_las_path);
    test_read_with_format!(las_format_9, 9, RawLASReader, get_test_las_path);
    test_read_with_format!(las_format_10, 10, RawLASReader, get_test_las_path);

    test_read_with_format!(laz_format_0, 0, RawLAZReader, get_test_laz_path);
    test_read_with_format!(laz_format_1, 1, RawLAZReader, get_test_laz_path);
    test_read_with_format!(laz_format_2, 2, RawLAZReader, get_test_laz_path);
    test_read_with_format!(laz_format_3, 3, RawLAZReader, get_test_laz_path);
    test_read_with_format!(laz_format_4, 4, RawLAZReader, get_test_laz_path);
    test_read_with_format!(laz_format_5, 5, RawLAZReader, get_test_laz_path);

    // There is currently a bug in `laz-rs` when seeking into files with point record format 6 or higher, so they are
    // still unsupported in pasture. See this issue here: https://github.com/laz-rs/laz-rs/issues/46

    // test_read_with_format!(laz_format_6, 6, RawLAZReader, get_test_laz_path);
    // test_read_with_format!(laz_format_7, 7, RawLAZReader, get_test_laz_path);
    // test_read_with_format!(laz_format_8, 8, RawLAZReader, get_test_laz_path);

    // Formats 9 and 10 seem to parse waveform data differently when using laz-rs, so they are unsupported for now
}
