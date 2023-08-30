use std::collections::HashMap;
use std::convert::TryInto;
use std::io::{Read, Seek, SeekFrom};

use anyhow::{anyhow, Context, Result};
use las_rs::Header;
use las_rs::{raw, Builder, Vlr};
use laz::LasZipDecompressor;
use pasture_core::containers::OwningBuffer;
use pasture_core::{layout::PointLayout, meta::Metadata};

use super::{map_laz_err, point_layout_from_las_metadata, LASMetadata, PointParser};
use crate::base::{PointReader, SeekToPoint};

/// Is the given VLR the LASzip VLR? Function taken from the `las` crate because it is not exported there
fn is_laszip_vlr(vlr: &Vlr) -> bool {
    if &vlr.user_id == laz::LazVlr::USER_ID && vlr.record_id == laz::LazVlr::RECORD_ID {
        true
    } else {
        false
    }
}

/// Parse a single chunk of points by reading the point data from the reader, running the given parser
/// and outputting the parsed data into the output_points_buffer
fn parse_chunk(
    input_points_buffer: &[u8],
    num_points_in_chunk: usize,
    output_points_buffer: &mut [u8],
    output_point_layout: &PointLayout,
    parser: &mut PointParser,
) {
    // This method assumes that self.reader is currently at the starting point of the chunk to read
    let input_point_size = input_points_buffer.len() / num_points_in_chunk;
    let input_point_chunks = input_points_buffer.chunks(input_point_size);

    let output_point_size = output_point_layout.size_of_point_entry() as usize;
    let output_point_chunks = output_points_buffer.chunks_mut(output_point_size);

    for (input_chunk, output_chunk) in input_point_chunks.zip(output_point_chunks) {
        unsafe {
            parser.parse_one(input_chunk, output_chunk);
        }
    }
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
    current_point_index: usize,
    offset_to_first_point_in_file: u64,
    size_of_point_in_file: u64,
    default_parser: PointParser,
    custom_parsers: HashMap<PointLayout, PointParser>,
    //TODO Add an option to not convert the position fields into world space
}

impl<T: Read + Seek> RawLASReader<T> {
    pub fn from_read(mut read: T) -> Result<Self> {
        let raw_header = raw::Header::read_from(&mut read)?;
        let offset_to_first_point_in_file = raw_header.offset_to_point_data as u64;
        let size_of_point_in_file = raw_header.point_data_record_length as u64;

        // Manually read the VLRs
        read.seek(SeekFrom::Start(raw_header.header_size as u64))?;
        let vlrs = (0..raw_header.number_of_variable_length_records as usize)
            .map(|_| {
                las_rs::raw::Vlr::read_from(&mut read, false)
                    .map(|raw_vlr| las_rs::Vlr::new(raw_vlr))
            })
            .collect::<Result<Vec<_>, _>>()
            .context("Failed to read VLRs")?;

        let mut builder = Builder::new(raw_header).context("Invalid LAS header")?;
        builder.vlrs = vlrs;
        let header = builder.into_header().context("Invalid LAS header")?;

        let metadata: LASMetadata = header
            .clone()
            .try_into()
            .context("Failed to parse LAS header")?;
        let point_layout = point_layout_from_las_metadata(&metadata, false)?;

        read.seek(SeekFrom::Start(offset_to_first_point_in_file as u64))?;

        let default_parser =
            PointParser::build(&metadata, &point_layout).context("Failed to build point parser")?;

        Ok(Self {
            reader: read,
            metadata: metadata,
            layout: point_layout,
            current_point_index: 0,
            offset_to_first_point_in_file,
            size_of_point_in_file,
            default_parser,
            custom_parsers: Default::default(),
        })
    }

    #[cfg(test)]
    pub fn las_metadata(&self) -> &LASMetadata {
        &self.metadata
    }

    fn read_into_default_layout<'a, 'b, B: OwningBuffer<'a>>(
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

        let size_of_input_chunk = num_points_to_read * self.size_of_point_in_file as usize;
        let mut input_buffer = vec![0; size_of_input_chunk];
        self.reader
            .read_exact(&mut input_buffer)
            .context("Failed to read chunk of points")?;

        // Read into chunks of a fixed size. Within each chunk, read all data into an untyped buffer
        // then push the untyped data into 'buffer'
        let chunk_size = 50_000;
        let point_size = self.layout.size_of_point_entry() as usize;
        let chunk_bytes = point_size as usize * chunk_size;
        let num_chunks = (num_points_to_read + chunk_size - 1) / chunk_size;
        let mut output_points_chunk: Vec<u8> = vec![0; chunk_bytes];

        for chunk_index in 0..num_chunks {
            let points_in_chunk =
                std::cmp::min(chunk_size, num_points_to_read - (chunk_index * chunk_size));
            let bytes_in_chunk = points_in_chunk * point_size;

            parse_chunk(
                &input_buffer,
                points_in_chunk,
                &mut output_points_chunk[..],
                &self.layout,
                &mut self.default_parser,
            );

            // Safe because we know that `point_buffer` has the default point layout for the current LAS point
            // record format, and `parse_chunk` is guaranteed to output data in the matching binary format
            unsafe {
                point_buffer.push_points(&output_points_chunk[0..bytes_in_chunk]);
            }
        }

        self.current_point_index += num_points_to_read;

        Ok(num_points_to_read)
    }

    fn read_into_custom_layout<'a, 'b, B: OwningBuffer<'a>>(
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

        let size_of_input_chunk = num_points_to_read * self.size_of_point_in_file as usize;
        let mut input_buffer = vec![0; size_of_input_chunk];
        self.reader
            .read_exact(&mut input_buffer)
            .context("Failed to read chunk of points")?;

        // Read in interleaved chunks, even if the `point_buffer` is not interleaved. `push_points_interleaved` will
        // handle the memory transpose in this case
        let chunk_size = 50_000;
        let point_size = point_buffer.point_layout().size_of_point_entry() as usize;
        let chunk_bytes = point_size * chunk_size;
        let num_chunks = (num_points_to_read + chunk_size - 1) / chunk_size;
        let mut points_chunk: Vec<u8> = vec![0; chunk_bytes];

        // Parser construction can be costly, so we cache parsers for known point layouts to amortize cost
        // over multiple `read_into_custom_layout` calls
        let mut parser = match self.custom_parsers.get_mut(point_buffer.point_layout()) {
            Some(parser) => parser,
            None => {
                let new_parser = PointParser::build(&self.metadata, point_buffer.point_layout())
                    .context("Failed to build point parser")?;
                self.custom_parsers
                    .insert(point_buffer.point_layout().clone(), new_parser);
                self.custom_parsers
                    .get_mut(point_buffer.point_layout())
                    .unwrap()
            }
        };

        for chunk_index in 0..num_chunks {
            let points_in_chunk =
                std::cmp::min(chunk_size, num_points_to_read - (chunk_index * chunk_size));
            let bytes_in_chunk = points_in_chunk * point_size;

            parse_chunk(
                &input_buffer,
                points_in_chunk,
                &mut points_chunk[..],
                point_buffer.point_layout(),
                &mut parser,
            );

            // Safe because `parse_chunk` is guaranteed to output data in the matching `PointLayout` for
            // `point_buffer`
            unsafe {
                point_buffer.push_points(&points_chunk[0..bytes_in_chunk]);
            }
        }

        self.current_point_index += num_points_to_read;

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
    fn read_into<'a, 'b, B: OwningBuffer<'a>>(
        &mut self,
        point_buffer: &'b mut B,
        count: usize,
    ) -> Result<usize>
    where
        'a: 'b,
    {
        if *point_buffer.point_layout() != self.layout {
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
    current_point_index: usize,
    size_of_point_in_file: u64,
    default_parser: PointParser,
}

impl<'a, T: Read + Seek + Send + 'a> RawLAZReader<'a, T> {
    pub fn from_read(mut read: T) -> Result<Self> {
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

        let header = header_builder.into_header()?;
        if header.point_format().has_waveform {
            return Err(anyhow!(
                "Compressed LAZ files with wave packet data are currently not supported!"
            ));
        }
        if header.point_format().is_extended {
            return Err(anyhow!(
                "Compressed LAZ files with extended formats (6-10) are currently not supported!"
            ));
        }

        let metadata: LASMetadata = header
            .clone()
            .try_into()
            .context("Could not parse LAS header")?;
        let point_layout = point_layout_from_las_metadata(&metadata, false)?;

        read.seek(SeekFrom::Start(offset_to_first_point_in_file as u64))?;

        let laszip_vlr = match header.vlrs().iter().find(|vlr| is_laszip_vlr(*vlr)) {
            None => Err(anyhow!(
                "RawLAZReader::new: LAZ variable length record not found in file!"
            )),
            Some(ref vlr) => {
                let laz_record =
                    laz::las::laszip::LazVlr::from_buffer(&vlr.data).map_err(map_laz_err)?;
                Ok(laz_record)
            }
        }?;
        let reader = LasZipDecompressor::new(read, laszip_vlr).map_err(map_laz_err)?;

        let default_parser =
            PointParser::build(&metadata, &point_layout).context("Failed to build point parser")?;

        Ok(Self {
            reader,
            metadata: metadata,
            layout: point_layout,
            current_point_index: 0,
            size_of_point_in_file,
            default_parser,
        })
    }

    #[cfg(test)]
    pub fn las_metadata(&self) -> &LASMetadata {
        &self.metadata
    }

    fn read_chunk_default_layout(
        &mut self,
        chunk_buffer: &mut [u8],
        decompression_buffer: &mut [u8],
        num_points_in_chunk: usize,
    ) -> Result<()> {
        let bytes_in_chunk = num_points_in_chunk * self.size_of_point_in_file as usize;

        self.reader
            .decompress_many(&mut decompression_buffer[0..bytes_in_chunk])?;

        let input_points_chunk = &decompression_buffer[..bytes_in_chunk];

        let num_output_bytes = num_points_in_chunk * self.layout.size_of_point_entry() as usize;
        let output_points_chunk = &mut chunk_buffer[..num_output_bytes];

        parse_chunk(
            input_points_chunk,
            num_points_in_chunk,
            output_points_chunk,
            &self.layout,
            &mut self.default_parser,
        );

        Ok(())
    }

    fn read_chunk_custom_layout(
        &mut self,
        chunk_buffer: &mut [u8],
        decompression_buffer: &mut [u8],
        num_points_in_chunk: usize,
        target_layout: &PointLayout,
    ) -> Result<()> {
        let bytes_in_chunk = num_points_in_chunk * self.size_of_point_in_file as usize;

        self.reader
            .decompress_many(&mut decompression_buffer[0..bytes_in_chunk])?;

        let input_points_chunk = &decompression_buffer[..bytes_in_chunk];

        let num_output_bytes = num_points_in_chunk * target_layout.size_of_point_entry() as usize;
        let output_points_chunk = &mut chunk_buffer[..num_output_bytes];

        let mut custom_parser = PointParser::build(&self.metadata, target_layout)
            .context("Failed to build point parser")?;

        parse_chunk(
            input_points_chunk,
            num_points_in_chunk,
            output_points_chunk,
            &target_layout,
            &mut custom_parser,
        );

        Ok(())
    }

    fn read_into_default_layout<'b, 'c, B: OwningBuffer<'b>>(
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

        // Read into chunks of a fixed size. Within each chunk, read all data into an untyped buffer
        // then push the untyped data into 'buffer'
        let chunk_size = 50_000;
        let point_size = self.layout.size_of_point_entry() as usize;
        let chunk_bytes = point_size as usize * chunk_size;
        let num_chunks = (num_points_to_read + chunk_size - 1) / chunk_size;

        let decompression_chunk_size = self.size_of_point_in_file as usize * chunk_size;
        let mut decompression_chunk: Vec<u8> = vec![0; decompression_chunk_size];

        // If the buffer is interleaved, we can directly read into the memory of the buffer and save one memcpy per chunk
        // if point_buffer.as_interleaved_mut().is_some() {
        //     let first_new_point_index = point_buffer.len();
        //     point_buffer.resize(num_points_to_read);

        //     let interleaved_buffer = point_buffer.as_interleaved_mut().unwrap();

        //     for chunk_index in 0..num_chunks {
        //         let points_in_chunk =
        //             std::cmp::min(chunk_size, num_points_to_read - (chunk_index * chunk_size));
        //         let first_point_index = first_new_point_index + chunk_index * chunk_size;
        //         let end_of_chunk_index = first_point_index + points_in_chunk;

        //         self.read_chunk_default_layout(
        //             interleaved_buffer.get_raw_points_mut(first_point_index..end_of_chunk_index),
        //             &mut decompression_chunk,
        //             points_in_chunk,
        //         )?;
        //     }
        // } else {
        let mut points_chunk: Vec<u8> = vec![0; chunk_bytes];

        for chunk_index in 0..num_chunks {
            let points_in_chunk =
                std::cmp::min(chunk_size, num_points_to_read - (chunk_index * chunk_size));
            let bytes_in_chunk = points_in_chunk * point_size;

            self.read_chunk_default_layout(
                &mut points_chunk[..],
                &mut decompression_chunk[..],
                points_in_chunk,
            )?;

            // Safe because `read_chunk_default_layout` outputs data in the matching `PointLayout` of `point_buffer`
            unsafe {
                point_buffer.push_points(&points_chunk[0..bytes_in_chunk]);
            }
        }
        // }

        self.current_point_index += num_points_to_read;

        Ok(num_points_to_read)
    }

    fn read_into_custom_layout<'b, 'c, B: OwningBuffer<'b>>(
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

        // Read in interleaved chunks, even if the `point_buffer` is not interleaved. `push_points_interleaved` will
        // handle the memory transpose in this case
        let chunk_size = 50_000;
        let point_size = point_buffer.point_layout().size_of_point_entry() as usize;
        let chunk_bytes = point_size * chunk_size;
        let num_chunks = (num_points_to_read + chunk_size - 1) / chunk_size;
        let mut points_chunk: Vec<u8> = vec![0; chunk_bytes];

        let decompression_chunk_size = self.size_of_point_in_file as usize * chunk_size;
        let mut decompression_chunk: Vec<u8> = vec![0; decompression_chunk_size];

        for chunk_index in 0..num_chunks {
            let points_in_chunk =
                std::cmp::min(chunk_size, num_points_to_read - (chunk_index * chunk_size));
            let bytes_in_chunk = points_in_chunk * point_size;

            self.read_chunk_custom_layout(
                &mut points_chunk[..],
                &mut decompression_chunk[..],
                points_in_chunk,
                point_buffer.point_layout(),
            )?;

            // Safe because `read_chunk_custom_layout` outputs data in the matching `PointLayout` of `point_buffer`
            unsafe {
                point_buffer.push_points(&points_chunk[0..bytes_in_chunk]);
            }
        }

        self.current_point_index += num_points_to_read;

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
    fn read_into<'b, 'c, B: OwningBuffer<'b>>(
        &mut self,
        point_buffer: &'c mut B,
        count: usize,
    ) -> Result<usize>
    where
        'b: 'c,
    {
        if *point_buffer.point_layout() != self.layout {
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
                    let mut reader = $reader::from_read(read)?;

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
                    let mut reader = $reader::from_read(read)?;
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
                    let mut reader = $reader::from_read(read)?;
                    let format = Format::new($format)?;

                    let layout = point_layout_from_las_metadata(reader.las_metadata(), false)?;
                    let mut buffer = VectorBuffer::new_from_layout(layout);

                    reader.read_into(&mut buffer, 10)?;
                    compare_to_reference_data(&buffer, format);

                    assert_eq!(10, reader.point_index()?);
                    assert_eq!(0, reader.remaining_points());

                    Ok(())
                }

                #[test]
                fn test_raw_las_reader_read_into_interleaved_in_multiple_chunks() -> Result<()> {
                    let read = BufReader::new(File::open(get_test_file_path())?);
                    let mut reader = $reader::from_read(read)?;
                    let format = Format::new($format)?;

                    let layout = point_layout_from_las_metadata(reader.las_metadata(), false)?;
                    let mut buffer = VectorBuffer::new_from_layout(layout);

                    const POINTS_PER_CHUNK: usize = 5;
                    const NUM_CHUNKS: usize = test_data_point_count() / POINTS_PER_CHUNK;
                    for chunk in 0..NUM_CHUNKS {
                        buffer.clear();
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
                    let mut reader = $reader::from_read(read)?;
                    let format = Format::new($format)?;

                    let layout = point_layout_from_las_metadata(reader.las_metadata(), false)?;
                    let mut buffer = HashMapBuffer::new_from_layout(layout);

                    reader.read_into(&mut buffer, 10)?;
                    compare_to_reference_data(&buffer, format);

                    assert_eq!(10, reader.point_index()?);
                    assert_eq!(0, reader.remaining_points());

                    Ok(())
                }

                #[test]
                fn test_raw_las_reader_read_into_columnar_in_multiple_chunks() -> Result<()> {
                    let read = BufReader::new(File::open(get_test_file_path())?);
                    let mut reader = $reader::from_read(read)?;
                    let format = Format::new($format)?;

                    let layout = point_layout_from_las_metadata(reader.las_metadata(), false)?;
                    let mut buffer = HashMapBuffer::new_from_layout(layout);

                    const POINTS_PER_CHUNK: usize = 5;
                    const NUM_CHUNKS: usize = test_data_point_count() / POINTS_PER_CHUNK;
                    for chunk in 0..NUM_CHUNKS {
                        buffer.clear();
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
                    let mut reader = $reader::from_read(read)?;

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
                    let mut reader = $reader::from_read(read)?;

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

                    const POINTS_PER_CHUNK: usize = 5;
                    const NUM_CHUNKS: usize = test_data_point_count() / POINTS_PER_CHUNK;
                    for _ in 0..NUM_CHUNKS {
                        reader.read_into(&mut buffer, POINTS_PER_CHUNK)?;
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
                    let mut reader = $reader::from_read(read)?;
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
                    let mut reader = $reader::from_read(read)?;

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
                    let mut reader = RawLASReader::from_read(read)?;
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
    // Formats 4,5,9,10 have wave packet data, which is currently unsupported by laz-rs
    // Format 6,7,8 seem to be unsupported by LASzip and give weird results with laz-rs (e.g. seek does not work correctly)
    // test_read_with_format!(laz_format_4, 4, RawLAZReader);
    // test_read_with_format!(laz_format_5, 5, RawLAZReader);
    // test_read_with_format!(laz_format_6, 6, RawLAZReader, get_test_laz_path);
    // test_read_with_format!(laz_format_7, 7, RawLAZReader, get_test_laz_path);
    // test_read_with_format!(laz_format_8, 8, RawLAZReader, get_test_laz_path);
    // test_read_with_format!(laz_format_9, 9, RawLAZReader);
    // test_read_with_format!(laz_format_10, 10, RawLAZReader);

    //######### TODO ###########
    // We have tests now for various formats and various conversions. We should extend them for a wider range, maybe even
    // fuzz-test (though this is more effort to setup...)
    // Also include comparisons for the additional attributes in the '_read_into_different_attribute_...' tests
}
