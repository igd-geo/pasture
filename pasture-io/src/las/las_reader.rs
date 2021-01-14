use std::convert::From;
use std::{fmt::Debug, io::SeekFrom, path::Path};

use anyhow::{bail, Result};
use las::{Read, Reader};

use crate::base::{PointReader, SeekToPoint};
use pasture_core::{
    containers::{
        InterleavedPointView, InterleavedVecPointStorage, PointBuffer, PointBufferWriteable,
    },
    layout::PointLayout,
    meta::Metadata,
};
use pasture_core::{
    layout::{conversion::RawPointConverter, PointType},
    util::view_raw_bytes,
};

use super::{
    point_layout_from_las_point_format, LASMetadata, LasPointFormat0, LasPointFormat1,
    LasPointFormat2, LasPointFormat3, LasPointFormat4, LasPointFormat5,
};

/// `PointReader` implementation for LAS/LAZ files
pub struct LASReader {
    reader: Reader,
    metadata: LASMetadata,
    layout: PointLayout,
    current_point_index: usize,
}

impl LASReader {
    // TODO LAS files store 32-bit integer coordinates in local space internally, but we almost always want
    // floating point values in world space instead. The conversion from integer to float in world space will
    // have to happen automatically during reading, and the default datatype of attribute POSITION_3D will be
    // Vector3<f64>. However, we should make it possible to switch this default conversion off for cases where
    // we want to read just the raw i32 data!

    /// Creates a new `LASReader` by opening the file at the given `path`.
    ///
    /// # Errors
    ///
    /// If `path` does not exist, cannot be opened or does not point to a valid LAS/LAZ file, an error is returned.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let las_reader = Reader::from_path(path)?;
        let metadata: LASMetadata = las_reader.header().into();
        let layout = point_layout_from_las_point_format(las_reader.header().point_format())?;
        Ok(Self {
            reader: las_reader,
            metadata,
            layout,
            current_point_index: 0,
        })
    }

    /// Creates a new `LASReader` from the given `std::io::Read`.
    ///
    /// # Errors
    ///
    /// If the given `Read` does not represent a valid LAS/LAZ file, an error is returned.
    pub fn new<R: std::io::Read + std::io::Seek + Send + Debug + 'static>(
        reader: R,
    ) -> Result<Self> {
        let las_reader = Reader::new(reader)?;
        let metadata: LASMetadata = las_reader.header().into();
        let layout = point_layout_from_las_point_format(las_reader.header().point_format())?;
        Ok(Self {
            reader: las_reader,
            metadata,
            layout,
            current_point_index: 0,
        })
    }

    /// Returns the number of remaining points that can be read from this `LASReader`
    pub fn remaining_points(&self) -> usize {
        self.metadata.point_count() - self.current_point_index
    }

    /// Read points using the given `PointType`
    fn do_read<T: From<las::Point> + PointType>(
        &mut self,
        count: usize,
        buffer: &mut InterleavedVecPointStorage,
    ) -> Result<()> {
        assert_eq!(
            buffer.point_layout().size_of_point_entry(),
            std::mem::size_of::<T>() as u64
        );
        for _ in 0..count {
            let las_point = self
                .reader
                .read()
                .expect("LASReader:do_read: las::Reader::read returned None")?;
            let typed_point: T = las_point.into();
            buffer.push_point_unchecked(typed_point);
        }

        Ok(())
    }

    fn do_convert<T: From<las::Point> + PointType>(
        &mut self,
        count: usize,
        buffer: &mut [u8],
        converter: &RawPointConverter,
        target_layout: &PointLayout,
    ) -> Result<()> {
        let target_stride = target_layout.size_of_point_entry() as usize;
        assert_eq!(buffer.len(), count * target_stride);

        for point_index in 0..count {
            let las_point = self
                .reader
                .read()
                .expect("LASReader:do_read: las::Reader::read returned None")?;
            let typed_point: T = las_point.into();

            let buffer_start = point_index * target_stride;
            let buffer_end = (point_index + 1) * target_stride;
            unsafe {
                let typed_point_bytes = view_raw_bytes(&typed_point);
                converter.convert(typed_point_bytes, &mut buffer[buffer_start..buffer_end]);
            }
        }

        Ok(())
    }
}

impl PointReader for LASReader {
    fn read(&mut self, count: usize) -> Result<Box<dyn pasture_core::containers::PointBuffer>> {
        let num_points_to_read = usize::min(count, self.remaining_points());
        let mut buffer =
            InterleavedVecPointStorage::with_capacity(num_points_to_read, self.layout.clone());

        // TODO How does reading work? I see two options:
        // Option 1) Define some Rust types that correspond to a single point in the 10 different LAS point formats and
        //           then run a conversion from las::Point to this type and call buffer.push_point_unchecked
        // Option 2) Allocate a raw buffer of the same size as self.layout.size() and use a series of conversion functions
        //           from the different LAS attributes into the memory regions inside the buffer as defined by self.layout

        // This is option 1:
        match self.metadata.point_format() {
            0 => self.do_read::<LasPointFormat0>(num_points_to_read, &mut buffer)?,
            1 => self.do_read::<LasPointFormat1>(num_points_to_read, &mut buffer)?,
            2 => self.do_read::<LasPointFormat2>(num_points_to_read, &mut buffer)?,
            3 => self.do_read::<LasPointFormat3>(num_points_to_read, &mut buffer)?,
            4 => self.do_read::<LasPointFormat4>(num_points_to_read, &mut buffer)?,
            5 => self.do_read::<LasPointFormat5>(num_points_to_read, &mut buffer)?,
            _ => panic!(
                "Currently unsupported point format {}",
                self.metadata.point_format()
            ),
        }

        Ok(Box::new(buffer))
    }

    fn read_into(
        &mut self,
        point_buffer: &mut dyn PointBufferWriteable,
        count: usize,
    ) -> Result<usize> {
        let num_points_to_read = usize::min(count, self.remaining_points());

        let target_point_size = point_buffer.point_layout().size_of_point_entry() as usize;
        let converter = RawPointConverter::from_to(&self.layout, point_buffer.point_layout());

        // TODO This is a first working version of read_into, but it is not fast because it does two conversions
        // (one from las::Point into LasPointFormatX and then from LasPointFormatX into target_layout)
        // If we write a custom reader for LAS data (not using las::Point but instead the raw point bytes), we can
        // circumvent this in the future and only read the bytes that we care about

        // Read in chunks of `chunk_size` point, this way we don't have to call PointBuffer::push_raw_points once for
        // every point. At the same time, a reasonable `chunk_size` keeps the memory overhead low!
        let chunk_size: usize = 50000;
        let num_chunks = (num_points_to_read + chunk_size - 1) / chunk_size;
        let mut chunk_buffer: Vec<u8> = vec![0; chunk_size * target_point_size];
        for chunk_index in 0..num_chunks {
            let num_points_in_chunk =
                usize::min(chunk_size, num_points_to_read - (chunk_index * chunk_size));
            let buffer_end = num_points_in_chunk * target_point_size;
            let buffer_slice = &mut chunk_buffer[0..buffer_end];

            match self.metadata.point_format() {
                0 => self.do_convert::<LasPointFormat0>(
                    num_points_to_read,
                    buffer_slice,
                    &converter,
                    point_buffer.point_layout(),
                )?,
                1 => self.do_convert::<LasPointFormat1>(
                    num_points_to_read,
                    buffer_slice,
                    &converter,
                    point_buffer.point_layout(),
                )?,
                2 => self.do_convert::<LasPointFormat2>(
                    num_points_to_read,
                    buffer_slice,
                    &converter,
                    point_buffer.point_layout(),
                )?,
                3 => self.do_convert::<LasPointFormat3>(
                    num_points_to_read,
                    buffer_slice,
                    &converter,
                    point_buffer.point_layout(),
                )?,
                4 => self.do_convert::<LasPointFormat4>(
                    num_points_to_read,
                    buffer_slice,
                    &converter,
                    point_buffer.point_layout(),
                )?,
                5 => self.do_convert::<LasPointFormat5>(
                    num_points_to_read,
                    buffer_slice,
                    &converter,
                    point_buffer.point_layout(),
                )?,
                _ => panic!(
                    "Currently unsupported point format {}",
                    self.metadata.point_format()
                ),
            }

            let chunk_view = InterleavedPointView::from_raw_slice(
                buffer_slice,
                point_buffer.point_layout().clone(),
            );
            point_buffer.push_points_interleaved(&chunk_view);
        }

        Ok(num_points_to_read)
    }

    fn get_metadata(&self) -> &dyn Metadata {
        &self.metadata
    }

    fn get_default_point_layout(&self) -> &PointLayout {
        &self.layout
    }
}

impl SeekToPoint for LASReader {
    fn seek_point(&mut self, position: SeekFrom) -> Result<usize> {
        let index_from_start = match position {
            SeekFrom::Current(offset_from_current) => {
                self.current_point_index as i64 + offset_from_current
            }
            SeekFrom::End(offset_from_end) => self.metadata.point_count() as i64 + offset_from_end,
            SeekFrom::Start(offset_from_start) => offset_from_start as i64,
        };

        if index_from_start < 0 {
            bail!(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Cannot seek before point index 0!"
            ));
        }

        let actual_index = usize::min(index_from_start as usize, self.metadata.point_count());
        self.reader.seek(actual_index as u64)?;
        Ok(actual_index)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use pasture_core::{containers::*, layout::attributes, nalgebra::Vector3};

    fn format_has_gps_times(format: u8) -> bool {
        match format {
            1 => true,
            3..=10 => true,
            _ => false,
        }
    }

    fn format_has_colors(format: u8) -> bool {
        match format {
            2..=3 => true,
            5 => true,
            7..=8 => true,
            10 => true,
            _ => false,
        }
    }

    fn expected_positions() -> Vec<Vector3<f64>> {
        vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(2.0, 2.0, 2.0),
            Vector3::new(3.0, 3.0, 3.0),
            Vector3::new(4.0, 4.0, 4.0),
            Vector3::new(5.0, 5.0, 5.0),
            Vector3::new(6.0, 6.0, 6.0),
            Vector3::new(7.0, 7.0, 7.0),
            Vector3::new(8.0, 8.0, 8.0),
            Vector3::new(9.0, 9.0, 9.0),
        ]
    }

    fn expected_intensities() -> Vec<u16> {
        vec![
            0,
            255,
            2 * 255,
            3 * 255,
            4 * 255,
            5 * 255,
            6 * 255,
            7 * 255,
            8 * 255,
            9 * 255,
        ]
    }

    fn expected_return_numbers() -> Vec<u8> {
        vec![0, 1, 2, 3, 4, 5, 6, 7, 0, 1]
    }

    fn expected_number_of_returns() -> Vec<u8> {
        vec![0, 1, 2, 3, 4, 5, 6, 7, 0, 1]
    }

    fn expected_scan_direction_flags() -> Vec<bool> {
        vec![
            false, true, false, true, false, true, false, true, false, true,
        ]
    }

    fn expected_edge_of_flight_lines() -> Vec<bool> {
        vec![
            false, true, false, true, false, true, false, true, false, true,
        ]
    }

    fn expected_classifications() -> Vec<u8> {
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }

    fn expected_scan_angle_ranks() -> Vec<i8> {
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }

    fn expected_user_data() -> Vec<u8> {
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }

    fn expected_point_source_ids() -> Vec<u16> {
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }

    fn expected_gps_times() -> Vec<f64> {
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    }

    fn expected_colors() -> Vec<Vector3<u16>> {
        vec![
            Vector3::new(0, 1 << 4, 2 << 8),
            Vector3::new(1, 2 << 4, 3 << 8),
            Vector3::new(2, 3 << 4, 4 << 8),
            Vector3::new(3, 4 << 4, 5 << 8),
            Vector3::new(4, 5 << 4, 6 << 8),
            Vector3::new(5, 6 << 4, 7 << 8),
            Vector3::new(6, 7 << 4, 8 << 8),
            Vector3::new(7, 8 << 4, 9 << 8),
            Vector3::new(8, 9 << 4, 10 << 8),
            Vector3::new(9, 10 << 4, 11 << 8),
        ]
    }

    fn test_las_reader_data(points: &dyn PointBuffer, point_format: u8) {
        let positions =
            attributes::<Vector3<f64>>(points, &attributes::POSITION_3D).collect::<Vec<_>>();
        assert_eq!(expected_positions(), positions);

        let intensities = attributes::<u16>(points, &attributes::INTENSITY).collect::<Vec<_>>();
        assert_eq!(expected_intensities(), intensities);

        let return_numbers =
            attributes::<u8>(points, &attributes::RETURN_NUMBER).collect::<Vec<_>>();
        assert_eq!(expected_return_numbers(), return_numbers);

        let number_of_returns =
            attributes::<u8>(points, &attributes::NUMBER_OF_RETURNS).collect::<Vec<_>>();
        assert_eq!(expected_number_of_returns(), number_of_returns);

        let scan_direction_flags =
            attributes::<bool>(points, &attributes::SCAN_DIRECTION_FLAG).collect::<Vec<_>>();
        assert_eq!(expected_scan_direction_flags(), scan_direction_flags);

        let eof = attributes::<bool>(points, &attributes::EDGE_OF_FLIGHT_LINE).collect::<Vec<_>>();
        assert_eq!(expected_edge_of_flight_lines(), eof);

        let classifications =
            attributes::<u8>(points, &attributes::CLASSIFICATION).collect::<Vec<_>>();
        assert_eq!(expected_classifications(), classifications);

        let scan_angle_ranks =
            attributes::<i8>(points, &attributes::SCAN_ANGLE_RANK).collect::<Vec<_>>();
        assert_eq!(expected_scan_angle_ranks(), scan_angle_ranks);

        let user_data = attributes::<u8>(points, &attributes::USER_DATA).collect::<Vec<_>>();
        assert_eq!(expected_user_data(), user_data);

        let point_source_ids =
            attributes::<u16>(points, &attributes::POINT_SOURCE_ID).collect::<Vec<_>>();
        assert_eq!(expected_point_source_ids(), point_source_ids);

        if format_has_gps_times(point_format) {
            let gps_times = attributes::<f64>(points, &attributes::GPS_TIME).collect::<Vec<_>>();
            assert_eq!(expected_gps_times(), gps_times);
        }

        if format_has_colors(point_format) {
            let colors =
                attributes::<Vector3<u16>>(points, &attributes::COLOR_RGB).collect::<Vec<_>>();
            assert_eq!(expected_colors(), colors);
        }
    }

    #[test]
    fn test_read_las_with_format_0() -> Result<()> {
        // From https://stackoverflow.com/questions/30003921/how-can-i-locate-resources-for-testing-with-cargo
        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("resources/test/10_points_format_0.las");

        let mut reader = LASReader::from_path(&test_file_path)?;
        assert_eq!(10, reader.remaining_points());

        let points = reader.read(10)?;

        test_las_reader_data(points.as_ref(), 0);

        Ok(())
    }

    #[test]
    fn test_read_las_with_format_1() -> Result<()> {
        // From https://stackoverflow.com/questions/30003921/how-can-i-locate-resources-for-testing-with-cargo
        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("resources/test/10_points_format_1.las");

        let mut reader = LASReader::from_path(&test_file_path)?;
        assert_eq!(10, reader.remaining_points());

        let points = reader.read(10)?;

        test_las_reader_data(points.as_ref(), 1);

        Ok(())
    }

    #[test]
    fn test_read_las_with_format_2() -> Result<()> {
        // From https://stackoverflow.com/questions/30003921/how-can-i-locate-resources-for-testing-with-cargo
        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("resources/test/10_points_format_2.las");

        let mut reader = LASReader::from_path(&test_file_path)?;
        assert_eq!(10, reader.remaining_points());

        let points = reader.read(10)?;

        test_las_reader_data(points.as_ref(), 2);

        Ok(())
    }

    #[test]
    fn test_read_las_with_format_3() -> Result<()> {
        // From https://stackoverflow.com/questions/30003921/how-can-i-locate-resources-for-testing-with-cargo
        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("resources/test/10_points_format_3.las");

        let mut reader = LASReader::from_path(&test_file_path)?;
        assert_eq!(10, reader.remaining_points());

        let points = reader.read(10)?;

        test_las_reader_data(points.as_ref(), 3);

        Ok(())
    }

    #[test]
    fn test_read_las_with_format_4() -> Result<()> {
        // From https://stackoverflow.com/questions/30003921/how-can-i-locate-resources-for-testing-with-cargo
        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("resources/test/10_points_format_4.las");

        let mut reader = LASReader::from_path(&test_file_path)?;
        assert_eq!(10, reader.remaining_points());

        let points = reader.read(10)?;

        test_las_reader_data(points.as_ref(), 4);

        Ok(())
    }

    #[test]
    fn test_read_las_with_format_5() -> Result<()> {
        // From https://stackoverflow.com/questions/30003921/how-can-i-locate-resources-for-testing-with-cargo
        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("resources/test/10_points_format_5.las");

        let mut reader = LASReader::from_path(&test_file_path)?;
        assert_eq!(10, reader.remaining_points());

        let points = reader.read(10)?;

        test_las_reader_data(points.as_ref(), 5);

        Ok(())
    }
}
