use std::convert::From;
use std::{fmt::Debug, io::SeekFrom, path::Path};

use anyhow::{bail, Result};
use las::{point::Format, Read, Reader};

use crate::base::{PointReader, SeekToPoint};
use pasture_core::math::AABB;
use pasture_core::nalgebra::Point3;
use pasture_core::{
    containers::{InterleavedPointBuffer, InterleavedVecPointStorage, PointBuffer},
    layout::PointLayout,
    meta::Metadata,
};

use super::{LASPoint_Format0, LASPoint_Format1, LASPoint_Format2, point_layout_from_las_point_format};

/// `Metadata` implementation for LAS/LAZ files
#[derive(Debug, Clone)]
pub struct LASMetadata {
    bounds: AABB<f64>,
    point_count: usize,
    point_format: u8,
}

impl LASMetadata {
    /// Creates a new `LASMetadata` with the given `bounds` and `point_count`
    ///
    /// Example:
    /// ```
    /// use pasture_io::las::LASMetadata;
    /// use pasture_core::{
    ///     math::AABB,
    ///     nalgebra::Point3,
    /// };
    ///
    /// let las_meta = LASMetadata::new(
    ///     AABB::from_min_max(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
    ///     1024,
    ///     0,
    /// );
    /// ```
    pub fn new(bounds: AABB<f64>, point_count: usize, point_format: u8) -> Self {
        Self {
            bounds,
            point_count,
            point_format,
        }
    }
}

impl Metadata for LASMetadata {
    fn bounds(&self) -> Option<AABB<f64>> {
        Some(self.bounds)
    }

    fn get_named_field(&self, field_name: &str) -> Option<&dyn std::any::Any> {
        todo!()
    }
}

impl From<&las::Header> for LASMetadata {
    fn from(header: &las::Header) -> Self {
        let min_point = Point3::new(
            header.bounds().min.x,
            header.bounds().min.y,
            header.bounds().min.z,
        );
        let max_point = Point3::new(
            header.bounds().max.x,
            header.bounds().max.y,
            header.bounds().max.z,
        );
        Self {
            bounds: AABB::from_min_max_unchecked(min_point, max_point),
            point_count: header.number_of_points() as usize,
            point_format: header
                .point_format()
                .to_u8()
                .expect("Invalid LAS point format"),
        }
    }
}

impl From<las::Header> for LASMetadata {
    fn from(header: las::Header) -> Self {
        (&header).into()
    }
}

struct LASReader {
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
        self.metadata.point_count - self.current_point_index
    }

    /// Read points in LAS point format 0
    fn do_read_format_0(
        &mut self,
        count: usize,
        buffer: &mut InterleavedVecPointStorage,
    ) -> Result<()> {
        for _ in 0..count {
            let las_point = self
                .reader
                .read()
                .expect("LASReader:do_read_format_0: las::Reader::read returned None")?;
            let typed_point: LASPoint_Format0 = las_point.into();
            buffer.push_point_unchecked(typed_point);
        }

        Ok(())
    }

    fn do_read_format_1(&mut self, count: usize, buffer: &mut InterleavedVecPointStorage) -> Result<()> {
        for _ in 0..count {
            let las_point = self
                .reader
                .read()
                .expect("LASReader:do_read_format_1: las::Reader::read returned None")?;
            let typed_point: LASPoint_Format1 = las_point.into();
            buffer.push_point_unchecked(typed_point);
        }

        Ok(())
    }

    fn do_read_format_2(&mut self, count: usize, buffer: &mut InterleavedVecPointStorage) -> Result<()> {
        for _ in 0..count {
            let las_point = self
                .reader
                .read()
                .expect("LASReader:do_read_format_2: las::Reader::read returned None")?;
            let typed_point: LASPoint_Format2 = las_point.into();
            buffer.push_point_unchecked(typed_point);
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
        match self.metadata.point_format {
            0 => self.do_read_format_0(num_points_to_read, &mut buffer)?,
            1 => self.do_read_format_1(num_points_to_read, &mut buffer)?,
            2 => self.do_read_format_2(num_points_to_read, &mut buffer)?,
            _ => panic!(
                "Currently unsupported point format {}",
                self.metadata.point_format
            ),
        }

        Ok(Box::new(buffer))
    }

    fn read_into(&mut self, point_buffer: &mut dyn PointBuffer, _count: usize) -> Result<usize> {
        let _target_layout = point_buffer.point_layout();

        todo!()
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
            SeekFrom::End(offset_from_end) => self.metadata.point_count as i64 + offset_from_end,
            SeekFrom::Start(offset_from_start) => offset_from_start as i64,
        };

        if index_from_start < 0 {
            bail!(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Cannot seek before point index 0!"
            ));
        }

        let actual_index = usize::min(index_from_start as usize, self.metadata.point_count);
        self.reader.seek(actual_index as u64)?;
        Ok(actual_index) 
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pasture_core::{nalgebra::Vector3, containers::*};

    #[test]
    fn test_read_las() -> Result<()> {
        let mut reader = LASReader::from_path("/home/pbormann/data/geodata/pointclouds/datasets/navvis_m6_3rdFloor/navvis_m6_HQ3rdFloor.laz")?;
        println!("Num points: {}", reader.remaining_points());

        let points = reader.read(1024)?;
        let positions = attributes::<Vector3<f64>>(points.as_ref(), &pasture_core::layout::attributes::POSITION_3D).collect::<Vec<_>>();
        println!("First position: {}", positions[0]);

        Ok(())
    }

}
