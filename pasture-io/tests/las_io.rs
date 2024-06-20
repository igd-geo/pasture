use std::io::{Cursor, Seek, SeekFrom};

use anyhow::{Context, Result};
use common::TestLASPointDistribution;
use itertools::Itertools;
use pasture_core::{
    containers::{BorrowedBuffer, BorrowedBufferExt, HashMapBuffer, VectorBuffer},
    layout::{
        attributes::{CLASSIFICATION, NORMAL, POSITION_3D},
        PointType,
    },
    nalgebra::Vector3,
};
use pasture_derive::PointType;
use pasture_io::{
    base::{PointReader, PointWriter},
    las::{
        is_known_las_attribute, LASReader, LASWriter, LasPointFormat0, LasPointFormat1,
        LasPointFormat10, LasPointFormat2, LasPointFormat3, LasPointFormat4, LasPointFormat5,
        LasPointFormat6, LasPointFormat7, LasPointFormat8, LasPointFormat9,
    },
};
use rand::{prelude::Distribution, thread_rng, Rng};

use crate::common::compare_attributes_dynamically_typed;

mod common;

fn write_large_file<T: PointType + PartialEq + std::fmt::Debug>(
    count: usize,
    compressed: bool,
) -> Result<()>
where
    TestLASPointDistribution: Distribution<T>,
{
    let rng = thread_rng();
    let expected_points = rng
        .sample_iter::<T, _>(TestLASPointDistribution)
        .take(count)
        .collect::<VectorBuffer>();

    let point_layout = T::layout();

    let mut in_memory_buffer: Cursor<Vec<u8>> = Cursor::new(Vec::default());

    {
        let mut writer =
            LASWriter::from_writer_and_point_layout(in_memory_buffer, &point_layout, compressed)?;
        writer.write(&expected_points)?;
        writer.flush()?;
        in_memory_buffer = writer.into_inner()?;
    }

    {
        in_memory_buffer.set_position(0);
        let mut reader = LASReader::from_read(in_memory_buffer, compressed, false)?;
        let actual_data = reader.read::<VectorBuffer>(count)?;

        assert_eq!(expected_points.len(), actual_data.len());
        for (idx, (expected_point, actual_point)) in expected_points
            .view::<T>()
            .into_iter()
            .zip(actual_data.view::<T>().into_iter())
            .enumerate()
        {
            assert_eq!(expected_point, actual_point, "Point {idx} does not match");
        }
    }

    Ok(())
}

#[repr(C, packed)]
#[derive(Copy, Clone, Debug, bytemuck::AnyBitPattern, bytemuck::NoUninit, PointType)]
struct PointTypeWithUnsupportedAttribute {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_NORMAL)]
    pub normal: Vector3<f32>,
    #[pasture(BUILTIN_CLASSIFICATION)]
    pub classification: u8,
}

impl Distribution<PointTypeWithUnsupportedAttribute> for TestLASPointDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> PointTypeWithUnsupportedAttribute {
        PointTypeWithUnsupportedAttribute {
            position: Vector3::new(
                rng.gen_range(-1000..1000) as f64,
                rng.gen_range(-1000..1000) as f64,
                rng.gen_range(-1000..1000) as f64,
            ),
            normal: Vector3::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            ),
            classification: rng.gen(),
        }
    }
}

fn write_large_file_with_unsupported_attribute(count: usize, compressed: bool) -> Result<()> {
    let rng = thread_rng();
    let expected_points = rng
        .sample_iter::<PointTypeWithUnsupportedAttribute, _>(TestLASPointDistribution)
        .take(count)
        .collect::<VectorBuffer>();

    let mut in_memory_buffer: Cursor<Vec<u8>> = Cursor::new(Vec::default());

    {
        let mut writer = LASWriter::from_writer_and_point_layout(
            in_memory_buffer,
            &PointTypeWithUnsupportedAttribute::layout(),
            compressed,
        )?;
        writer.write(&expected_points)?;
        writer.flush()?;
        in_memory_buffer = writer.into_inner()?;
    }

    {
        in_memory_buffer.set_position(0);
        let mut reader = LASReader::from_read(in_memory_buffer, compressed, false)?;
        let actual_data = reader.read::<VectorBuffer>(count)?;

        assert_eq!(expected_points.len(), actual_data.len());

        let expected_positions = expected_points
            .view_attribute::<Vector3<f64>>(&POSITION_3D)
            .into_iter()
            .collect_vec();
        let actual_positions = actual_data
            .view_attribute::<Vector3<f64>>(&POSITION_3D)
            .into_iter()
            .collect_vec();
        assert_eq!(expected_positions, actual_positions);

        let expected_classifications = expected_points
            .view_attribute::<u8>(&CLASSIFICATION)
            .into_iter()
            .collect_vec();
        let actual_classifications = actual_data
            .view_attribute::<u8>(&CLASSIFICATION)
            .into_iter()
            .collect_vec();
        assert_eq!(expected_classifications, actual_classifications);

        assert!(!actual_data.point_layout().has_attribute(&NORMAL));
    }

    Ok(())
}

/// A rather complex `PointType` where each attribute requires some conversion because it does not
/// have the default datatype that pasture expects
#[repr(C, packed)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::AnyBitPattern, bytemuck::NoUninit, PointType)]
struct ComplexPointTypeWithConversions {
    #[pasture(BUILTIN_CLASSIFICATION)]
    pub classification: u16,
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: i64,
    #[pasture(BUILTIN_COLOR_RGB)]
    pub color: Vector3<f64>,
    #[pasture(BUILTIN_GPS_TIME)]
    pub gps_time: f32,
    #[pasture(BUILTIN_USER_DATA)]
    pub user_data: i32,
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f32>,
}

impl Distribution<ComplexPointTypeWithConversions> for TestLASPointDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> ComplexPointTypeWithConversions {
        // Generate data as if it were in the default type of the attribute in an LAS file, so that we
        // can perform comparisons on the data after reading it from LAS without any precision loss
        ComplexPointTypeWithConversions {
            position: Vector3::new(
                rng.gen_range(-1000..1000) as f32,
                rng.gen_range(-1000..1000) as f32,
                rng.gen_range(-1000..1000) as f32,
            ),
            classification: rng.gen_range(0..=255) as u16,
            color: Vector3::new(
                rng.gen_range(0..32000) as f64,
                rng.gen_range(0..32000) as f64,
                rng.gen_range(0..32000) as f64,
            ),
            gps_time: rng.gen(),
            intensity: rng.gen_range(0..32000) as i64,
            user_data: rng.gen_range(0..=255),
        }
    }
}

/// Write a large LAS/LAZ file (to check that chunked writing works correctly), using a `PointType` that does not
/// necessarily match any of the known LAS types
fn write_large_file_with_custom_format<T: PointType + PartialEq + std::fmt::Debug>(
    count: usize,
    compressed: bool,
) -> Result<()>
where
    TestLASPointDistribution: Distribution<T>,
{
    let rng = thread_rng();
    let expected_points = rng
        .sample_iter::<T, _>(TestLASPointDistribution)
        .take(count)
        .collect::<HashMapBuffer>();

    let point_layout = T::layout();

    let mut in_memory_buffer: Cursor<Vec<u8>> = Cursor::new(Vec::default());

    {
        let mut writer =
            LASWriter::from_writer_and_point_layout(in_memory_buffer, &point_layout, compressed)?;
        writer.write(&expected_points)?;
        writer.flush()?;
        in_memory_buffer = writer.into_inner()?;
    }

    {
        in_memory_buffer.set_position(0);
        let mut reader = LASReader::from_read(in_memory_buffer, compressed, false)?;
        let actual_data = reader.read::<HashMapBuffer>(count)?;

        assert_eq!(expected_points.len(), actual_data.len());
        for attribute in point_layout
            .attributes()
            .filter(|a| is_known_las_attribute(a.attribute_definition()))
        {
            compare_attributes_dynamically_typed(
                attribute.attribute_definition(),
                &expected_points,
                &actual_data,
            );
        }
    }
    Ok(())
}

#[test]
fn test_write_large_file_format0() -> Result<()> {
    const COUNT: usize = 333333;
    write_large_file::<LasPointFormat0>(COUNT, false).context("Writing large LAS file failed")?;
    write_large_file::<LasPointFormat0>(COUNT, true).context("Writing large LAZ file failed")?;
    Ok(())
}

#[test]
fn test_write_large_file_format1() -> Result<()> {
    const COUNT: usize = 333333;
    write_large_file::<LasPointFormat1>(COUNT, false).context("Writing large LAS file failed")?;
    write_large_file::<LasPointFormat1>(COUNT, true).context("Writing large LAZ file failed")?;
    Ok(())
}

#[test]
fn test_write_large_file_format2() -> Result<()> {
    const COUNT: usize = 333333;
    write_large_file::<LasPointFormat2>(COUNT, false).context("Writing large LAS file failed")?;
    write_large_file::<LasPointFormat2>(COUNT, true).context("Writing large LAZ file failed")?;
    Ok(())
}

#[test]
fn test_write_large_file_format3() -> Result<()> {
    const COUNT: usize = 333333;
    write_large_file::<LasPointFormat3>(COUNT, false).context("Writing large LAS file failed")?;
    write_large_file::<LasPointFormat3>(COUNT, true).context("Writing large LAZ file failed")?;
    Ok(())
}

#[test]
fn test_write_large_file_format4() -> Result<()> {
    const COUNT: usize = 333333;
    write_large_file::<LasPointFormat4>(COUNT, false).context("Writing large LAS file failed")?;
    write_large_file::<LasPointFormat4>(COUNT, true).context("Writing large LAZ file failed")?;
    Ok(())
}

#[test]
fn test_write_large_file_format5() -> Result<()> {
    const COUNT: usize = 333333;
    write_large_file::<LasPointFormat5>(COUNT, false).context("Writing large LAS file failed")?;
    write_large_file::<LasPointFormat5>(COUNT, true).context("Writing large LAZ file failed")?;
    Ok(())
}

#[test]
fn test_write_large_file_format6() -> Result<()> {
    const COUNT: usize = 333333;
    write_large_file::<LasPointFormat6>(COUNT, false).context("Writing large LAS file failed")?;
    write_large_file::<LasPointFormat6>(COUNT, true).context("Writing large LAZ file failed")?;
    Ok(())
}

#[test]
fn test_write_large_file_format7() -> Result<()> {
    const COUNT: usize = 333333;
    write_large_file::<LasPointFormat7>(COUNT, false).context("Writing large LAS file failed")?;
    write_large_file::<LasPointFormat7>(COUNT, true).context("Writing large LAZ file failed")?;
    Ok(())
}

#[test]
fn test_write_large_file_format8() -> Result<()> {
    const COUNT: usize = 333333;
    write_large_file::<LasPointFormat8>(COUNT, false).context("Writing large LAS file failed")?;
    write_large_file::<LasPointFormat8>(COUNT, true).context("Writing large LAZ file failed")?;
    Ok(())
}

#[test]
fn test_write_large_file_format9() -> Result<()> {
    const COUNT: usize = 333333;
    write_large_file::<LasPointFormat9>(COUNT, false).context("Writing large LAS file failed")?;
    // LAZ with point format 9 currently unsupported due to problems with waveform data
    Ok(())
}

#[test]
fn test_write_large_file_format10() -> Result<()> {
    const COUNT: usize = 333333;
    write_large_file::<LasPointFormat10>(COUNT, false).context("Writing large LAS file failed")?;
    // LAZ with point format 10 currently unsupported due to problems with waveform data
    Ok(())
}

#[test]
fn test_write_large_file_with_unsupported_attribute() -> Result<()> {
    const COUNT: usize = 333333;
    write_large_file_with_unsupported_attribute(COUNT, false)
        .context("Writing large LAS file with unsupported attribute failed")?;
    write_large_file_with_unsupported_attribute(COUNT, true)
        .context("Writing large LAZ file with unsupported attribute failed")?;
    Ok(())
}

#[test]
fn test_write_large_file_with_custom_format() -> Result<()> {
    const COUNT: usize = 333333;
    write_large_file_with_custom_format::<ComplexPointTypeWithConversions>(COUNT, false)
        .context("Writing large LAS file with custom format failed")?;
    write_large_file_with_custom_format::<ComplexPointTypeWithConversions>(COUNT, true)
        .context("Writing large LAZ file with custom format failed")?;
    Ok(())
}

#[test]
fn test_las_laz_readers_are_equivalent() -> Result<()> {
    // Test that writing and reading points as LAS and as LAZ gives the same result
    let rng = thread_rng();
    let expected_points = rng
        .sample_iter::<LasPointFormat0, _>(TestLASPointDistribution)
        .take(345)
        .collect::<VectorBuffer>();

    let point_layout = LasPointFormat0::layout();

    let mut las_buffer: Cursor<Vec<u8>> = Cursor::new(Vec::default());
    let mut laz_buffer: Cursor<Vec<u8>> = Cursor::new(Vec::default());

    {
        let mut writer = LASWriter::from_writer_and_point_layout(las_buffer, &point_layout, false)?;
        writer.write(&expected_points)?;
        writer.flush()?;
        las_buffer = writer.into_inner()?;
    }

    {
        let mut writer = LASWriter::from_writer_and_point_layout(laz_buffer, &point_layout, true)?;
        writer.write(&expected_points)?;
        writer.flush()?;
        laz_buffer = writer.into_inner()?;
    }

    let points_from_las_file = {
        las_buffer.seek(SeekFrom::Start(0))?;
        let mut reader = LASReader::from_read(las_buffer, false, false)?;
        reader.read::<VectorBuffer>(reader.remaining_points())?
    };

    let points_from_laz_file = {
        laz_buffer.seek(SeekFrom::Start(0))?;
        let mut reader = LASReader::from_read(laz_buffer, true, false)?;
        reader.read::<VectorBuffer>(reader.remaining_points())?
    };

    assert_eq!(expected_points, points_from_las_file);
    assert_eq!(expected_points, points_from_laz_file);

    Ok(())
}
