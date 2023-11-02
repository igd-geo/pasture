/**
 * This example demonstrates external memory buffers in pasture, and how we can use them to
 * map a LAS file directly into memory and view its point data through an `ExternalMemoryBuffer`.
 * This is probably the fastest way to read LAS and other fixed-width binary formats, as no
 * parsing takes place!
 */
use std::{
    fs::File,
    io::{BufWriter, Cursor, Write},
};

use anyhow::{bail, Result};
use memmap2::{Advice, MmapOptions};
use pasture_core::{
    containers::{BorrowedBuffer, ExternalMemoryBuffer},
    layout::{attributes::POSITION_3D, PointAttributeDataType},
    nalgebra::Vector3,
};
use pasture_io::las::{point_layout_from_las_metadata, LASReader};

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 2 {
        bail!("Usage: fast_las_parsing <INPUT_FILE>");
    }

    let file_path = &args[1];
    let file = unsafe { MmapOptions::new().map(&File::open(file_path)?)? };
    file.advise(Advice::Sequential)?;

    let las_metadata = LASReader::from_read(Cursor::new(&file), false, false)?
        .las_metadata()
        .clone();

    let las_header = las_metadata.raw_las_header().unwrap();
    let raw_header = las_header.clone().into_raw()?;

    let offset_to_point_data = raw_header.offset_to_point_data as usize;
    let size_of_point_data =
        raw_header.point_data_record_length as usize * raw_header.number_of_point_records as usize;

    let points_memory = &file[offset_to_point_data..(offset_to_point_data + size_of_point_data)];
    let point_layout = point_layout_from_las_metadata(&las_metadata, true)?;

    // now we can get a very efficient point buffer that points to the actual LAS memory!
    let buffer = ExternalMemoryBuffer::new(points_memory, point_layout);

    let stdout = std::io::stdout().lock();
    let mut stdout = BufWriter::new(stdout);

    for position in buffer
        .view_attribute::<Vector3<i32>>(
            &POSITION_3D.with_custom_datatype(PointAttributeDataType::Vec3i32),
        )
        .into_iter()
    {
        stdout.write_all(bytemuck::bytes_of(&position))?;
    }

    Ok(())
}
