#![warn(clippy::all)]

use anyhow::Result;
use pasture_core::containers::{InterleavedVecPointStorage, PointBufferWriteable};
use pasture_io::base::PointReader;
use pasture_io::las::LASReader;

fn main() -> Result<()> {
    let mut reader = LASReader::from_path("/home/pbormann/data/geodata/pointclouds/datasets/navvis_m6_3rdFloor/navvis_m6_HQ3rdFloor.laz")?;

    let total_points = reader.remaining_points();
    let mut remaining_points = total_points;
    let chunk_size = 50_000;
    let mut buffer = InterleavedVecPointStorage::with_capacity(
        chunk_size,
        reader.get_default_point_layout().clone(),
    );
    let chunks = (remaining_points + (chunk_size - 1)) / chunk_size;

    for _ in 0..chunks {
        let points_in_chunk = std::cmp::min(chunk_size, remaining_points);
        reader.read_into(&mut buffer, points_in_chunk)?;
        eprintln!("Read {} points", total_points - remaining_points);
        remaining_points -= points_in_chunk;
        buffer.clear();
    }

    Ok(())
}
