use std::io::Write;

use anyhow::Result;
use pasture_core::{
    containers::{BorrowedBuffer, InterleavedBuffer, VectorBuffer},
    layout::attributes::CLASSIFICATION,
};
use pasture_io::{base::PointReader, las::LASReader};

fn main() -> Result<()> {
    let mut las_reader = LASReader::from_path("pointcloud.las", false)?;
    let points = las_reader.read::<VectorBuffer>(las_reader.remaining_points())?;

    const CLASS_BUILDING: u8 = 6;
    for (index, _) in points
        .view_attribute::<u8>(&CLASSIFICATION)
        .into_iter()
        .enumerate()
        .filter(|(_, c)| *c == CLASS_BUILDING)
    {
        std::io::stdout().write_all(points.get_point_ref(index))?;
    }

    Ok(())
}
