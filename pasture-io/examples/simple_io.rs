use anyhow::{bail, Context, Result};
use pasture_core::{
    containers::{BorrowedBuffer, VectorBuffer},
    layout::attributes::POSITION_3D,
    nalgebra::Vector3,
};
use pasture_io::base::{read_all, write_all};

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 3 {
        bail!("Usage: simple_io <INPUT_FILE> <OUTPUT_FILE>");
    }

    // Reading a point cloud file is as simple as calling `read_all`
    let points = read_all::<VectorBuffer, _>(args[1].as_str()).context("Failed to read points")?;

    if points.point_layout().has_attribute(&POSITION_3D) {
        for position in points
            .view_attribute::<Vector3<f64>>(&POSITION_3D)
            .into_iter()
            .take(10)
        {
            println!("({};{};{})", position.x, position.y, position.z);
        }
    } else {
        println!("Point cloud files has no positions!");
    }

    // Writing all points from a buffer to a file is also easy: Just call `write_all`
    write_all(&points, args[2].as_str()).context("Failed to write points")?;

    Ok(())
}
