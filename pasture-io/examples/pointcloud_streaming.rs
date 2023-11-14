use anyhow::{Context, Result};
use pasture_core::{
    containers::BorrowedBuffer, layout::attributes::POSITION_3D, nalgebra::Vector3,
};
use pasture_io::base::Pointcloud;

fn main() -> Result<()> {
    let pointcloud = Pointcloud::from_dir(
        "/Users/pbormann/data/projects/progressive_indexing/experiment_data/ca13/las",
        None,
    )
    .context("Failed to load pointcloud")?;

    println!("Files: {:#?}", pointcloud.files());
    println!("Points: {}", pointcloud.num_points());

    let mut min = Vector3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut max = Vector3::new(f64::MIN, f64::MIN, f64::MIN);

    pointcloud.stream(1_000_000).for_each(|maybe_chunk| {
        let (chunk_info, data) = maybe_chunk.expect("Failed to load next chunk");
        let positions = data.view_attribute::<Vector3<f64>>(&POSITION_3D);
        for position in positions
            .into_iter()
            .take(chunk_info.point_range_in_file.len())
        {
            min.x = min.x.min(position.x);
            min.y = min.y.min(position.y);
            min.z = min.z.min(position.z);
            max.x = max.x.max(position.x);
            max.y = max.y.max(position.y);
            max.z = max.z.max(position.z);
        }
    });

    println!("Min: {min}");
    println!("Max: {max}");

    Ok(())
}
