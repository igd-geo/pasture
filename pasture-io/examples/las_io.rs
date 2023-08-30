use anyhow::{bail, Context, Result};
use pasture_core::{
    containers::{BorrowedBuffer, HashMapBuffer, VectorBuffer},
    layout::{attributes::POSITION_3D, PointLayout},
    nalgebra::Vector3,
};
use pasture_io::{base::PointReader, las::LASReader};

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 2 {
        bail!("Usage: las_io <INPUT_FILE>");
    }

    let input_file = args[1].as_str();
    // Reading a LAS/LAZ file works with the `LASReader` type. You also need the `pasture_io::base::PointReader` trait in scope
    // to access the `read` function
    let mut las_reader = LASReader::from_path(input_file).context("Could not open LAS file")?;
    // We have to tell `read` what kind of buffer we want our data to be read into. This is a change from the
    // old pasture buffer API, where `read` returned an opaque buffer type
    let points = las_reader
        .read::<VectorBuffer>(10)
        .context("Error while reading points")?;

    // Print the first 10 positions as Vector3<f64> values
    for position in points
        .view_attribute::<Vector3<f64>>(&POSITION_3D)
        .into_iter()
        .take(10)
    {
        println!("{}", position);
    }

    // Let's do the same thing, but read data into an existing buffer with a custom `PointLayout`!
    drop(las_reader);
    las_reader = LASReader::from_path(input_file).context("Could not open LAS file")?;

    // For this, we first allocate a  point buffer manually. Since we only care for a single point attribute, the columnar
    // memory layout is a good candidate. We also have to specify the `PointLayout` that we want, which in our case only includes
    // the `POSITION_3D` attribute
    let layout = PointLayout::from_attributes(&[POSITION_3D]);
    let mut points = HashMapBuffer::with_capacity(10, layout);
    // The `LASReader` reads the data into the memory layout of our custom point buffer
    las_reader
        .read_into(&mut points, 10)
        .context("Can't read points in custom layout")?;

    // Since we know the type of our point buffer, we can do more than just obtain an iterator. The columnar memory layout
    // allows iterating over positions by reference using a call to `iter()` (similar to how vectors and slices work):
    for position in points.view_attribute::<Vector3<f64>>(&POSITION_3D).iter() {
        // Notice how `position` is a *borrow* now, instead of a copy like with `view_attribute(...).into_iter()`
        println!("{}", position);
    }

    Ok(())
}
