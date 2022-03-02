# pasture

A Rust library for working with point cloud data. It features:
-  Fine-grained support for arbitrary point attributes, similar to [PDAL](https://pdal.io/), but with added type safety
-  A very flexible memory model, natively supporting both Array-of-Structs (AoS) and Struct-of-Arrays (SoA) memory layouts
-  Support for reading and writing various point cloud formats with the `pasture-io` crate (such as `LAS`, `LAZ`, `3D Tiles`, as well as ASCII files)
-  A growing set of algorithms with the `pasture-algorithms` crate

To this end, `pasture` chooses flexibility over simplicity. If you are looking for something small and simple, for example to work with LAS files, try a crate like [`las`](https://crates.io/crates/las). If you are planning to implement high-performance tools and services that will work with very large point cloud data, `pasture` is what you are looking for!

# Usage 

Add this to your `Cargo.toml`:
```
[dependencies]
pasture-core = "0.1.0"
# You probably also want I/O support
pasture-io = "0.1.0"
```

Here is an example on how to load a pointcloud from an LAS file and do something with it:

```Rust
use anyhow::{Context, Result};
use pasture_core::{
    containers::PointBufferExt, layout::attributes::POSITION_3D, nalgebra::Vector3,
};
use pasture_io::{base::PointReader, las::LASReader};

fn main() -> Result<()> {
    let mut las_reader = LASReader::from_path("pointcloud.las").context("Could not open LAS file")?;
    let num_points = las_reader.remaining_points();
    let points = las_reader
        .read(10)
        .context("Error while reading points")?;

    // Print the first 10 positions as Vector3<f64> values
    for position in points.iter_attribute::<Vector3<f64>>(&POSITION_3D).take(10) {
        println!("{}", position);
    }

    Ok(())
}
```

For more examples, check out the [`pasture_core` examples](pasture_core/examples) and the [`pasture_io` examples](pasture_io/examples).

# Development

`pasture` is in the early stages of development and bugs may occur. The GPU features of `pasture_core` are highly unstable and under active development. 

# License

`pasture` is distributed under the terms of the Apache License (Version 2.0). See [LICENSE](LICENSE) for details. 