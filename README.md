# pasture

A Rust library for working with point cloud data. It features:
-  Fine-grained support for arbitrary point attributes, similar to [PDAL](https://pdal.io/), but with added type safety
-  A very flexible memory model, natively supporting both Array-of-Structs (AoS) and Struct-of-Arrays (SoA) memory layouts (which `pasture` calls 'interleaved' and 'columnar')
-  Support for reading and writing various point cloud formats with the `pasture-io` crate (such as `LAS`, `LAZ`, `3D Tiles`, as well as ASCII files)
-  A growing set of algorithms with the `pasture-algorithms` crate

To this end, `pasture` chooses flexibility over simplicity. If you are looking for something small and simple, for example to work with LAS files, try a crate like [`las`](https://crates.io/crates/las). If you are planning to implement high-performance tools and services that will work with very large point cloud data, `pasture` is what you are looking for!

# Usage 

Add this to your `Cargo.toml`:
```
[dependencies]
pasture-core = "0.4.0"
# You probably also want I/O support
pasture-io = "0.4.0"
```

Here is an example on how to load a pointcloud from an LAS file and do something with it:

```Rust
use anyhow::{bail, Context, Result};
use pasture_core::{
    containers::{BorrowedBuffer, VectorBuffer},
    layout::attributes::POSITION_3D,
    nalgebra::Vector3,
};
use pasture_io::base::{read_all};

fn main() -> Result<()> {
    // Reading a point cloud file is as simple as calling `read_all`
    let points = read_all::<VectorBuffer, _>("pointcloud.las").context("Failed to read points")?;

    if points.point_layout().has_attribute(&POSITION_3D) {
        for position in points
            .view_attribute::<Vector3<f64>>(&POSITION_3D)
            .into_iter()
            .take(10)
        {
            println!("({};{};{})", position.x, position.y, position.z);
        }
    } else {
        bail!("Point cloud files has no positions!");
    }

    Ok(())
}

```

For more examples, check out the [`pasture_core` examples](pasture-core/examples) and the [`pasture_io` examples](pasture-io/examples).

## Migration from versions < 0.4

With version `0.4`, the buffer API of `pasture-core` was rewritten. If you are migrating from an earlier version, here are some guidelines for migration. Also check out the documentation of the [`containers` module](https://docs.rs/pasture-core/latest/pasture_core/containers/index.html).

### New buffer types

The main buffer types were renamed:
* `InterleavedVecPointStorage` is now `VectorBuffer`
* `PerAttributeVecPointStorage` is now `HashMapBuffer`

The trait structure is also different:
* `PointBuffer` and `PointBufferWriteable` are replaced by `BorrowedBuffer`, `BorrowedMutBuffer`, and `OwningBuffer`, which define the ownership model of the buffer memory
* `InterleavedPointBuffer` and `InterleavedPointBufferMut` are now `InterleavedBuffer` and `InterleavedBufferMut`
* `PerAttributePointBuffer` and `PerAttributePointBufferMut` are now `ColumnarBuffer` and `ColumnarBufferMut`. In general, the term `PerAttribute` is replaced by the more common term `Columnar`

There are no more extension traits (e.g. `PointBufferExt`). To get/set strongly typed point data, you now use *views* which can be obtained through the `BorrowedBuffer` and `BorrowedBufferMut` traits:

```
let view = buffer.view_attribute::<Vector3<f64>>(&POSITION_3D);
```

Views support strongly typed access to the data and are convertible to iterators. 

### New interface for readers and writers

The `PointReader` and `PointWriter` traits are no longer object safe. Instead, they have `read` and `read_into` methods that are strongly typed over the buffer type for improved efficiency. There is a `GenericPointReader` type, which uses static dispatch and encapsulates readers for LAS, LAZ, and 3D Tiles. 

# Development

`pasture` is in the early stages of development and bugs may occur. 

# License

`pasture` is distributed under the terms of the Apache License (Version 2.0). See [LICENSE](LICENSE) for details. 