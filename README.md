# pasture

A Rust library for working with point cloud data. It features:
-  Fine-grained support for arbitrary point attributes, similar to [PDAL](https://pdal.io/), but with added type safety
-  A very flexible memory model, natively supporting both Array-of-Structs (AoS) and Struct-of-Arrays (SoA) memory layouts
-  Support for reading and writing various point cloud formats with the `pasture-io` crate
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

# Development

`pasture` is in the early stages of development and as such, bugs might occur. 

# License

`pasture` is distributed under the terms of the Apacke License (Version 2.0). See [LICENSE](LICENSE) for details. 