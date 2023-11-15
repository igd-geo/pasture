# Unreleased
- Add `BorrowedBuffer::view_attribute_checked` that returns None rather than panicking if the data type does not match the requested one.
- Require the user to specify the underlying floating point type of their point cloud's `POSITION_3D` when running `compute_centroid`.

# 0.4.0 

- Major overhaul of the buffer API in `pasture-core`. This is a breaking change for previous `pasture` versions
    - Point buffers are now more consistent with how regular Rust containers work (in particular `Vec<T>` and slices)
    - Moved to using the `bytemuck` crate for all the unsafe byte casting, which uncovered several instances of undefined behavior. As a consequence, `pasture-core` no longer supports `bool` as a valid type for point attributes. 
    - Naming of the buffer types and traits should be more consistent and easier to understand. No more `InterleavedVecPointStorage`, this is now `VectorBuffer`. 
    - Introduced a new buffer type called `ExternalMemoryBuffer` which allows using existing memory as the memory source for a `pasture` point buffer. This should enable optimizations for I/O-heavy code, for example when using `mmap` or working with GPU buffers
- Support for reading LAS files with extra bytes