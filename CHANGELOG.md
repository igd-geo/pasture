# 0.5.0

- Allows creating point and attribute views on trait objects, i.e. `dyn BorrowedBuffer` and the likes
- `serde` is now an optional dependency in `pasture-core`
- Fixed some bugs with LAS reading/writing
    - The `LargeFile` header entry is now only set for LAS files that actually support it, i.e. LAS files with version 1.4
    - Compressed LAZ files now correctly write the LAZ compression bit into the point format

# 0.4.0 

- Major overhaul of the buffer API in `pasture-core`. This is a breaking change for previous `pasture` versions
    - Point buffers are now more consistent with how regular Rust containers work (in particular `Vec<T>` and slices)
    - Moved to using the `bytemuck` crate for all the unsafe byte casting, which uncovered several instances of undefined behavior. As a consequence, `pasture-core` no longer supports `bool` as a valid type for point attributes. 
    - Naming of the buffer types and traits should be more consistent and easier to understand. No more `InterleavedVecPointStorage`, this is now `VectorBuffer`. 
    - Introduced a new buffer type called `ExternalMemoryBuffer` which allows using existing memory as the memory source for a `pasture` point buffer. This should enable optimizations for I/O-heavy code, for example when using `mmap` or working with GPU buffers
- Support for reading LAS files with extra bytes