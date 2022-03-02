# Pasture GPU Octree
## TODO
- Runtime of octree construction can be improved, bottleneck are the first few iterations of the compute shader,
  where the partition of the node indices runs on a large set of numbers (whole buffer at first iteration).
  - Possible way of accelerating the partitioning is via a GPU sort (bitonic sort, radix sort, odd-even-merge sort) on the morton codes of the points.
  - As of now, there are no atomic operations supported by wgpu, but since 2021 they are officially part of the WebGPU spec. In the near future wgpu should allow for the use of atomic operations, which should make implementing sorting easier
- Currently nearest_neighbor search runs recursively and single threaded. When compiling for release this is not really a bottleneck as of now, but the runtime when building in dev mode could improve radically.
  - It's possible to enhance the nearest neighbor search to run multi-threaded in the future, for example through the use of the rayon crate.
  - For that to function it could be possible to restructure the use of some resources in the algorithm to get the ownership right, when working with multiple threads.
- As of now, the compute shader of the construction algorithm terminates prematurely on large input sizes (point clouds with over 3-4 million points).
  - As I was able to reproduce a Vulkan Validation error, that is not captured by wgpu, I submitted an issue in their repo (https://github.com/gfx-rs/wgpu/issues/2484)
  - possible sources for this can be:
    - Illegal access of any resource by the compute shader on large files
    - Termination of gpu process due to timing constraints
    - An error in wgpu, as the error in the issue occurs when polling the gpu.
  - to further investigate the issue I will remain participating on it

If any questions arise, I can be contacted via email: jannis.neus@live.de

## Building the octree

To build the GpuOctree module, pasture needs to be build with the `gpu` feature enabled:

```
cargo build --features gpu
```

## Using octrees

There is an example file in pasture-algorithms that shows how to use this module.
