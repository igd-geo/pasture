#![warn(clippy::all)]
//! Algorithms that operate on point-buffers.
//!
//! Pasture contains algorithms that can manipulate the point cloud data or
//! calculate results based on them.

// Algorithm to calculate the bounding box of a point cloud.
pub mod bounds;
// Get the minimum and maximum value of a specific attribute in a point cloud.
pub mod minmax;
// Algorithm to calculate the convex hull of a point cloud.
pub mod convexhull;
// Contains ransac line- and plane-segmentation algorithms in serial and parallel that can be used
// to get the best line-/plane-model and the corresponding inlier indices.
pub mod segmentation;
// Contains an algorithm to reproject coordinate systems
pub mod reprojection;
// Contains voxel-grid-filter function to downsample a given point buffer.
pub mod voxel_grid;
