#![warn(clippy::all)]

//! Core data structures for working with point cloud data
//!
//! Pasture provides data structures for reading, writing and in-memory handling of arbitrary point cloud data.
//! The best way to get started with Pasture is to look at the [example code](https://github.com/Mortano/pasture/tree/main/pasture-core/examples).
//! For understanding Pasture, it is best to look at the [PointLayout](crate::layout::PointLayout) type and the [containers](crate::containers) module.

pub extern crate nalgebra;
extern crate self as pasture_core;

pub mod containers;
/// Defines attributes and data layout of point cloud data
pub mod layout;
/// Useful mathematical tools when working with point clooud data
pub mod math;
/// Data structures for handling point cloud metadata
pub mod meta;
/// Utilities
pub mod util;
#[cfg(feature = "gpu")]
pub mod gpu;
