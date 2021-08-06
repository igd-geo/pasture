#![warn(clippy::all)]
//! Algorithms that operate on point-buffers.
//!
//! Pasture contains algorithms that can manipulate the point cloud data or
//! calculate results based on them.

// Algorithm to calculate the bounding box of a point cloud.
pub mod bounds;
// Get the minimum and maximum value of a specific attribute in a point cloud.
pub mod minmax;
// Contains ransac line- and plane-segmentation algorithms in serial and parallel that can be used
// to get the best line-/plane-model and the corresponding inlier indices.
pub mod segmentation;
// Contains methods to compute various structure measures using the structure tensor. The definitions for the structure measures
// are taken from the paper 'FEATURE RELEVANCE ASSESSMENT FOR THE SEMANTIC INTERPRETATION OF 3D POINT CLOUD DATA' (Martin Weinmann, Boris Jutzi, Clément Mallet, ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences, Volume II-5/W2, 2013)
pub mod structure_measures;
