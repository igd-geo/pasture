//! Provides the core data structures and traits for storing point cloud data in memory.
//!
//! Pasture exposes a hierarchy of traits that define the capabilities of the in-memory buffers that
//! the point data is stored in. The most basic trait is [PointBuffer], which is implemented by every
//! specific point cloud buffer in Pasture. It provides basic access to the point data without making
//! any assumptions about the memory layout of the point data. More specific types include [InterleavedPointBuffer]
//! and [PerAttributePointBuffer], which store data in an Interleaved or PerAttribute memory layout.
//! For an explanation of these memory layouts, see the [PointLayout](crate::layout::PointLayout) type.
//!
//! On top of these traits, Pasture provides some specific implementations for storing contiguous
//! point data in [Interleaved](InterleavedVecPointStorage) or [PerAttribute](PerAttributeVecPointStorage)
//! layouts, as well as [non-owning](InterleavedPointView) and [sliced](InterleavedPointBufferSlice) versions
//! of these buffers.
//!
//! Lastly, this module exposes some helper functions for iterating over the point data inside any of
//! these buffers.

mod point_buffer;
pub use self::point_buffer::*;

mod point_view;
pub use self::point_view::*;

mod attribute_iterators;
pub use self::attribute_iterators::*;

mod attribute_range;
pub use self::attribute_range::*;

mod point_iterators;
pub use self::point_iterators::*;

mod vec_buffers;
pub use self::vec_buffers::*;

mod slice_buffers;
pub use self::slice_buffers::*;

mod untyped_point;
pub use self::untyped_point::*;
