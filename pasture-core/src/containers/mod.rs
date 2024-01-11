//! Defines traits for the different types of buffers that pasture supports, as well as some
//! common implementations.
//!
//! # The buffer hierarchy in pasture
//!
//! At its core, pasture distinguishes between two types of buffer properties:
//! 1) Who owns the memory of the buffer?
//! 2) How is the point data layed out in memory?
//!
//! For the first property, there are three options:
//! 1) Memory is owned by the buffer itself (like in a `Vec<T>`)
//! 2) Memory is borrowed mutably (like in a `&mut [T]`)
//! 3) Memory is borrowed immutably (like in a `&[T]`)
//!
//! For the second property, pasture knowns three different types of memory layouts:
//! 1) Unknown memory layout: No guarantees about the memory layout, reading and writing point data works
//!    exclusively by value
//! 2) Interleaved memory layout: All attributes for a single point are stored together in memory. This
//!    is similar to storing a `Vec<T>` where `T` is some `struct` whose members are the point attributes
//!    (like `POSITION_3D`, `CLASSIFICATION` etc.)
//! 3) Columnar memory layout: Data for the same attribute of multiple points is stored together in
//!    memory. This is sometimes called a 'struct-of-arrays' memory layout and is similar to how a column-oriented
//!    database stores its records.
//!
//! Based on these two properties, pasture defines the following set of traits:
//!
//! ## Memory ownership traits
//!
//! Each buffer for point data (simply referred to as a 'point buffer' from here on) has to implement at
//! least one of the memory ownership traits [`BorrowedBuffer`], [`BorrowedMutBuffer`], and [`OwningBuffer`].  
//! These correspond to the three ways of ownership of the buffer memory and form a hierarchy, where
//! [`BorrowedMutBuffer`] implies [`BorrowedBuffer`], and [`OwningBuffer`] implies [`BorrowedMutBuffer`].
//!
//! ## Memory layout traits
//!
//! Optionally, a point buffer can implement one or more of the memory layout traits [`InterleavedBuffer`],
//! [`InterleavedBufferMut`], [`ColumnarBuffer`] and [`ColumnarBufferMut`]. The distinction between immutable
//! and mutable memory layout is important for slicing, which is explained in a later section.
//!
//! If the buffer type stores point data in an interleaved layout, [`InterleavedBuffer`] can be implemented,
//! which allows accessing point data by borrow (or mutable borrow through [`InterleavedBufferMut`]), either
//! for a single point or a range of points.
//!
//! If the buffer type stores point data in a columnar layout, [`ColumnarBuffer`] can be implemented, which
//! allows accessing attribute data by borrow (or mutable borrow through [`ColumnarBufferMut`]), either for
//! a single point or a range of points.
//!
//! To illustrate this, here is an example. Given the following point type:
//!
//! ```
//! use nalgebra::Vector3;
//! // (skipped necessary derives for brevity)
//! struct Point {
//!    position: Vector3<f64>,
//!    classification: u8,
//! }
//! ```
//!
//! An interleaved buffer allows accessing the point data as a `&[Point]`, whereas a columnar buffer allows
//! accessing e.g. all positions as a `&[Vector3<f64>]`. In practice, the point buffer traits work with
//! **untyped** data, so they will always return `&[u8]` or `&mut [u8]` instead of strongly typed values!
//!
//! # Slicing buffers
//!
//! Just as with a Rust `Vec<T>`, pasture point buffers can support slicing through the [`SliceBuffer`] and
//! [`SliceBufferMut`] traits. Given some buffer type `T`, calling `slice(range)` on the buffer will yield
//! an immutable slice to the point data of that buffer. The slice keeps its memory layout, so if `T`
//! implements `InterleavedBuffer`, the slice will also implement `InterleavedBuffer` (and conversely for
//! all the other memory layout traits). Due to limitations of the [`std::ops::Index`] trait in Rust, a separate trait
//! is required for slicing point buffers, meaning you cannot use the `[]` operator for slicing and instead
//! have to call `slice()` or `slice_mut()` explicitly.
//!
//! # Raw vs. typed memory
//!
//! Since the pasture point buffers store dynamically typed data (i.e. point data whose attributes are only
//! known at runtime), the API of all the buffer traits works with byte slices (`&[u8]` and `&mut [u8]`) instead
//! of strongly typed data. Whenever possible and viable (due to performance reasons), runtime type checking
//! is performed using the [`PointLayout`](crate::layout::PointLayout), which is part of every point buffer (it is mandated by the
//! [`BorrowedBuffer`] trait). All methods that perform no type checking are marked `unsafe`. See their documentation for information
//! about invariants that must hold when calling these functions.
//!
//! Depending on your application, you might be able to work almost exclusively with strongly typed point data.
//! In this case, all point buffers provide `view` and `view_attribute` methods (with mutable variants) that allow
//! access to the point buffer through a strongly typed interface. See the `buffer_views` module for more information
//! on buffer views.
//!
//! # Specific buffer types
//!
//! Currently, pasture provides three specific buffer implementations:
//! - [`VectorBuffer`], an owning, interleaved point buffer using a `Vec<u8>` as its underlying storage
//! - [`HashMapBuffer`], an owning, columnar point buffer using a `HashMap<PointAttributeDefinition, Vec<u8>>` as its
//!   underlying storage
//! - [`ExternalMemoryBuffer`], a non-owning (though potentially mutable) interleaved point buffer
//!   which uses an arbitrary external memory resource for its underlying storage

mod buffers;
pub use self::buffers::*;

mod attribute_iterators;
pub use self::attribute_iterators::*;

mod raw_attribute_view;
pub use self::raw_attribute_view::*;

mod point_iterators;
pub use self::point_iterators::*;

mod buffer_views;
pub use self::buffer_views::*;

mod untyped_point;
pub use self::untyped_point::*;

mod slice;
pub use self::slice::*;
