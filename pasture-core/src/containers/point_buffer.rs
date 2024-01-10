use anyhow::Result;
use std::{collections::HashMap, iter::FromIterator, ops::Range};

use crate::layout::{
    PointAttributeDefinition, PointAttributeMember, PointLayout, PointType, PrimitiveType,
};

use super::{
    buffer_views::{AttributeView, AttributeViewMut, PointView, PointViewMut},
    AttributeViewConverting, BufferSliceColumnar, BufferSliceColumnarMut, BufferSliceInterleaved,
    BufferSliceInterleavedMut, RawAttributeView, RawAttributeViewMut, SliceBuffer, SliceBufferMut,
};

/// Base trait for all point buffers in pasture. The only assumption this trait makes is that the
/// underlying memory can be borrowed by the buffer. Provides point and attribute accessors by
/// untyped value (i.e. copying into a provided `&mut [u8]`)
pub trait BorrowedBuffer<'a> {
    /// Returns the length of this buffer, i.e. the number of points
    ///
    /// # Example
    ///
    /// ```
    /// use pasture_core::containers::*;
    /// use pasture_core::layout::*;
    ///
    /// let buffer = VectorBuffer::new_from_layout(PointLayout::default());
    /// assert_eq!(0, buffer.len());
    /// ```
    fn len(&self) -> usize;
    /// Returns `true` if this buffer does not store any points
    ///
    /// # Example
    ///
    /// ```
    /// use pasture_core::containers::*;
    /// use pasture_core::layout::*;
    ///
    /// let buffer = VectorBuffer::new_from_layout(PointLayout::default());
    /// assert!(buffer.is_empty());
    /// ```
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Returns the `PointLayout` of this buffer. The `PointLayout` describes the structure of a single
    /// point at runtime.
    ///
    /// # Example
    ///
    /// ```
    /// use pasture_core::containers::*;
    /// use pasture_core::layout::*;
    ///
    /// let layout = PointLayout::from_attributes(&[attributes::POSITION_3D, attributes::INTENSITY]);
    /// let buffer = VectorBuffer::new_from_layout(layout.clone());
    /// assert_eq!(layout, *buffer.point_layout());
    /// ```
    fn point_layout(&self) -> &PointLayout;
    /// Writes the data for the point at `index` from this buffer into `data`
    ///
    /// # Panics
    ///
    /// May panic if `index` is out of bounds.
    /// May panic if `data.len()` does not equal `self.point_layout().size_of_point_entry()`
    fn get_point(&self, index: usize, data: &mut [u8]);
    /// Writes the data for the given `range` of points from this buffer into `data`
    ///
    /// # Panics
    ///
    /// May panic if `range` is out of bounds.
    /// May panic if `data.len()` does not equal `range.len() * self.point_layout().size_of_point_entry()`
    fn get_point_range(&self, range: Range<usize>, data: &mut [u8]);
    /// Writes the data for the given `attribute` of the point at `index` into `data`
    ///
    /// # Panics
    ///
    /// May panic if `attribute` is not part of the `PointLayout` of this buffer. It is implementation-defined
    /// whether data type conversions are supported (i.e. the buffer stores positions as `Vector3<f64>`, but
    /// `get_attribute` is called with `Vector3<f32>` as the attribute data type), but generally assume that
    /// conversions are *not* possible!
    /// May panic if `data.len()` does not equal the size of a single value of the data type of `attribute`
    fn get_attribute(&self, attribute: &PointAttributeDefinition, index: usize, data: &mut [u8]) {
        let attribute_member = self
            .point_layout()
            .get_attribute(attribute)
            .expect("Attribute not found in PointLayout of this buffer");
        // Is safe because we get the `attribute_member` from this buffer's point layout
        unsafe {
            self.get_attribute_unchecked(attribute_member, index, data);
        }
    }
    /// Writes the data for the given `attribute` of the given range of points into `data`. The attribute data will
    /// be tightly packed in `data`, irregardless of the memory layout of this buffer. The attribute values might not
    /// be correctly aligned, if the alignment requirement of `attribute.datatype()` is larger than the size of the
    /// attribute. All of the built-in attributes in pasture have alignments that are less than or equal to their size,
    /// so for built-in attributes you can assume that the attributes in `data` are correctly aligned.
    ///
    /// # Panics
    ///
    /// May panic if `attribute` is not part of the `PointLayout` of this buffer.
    /// May panic if `point_range` is out of bounds.
    /// May panic if `data.len()` does not equal `attribute.size() * point_range.len()`
    fn get_attribute_range(
        &self,
        attribute: &PointAttributeDefinition,
        point_range: Range<usize>,
        data: &mut [u8],
    ) {
        let attribute_member = self
            .point_layout()
            .get_attribute(attribute)
            .expect("Attribute not found in PointLayout of this buffer");
        let attribute_size = attribute_member.size() as usize;
        let first_point = point_range.start;
        for point_index in point_range {
            let zero_based_index = point_index - first_point;
            let data_slice = &mut data
                [(zero_based_index * attribute_size)..((zero_based_index + 1) * attribute_size)];
            // Safe because we checked the attribute and size of the buffer slice
            unsafe {
                self.get_attribute_unchecked(attribute_member, point_index, data_slice);
            }
        }
    }

    /// Like `get_attribute`, but performs no check whether the attribute actually is part of this buffers `PointLayout`
    /// or not. Because of this, this function accepts a `PointAttributeMember` instead of a `PointAttributeDefinition`,
    /// and this `PointAttributeMember` must come from the `PointLayout` of this buffer! The benefit over `get_attribute`
    /// is that this function skips the include checks and thus will be faster if you repeatedly want to get data for a
    /// single attribute
    ///
    /// # Safety
    ///
    /// Requires `attribute_member` to be a part of this buffer's `PointLayout`
    unsafe fn get_attribute_unchecked(
        &self,
        attribute_member: &PointAttributeMember,
        index: usize,
        data: &mut [u8],
    );

    /// Try to get a reference to `self` as an `InterleavedBuffer`. Returns `None` if `self` does not
    /// implement `InterleavedBuffer`
    fn as_interleaved(&self) -> Option<&dyn InterleavedBuffer<'a>> {
        None
    }

    /// Try to get a reference to `self` as a `ColumnarBuffer`. Returns `None` if `self` does not
    /// implement `ColumnarBuffer`
    fn as_columnar(&self) -> Option<&dyn ColumnarBuffer<'a>> {
        None
    }
}

/// Trait for a point buffer that mutably borrows its memory. Compared to [`BorrowedBuffer`], buffers that implement
/// this trait support the following additional capabilities:
/// - Manipulating point and attribute data in-place through `set_point` and `set_attribute`
/// - Shuffling data through `swap`
/// - Mutable views to points and attributes through `view_mut` and `view_attribute_mut`
pub trait BorrowedMutBuffer<'a>: BorrowedBuffer<'a> {
    /// Sets the data for the point at the given `index`
    ///
    /// # Safety
    ///
    /// Requires that `point_data` contains memory for a single point with the same `PointLayout` as `self.point_layout()`.
    /// This property is not enforced at runtime, so this function is very unsafe!
    ///
    /// # Panics
    ///
    /// May panic if `index` is out of bounds.<br>
    /// May panic if `point_data.len()` does not equal `self.point_layout().size_of_point_record()`
    unsafe fn set_point(&mut self, index: usize, point_data: &[u8]);
    /// Sets the data for the given range of points. This function will generally be more efficient than calling [`set_point`]
    /// multiple times, as [`set_point`] performs size and index checks on every call, whereas this function can perform them
    /// only once. Assumes that the `point_data` is tightly packed, i.e. stored in an interleaved format with point alignment
    /// of `1`.
    ///
    /// # Safety
    ///
    /// `point_data` must contain correctly initialized data for `point_range.len()` points in interleaved layout. The data
    /// must be tightly packed (i.e. no padding bytes between adjacent points).
    ///
    /// # Panics
    ///
    /// May panic if `point_range` is out of bounds.<br>
    /// May panic if `point_data.len()` does not equal `point_range.len() * self.point_layout().size_of_point_record()`
    unsafe fn set_point_range(&mut self, point_range: Range<usize>, point_data: &[u8]);
    /// Sets the data for the given `attribute` of the point at `index`
    ///
    /// # Safety
    ///
    /// Requires that `attribute_data` contains memory for a single value of the data type of `attribute`. This property
    /// is not enforced at runtime, so this function is very unsafe!
    ///
    /// # Panics
    ///
    /// May panic if `attribute` is not part of the `PointLayout` of this buffer.<br>
    /// May panic if `index` is out of bounds.<br>
    /// May panic if `attribute_data.len()` does not match the size of the data type of `attribute`
    unsafe fn set_attribute(
        &mut self,
        attribute: &PointAttributeDefinition,
        index: usize,
        attribute_data: &[u8],
    );
    /// Sets the data for the given `attribute` for all points in `point_range`. This function will generally be more efficient
    /// than calling [`set_attribute`] multiple times, as [`set_attribute`] has to perform type and bounds checks on each call.
    ///
    /// # Safety
    ///
    /// Requires that `attribute_data` contains data for `point_range.len()` attribute values. The data must be tightly packed,
    /// i.e. there must be no padding bytes between adjacent values.
    ///
    /// # Panics
    ///
    /// May panic if `attribute` is not part of the `PointLayout` of this buffer.<br>
    /// May panic if `point_range` is out of bounds.<br>
    /// May panic if `attribute_data.len()` does not equal `point_range.len() * attribute.size()`
    unsafe fn set_attribute_range(
        &mut self,
        attribute: &PointAttributeDefinition,
        point_range: Range<usize>,
        attribute_data: &[u8],
    );
    /// Swaps the two points at `from_index` and `to_index`. Implementations should allow the case where `from_index == to_index`
    ///
    /// # Panics
    ///
    /// May panic if any of `from_index` or `to_index` is out of bounds
    fn swap(&mut self, from_index: usize, to_index: usize);

    /// Apply a transformation function to the given `attribute` of all points within this buffer. This function is
    /// helpful if you want to modify a single attribute of a buffer in-place and works for buffers of all memory
    /// layouts. For columnar buffers, prefer using `get_attribute_range_mut` to modify attribute data in-place.
    ///
    /// This function does not support attribute type conversion, so the type `T` must match the `PointAttributeDataType`
    /// of `attribute`!
    ///
    /// The conversion function takes the current value of the attribute as a strongly typed `T` and returns the new value
    /// for the attribute. It also takes the index of the point within the buffer, so that `func` can access additional
    /// data.
    ///
    /// # Panics
    ///
    /// If `attribute` is not part of the `PointLayout` of this buffer.<br>
    /// If `T::data_type()` does not equal `attribute.datatype()`
    fn transform_attribute<'b, T: PrimitiveType, F: Fn(usize, T) -> T>(
        &'b mut self,
        attribute: &PointAttributeDefinition,
        func: F,
    ) where
        Self: Sized,
        'a: 'b,
    {
        let num_points = self.len();
        let mut attribute_view = self.view_attribute_mut(attribute);
        for point_index in 0..num_points {
            let attribute_value = attribute_view.at(point_index);
            attribute_view.set_at(point_index, func(point_index, attribute_value));
        }
    }

    /// Try to get a mutable reference to `self` as an `InterleavedBufferMut`. Returns `None` if `self` does not
    /// implement `InterleavedBufferMut`
    fn as_interleaved_mut(&mut self) -> Option<&mut dyn InterleavedBufferMut<'a>> {
        None
    }

    /// Try to get a mutable reference to `self` as a `ColumnarBufferMut`. Returns `None` if `self` does not
    /// implement `ColumnarBufferMut`
    fn as_columnar_mut(&mut self) -> Option<&mut dyn ColumnarBufferMut<'a>> {
        None
    }
}

/// Trait for point buffers that own their memory. Compared to [`BorrowedMutBuffer`], buffers that implement
/// this trait support the following additional capabilities:
/// - Pushing point data into the buffer using `push_points`
/// - Appending other buffers to the end of this buffer using `append`, `append_interleaved`, and `append_columnar`
/// - Resizing and clearing the contents of the buffer using `resize` and `clear`
pub trait OwningBuffer<'a>: BorrowedMutBuffer<'a> {
    /// Push the raw memory for a range of points into this buffer. Works similar to `Vec::push`
    ///
    /// # Safety
    ///
    /// `point_bytes` must contain the raw memory for a whole number of points in the `PointLayout` of this buffer.
    /// This property is not checked at runtime, so this function is very unsafe!
    ///
    /// # Panics
    ///
    /// May panic if `point_bytes.len()` is not a multiple of `self.point_layout().size_of_point_record()`
    unsafe fn push_points(&mut self, point_bytes: &[u8]);
    /// Appends data from the given buffer to the end of this buffer. Makes no assumptions about the memory
    /// layout of `other`. If you know the memory layout of `other`, consider using `append_interleaved` or
    /// `append_columnar`instead, as these will give better performance.
    ///
    /// # Panics
    ///
    /// If `self.point_layout()` does not equal `other.point_layout()`
    fn append<'b, B: BorrowedBuffer<'b>>(&mut self, other: &'_ B) {
        assert_eq!(self.point_layout(), other.point_layout());
        let old_self_len = self.len();
        self.resize(old_self_len + other.len());
        let mut point_buffer = vec![0; self.point_layout().size_of_point_entry() as usize];
        for point_index in 0..other.len() {
            other.get_point(point_index, &mut point_buffer);
            // Is safe because we assert that the point layouts of self and other match
            unsafe {
                self.set_point(old_self_len + point_index, &point_buffer);
            }
        }
    }
    /// Appends data from the given interleaved buffer to the end of this buffer
    ///
    /// # Note
    ///
    /// Why is there no single `append` function? As far as I understand the currently Rust rules, we can't
    /// state that two traits are mutually exclusive. So in principle there could be some point buffer type
    /// that implements both `InterleavedBuffer` and `ColumnarBuffer`. So we can't auto-detect from the type
    /// `B` whether we should use an implementation that assumes interleaved memory layout, or one that assumes
    /// columnar memory layout. We could always be conservative and assume neither layout and use the `get_point`
    /// and `set_point` API, but this is pessimistic and has suboptimal performance. So instead, we provide
    /// two independent functions that allow more optimal implementations if the memory layouts of `Self` and
    /// `B` match.
    ///
    /// # Panics
    ///
    /// If `self.point_layout()` does not equal `other.point_layout()`
    fn append_interleaved<'b, B: InterleavedBuffer<'b>>(&mut self, other: &'_ B);
    /// Appends data from the given columnar buffer to the end of this buffer
    ///
    /// # Panics
    ///
    /// If `self.point_layout()` does not equal `other.point_layout()`
    fn append_columnar<'b, B: ColumnarBuffer<'b>>(&mut self, other: &'_ B);
    /// Resize this buffer to contain exactly `count` points. If `count` is less than `self.len()`, point data
    /// is removed, if `count` is greater than `self.len()` new points are default-constructed (i.e. zero-initialized).
    fn resize(&mut self, count: usize);
    /// Clears the contents of this buffer, removing all point data and setting the length to `0`
    fn clear(&mut self);
}

/// Extension trait for `BorrowedBuffer` that allows obtaining strongly-typed views over points and
/// attributes.
///
/// # Notes
///
/// The `view...` methods on this type are implemented in an extension trait and not the base trait
/// `BorrowedBuffer` so that we retain the option to create trait objects for types implementing
/// `BorrowedBuffer`, while also allowing both static types `T: BorrowedBuffer` and dynamic trait object
/// types (`dyn BorrowedBuffer`) to be used for views. I.e. this makes the following code possible:
///
/// ```ignore
/// let layout = ...;
/// let buffer = VectorBuffer::new_from_layout(layout);
/// let view_from_sized_type = buffer.view::<Vector3<f64>>(&POSITION_3D);
///
/// // In previous pasture version, this code was not possible because views required sized types:
/// let buffer_trait_object: &dyn InterleavedBuffer = buffer.as_interleaved().unwrap();
/// let view_from_trait_object = buffer_trait_object.view::<Vector3<f64>>(&POSITION_3D);
/// ```
pub trait BorrowedBufferExt<'a>: BorrowedBuffer<'a> {
    /// Get a strongly typed view of the point data of this buffer
    ///
    /// # Panics
    ///
    /// Panics if `T::layout()` does not match the `PointLayout` of this buffer
    fn view<'b, T: PointType>(&'b self) -> PointView<'a, 'b, Self, T>
    where
        'a: 'b,
    {
        PointView::new(self)
    }

    /// Gets a strongly typed view of the `attribute` of all points in this buffer
    ///
    /// # Panics
    ///
    /// If `attribute` is not part of the `PointLayout` of this buffer.
    /// If `T::data_type()` does not match the data type of the attribute within the buffer
    fn view_attribute<'b, T: PrimitiveType>(
        &'b self,
        attribute: &PointAttributeDefinition,
    ) -> AttributeView<'a, 'b, Self, T>
    where
        'a: 'b,
    {
        AttributeView::new(self, attribute)
    }

    /// Like `view_attribute`, but allows `T::data_type()` to be different from the data type of  
    /// the `attribute` within this buffer.
    ///
    /// # Panics
    ///
    /// If `T::data_type()` does not match the data type of `attribute`
    fn view_attribute_with_conversion<'b, T: PrimitiveType>(
        &'b self,
        attribute: &PointAttributeDefinition,
    ) -> Result<AttributeViewConverting<'a, 'b, Self, T>>
    where
        'a: 'b,
    {
        AttributeViewConverting::new(self, attribute)
    }
}

impl<'a, T: BorrowedBuffer<'a>> BorrowedBufferExt<'a> for T {}
impl<'a> BorrowedBufferExt<'a> for dyn BorrowedBuffer<'a> + 'a {}
impl<'a> BorrowedBufferExt<'a> for dyn BorrowedMutBuffer<'a> + 'a {}
// TODO Make OwningBuffer object safe, e.g. by moving the append functions to another extension trait
// Open question how to deal with append_interleaved / append_columnar
// impl<'a> BorrowedBufferExt<'a> for dyn OwningBuffer<'a> + 'a {}
impl<'a> BorrowedBufferExt<'a> for dyn InterleavedBuffer<'a> + 'a {}
impl<'a> BorrowedBufferExt<'a> for dyn InterleavedBufferMut<'a> + 'a {}
impl<'a> BorrowedBufferExt<'a> for dyn ColumnarBuffer<'a> + 'a {}
impl<'a> BorrowedBufferExt<'a> for dyn ColumnarBufferMut<'a> + 'a {}

/// Extension trait for `BorrowedMutBuffer` that allows obtaining strongly-typed views over points and
/// attributes.
pub trait BorrowedMutBufferExt<'a>: BorrowedMutBuffer<'a> {
    /// Get a strongly typed view of the point data of this buffer. This view allows mutating the point data!
    ///
    /// # Panics
    ///
    /// If `T::point_layout()` does not match `self.point_layout()`
    fn view_mut<'b, T: PointType>(&'b mut self) -> PointViewMut<'a, 'b, Self, T>
    where
        Self: Sized,
        'a: 'b,
    {
        PointViewMut::new(self)
    }

    /// Get a strongly typed view of the `attribute` of all points in this buffer. This view allows mutating
    /// the attribute data!
    ///
    /// # Panics
    ///
    /// If `attribute` is not part of the `PointLayout` of this buffer.<br>
    /// If `T::data_type()` does not match `attribute.datatype()`
    fn view_attribute_mut<'b, T: PrimitiveType>(
        &'b mut self,
        attribute: &PointAttributeDefinition,
    ) -> AttributeViewMut<'a, 'b, Self, T>
    where
        Self: Sized,
        'a: 'b,
    {
        AttributeViewMut::new(self, attribute)
    }
}

impl<'a, T: BorrowedMutBuffer<'a>> BorrowedMutBufferExt<'a> for T {}
impl<'a> BorrowedMutBufferExt<'a> for dyn BorrowedMutBuffer<'a> + 'a {}
// TODO impl for owning buffer
impl<'a> BorrowedMutBufferExt<'a> for dyn InterleavedBufferMut<'a> + 'a {}
impl<'a> BorrowedMutBufferExt<'a> for dyn ColumnarBufferMut<'a> + 'a {}

/// Trait for all buffers that can be default-constructed from a given `PointLayout`. This trait is helpful for generic
/// code that needs to construct an generic buffer type
pub trait MakeBufferFromLayout<'a>: BorrowedBuffer<'a> + Sized {
    /// Creates a new empty buffer from the given `PointLayout`
    fn new_from_layout(point_layout: PointLayout) -> Self;
}

/// Trait for point buffers that store their point data in interleaved memory layout. This allows accessing
/// point data by reference
pub trait InterleavedBuffer<'a>: BorrowedBuffer<'a> {
    /// Get an immutable slice of the point memory of the point at `index`
    ///
    /// # Lifetimes
    ///
    /// Has a more relaxed lifetime bound than the underlying buffer, since we should be able to borrow point
    /// data for a lifetime `'b` that is potentially shorter than the lifetime `'a` of the `BorrowedBuffer`
    ///
    /// # Panics
    ///
    /// Should panic if `index` is out of bounds
    fn get_point_ref<'b>(&'b self, index: usize) -> &'b [u8]
    where
        'a: 'b;
    /// Get an immutable slice of the memory for the given `range` of points. This is the range-version of [`get_point_ref`],
    /// see its documentation for more details
    ///
    /// # Panics
    ///
    /// If `range` is out of bounds
    fn get_point_range_ref<'b>(&'b self, range: Range<usize>) -> &'b [u8]
    where
        'a: 'b;

    /// Get a raw view over the given `attribute` from this point buffer. Unlike the typed view that `view_attribute`
    /// returns, this view dereferences to byte slices, but it is potentially more efficient to use than calling
    /// `get_attribute` repeatedly
    fn view_raw_attribute<'b>(&'b self, attribute: &PointAttributeMember) -> RawAttributeView<'b>
    where
        'a: 'b,
    {
        RawAttributeView::from_interleaved_buffer(self, attribute)
    }
}

/// Trait for buffers that store point data in interleaved memory layout and also borrow their memory mutably. Compared
/// to [`InterleavedBuffer`], this allows accessing point data by mutable reference!
pub trait InterleavedBufferMut<'a>: InterleavedBuffer<'a> + BorrowedMutBuffer<'a> {
    /// Get a mutable slice of the point memory of the point at `index`. This is the mutable version of [`InterleavedBuffer::get_point_ref`]
    ///
    /// # Panics
    ///
    /// Should panic if `index` is out of bounds
    fn get_point_mut<'b>(&'b mut self, index: usize) -> &'b mut [u8]
    where
        'a: 'b;
    /// Get a mutable slice of the memory for the given `range` of points. This is the mutable version of [`InterleavedBuffer::get_point_range_ref`]
    ///
    /// # Panics
    ///
    /// Should panic if `index` is out of bounds
    fn get_point_range_mut<'b>(&'b mut self, range: Range<usize>) -> &'b mut [u8]
    where
        'a: 'b;

    /// Like `view_raw_attribute`, but returns mutable byte slices of the attribute data
    fn view_raw_attribute_mut<'b>(
        &'b mut self,
        attribute: &PointAttributeMember,
    ) -> RawAttributeViewMut<'b>
    where
        'a: 'b,
    {
        RawAttributeViewMut::from_interleaved_buffer(self, attribute)
    }
}

/// Trait for point buffers that store their point data in columnar memory layout. This allows accessing point attributes
/// by reference
pub trait ColumnarBuffer<'a>: BorrowedBuffer<'a> {
    /// Get an immutable slice to the memory of the given `attribute` for the point at `index`. See [`InterleavedBuffer::get_point_ref`] for an explanation of the lifetime bounds.
    ///
    /// # Panics
    ///
    /// Should panic if `attribute` is not part of the `PointLayout` of this buffer.<br>
    /// Should panic if `index` is out of bounds.
    fn get_attribute_ref<'b>(
        &'b self,
        attribute: &PointAttributeDefinition,
        index: usize,
    ) -> &'b [u8]
    where
        'a: 'b;
    /// Get an immutable slice to the memory for the `attribute` of the given `range` of points
    ///
    /// # Panics
    ///
    /// Should panic if `attribute` is not part of the `PointLayout` of this buffer.<br>
    /// Should panic if `range` is out of bounds.
    fn get_attribute_range_ref<'b>(
        &'b self,
        attribute: &PointAttributeDefinition,
        range: Range<usize>,
    ) -> &'b [u8]
    where
        'a: 'b;

    /// Get a raw view over the given `attribute` from this point buffer. Unlike the typed view that `view_attribute`
    /// returns, this view dereferences to byte slices, but it is potentially more efficient to use than calling
    /// `get_attribute` repeatedly
    fn view_raw_attribute<'b>(&'b self, attribute: &PointAttributeMember) -> RawAttributeView<'b>
    where
        'a: 'b,
    {
        RawAttributeView::from_columnar_buffer(self, attribute.attribute_definition())
    }
}

/// Trait for buffers that store point data in columnar memory layout and also borrow their memory mutably. Compared
/// to [`ColumnarBuffer`], this allows accessing point attributes by mutable reference!
pub trait ColumnarBufferMut<'a>: ColumnarBuffer<'a> + BorrowedMutBuffer<'a> {
    /// Get a mutable slice to the memory of the given `attribute` for the point at `index`. This is the mutable
    /// version of [`ColumnarBuffer::get_attribute_ref`]
    ///
    /// # Panics
    ///
    /// Should panic if `attribute` is not part of the `PointLayout` of this buffer.<br>
    /// Should panic if `index` is out of bounds.
    fn get_attribute_mut<'b>(
        &'b mut self,
        attribute: &PointAttributeDefinition,
        index: usize,
    ) -> &'b mut [u8]
    where
        'a: 'b;
    /// Get a mutable slice to the memory for the `attribute` of the given `range` of points. This is the mutable
    /// version of [`ColumnarBuffer::get_attribute_range_ref`]
    ///
    /// # Panics
    ///
    /// Should panic if `attribute` is not part of the `PointLayout` of this buffer.<br>
    /// Should panic if `range` is out of bounds.
    fn get_attribute_range_mut<'b>(
        &'b mut self,
        attribute: &PointAttributeDefinition,
        range: Range<usize>,
    ) -> &'b mut [u8]
    where
        'a: 'b;

    /// Like `view_raw_attribute`, but returns mutable byte slices of the attribute data
    fn view_raw_attribute_mut<'b>(
        &'b mut self,
        attribute: &PointAttributeMember,
    ) -> RawAttributeViewMut<'b>
    where
        'a: 'b,
    {
        RawAttributeViewMut::from_columnar_buffer(self, attribute.attribute_definition())
    }
}

/// A point buffer that uses a `Vec<u8>` as its underlying storage. It stores point data in interleaved memory
/// layout and generally behaves like an untyped vector.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VectorBuffer {
    storage: Vec<u8>,
    point_layout: PointLayout,
    length: usize,
}

impl VectorBuffer {
    /// Creates a new `VectorBuffer` with the given `capacity` and `point_layout`. This preallocates enough memory
    /// to store at least `capacity` points
    pub fn with_capacity(capacity: usize, point_layout: PointLayout) -> Self {
        let required_bytes = capacity * point_layout.size_of_point_entry() as usize;
        Self {
            point_layout,
            storage: Vec::with_capacity(required_bytes),
            length: 0,
        }
    }

    fn get_byte_range_of_point(&self, point_index: usize) -> Range<usize> {
        let size_of_point = self.point_layout.size_of_point_entry() as usize;
        (point_index * size_of_point)..((point_index + 1) * size_of_point)
    }

    fn get_byte_range_of_points(&self, points_range: Range<usize>) -> Range<usize> {
        let size_of_point = self.point_layout.size_of_point_entry() as usize;
        (points_range.start * size_of_point)..(points_range.end * size_of_point)
    }

    fn get_byte_range_of_attribute(
        &self,
        point_index: usize,
        attribute: &PointAttributeMember,
    ) -> Range<usize> {
        let start_byte = (point_index * self.point_layout.size_of_point_entry() as usize)
            + attribute.offset() as usize;
        let end_byte = start_byte + attribute.size() as usize;
        start_byte..end_byte
    }
}

impl<'a> MakeBufferFromLayout<'a> for VectorBuffer {
    fn new_from_layout(point_layout: PointLayout) -> Self {
        Self {
            point_layout,
            storage: Default::default(),
            length: 0,
        }
    }
}

impl<'a> BorrowedBuffer<'a> for VectorBuffer
where
    VectorBuffer: 'a,
{
    fn len(&self) -> usize {
        self.length
    }

    fn point_layout(&self) -> &PointLayout {
        &self.point_layout
    }

    fn get_point(&self, index: usize, data: &mut [u8]) {
        let point_ref = self.get_point_ref(index);
        data.copy_from_slice(point_ref);
    }

    fn get_point_range(&self, range: Range<usize>, data: &mut [u8]) {
        let points_ref = self.get_point_range_ref(range);
        data.copy_from_slice(points_ref);
    }

    unsafe fn get_attribute_unchecked(
        &self,
        attribute_member: &PointAttributeMember,
        index: usize,
        data: &mut [u8],
    ) {
        let byte_range = self.get_byte_range_of_attribute(index, attribute_member);
        data.copy_from_slice(&self.storage[byte_range]);
    }

    fn as_interleaved(&self) -> Option<&dyn InterleavedBuffer<'a>> {
        Some(self)
    }
}

impl<'a> BorrowedMutBuffer<'a> for VectorBuffer
where
    VectorBuffer: 'a,
{
    unsafe fn set_point(&mut self, index: usize, point_data: &[u8]) {
        let point_bytes = self.get_point_mut(index);
        point_bytes.copy_from_slice(point_data);
    }

    unsafe fn set_attribute(
        &mut self,
        attribute: &PointAttributeDefinition,
        index: usize,
        attribute_data: &[u8],
    ) {
        let attribute_member = self
            .point_layout
            .get_attribute(attribute)
            .expect("Attribute not found in PointLayout of this buffer");
        let attribute_byte_range = self.get_byte_range_of_attribute(index, attribute_member);
        let attribute_bytes = &mut self.storage[attribute_byte_range];
        attribute_bytes.copy_from_slice(attribute_data);
    }

    fn swap(&mut self, from_index: usize, to_index: usize) {
        assert!(from_index < self.len());
        assert!(to_index < self.len());
        if from_index == to_index {
            return;
        }
        let size_of_point = self.point_layout.size_of_point_entry() as usize;
        // Is safe as long as 'from_index' and 'to_index' are not out of bounds, which is asserted
        unsafe {
            let from_ptr = self.storage.as_mut_ptr().add(from_index * size_of_point);
            let to_ptr = self.storage.as_mut_ptr().add(to_index * size_of_point);
            std::ptr::swap_nonoverlapping(from_ptr, to_ptr, size_of_point);
        }
    }

    unsafe fn set_point_range(&mut self, point_range: Range<usize>, point_data: &[u8]) {
        let point_bytes = self.get_point_range_mut(point_range);
        point_bytes.copy_from_slice(point_data);
    }

    unsafe fn set_attribute_range(
        &mut self,
        attribute: &PointAttributeDefinition,
        point_range: Range<usize>,
        attribute_data: &[u8],
    ) {
        let attribute_member = self
            .point_layout
            .get_attribute(attribute)
            .expect("Attribute not found in PointLayout of this buffer");
        let attribute_size = attribute_member.size() as usize;
        let first_point = point_range.start;
        for point_index in point_range {
            let zero_based_index = point_index - first_point;
            let src_slice = &attribute_data
                [(zero_based_index * attribute_size)..((zero_based_index + 1) * attribute_size)];
            let attribute_byte_range =
                self.get_byte_range_of_attribute(point_index, attribute_member);
            let attribute_bytes = &mut self.storage[attribute_byte_range];
            attribute_bytes.copy_from_slice(src_slice);
        }
    }

    fn as_interleaved_mut(&mut self) -> Option<&mut dyn InterleavedBufferMut<'a>> {
        Some(self)
    }
}

impl<'a> OwningBuffer<'a> for VectorBuffer
where
    VectorBuffer: 'a,
{
    unsafe fn push_points(&mut self, point_bytes: &[u8]) {
        let size_of_point = self.point_layout.size_of_point_entry() as usize;
        if size_of_point == 0 {
            assert_eq!(0, point_bytes.len());
        } else {
            assert_eq!(point_bytes.len() % size_of_point, 0);
            self.storage.extend_from_slice(point_bytes);
            self.length += point_bytes.len() / size_of_point;
        }
    }

    fn resize(&mut self, count: usize) {
        let size_of_point = self.point_layout.size_of_point_entry() as usize;
        self.storage.resize(count * size_of_point, 0);
        self.length = count;
    }

    fn clear(&mut self) {
        self.storage.clear();
        self.length = 0;
    }

    fn append_interleaved<'b, B: InterleavedBuffer<'b>>(&mut self, other: &'_ B) {
        assert_eq!(self.point_layout(), other.point_layout());
        // Is safe because we checked that the two `PointLayout`s match
        unsafe {
            self.push_points(other.get_point_range_ref(0..other.len()));
        }
    }

    fn append_columnar<'b, B: ColumnarBuffer<'b>>(&mut self, other: &'_ B) {
        assert_eq!(self.point_layout(), other.point_layout());
        let previous_self_len = self.len();
        self.resize(previous_self_len + other.len());
        for point_index in 0..other.len() {
            let self_memory = self.get_point_mut(previous_self_len + point_index);
            other.get_point(point_index, self_memory);
        }
    }
}

impl<'a> InterleavedBuffer<'a> for VectorBuffer
where
    VectorBuffer: 'a,
{
    fn get_point_ref<'b>(&'b self, index: usize) -> &'b [u8]
    where
        'a: 'b,
    {
        &self.storage[self.get_byte_range_of_point(index)]
    }

    fn get_point_range_ref<'b>(&'b self, range: Range<usize>) -> &'b [u8]
    where
        'a: 'b,
    {
        &self.storage[self.get_byte_range_of_points(range)]
    }
}

impl<'a> InterleavedBufferMut<'a> for VectorBuffer
where
    VectorBuffer: 'a,
{
    fn get_point_mut<'b>(&'b mut self, index: usize) -> &'b mut [u8]
    where
        'a: 'b,
    {
        let byte_range = self.get_byte_range_of_point(index);
        &mut self.storage[byte_range]
    }

    fn get_point_range_mut<'b>(&'b mut self, range: Range<usize>) -> &'b mut [u8]
    where
        'a: 'b,
    {
        let byte_range = self.get_byte_range_of_points(range);
        &mut self.storage[byte_range]
    }
}

impl<'a> SliceBuffer<'a> for VectorBuffer
where
    Self: 'a,
{
    type SliceType = BufferSliceInterleaved<'a, Self>;

    fn slice(&'a self, range: Range<usize>) -> Self::SliceType {
        BufferSliceInterleaved::new(self, range)
    }
}

impl<'a> SliceBufferMut<'a> for VectorBuffer
where
    Self: 'a,
{
    type SliceTypeMut = BufferSliceInterleavedMut<'a, Self>;

    fn slice_mut(&'a mut self, range: Range<usize>) -> Self::SliceTypeMut {
        BufferSliceInterleavedMut::new(self, range)
    }
}

impl<T: PointType> FromIterator<T> for VectorBuffer {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let point_layout = T::layout();
        let iter = iter.into_iter();
        let (_, maybe_known_length) = iter.size_hint();
        if let Some(known_length) = maybe_known_length {
            let num_bytes = known_length * point_layout.size_of_point_entry() as usize;
            let storage = vec![0; num_bytes];
            let mut buffer = Self {
                point_layout,
                storage,
                length: known_length,
            };
            // Overwrite the preallocated memory of the buffer with the points in the iterator:
            iter.enumerate().for_each(|(index, point)| {
                let point_bytes = bytemuck::bytes_of(&point);
                // Safe because we created `buffer` from `T::layout()`, so we know the layouts match
                unsafe {
                    buffer.set_point(index, point_bytes);
                }
            });
            buffer
        } else {
            let mut buffer = Self {
                point_layout,
                storage: Default::default(),
                length: 0,
            };
            iter.for_each(|point| {
                let point_bytes = bytemuck::bytes_of(&point);
                // Safe because we know that `buffer` has the same layout as `T`
                unsafe {
                    buffer.push_points(point_bytes);
                }
            });
            buffer
        }
    }
}

/// Helper struct to push point data into a `HashMapBuffer` attribute by attribute. This allows constructing a point buffer
/// from multiple ranges of attribute data, since regular point buffers do not allow pushing just a single attribute into
/// the buffer, as buffers always have to store complete points (even with columnar memory layout)
pub struct HashMapBufferAttributePusher<'a> {
    attributes_storage: HashMap<PointAttributeDefinition, Vec<u8>>,
    num_new_points: Option<usize>,
    buffer: &'a mut HashMapBuffer,
}

impl<'a> HashMapBufferAttributePusher<'a> {
    pub(crate) fn new(buffer: &'a mut HashMapBuffer) -> Self {
        let attributes_storage = buffer
            .point_layout()
            .attributes()
            .map(|attribute| (attribute.attribute_definition().clone(), vec![]))
            .collect();
        Self {
            attributes_storage,
            num_new_points: None,
            buffer,
        }
    }

    /// Push a range of values for the given `attribute` into the underlying buffer. The first range of values that
    /// is pushed in this way determines the expected number of points that will be added to the buffer. Consecutive
    /// calls to `push_attribute_range` will assert that `data.len()` matches the expected count.
    ///
    /// # Panics
    ///
    /// If `attribute` is not part of the `PointLayout` of the underlying buffer.<br>
    /// If `T::data_type()` does not match `attribute.datatype()`.<br>
    /// If this is not the first call to `push_attribute_range`, and `data.len()` does not match the length of the
    /// data that was passed to the first invocation of `push_attribute_range`
    pub fn push_attribute_range<T: PrimitiveType>(
        &mut self,
        attribute: &PointAttributeDefinition,
        data: &[T],
    ) {
        assert_eq!(T::data_type(), attribute.datatype());
        let storage = self
            .attributes_storage
            .get_mut(attribute)
            .expect("Attribute not found in PointLayout of this buffer");
        if let Some(point_count) = self.num_new_points {
            assert_eq!(point_count, data.len());
        } else {
            self.num_new_points = Some(data.len());
        }
        storage.extend_from_slice(bytemuck::cast_slice(data));
    }

    /// Commit all pushed data into the underlying buffer. This function checks that there is the correct amount
    /// of data for all expected attributes in the `PointLayout` of the underlying buffer and will panic otherwise
    ///
    /// # Panics
    ///
    /// If there is missing data for at least one of the attributes in the `PointLayout` of the underlying buffer,
    /// i.e. if `push_attribute_range` was not called for at least one of these attributes.
    pub fn done(self) {
        let num_new_points = self.num_new_points.unwrap_or(0);
        if num_new_points == 0 {
            return;
        }

        // Check that all attributes are complete! We don't have to check the exact size of the vectors,
        // as this is checked in `push_attribute_range`, it is sufficient to verify that no vector is empty
        assert!(self
            .attributes_storage
            .values()
            .all(|vector| !vector.is_empty()));

        for (attribute, mut data) in self.attributes_storage {
            // Can safely unwrap, because self.attributes_storage was initialized from the `PointLayout` of the buffer!
            let buffer_storage = self.buffer.attributes_storage.get_mut(&attribute).unwrap();
            buffer_storage.append(&mut data);
        }

        self.buffer.length += num_new_points;
    }
}

/// A point buffer that stores point data in columnar memory layout, using a `HashMap<PointAttributeDefinition, Vec<u8>>` as
/// its underlying storage
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashMapBuffer {
    attributes_storage: HashMap<PointAttributeDefinition, Vec<u8>>,
    point_layout: PointLayout,
    length: usize,
}

impl HashMapBuffer {
    /// Creates a new `HashMapBuffer` with the given `capacity` and `point_layout`. It preallocates enough memory to store
    /// at least `capacity` points
    pub fn with_capacity(capacity: usize, point_layout: PointLayout) -> Self {
        let attributes_storage = point_layout
            .attributes()
            .map(|attribute| {
                let bytes_for_attribute = capacity * attribute.size() as usize;
                (
                    attribute.attribute_definition().clone(),
                    Vec::with_capacity(bytes_for_attribute),
                )
            })
            .collect();
        Self {
            attributes_storage,
            point_layout,
            length: 0,
        }
    }

    /// Create a new helper object through which ranges of attribute data can be pushed into this buffer
    pub fn begin_push_attributes(&mut self) -> HashMapBufferAttributePusher<'_> {
        HashMapBufferAttributePusher::new(self)
    }

    /// Like `Iterator::filter`, but filters into a point buffer of type `B`
    pub fn filter<
        B: for<'a> OwningBuffer<'a> + for<'a> MakeBufferFromLayout<'a>,
        F: Fn(usize) -> bool,
    >(
        &self,
        predicate: F,
    ) -> B {
        let num_matches = (0..self.len()).filter(|idx| predicate(*idx)).count();
        let mut filtered_points = B::new_from_layout(self.point_layout.clone());
        filtered_points.resize(num_matches);
        self.filter_into(&mut filtered_points, predicate, Some(num_matches));
        filtered_points
    }

    /// Like `filter`, but writes the filtered points into the given `buffer`
    ///
    /// # panics
    ///
    /// If `buffer.len()` is less than the number of matching points according to `predicate`
    /// If the `PointLayout` of `buffer` does not match the `PointLayout` of `self`
    pub fn filter_into<B: for<'a> BorrowedMutBuffer<'a>, F: Fn(usize) -> bool>(
        &self,
        buffer: &mut B,
        predicate: F,
        num_matches_hint: Option<usize>,
    ) {
        if buffer.point_layout() != self.point_layout() {
            panic!("PointLayouts must match");
        }
        let num_matches = num_matches_hint
            .unwrap_or_else(|| (0..self.len()).filter(|idx| predicate(*idx)).count());
        if buffer.len() < num_matches {
            panic!("buffer.len() must be at least as large as the number of predicate matches");
        }
        if let Some(columnar_buffer) = buffer.as_columnar_mut() {
            for attribute in self.point_layout.attributes() {
                let src_attribute_data =
                    self.get_attribute_range_ref(attribute.attribute_definition(), 0..self.len());
                let dst_attribute_data = columnar_buffer
                    .get_attribute_range_mut(attribute.attribute_definition(), 0..num_matches);
                let stride = attribute.size() as usize;
                for (dst_index, src_index) in
                    (0..self.len()).filter(|idx| predicate(*idx)).enumerate()
                {
                    dst_attribute_data[(dst_index * stride)..((dst_index + 1) * stride)]
                        .copy_from_slice(
                            &src_attribute_data[(src_index * stride)..((src_index + 1) * stride)],
                        );
                }
            }
        } else if let Some(interleaved_buffer) = buffer.as_interleaved_mut() {
            let dst_data = interleaved_buffer.get_point_range_mut(0..num_matches);
            for attribute in self.point_layout.attributes() {
                let src_attribute_data =
                    self.get_attribute_range_ref(attribute.attribute_definition(), 0..self.len());
                let src_stride = attribute.size() as usize;
                let dst_offset = attribute.offset() as usize;
                let dst_stride = self.point_layout.size_of_point_entry() as usize;
                for (dst_index, src_index) in
                    (0..self.len()).filter(|idx| predicate(*idx)).enumerate()
                {
                    let dst_attribute_start = dst_offset + (dst_index * dst_stride);
                    let dst_point_range = dst_attribute_start..(dst_attribute_start + src_stride);
                    dst_data[dst_point_range].copy_from_slice(
                        &src_attribute_data
                            [(src_index * src_stride)..((src_index + 1) * src_stride)],
                    );
                }
            }
        } else {
            unimplemented!()
        }
    }

    fn get_byte_range_for_attribute(
        point_index: usize,
        attribute: &PointAttributeDefinition,
    ) -> Range<usize> {
        let attribute_size = attribute.size() as usize;
        (point_index * attribute_size)..((point_index + 1) * attribute_size)
    }

    fn get_byte_range_for_attributes(
        points_range: Range<usize>,
        attribute: &PointAttributeDefinition,
    ) -> Range<usize> {
        let attribute_size = attribute.size() as usize;
        (points_range.start * attribute_size)..(points_range.end * attribute_size)
    }
}

impl<'a> MakeBufferFromLayout<'a> for HashMapBuffer {
    fn new_from_layout(point_layout: PointLayout) -> Self {
        let attributes_storage = point_layout
            .attributes()
            .map(|attribute| (attribute.attribute_definition().clone(), Vec::default()))
            .collect();
        Self {
            attributes_storage,
            point_layout,
            length: 0,
        }
    }
}

impl<'a> BorrowedBuffer<'a> for HashMapBuffer
where
    HashMapBuffer: 'a,
{
    fn len(&self) -> usize {
        self.length
    }

    fn point_layout(&self) -> &PointLayout {
        &self.point_layout
    }

    fn get_point(&self, index: usize, data: &mut [u8]) {
        for attribute in self.point_layout.attributes() {
            let attribute_storage = self
                .attributes_storage
                .get(attribute.attribute_definition())
                .expect("Attribute not found within storage of this PointBuffer");
            let src_slice = &attribute_storage
                [Self::get_byte_range_for_attribute(index, attribute.attribute_definition())];
            let dst_slice = &mut data[attribute.byte_range_within_point()];
            dst_slice.copy_from_slice(src_slice);
        }
    }

    fn get_point_range(&self, range: Range<usize>, data: &mut [u8]) {
        let size_of_point = self.point_layout.size_of_point_entry() as usize;
        for attribute in self.point_layout.attributes() {
            let attribute_storage = self
                .attributes_storage
                .get(attribute.attribute_definition())
                .expect("Attribute not found within storage of this PointBuffer");
            for point_index in range.clone() {
                let src_slice = &attribute_storage[Self::get_byte_range_for_attribute(
                    point_index,
                    attribute.attribute_definition(),
                )];
                let dst_point_slice = &mut data[((point_index - range.start) * size_of_point)..];
                let dst_slice = &mut dst_point_slice[attribute.byte_range_within_point()];
                dst_slice.copy_from_slice(src_slice);
            }
        }
    }

    fn get_attribute(&self, attribute: &PointAttributeDefinition, index: usize, data: &mut [u8]) {
        let memory = self
            .attributes_storage
            .get(attribute)
            .expect("Attribute not found in PointLayout of this buffer");
        let attribute_byte_range = Self::get_byte_range_for_attribute(index, attribute);
        data.copy_from_slice(&memory[attribute_byte_range]);
    }

    unsafe fn get_attribute_unchecked(
        &self,
        attribute_member: &PointAttributeMember,
        index: usize,
        data: &mut [u8],
    ) {
        let memory = self
            .attributes_storage
            .get(attribute_member.attribute_definition())
            .expect("Attribute not found in PointLayout of this buffer");
        let attribute_byte_range =
            Self::get_byte_range_for_attribute(index, attribute_member.attribute_definition());
        data.copy_from_slice(&memory[attribute_byte_range]);
    }

    fn as_columnar(&self) -> Option<&dyn ColumnarBuffer<'a>> {
        Some(self)
    }
}

impl<'a> BorrowedMutBuffer<'a> for HashMapBuffer
where
    HashMapBuffer: 'a,
{
    unsafe fn set_point(&mut self, index: usize, point_data: &[u8]) {
        for attribute in self.point_layout.attributes() {
            let attribute_definition = attribute.attribute_definition();
            let attribute_byte_range =
                Self::get_byte_range_for_attribute(index, attribute_definition);
            let attribute_storage = self
                .attributes_storage
                .get_mut(attribute_definition)
                .expect("Attribute not found within storage of this PointBuffer");
            let dst_slice = &mut attribute_storage[attribute_byte_range];
            let src_slice = &point_data[attribute.byte_range_within_point()];
            dst_slice.copy_from_slice(src_slice);
        }
    }

    unsafe fn set_attribute(
        &mut self,
        attribute: &PointAttributeDefinition,
        index: usize,
        attribute_data: &[u8],
    ) {
        let attribute_byte_range = Self::get_byte_range_for_attribute(index, attribute);
        let attribute_storage = self
            .attributes_storage
            .get_mut(attribute)
            .expect("Attribute not found in PointLayout of this buffer");
        let attribute_bytes = &mut attribute_storage[attribute_byte_range];
        attribute_bytes.copy_from_slice(attribute_data);
    }

    fn swap(&mut self, from_index: usize, to_index: usize) {
        assert!(from_index < self.len());
        assert!(to_index < self.len());
        if from_index == to_index {
            return;
        }
        for (attribute, storage) in self.attributes_storage.iter_mut() {
            let src_byte_range = Self::get_byte_range_for_attribute(from_index, attribute);
            let dst_byte_range = Self::get_byte_range_for_attribute(to_index, attribute);
            // Is safe as long as 'from_index' and 'to_index' are not out of bounds, which is asserted
            unsafe {
                let src_ptr = storage.as_mut_ptr().add(src_byte_range.start);
                let dst_ptr = storage.as_mut_ptr().add(dst_byte_range.start);
                std::ptr::swap_nonoverlapping(src_ptr, dst_ptr, attribute.size() as usize);
            }
        }
    }

    unsafe fn set_point_range(&mut self, point_range: Range<usize>, point_data: &[u8]) {
        let size_of_point = self.point_layout.size_of_point_entry() as usize;
        let first_point = point_range.start;
        for attribute in self.point_layout.attributes() {
            let attribute_definition = attribute.attribute_definition();
            let attribute_storage = self
                .attributes_storage
                .get_mut(attribute_definition)
                .expect("Attribute not found within storage of this PointBuffer");
            for point_index in point_range.clone() {
                let zero_based_index = point_index - first_point;
                let attribute_byte_range =
                    Self::get_byte_range_for_attribute(point_index, attribute_definition);

                let dst_slice = &mut attribute_storage[attribute_byte_range];
                let src_point_slice = &point_data
                    [(zero_based_index * size_of_point)..((zero_based_index + 1) * size_of_point)];
                let src_slice = &src_point_slice[attribute.byte_range_within_point()];
                dst_slice.copy_from_slice(src_slice);
            }
        }
    }

    unsafe fn set_attribute_range(
        &mut self,
        attribute: &PointAttributeDefinition,
        point_range: Range<usize>,
        attribute_data: &[u8],
    ) {
        let attribute_range = self.get_attribute_range_mut(attribute, point_range);
        attribute_range.copy_from_slice(attribute_data);
    }

    fn as_columnar_mut(&mut self) -> Option<&mut dyn ColumnarBufferMut<'a>> {
        Some(self)
    }
}

impl<'a> OwningBuffer<'a> for HashMapBuffer
where
    HashMapBuffer: 'a,
{
    unsafe fn push_points(&mut self, point_bytes: &[u8]) {
        let point_size = self.point_layout.size_of_point_entry() as usize;
        assert_eq!(point_bytes.len() % point_size, 0);
        let num_points_added = point_bytes.len() / point_size;
        for attribute in self.point_layout.attributes() {
            let storage = self
                .attributes_storage
                .get_mut(attribute.attribute_definition())
                .expect("Attribute not found in storage of this buffer");
            for index in 0..num_points_added {
                let point_bytes = &point_bytes[(index * point_size)..((index + 1) * point_size)];
                let attribute_bytes = &point_bytes[attribute.byte_range_within_point()];
                storage.extend_from_slice(attribute_bytes);
            }
        }
        self.length += num_points_added;
    }

    fn resize(&mut self, count: usize) {
        for (attribute, storage) in self.attributes_storage.iter_mut() {
            let new_num_bytes = count * attribute.size() as usize;
            storage.resize(new_num_bytes, 0);
        }
        self.length = count;
    }

    fn clear(&mut self) {
        for storage in self.attributes_storage.values_mut() {
            storage.clear();
        }
        self.length = 0;
    }

    fn append_interleaved<'b, B: InterleavedBuffer<'b>>(&mut self, other: &'_ B) {
        assert_eq!(self.point_layout(), other.point_layout());
        // Safe because we checked that the point layouts match
        unsafe {
            self.push_points(other.get_point_range_ref(0..other.len()));
        }
    }

    fn append_columnar<'b, B: ColumnarBuffer<'b>>(&mut self, other: &'_ B) {
        assert_eq!(self.point_layout(), other.point_layout());
        for attribute in self.point_layout.attributes() {
            let storage = self
                .attributes_storage
                .get_mut(attribute.attribute_definition())
                .expect("Attribute not found in storage of this buffer");
            storage.extend_from_slice(
                other.get_attribute_range_ref(attribute.attribute_definition(), 0..other.len()),
            );
        }
        self.length += other.len();
    }
}

impl<'a> ColumnarBuffer<'a> for HashMapBuffer
where
    HashMapBuffer: 'a,
{
    fn get_attribute_ref<'b>(
        &'b self,
        attribute: &PointAttributeDefinition,
        index: usize,
    ) -> &'b [u8]
    where
        'a: 'b,
    {
        let storage_of_attribute = self
            .attributes_storage
            .get(attribute)
            .expect("Attribute not found in PointLayout of this buffer");
        &storage_of_attribute[Self::get_byte_range_for_attribute(index, attribute)]
    }

    fn get_attribute_range_ref<'b>(
        &'b self,
        attribute: &PointAttributeDefinition,
        range: Range<usize>,
    ) -> &'b [u8]
    where
        'a: 'b,
    {
        let storage_of_attribute = self
            .attributes_storage
            .get(attribute)
            .expect("Attribute not found in PointLayout of this buffer");
        &storage_of_attribute[Self::get_byte_range_for_attributes(range, attribute)]
    }
}

impl<'a> ColumnarBufferMut<'a> for HashMapBuffer
where
    HashMapBuffer: 'a,
{
    fn get_attribute_mut<'b>(
        &'b mut self,
        attribute: &PointAttributeDefinition,
        index: usize,
    ) -> &'b mut [u8]
    where
        'a: 'b,
    {
        let byte_range = Self::get_byte_range_for_attribute(index, attribute);
        let storage_of_attribute = self
            .attributes_storage
            .get_mut(attribute)
            .expect("Attribute not found in PointLayout of this buffer");
        &mut storage_of_attribute[byte_range]
    }

    fn get_attribute_range_mut<'b>(
        &'b mut self,
        attribute: &PointAttributeDefinition,
        range: Range<usize>,
    ) -> &'b mut [u8]
    where
        'a: 'b,
    {
        let byte_range = Self::get_byte_range_for_attributes(range, attribute);
        let storage_of_attribute = self
            .attributes_storage
            .get_mut(attribute)
            .expect("Attribute not found in PointLayout of this buffer");
        &mut storage_of_attribute[byte_range]
    }
}

impl<'a> SliceBuffer<'a> for HashMapBuffer
where
    Self: 'a,
{
    type SliceType = BufferSliceColumnar<'a, Self>;

    fn slice(&'a self, range: Range<usize>) -> Self::SliceType {
        BufferSliceColumnar::new(self, range)
    }
}

impl<'a> SliceBufferMut<'a> for HashMapBuffer {
    type SliceTypeMut = BufferSliceColumnarMut<'a, Self>;

    fn slice_mut(&'a mut self, range: Range<usize>) -> Self::SliceTypeMut {
        BufferSliceColumnarMut::new(self, range)
    }
}

impl<T: PointType> FromIterator<T> for HashMapBuffer {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let point_layout = T::layout();
        let mut buffer = Self::new_from_layout(point_layout);
        iter.into_iter().for_each(|point| {
            let point_bytes = bytemuck::bytes_of(&point);
            // Safe because we know that `buffer` has the same `PointLayout` as `T`
            unsafe {
                buffer.push_points(point_bytes);
            }
        });
        buffer
    }
}

/// A point buffer that stores point data in interleaved memory layout in an externally borrowed memory resource.
/// This can be any type that is convertible to a `&[u8]`. If `T` also is convertible to a `&mut [u8]`, this buffer
/// also implements [`BorrowedMutBuffer`]
pub struct ExternalMemoryBuffer<T: AsRef<[u8]>> {
    external_memory: T,
    point_layout: PointLayout,
    length: usize,
}

impl<T: AsRef<[u8]>> ExternalMemoryBuffer<T> {
    /// Creates a new `ExternalMemoryBuffer` from the given `external_memory` resource and the given `PointLayout`
    pub fn new(external_memory: T, point_layout: PointLayout) -> Self {
        let length = match point_layout.size_of_point_entry() {
            0 => {
                assert_eq!(0, external_memory.as_ref().len());
                0
            }
            point_size => {
                assert!(external_memory.as_ref().len() % point_size as usize == 0);
                external_memory.as_ref().len() / point_size as usize
            }
        };
        Self {
            external_memory,
            point_layout,
            length,
        }
    }

    fn get_byte_range_for_point(&self, point_index: usize) -> Range<usize> {
        let size_of_point = self.point_layout.size_of_point_entry() as usize;
        (point_index * size_of_point)..((point_index + 1) * size_of_point)
    }

    fn get_byte_range_for_point_range(&self, point_range: Range<usize>) -> Range<usize> {
        let size_of_point = self.point_layout.size_of_point_entry() as usize;
        (point_range.start * size_of_point)..(point_range.end * size_of_point)
    }

    fn get_byte_range_of_attribute(
        &self,
        point_index: usize,
        attribute: &PointAttributeMember,
    ) -> Range<usize> {
        let start_byte = (point_index * self.point_layout.size_of_point_entry() as usize)
            + attribute.offset() as usize;
        let end_byte = start_byte + attribute.size() as usize;
        start_byte..end_byte
    }
}

impl<'a, T: AsRef<[u8]>> BorrowedBuffer<'a> for ExternalMemoryBuffer<T>
where
    ExternalMemoryBuffer<T>: 'a,
{
    fn len(&self) -> usize {
        self.length
    }

    fn point_layout(&self) -> &PointLayout {
        &self.point_layout
    }

    fn get_point(&self, index: usize, data: &mut [u8]) {
        let point_bytes = &self.external_memory.as_ref()[self.get_byte_range_for_point(index)];
        data.copy_from_slice(point_bytes);
    }

    fn get_point_range(&self, range: Range<usize>, data: &mut [u8]) {
        let point_bytes =
            &self.external_memory.as_ref()[self.get_byte_range_for_point_range(range)];
        data.copy_from_slice(point_bytes);
    }

    unsafe fn get_attribute_unchecked(
        &self,
        attribute_member: &PointAttributeMember,
        index: usize,
        data: &mut [u8],
    ) {
        let attribute_bytes_range = self.get_byte_range_of_attribute(index, attribute_member);
        let attribute_bytes = &self.external_memory.as_ref()[attribute_bytes_range];
        data.copy_from_slice(attribute_bytes);
    }

    fn as_interleaved(&self) -> Option<&dyn InterleavedBuffer<'a>> {
        Some(self)
    }
}

impl<'a, T: AsMut<[u8]> + AsRef<[u8]>> BorrowedMutBuffer<'a> for ExternalMemoryBuffer<T>
where
    ExternalMemoryBuffer<T>: 'a,
{
    unsafe fn set_point(&mut self, index: usize, point_data: &[u8]) {
        let point_byte_range = self.get_byte_range_for_point(index);
        let point_memory = &mut self.external_memory.as_mut()[point_byte_range];
        point_memory.copy_from_slice(point_data);
    }

    unsafe fn set_attribute(
        &mut self,
        attribute: &PointAttributeDefinition,
        index: usize,
        attribute_data: &[u8],
    ) {
        let attribute_member = self
            .point_layout
            .get_attribute(attribute)
            .expect("Attribute not found in PointLayout of this buffer");
        let attribute_byte_range = self.get_byte_range_of_attribute(index, attribute_member);
        let attribute_bytes = &mut self.external_memory.as_mut()[attribute_byte_range];
        attribute_bytes.copy_from_slice(attribute_data);
    }

    fn swap(&mut self, from_index: usize, to_index: usize) {
        assert!(from_index < self.len());
        assert!(to_index < self.len());
        if from_index == to_index {
            return;
        }
        let size_of_point = self.point_layout.size_of_point_entry() as usize;
        // Is safe if neither `from_index` nor `to_index` is out of bounds, which is asserted
        unsafe {
            let from_ptr = self
                .external_memory
                .as_mut()
                .as_mut_ptr()
                .add(from_index * size_of_point);
            let to_ptr = self
                .external_memory
                .as_mut()
                .as_mut_ptr()
                .add(to_index * size_of_point);
            std::ptr::swap_nonoverlapping(from_ptr, to_ptr, size_of_point);
        }
    }

    unsafe fn set_point_range(&mut self, point_range: Range<usize>, point_data: &[u8]) {
        let point_byte_range = self.get_byte_range_for_point_range(point_range);
        let point_memory = &mut self.external_memory.as_mut()[point_byte_range];
        point_memory.copy_from_slice(point_data);
    }

    unsafe fn set_attribute_range(
        &mut self,
        attribute: &PointAttributeDefinition,
        point_range: Range<usize>,
        attribute_data: &[u8],
    ) {
        let attribute_member = self
            .point_layout
            .get_attribute(attribute)
            .expect("Attribute not found in PointLayout of this buffer");
        let attribute_size = attribute_member.size() as usize;
        let first_point = point_range.start;
        for point_index in point_range {
            let zero_based_index = point_index - first_point;
            let src_slice = &attribute_data
                [(zero_based_index * attribute_size)..((zero_based_index + 1) * attribute_size)];
            let attribute_byte_range =
                self.get_byte_range_of_attribute(point_index, attribute_member);
            let attribute_bytes = &mut self.external_memory.as_mut()[attribute_byte_range];
            attribute_bytes.copy_from_slice(src_slice);
        }
    }

    fn as_interleaved_mut(&mut self) -> Option<&mut dyn InterleavedBufferMut<'a>> {
        Some(self)
    }
}

impl<'a, T: AsRef<[u8]>> InterleavedBuffer<'a> for ExternalMemoryBuffer<T>
where
    ExternalMemoryBuffer<T>: 'a,
{
    fn get_point_ref<'b>(&'b self, index: usize) -> &'b [u8]
    where
        'a: 'b,
    {
        let memory = self.external_memory.as_ref();
        &memory[self.get_byte_range_for_point(index)]
    }

    fn get_point_range_ref<'b>(&'b self, range: Range<usize>) -> &'b [u8]
    where
        'a: 'b,
    {
        let memory = self.external_memory.as_ref();
        &memory[self.get_byte_range_for_point_range(range)]
    }
}

impl<'a, T: AsRef<[u8]> + AsMut<[u8]>> InterleavedBufferMut<'a> for ExternalMemoryBuffer<T>
where
    ExternalMemoryBuffer<T>: 'a,
{
    fn get_point_mut<'b>(&'b mut self, index: usize) -> &'b mut [u8]
    where
        'a: 'b,
    {
        let byte_range = self.get_byte_range_for_point(index);
        let memory = self.external_memory.as_mut();
        &mut memory[byte_range]
    }

    fn get_point_range_mut<'b>(&'b mut self, range: Range<usize>) -> &'b mut [u8]
    where
        'a: 'b,
    {
        let byte_range = self.get_byte_range_for_point_range(range);
        let memory = self.external_memory.as_mut();
        &mut memory[byte_range]
    }
}

impl<'a, T: AsRef<[u8]> + 'a> SliceBuffer<'a> for ExternalMemoryBuffer<T>
where
    Self: 'a,
{
    type SliceType = BufferSliceInterleaved<'a, Self>;

    fn slice(&'a self, range: Range<usize>) -> Self::SliceType {
        BufferSliceInterleaved::new(self, range)
    }
}

impl<'a, T: AsRef<[u8]> + AsMut<[u8]> + 'a> SliceBufferMut<'a> for ExternalMemoryBuffer<T> {
    type SliceTypeMut = BufferSliceInterleavedMut<'a, Self>;
    fn slice_mut(&'a mut self, range: Range<usize>) -> Self::SliceTypeMut {
        BufferSliceInterleavedMut::new(self, range)
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use nalgebra::Vector3;
    use rand::{prelude::Distribution, thread_rng, Rng};

    use crate::layout::{attributes::POSITION_3D, PointAttributeDataType};
    use crate::test_utils::*;

    use super::*;

    fn compare_attributes_typed<'a, U: PrimitiveType + std::fmt::Debug + PartialEq>(
        buffer: &'a impl BorrowedBuffer<'a>,
        attribute: &PointAttributeDefinition,
        expected_points: &'a impl BorrowedBuffer<'a>,
    ) {
        let collected_values = buffer
            .view_attribute::<U>(attribute)
            .into_iter()
            .collect::<Vec<_>>();
        let expected_values = expected_points
            .view_attribute::<U>(attribute)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(expected_values, collected_values);
    }

    /// Compare the given point attribute using the static type corresponding to the attribute's `PointAttributeDataType`
    fn compare_attributes<'a>(
        buffer: &'a impl BorrowedBuffer<'a>,
        attribute: &PointAttributeDefinition,
        expected_points: &'a impl BorrowedBuffer<'a>,
    ) {
        match attribute.datatype() {
            PointAttributeDataType::F32 => {
                compare_attributes_typed::<f32>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::F64 => {
                compare_attributes_typed::<f64>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::I16 => {
                compare_attributes_typed::<i16>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::I32 => {
                compare_attributes_typed::<i32>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::I64 => {
                compare_attributes_typed::<i64>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::I8 => {
                compare_attributes_typed::<i8>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::U16 => {
                compare_attributes_typed::<u16>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::U32 => {
                compare_attributes_typed::<u32>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::U64 => {
                compare_attributes_typed::<u64>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::U8 => {
                compare_attributes_typed::<u8>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::Vec3f32 => {
                compare_attributes_typed::<Vector3<f32>>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::Vec3f64 => {
                compare_attributes_typed::<Vector3<f64>>(buffer, attribute, expected_points);
            }
            PointAttributeDataType::Vec3i32 => {
                compare_attributes_typed::<Vector3<i32>>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::Vec3u16 => {
                compare_attributes_typed::<Vector3<u16>>(buffer, attribute, expected_points)
            }
            PointAttributeDataType::Vec3u8 => {
                compare_attributes_typed::<Vector3<u8>>(buffer, attribute, expected_points)
            }
            _ => unimplemented!(),
        }
    }

    fn test_vector_buffer_with_type<T: PointType + std::fmt::Debug + PartialEq + Copy + Clone>()
    where
        DefaultPointDistribution: Distribution<T>,
    {
        const COUNT: usize = 16;
        let test_data: Vec<T> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();
        let overwrite_data: Vec<T> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();

        let test_data_as_buffer = test_data.iter().copied().collect::<VectorBuffer>();

        {
            let mut buffer = VectorBuffer::new_from_layout(T::layout());
            assert_eq!(0, buffer.len());
            assert_eq!(T::layout(), *buffer.point_layout());
            assert_eq!(0, buffer.view::<T>().into_iter().count());

            for (idx, point) in test_data.iter().enumerate() {
                buffer.view_mut().push_point(*point);
                assert_eq!(idx + 1, buffer.len());
                assert_eq!(*point, buffer.view().at(idx));
            }

            let mut collected_points = buffer.view().into_iter().collect::<Vec<_>>();
            assert_eq!(test_data, collected_points);

            let collected_points_by_ref = buffer.view().iter().copied().collect::<Vec<_>>();
            assert_eq!(test_data, collected_points_by_ref);

            for attribute in buffer.point_layout().attributes() {
                compare_attributes(
                    &buffer,
                    attribute.attribute_definition(),
                    &test_data_as_buffer,
                );
            }

            let slice = buffer.slice(1..2);
            assert_eq!(test_data[1], slice.view().at(0));

            for (idx, point) in overwrite_data.iter().enumerate() {
                *buffer.view_mut().at_mut(idx) = *point;
            }
            collected_points = buffer.view().iter().copied().collect();
            assert_eq!(overwrite_data, collected_points);
        }
    }

    fn test_hashmap_buffer_with_type<T: PointType + std::fmt::Debug + PartialEq + Copy + Clone>()
    where
        DefaultPointDistribution: Distribution<T>,
    {
        const COUNT: usize = 16;
        let test_data: Vec<T> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();
        let overwrite_data: Vec<T> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();

        let test_data_as_buffer = test_data.iter().copied().collect::<HashMapBuffer>();

        {
            let mut buffer = HashMapBuffer::new_from_layout(T::layout());
            assert_eq!(0, buffer.len());
            assert_eq!(T::layout(), *buffer.point_layout());
            assert_eq!(0, buffer.view::<T>().into_iter().count());

            for (idx, point) in test_data.iter().enumerate() {
                buffer.view_mut().push_point(*point);
                assert_eq!(idx + 1, buffer.len());
                assert_eq!(*point, buffer.view().at(idx));
            }

            let mut collected_points = buffer.view().into_iter().collect::<Vec<_>>();
            assert_eq!(test_data, collected_points);

            for attribute in buffer.point_layout().attributes() {
                compare_attributes(
                    &buffer,
                    attribute.attribute_definition(),
                    &test_data_as_buffer,
                );
            }

            let slice = buffer.slice(1..2);
            assert_eq!(test_data[1], slice.view().at(0));

            for (idx, point) in overwrite_data.iter().enumerate() {
                buffer.view_mut().set_at(idx, *point);
            }
            collected_points = buffer.view().into_iter().collect();
            assert_eq!(overwrite_data, collected_points);
        }
    }

    fn test_external_memory_buffer_with_type<
        T: PointType + std::fmt::Debug + PartialEq + Copy + Clone,
    >()
    where
        DefaultPointDistribution: Distribution<T>,
    {
        const COUNT: usize = 16;
        let test_data: Vec<T> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();
        let overwrite_data: Vec<T> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();

        let mut memory_of_buffer: Vec<u8> =
            vec![0; COUNT * T::layout().size_of_point_entry() as usize];
        let mut test_data_as_buffer = ExternalMemoryBuffer::new(&mut memory_of_buffer, T::layout());
        for (idx, point) in test_data.iter().copied().enumerate() {
            test_data_as_buffer.view_mut().set_at(idx, point);
        }

        {
            let mut buffer = VectorBuffer::new_from_layout(T::layout());
            assert_eq!(0, buffer.len());
            assert_eq!(T::layout(), *buffer.point_layout());
            assert_eq!(0, buffer.view::<T>().into_iter().count());

            for (idx, point) in test_data.iter().enumerate() {
                buffer.view_mut().push_point(*point);
                assert_eq!(idx + 1, buffer.len());
                assert_eq!(*point, buffer.view().at(idx));
            }

            let mut collected_points = buffer.view().into_iter().collect::<Vec<_>>();
            assert_eq!(test_data, collected_points);

            let collected_points_by_ref = buffer.view().iter().copied().collect::<Vec<_>>();
            assert_eq!(test_data, collected_points_by_ref);

            for attribute in buffer.point_layout().attributes() {
                compare_attributes(
                    &buffer,
                    attribute.attribute_definition(),
                    &test_data_as_buffer,
                );
            }

            let slice = buffer.slice(1..2);
            assert_eq!(test_data[1], slice.view().at(0));

            for (idx, point) in overwrite_data.iter().enumerate() {
                *buffer.view_mut().at_mut(idx) = *point;
            }
            collected_points = buffer.view().iter().copied().collect();
            assert_eq!(overwrite_data, collected_points);
        }
    }

    #[test]
    fn test_vector_buffer() {
        test_vector_buffer_with_type::<CustomPointTypeSmall>();
        test_vector_buffer_with_type::<CustomPointTypeBig>();
    }

    #[test]
    fn test_hash_map_buffer() {
        test_hashmap_buffer_with_type::<CustomPointTypeSmall>();
        test_hashmap_buffer_with_type::<CustomPointTypeBig>();
    }

    #[test]
    fn test_hash_map_buffer_mutate_attribute() {
        const COUNT: usize = 16;
        let test_data: Vec<CustomPointTypeBig> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();
        let overwrite_data: Vec<CustomPointTypeBig> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();

        let mut buffer = test_data.iter().copied().collect::<HashMapBuffer>();

        for (idx, attribute) in buffer
            .view_attribute_mut::<Vector3<f64>>(&POSITION_3D)
            .iter_mut()
            .enumerate()
        {
            *attribute = overwrite_data[idx].position;
        }

        let expected_positions = overwrite_data
            .iter()
            .map(|point| point.position)
            .collect::<Vec<_>>();
        let actual_positions = buffer
            .view_attribute(&POSITION_3D)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(expected_positions, actual_positions);
    }

    #[test]
    fn test_external_memory_buffer() {
        test_external_memory_buffer_with_type::<CustomPointTypeSmall>();
        test_external_memory_buffer_with_type::<CustomPointTypeBig>();
    }

    fn test_transform_attribute_generic<
        'a,
        B: BorrowedMutBuffer<'a> + FromIterator<CustomPointTypeBig> + 'a,
    >() {
        const COUNT: usize = 16;
        let test_data: Vec<CustomPointTypeBig> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();
        let overwrite_data: Vec<CustomPointTypeBig> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();

        let mut buffer = test_data.iter().copied().collect::<B>();
        // Overwrite the positions with the positions in `overwrite_data` using the `transform_attribute` function
        buffer.transform_attribute(&POSITION_3D, |index, _| -> Vector3<f64> {
            overwrite_data[index].position
        });

        let expected_positions = overwrite_data
            .iter()
            .map(|point| point.position)
            .collect::<Vec<_>>();
        let actual_positions = buffer
            .view_attribute(&POSITION_3D)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(expected_positions, actual_positions);
    }

    #[test]
    fn test_transform_attribute() {
        test_transform_attribute_generic::<VectorBuffer>();
        test_transform_attribute_generic::<HashMapBuffer>();
    }

    #[test]
    fn test_append() {
        const COUNT: usize = 16;
        let test_data: Vec<CustomPointTypeBig> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();

        let expected_buffer_interleaved = test_data.iter().copied().collect::<VectorBuffer>();
        let expected_buffer_columnar = test_data.iter().copied().collect::<HashMapBuffer>();

        {
            let mut vector_buffer = VectorBuffer::new_from_layout(CustomPointTypeBig::layout());
            vector_buffer.append(&expected_buffer_interleaved);
            assert_eq!(expected_buffer_interleaved, vector_buffer);
        }
        {
            let mut vector_buffer = VectorBuffer::new_from_layout(CustomPointTypeBig::layout());
            vector_buffer.append(&expected_buffer_columnar);
            assert_eq!(expected_buffer_interleaved, vector_buffer);
        }
        {
            let mut vector_buffer = VectorBuffer::new_from_layout(CustomPointTypeBig::layout());
            vector_buffer.append_interleaved(&expected_buffer_interleaved);
            assert_eq!(expected_buffer_interleaved, vector_buffer);
        }
        {
            let mut vector_buffer = VectorBuffer::new_from_layout(CustomPointTypeBig::layout());
            vector_buffer.append_columnar(&expected_buffer_columnar);
            assert_eq!(expected_buffer_interleaved, vector_buffer);
        }
        {
            let mut hashmap_buffer = HashMapBuffer::new_from_layout(CustomPointTypeBig::layout());
            hashmap_buffer.append(&expected_buffer_columnar);
            assert_eq!(expected_buffer_columnar, hashmap_buffer);
        }
        {
            let mut hashmap_buffer = HashMapBuffer::new_from_layout(CustomPointTypeBig::layout());
            hashmap_buffer.append(&expected_buffer_interleaved);
            assert_eq!(expected_buffer_columnar, hashmap_buffer);
        }
        {
            let mut hashmap_buffer = HashMapBuffer::new_from_layout(CustomPointTypeBig::layout());
            hashmap_buffer.append_columnar(&expected_buffer_columnar);
            assert_eq!(expected_buffer_columnar, hashmap_buffer);
        }
        {
            let mut hashmap_buffer: HashMapBuffer =
                HashMapBuffer::new_from_layout(CustomPointTypeBig::layout());
            hashmap_buffer.append_interleaved(&expected_buffer_interleaved);
            assert_eq!(expected_buffer_columnar, hashmap_buffer);
        }
    }

    #[test]
    fn test_buffers_from_empty_layout() {
        let empty_layout = PointLayout::default();

        {
            let buffer = VectorBuffer::new_from_layout(empty_layout.clone());
            assert_eq!(0, buffer.len());
        }
        {
            let buffer = HashMapBuffer::new_from_layout(empty_layout.clone());
            assert_eq!(0, buffer.len());
        }
        {
            let empty_memory = Vec::default();
            let buffer = ExternalMemoryBuffer::new(&empty_memory, empty_layout.clone());
            assert_eq!(0, buffer.len());
        }
    }

    fn test_buffer_set_point_range_generic<
        B: for<'a> BorrowedMutBuffer<'a>
            + FromIterator<CustomPointTypeBig>
            + for<'a> SliceBufferMut<'a>,
    >() {
        const COUNT: usize = 16;
        let test_data: Vec<CustomPointTypeBig> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();
        let overwrite_data: Vec<CustomPointTypeBig> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();
        let raw_overwrite_data: &[u8] = bytemuck::cast_slice(&overwrite_data);

        let mut buffer = test_data.iter().copied().collect::<B>();
        // Safe because we know the point layout of buffer is equal to that of CustomPointTypeBig
        unsafe {
            buffer.set_point_range(0..COUNT, raw_overwrite_data);
        }

        let actual_data = buffer
            .view::<CustomPointTypeBig>()
            .into_iter()
            .collect_vec();
        assert_eq!(overwrite_data, actual_data);

        // Do the same thing, but with a slice
        buffer = test_data.iter().copied().collect::<B>();
        let mut buffer_slice = buffer.slice_mut(0..COUNT);
        unsafe {
            buffer_slice.set_point_range(0..COUNT, raw_overwrite_data);
        }
        drop(buffer_slice);

        let actual_data = buffer
            .view::<CustomPointTypeBig>()
            .into_iter()
            .collect_vec();
        assert_eq!(overwrite_data, actual_data);
    }

    #[test]
    fn test_buffers_set_point_range() {
        test_buffer_set_point_range_generic::<VectorBuffer>();
        test_buffer_set_point_range_generic::<HashMapBuffer>();
    }

    fn test_buffer_get_point_range_generic<
        B: for<'a> BorrowedMutBuffer<'a>
            + FromIterator<CustomPointTypeBig>
            + for<'a> SliceBufferMut<'a>,
    >() {
        const COUNT: usize = 16;
        let test_data: Vec<CustomPointTypeBig> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();
        let raw_test_data: &[u8] = bytemuck::cast_slice(test_data.as_slice());
        let size_of_single_point = std::mem::size_of::<CustomPointTypeBig>();

        let buffer = test_data.iter().copied().collect::<B>();

        let mut actual_point_data = vec![0; raw_test_data.len()];
        buffer.get_point_range(0..COUNT, &mut actual_point_data);

        assert_eq!(raw_test_data, actual_point_data);

        // Check that subset ranges work correctly as well
        let subset_slice = &mut actual_point_data[..(6 * size_of_single_point)];
        buffer.get_point_range(2..8, subset_slice);
        assert_eq!(
            &raw_test_data[(2 * size_of_single_point)..(8 * size_of_single_point)],
            subset_slice
        );
    }

    #[test]
    fn test_buffer_get_point_range() {
        test_buffer_get_point_range_generic::<VectorBuffer>();
        test_buffer_get_point_range_generic::<HashMapBuffer>();
    }

    fn test_buffer_set_attribute_range_generic<
        B: for<'a> BorrowedMutBuffer<'a>
            + FromIterator<CustomPointTypeBig>
            + for<'a> SliceBufferMut<'a>,
    >() {
        const COUNT: usize = 16;
        let test_data: Vec<CustomPointTypeBig> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();
        let overwrite_data: Vec<CustomPointTypeBig> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();
        let overwrite_positions = overwrite_data
            .iter()
            .map(|point| point.position)
            .collect_vec();
        let overwrite_positions_raw_data: &[u8] = bytemuck::cast_slice(&overwrite_positions);

        let mut buffer = test_data.iter().copied().collect::<B>();
        // Safe because we know the point layout of buffer
        unsafe {
            buffer.set_attribute_range(&POSITION_3D, 0..COUNT, overwrite_positions_raw_data);
        }

        let actual_positions = buffer
            .view_attribute(&POSITION_3D)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(overwrite_positions, actual_positions);

        // Do the same test, but for a slice
        buffer = test_data.iter().copied().collect::<B>();
        let mut buffer_slice = buffer.slice_mut(0..test_data.len());
        unsafe {
            buffer_slice.set_attribute_range(&POSITION_3D, 0..COUNT, overwrite_positions_raw_data);
        }
        drop(buffer_slice);

        let actual_positions = buffer
            .view_attribute(&POSITION_3D)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(overwrite_positions, actual_positions);
    }

    #[test]
    fn test_buffers_set_attribute_range() {
        test_buffer_set_attribute_range_generic::<VectorBuffer>();
        test_buffer_set_attribute_range_generic::<HashMapBuffer>();
    }

    #[test]
    fn test_buffers_as_memory_layout_accessors() {
        let mut vector_buffer = VectorBuffer::new_from_layout(CustomPointTypeSmall::layout());
        let mut hashmap_buffer = HashMapBuffer::new_from_layout(CustomPointTypeSmall::layout());

        let mut memory: Vec<u8> = Vec::default();
        let mut external_memory_buffer =
            ExternalMemoryBuffer::new(memory.as_mut_slice(), CustomPointTypeSmall::layout());

        assert!(vector_buffer.as_interleaved().is_some());
        assert!(vector_buffer.slice(0..0).as_interleaved().is_some());
        assert!(vector_buffer.slice_mut(0..0).as_interleaved().is_some());
        assert!(hashmap_buffer.as_interleaved().is_none());
        assert!(hashmap_buffer.slice(0..0).as_interleaved().is_none());
        assert!(hashmap_buffer.slice_mut(0..0).as_interleaved().is_none());
        assert!(external_memory_buffer.as_interleaved().is_some());
        assert!(external_memory_buffer
            .slice(0..0)
            .as_interleaved()
            .is_some());
        assert!(external_memory_buffer
            .slice_mut(0..0)
            .as_interleaved()
            .is_some());

        assert!(vector_buffer.as_interleaved_mut().is_some());
        assert!(vector_buffer.slice_mut(0..0).as_interleaved_mut().is_some());
        assert!(hashmap_buffer.as_interleaved_mut().is_none());
        assert!(hashmap_buffer
            .slice_mut(0..0)
            .as_interleaved_mut()
            .is_none());
        assert!(external_memory_buffer.as_interleaved_mut().is_some());
        assert!(external_memory_buffer
            .slice_mut(0..0)
            .as_interleaved_mut()
            .is_some());

        assert!(vector_buffer.as_columnar().is_none());
        assert!(vector_buffer.slice(0..0).as_columnar().is_none());
        assert!(vector_buffer.slice_mut(0..0).as_columnar().is_none());
        assert!(hashmap_buffer.as_columnar().is_some());
        assert!(hashmap_buffer.slice(0..0).as_columnar().is_some());
        assert!(hashmap_buffer.slice_mut(0..0).as_columnar().is_some());
        assert!(external_memory_buffer.as_columnar().is_none());
        assert!(external_memory_buffer.slice(0..0).as_columnar().is_none());
        assert!(external_memory_buffer
            .slice_mut(0..0)
            .as_columnar()
            .is_none());

        assert!(vector_buffer.as_columnar_mut().is_none());
        assert!(vector_buffer.slice_mut(0..0).as_columnar_mut().is_none());
        assert!(hashmap_buffer.as_columnar_mut().is_some());
        assert!(hashmap_buffer.slice_mut(0..0).as_columnar_mut().is_some());
        assert!(external_memory_buffer.as_columnar_mut().is_none());
        assert!(external_memory_buffer
            .slice_mut(0..0)
            .as_columnar_mut()
            .is_none());
    }

    #[test]
    fn test_hash_map_buffer_filter() {
        const COUNT: usize = 16;
        let test_data: Vec<CustomPointTypeBig> = thread_rng()
            .sample_iter(DefaultPointDistribution)
            .take(COUNT)
            .collect();
        let even_points = test_data
            .iter()
            .enumerate()
            .filter_map(
                |(idx, point)| {
                    if idx % 2 == 0 {
                        Some(*point)
                    } else {
                        None
                    }
                },
            )
            .collect_vec();

        let src_buffer = test_data.iter().copied().collect::<HashMapBuffer>();

        let even_points_columnar = src_buffer.filter::<HashMapBuffer, _>(|idx| idx % 2 == 0);
        assert_eq!(
            even_points_columnar,
            even_points.iter().copied().collect::<HashMapBuffer>()
        );

        let even_points_interleaved = src_buffer.filter::<VectorBuffer, _>(|idx| idx % 2 == 0);
        assert_eq!(
            even_points_interleaved,
            even_points.iter().copied().collect::<VectorBuffer>()
        );
    }
}
