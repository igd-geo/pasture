use std::ops::Range;

use crate::{
    containers::{
        AttributeView, AttributeViewConverting, AttributeViewMut, PointView, PointViewMut,
        RawAttributeView, RawAttributeViewMut,
    },
    layout::{
        PointAttributeDefinition, PointAttributeMember, PointLayout, PointType, PrimitiveType,
    },
};

use anyhow::Result;

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
        'a: 'b,
    {
        AttributeViewMut::new(self, attribute)
    }

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
        'a: 'b,
    {
        let num_points = self.len();
        let mut attribute_view = self.view_attribute_mut(attribute);
        for point_index in 0..num_points {
            let attribute_value = attribute_view.at(point_index);
            attribute_view.set_at(point_index, func(point_index, attribute_value));
        }
    }
}

impl<'a, T: BorrowedMutBuffer<'a>> BorrowedMutBufferExt<'a> for T {}
impl<'a> BorrowedMutBufferExt<'a> for dyn BorrowedMutBuffer<'a> + 'a {}
// TODO impl for owning buffer
impl<'a> BorrowedMutBufferExt<'a> for dyn InterleavedBufferMut<'a> + 'a {}
impl<'a> BorrowedMutBufferExt<'a> for dyn ColumnarBufferMut<'a> + 'a {}

pub trait OwningBufferExt<'a>: OwningBuffer<'a> {
    /// Appends data from the given buffer to the end of this buffer
    ///
    /// # Panics
    ///
    /// If `self.point_layout()` does not equal `other.point_layout()`
    fn append<'b, B: BorrowedBuffer<'b> + ?Sized>(&mut self, other: &'_ B) {
        assert_eq!(self.point_layout(), other.point_layout());

        // There are a bunch of ways we can append data, depending on the memory layout. In general, if
        // the memory layout of this buffer and other match, we can get by with only a few copy operations
        // (one for interleaved layout, one per attribute for columnar layout)
        // If we know the specific layout of one buffer but not the other, we can get by without any allocations
        // If both buffers are neither interleaved nor columnar, we have to resort to the most general, but
        // slowest methods

        if let (Some(_), Some(other_interleaved)) =
            (self.as_interleaved_mut(), other.as_interleaved())
        {
            // The happy case where append is equal to Vec::append
            // Safe because both point layouts are equal
            unsafe {
                self.push_points(other_interleaved.get_point_range_ref(0..other.len()));
            }
            return;
        }

        let old_self_len = self.len();
        let new_self_len = old_self_len + other.len();
        self.resize(new_self_len);

        if let Some(self_interleaved) = self.as_interleaved_mut() {
            let point_size = self_interleaved.point_layout().size_of_point_entry() as usize;
            let new_points = self_interleaved.get_point_range_mut(old_self_len..new_self_len);
            for (index, new_point) in new_points.chunks_exact_mut(point_size).enumerate() {
                other.get_point(index, new_point);
            }
        } else if let Some(self_columnar) = self.as_columnar_mut() {
            if let Some(other_columnar) = other.as_columnar() {
                for attribute in other.point_layout().attributes() {
                    // Safe because point layouts are equal
                    unsafe {
                        self_columnar.set_attribute_range(
                            attribute.attribute_definition(),
                            old_self_len..new_self_len,
                            other_columnar.get_attribute_range_ref(
                                attribute.attribute_definition(),
                                0..other_columnar.len(),
                            ),
                        );
                    }
                }
            } else {
                for attribute in other.point_layout().attributes() {
                    let new_attributes = self_columnar.get_attribute_range_mut(
                        attribute.attribute_definition(),
                        old_self_len..new_self_len,
                    );
                    for (index, new_attribute) in new_attributes
                        .chunks_exact_mut(attribute.size() as usize)
                        .enumerate()
                    {
                        other.get_attribute(attribute.attribute_definition(), index, new_attribute);
                    }
                }
            }
        } else {
            let mut point_buffer = vec![0; self.point_layout().size_of_point_entry() as usize];
            for point_index in 0..other.len() {
                other.get_point(point_index, &mut point_buffer);
                // Is safe because we assert that the point layouts of self and other match
                unsafe {
                    self.set_point(old_self_len + point_index, &point_buffer);
                }
            }
        }
    }
}

impl<'a, T: OwningBuffer<'a>> OwningBufferExt<'a> for T {}
impl<'a> OwningBufferExt<'a> for dyn OwningBuffer<'a> + 'a {}

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
