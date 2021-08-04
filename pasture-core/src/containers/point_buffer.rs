use std::{mem::MaybeUninit, ops::Range};

use crate::{
    layout::{
        conversion::get_converter_for_attributes, PointAttributeDefinition, PointLayout, PointType,
        PrimitiveType,
    },
    util::view_raw_bytes,
};

use super::{
    attr1::AttributeIteratorByRef,
    attr1::{
        AttributeIteratorByMut, AttributeIteratorByValue, AttributeIteratorByValueWithConversion,
    },
    iterators::PointIteratorByMut,
    iterators::PointIteratorByRef,
    iterators::PointIteratorByValue,
    PerAttributePointBufferSlice, PerAttributePointBufferSliceMut,
};

// TODO Can we maybe impl<T: PointBufferWriteable> &T and provide some push<U> methods?

/// Base trait for all containers that store point data. A PointBuffer stores any number of point entries
/// with a layout defined by the PointBuffers associated PointLayout structure.
///
/// Users will rarely have to work with this base trait directly as it exposes the underlying memory-unsafe
/// API for fast point and attribute access. Instead, prefer specific PointBuffer implementations or point views!
pub trait PointBuffer {
    /// Get the data for a single point from this PointBuffer and store it inside the given memory region.
    /// Panics if point_index is out of bounds. buf must be at least as big as a single point entry in the
    /// corresponding PointLayout of this PointBuffer
    fn get_raw_point(&self, point_index: usize, buf: &mut [u8]);
    /// Get the data for the given attribute of a single point from this PointBuffer and store it inside the
    /// given memory region. Panics if point_index is out of bounds or if the attribute is not part of the point_layout
    /// of this PointBuffer. buf must be at least as big as a single attribute entry of the given attribute.
    fn get_raw_attribute(
        &self,
        point_index: usize,
        attribute: &PointAttributeDefinition,
        buf: &mut [u8],
    );
    /// Get the data for a range of points from this PointBuffer and stores it inside the given memory region.
    /// Panics if any index in index_range is out of bounds. buf must be at least as big as the size of a single
    /// point entry in the corresponding PointLayout of this PointBuffer multiplied by the number of point indices
    /// in index_range
    fn get_raw_points(&self, index_range: Range<usize>, buf: &mut [u8]);
    // Get the data for the given attribute for a range of points from this PointBuffer and stores it inside the
    // given memory region. Panics if any index in index_range is out of bounds or if the attribute is not part of
    // the point_layout of this PointBuffer. buf must be at least as big as the size of a single entry of the given
    // attribute multiplied by the number of point indices in index_range.
    fn get_raw_attribute_range(
        &self,
        index_range: Range<usize>,
        attribute: &PointAttributeDefinition,
        buf: &mut [u8],
    );

    /// Returns the number of unique point entries stored inside this PointBuffer
    fn len(&self) -> usize;
    /// Returns true if the associated `PointBuffer` is empty, i.e. it stores zero points
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Returns a reference to the underlying PointLayout of this PointBuffer
    fn point_layout(&self) -> &PointLayout;

    /// Try to downcast the associated `PointBuffer` into an `InterleavedPointBuffer`
    fn as_interleaved(&self) -> Option<&dyn InterleavedPointBuffer> {
        None
    }

    /// Try to downcast the associated `PointBuffer` into an `PerAttributePointBuffer`
    fn as_per_attribute(&self) -> Option<&dyn PerAttributePointBuffer> {
        None
    }
}

/// Trait for all mutable `PointBuffer`s, that is all `PointBuffer`s where it is possible to push points into. Distinguishing between
/// read-only `PointBuffer` and mutable `PointBufferMut` traits enables read-only, non-owning views of a `PointBuffer` with the same interface
/// as an owning `PointBuffer`!
pub trait PointBufferWriteable: PointBuffer {
    /// Sets the data for a single point within the associated `PointBufferWriteable` to the data in `buf`.
    ///
    /// # Panics
    ///
    /// Panics if `point_index` is out of bounds or the length of `buf` does not match the size of a single point in the
    /// `PointLayout`
    fn set_raw_point(&mut self, point_index: usize, buf: &[u8]);
    /// Sets the data for a single attribute of a isngle point within the associated `PointBufferWriteable` to the data in `buf`.
    ///
    /// # Panics
    ///
    /// Panics if `point_index` is out of bounds, `attribute` is not part of the `PointLayout`, or the length of `buf` does not
    /// match the size of a single point in the `PointLayout`
    fn set_raw_attribute(
        &mut self,
        point_index: usize,
        attribute: &PointAttributeDefinition,
        buf: &[u8],
    );

    /// Appends the given `points` to the end of the associated `PointBuffer`
    ///
    /// # Panics
    ///
    /// Panics if `points` is neither an `InterleavedPointBuffer` nor a `PerAttributePointBuffer`. *Note:* All the builtin buffers supplied by
    /// Pasture always implement at least one of these traits, so it is always safe to call `push` with any of the builtin buffers.
    fn push(&mut self, points: &dyn PointBuffer);

    /// Replaces the specified `range` in the associated `PointBuffer` with the given `replace_with` buffer.
    ///
    /// # Panics
    ///
    /// Panics if the starting point is greater than the end point or if the end point is greater than the length of the associated `PointBuffer`.
    /// Panics if the length of `replace_with` is less than the length of `range`
    /// Panics if `replace_with` is neither an `InterleavedPointBuffer` nor a `PerAttributePointBuffer`. *Note:* All the builtin buffers supplied by
    /// Pasture always implement at least one of these traits, so it is always safe to call `splice` with any of the builtin buffers.
    fn splice(&mut self, range: Range<usize>, replace_with: &dyn PointBuffer);

    /// Clears the contents of the associated `PointBufferMut`
    fn clear(&mut self);

    /// Resizes this buffer to the given number of `new_points`. This will trim the buffer if `new_points` is smaller
    /// than the current number of points, or create default-initialized points if `new_points` is larger.
    fn resize(&mut self, new_points: usize);
}

/// Trait for `PointBuffer` types that store point data in Interleaved memory layout. In an `InterleavedPointBuffer`, all attributes
/// for a single point are stored together in memory. To illustrate this, suppose the `PointLayout` of some point
/// type defines the default attributes `POSITION_3D` (`Vector3<f64>`), `INTENSITY` (`u16`) and `CLASSIFICATION` (`u8`). In
/// an `InterleavedPointBuffer`, the data layout is like this:<br>
/// `[Vector3<f64>, u16, u8, Vector3<f64>, u16, u8, ...]`<br>
/// ` |------Point 1-------| |------Point 2-------| |--...`<br>
pub trait InterleavedPointBuffer: PointBuffer {
    /// Returns a pointer to the raw memory of the point entry at the given index in this PointBuffer. In contrast
    /// to [get_raw_point](PointBuffer::get_raw_point), this function performs no copy operations and thus can
    /// yield better performance. Panics if point_index is out of bounds.
    fn get_raw_point_ref(&self, point_index: usize) -> &[u8];
    /// Returns a pointer to the raw memory of a range of point entries in this PointBuffer. In contrast to
    /// [get_raw_point](PointBuffer::get_raw_point), this function performs no copy operations and thus can
    /// yield better performance. Panics if any index in index_range is out of bounds.
    fn get_raw_points_ref(&self, index_range: Range<usize>) -> &[u8];
}

/// Trait for `InterleavedPointBuffer` types that provide mutable access to the point data
pub trait InterleavedPointBufferMut: InterleavedPointBuffer {
    /// Mutable version of [get_raw_point_ref](InterleavedPointBuffer::get_raw_point_ref)
    fn get_raw_point_mut(&mut self, point_index: usize) -> &mut [u8];
    /// Mutable version of [get_raw_points_ref](InterleavedPointBuffer::get_raw_points_ref)
    fn get_raw_points_mut(&mut self, index_range: Range<usize>) -> &mut [u8];
}

/// Trait for `PointBuffer` types that store point data in PerAttribute memory layout. In buffers of this type, the data for a single
/// attribute of all points in stored together in memory. To illustrate this, suppose the `PointLayout` of some point
/// type defines the default attributes `POSITION_3D` (`Vector3<f64>`), `INTENSITY` (`u16`) and `CLASSIFICATION` (`u8`). In
/// a `PerAttributePointBuffer`, the data layout is like this:<br>
/// `[Vector3<f64>, Vector3<f64>, Vector3<f64>, ...]`<br>
/// `[u16, u16, u16, ...]`<br>
/// `[u8, u8, u8, ...]`<br>
pub trait PerAttributePointBuffer: PointBuffer {
    /// Returns a pointer to the raw memory for the attribute entry of the given point in this PointBuffer. In contrast
    /// to [get_raw_attribute](PointBuffer::get_raw_attribute), this function performs no copy operations and
    /// thus can yield better performance. Panics if point_index is out of bounds or if the attribute is not part of
    /// the point_layout of this PointBuffer.
    fn get_raw_attribute_ref(
        &self,
        point_index: usize,
        attribute: &PointAttributeDefinition,
    ) -> &[u8];
    /// Returns a pointer to the raw memory for the given attribute of a range of points in this PointBuffer. In contrast
    /// to [get_raw_attribute_range](PointBuffer::get_raw_attribute_range), this function performs no copy operations
    /// and thus can yield better performance. Panics if any index in index_range is out of bounds or if the attribute is
    /// not part of the point_layout of this PointBuffer.
    fn get_raw_attribute_range_ref(
        &self,
        index_range: Range<usize>,
        attribute: &PointAttributeDefinition,
    ) -> &[u8];

    /// Returns a read-only slice of the associated `PerAttributePointBuffer`
    fn slice(&self, range: Range<usize>) -> PerAttributePointBufferSlice<'_>;
}

/// Trait for `PerAttributePointBuffer` types that provide mutable access to specific attributes
pub trait PerAttributePointBufferMut<'b>: PerAttributePointBuffer {
    /// Mutable version of [get_raw_attribute_ref](PerAttributePointBuffer::get_raw_attribute_ref)
    fn get_raw_attribute_mut(
        &mut self,
        point_index: usize,
        attribute: &PointAttributeDefinition,
    ) -> &mut [u8];
    /// Mutable version of [get_raw_attribute_range_ref](PerAttributePointBuffer::get_raw_attribute_range_ref)
    fn get_raw_attribute_range_mut(
        &mut self,
        index_range: Range<usize>,
        attribute: &PointAttributeDefinition,
    ) -> &mut [u8];
    /// Returns a mutable slice of the associated `PerAttributePointBufferMut`
    fn slice_mut(&'b mut self, range: Range<usize>) -> PerAttributePointBufferSliceMut<'b>;

    /// Splits the associated `PerAttributePointBufferMut` into multiple disjoint mutable slices, based on the given disjoint `ranges`. This
    /// function is similar to Vec::split_as_mut, but can split into more than two regions.
    ///
    /// # Note
    ///
    /// The lifetime bounds seem a bit weird for this function, but they are necessary. The lifetime of this `'b` of this trait determines
    /// how long the underlying buffer (i.e. whatever stores the point data) lives. The lifetime `'p` of this function enables nested
    /// slicing of buffers, because it is bounded to live *at most* as long as `'b`. If `'b` and `'p` were a single lifetime, then it would
    /// require that any buffer *reference* from which we want to obtain a slice lives as long as the buffer itself. This is restrictive in
    /// a way that is not necessary. Consider the following scenario in pseudo-code, annotated with lifetimes:
    ///
    /// ```ignore
    /// 'x: {
    /// // Assume 'buffer' contains some data
    /// let mut buffer : PerAttributeVecPointStorage = ...;
    ///     'y: {
    ///         // Obtain two slices to the lower and upper half of the buffer
    ///         let mut half_slices = buffer.disjunct_slices_mut(&[0..(buffer.len()/2), (buffer.len()/2)..buffer.len()]);
    ///         // ... do something with the slices ...
    ///         'z: {
    ///             // Split the half slices in half again
    ///             let quarter_len = buffer.len() / 4;
    ///             let half_len = buffer.len() / 2;
    ///             let mut lower_quarter_slices = half_slices[0].disjunct_slices_mut(&[0..quarter_len, quarter_len..half_len]);
    ///         }
    ///     }
    /// }
    /// ```
    /// In this scenario, `buffer` lives for `'x`, while `half_slices` only lives for `'y`. The underlying data in `half_slices` however lives
    /// as long as `buffer`. The separate lifetime bound `'p` on this function is what allows calling `disjunct_slices_mut` on `half_slices[0]`,
    /// even though `half_slices` only lives for `'y`.
    ///
    /// # Panics
    ///
    /// If any two ranges in `ranges` overlap, or if any range in `ranges` is out of bounds
    fn disjunct_slices_mut<'p>(
        &'p mut self,
        ranges: &[Range<usize>],
    ) -> Vec<PerAttributePointBufferSliceMut<'b>>
    where
        'b: 'p;

    /// Helper method to access this `PerAttributePointBufferMut` as a `PerAttributePointBuffer`
    fn as_per_attribute_point_buffer(&self) -> &dyn PerAttributePointBuffer;
}

/// Extension trait that provides generic methods for accessing point data in a `PointBuffer`
pub trait PointBufferExt<B: PointBuffer + ?Sized> {
    /// Returns the point at `index` from the associated `PointBuffer`, strongly typed to the `PointType` `T`
    fn get_point<T: PointType>(&self, index: usize) -> T;
    /// Returns the given `attribute` for the point at `index` from the associated `PointBuffer`, strongly typed to the `PrimitiveType` `T`
    fn get_attribute<T: PrimitiveType>(
        &self,
        attribute: &PointAttributeDefinition,
        index: usize,
    ) -> T;

    /// Returns an iterator over all points in the associated `PointBuffer`, strongly typed to the `PointType` `T`
    fn iter_point<T: PointType>(&self) -> PointIteratorByValue<'_, T, B>;
    /// Returns an iterator over the given `attribute` of all points in the associated `PointBuffer`, strongly typed to the `PrimitiveType` `T`.
    ///
    /// For iterating over multiple attributes at once, use the [attributes!] macro.
    ///
    /// # Panics
    ///
    /// Panics if `attribute` is not part of the `PointLayout` of the buffer.<br>
    /// Panics if the data type of `attribute` inside the associated `PointBuffer` is not equal to `T`. If you want a conversion, use `iter_attribute_as`.
    fn iter_attribute<'a, T: PrimitiveType>(
        &'a self,
        attribute: &'a PointAttributeDefinition,
    ) -> AttributeIteratorByValue<'a, T, B>;
    /// Returns an iterator over the given `attribute` of all points in the associated `PointBuffer`, converted to the `PrimitiveType` `T`. This iterator
    /// supports conversion of types, so it works even if the `attribute` inside the buffer is stored as some other type `U`, as long as there is a valid
    /// conversion from `U` to `T`. Regarding conversions, see the [conversions module](crate::layout::conversion).
    ///
    /// For iterating over multiple attributes at once, use the [attributes!] macro.
    ///
    /// # Panics
    ///
    /// Panics if `attribute` is not part of the `PointLayout` of the buffer.<br>
    /// Panics if no valid conversion exists from the type that the attribute is stored as inside the buffer into type `T`.
    fn iter_attribute_as<'a, T: PrimitiveType>(
        &'a self,
        attribute: &'a PointAttributeDefinition,
    ) -> AttributeIteratorByValueWithConversion<'a, T, B>;
}

impl<B: PointBuffer + ?Sized> PointBufferExt<B> for B {
    fn get_point<T: PointType>(&self, index: usize) -> T {
        let mut point = MaybeUninit::<T>::uninit();
        unsafe {
            self.get_raw_point(
                index,
                std::slice::from_raw_parts_mut(
                    point.as_mut_ptr() as *mut u8,
                    std::mem::size_of::<T>(),
                ),
            );
            point.assume_init()
        }
    }

    fn get_attribute<T: PrimitiveType>(
        &self,
        attribute: &PointAttributeDefinition,
        index: usize,
    ) -> T {
        let mut attribute_data = MaybeUninit::<T>::uninit();
        unsafe {
            self.get_raw_attribute(
                index,
                attribute,
                std::slice::from_raw_parts_mut(
                    attribute_data.as_mut_ptr() as *mut u8,
                    std::mem::size_of::<T>(),
                ),
            );
            attribute_data.assume_init()
        }
    }

    fn iter_point<T: PointType>(&self) -> PointIteratorByValue<'_, T, B> {
        PointIteratorByValue::new(self)
    }

    fn iter_attribute<'a, T: PrimitiveType>(
        &'a self,
        attribute: &'a PointAttributeDefinition,
    ) -> AttributeIteratorByValue<'a, T, B> {
        AttributeIteratorByValue::new(self, attribute)
    }

    fn iter_attribute_as<'a, T: PrimitiveType>(
        &'a self,
        attribute: &'a PointAttributeDefinition,
    ) -> AttributeIteratorByValueWithConversion<'a, T, B> {
        AttributeIteratorByValueWithConversion::new(self, attribute)
    }
}

/// Extension trait that provides generic methods for manipulating point and attribute data in a `PointBufferWriteable`
pub trait PointBufferWriteableExt<B: PointBufferWriteable + ?Sized> {
    /// Sets the point at the given index to the given `point`, strongly typed to the `PointType` `T`.
    /// # Panics
    /// If the `PointLayout` of `T` does not match the `PointLayout` of this buffer.
    /// If `index` is out of bounds.
    fn set_point<T: PointType>(&mut self, index: usize, point: T);
    /// Sets the given `attribute` at the given index to the given `attribute_value`, strongly typed to the `PrimitiveType` `T`
    /// # Panics
    /// If the `PointLayout` of this buffer does not contain the given `attribute`.
    /// If the `PointAttributeDataType` of `attribute` does not match the type `T`.
    /// If `index` is out of bounds.
    fn set_attribute<T: PrimitiveType>(
        &mut self,
        attribute: &PointAttributeDefinition,
        index: usize,
        attribute_value: T,
    );

    /// Applies the given transformation `func` to all values of the attribute with the given `attribute_name` in the buffer. The function gets a mutable
    /// borrow to the attribute value, through which the attribute value can be manipulated in-place. This is an easier alternative
    /// to calling `get_attribute` and `set_attribute` in a loop.
    /// **This method also performs data type conversions!** So e.g. if the underlying `PointBuffer` stores a POSITION_3D attribute as
    /// `Vector3<f32>`, you can call this method with a function accepting e.g. `Vector3<f64>` and the data will be converted internally
    /// using the default conversion (see `layout/conversion.rs` for more information on conversions).
    /// # Panics
    /// If no attribute with the name of `attribute` is present in this buffer.
    /// If no conversion between type `T` and the underlying data type of the `attribute` in this buffer can be performed.
    /// # Examples
    /// ```
    /// # use pasture_core::containers::*;
    /// # use pasture_core::layout::*;
    /// # use pasture_derive::PointType;
    ///
    /// #[repr(C)]
    /// #[derive(PointType, Debug, PartialEq, Eq)]
    /// struct MyPointType(#[pasture(BUILTIN_INTENSITY)] u16);
    ///
    /// {
    ///   let mut storage = PerAttributeVecPointStorage::new(MyPointType::layout());
    ///   storage.push_points(&[MyPointType(42), MyPointType(43)]);
    ///   storage.transform_attribute(attributes::INTENSITY.name(), |_index, intensity: &mut u16| {
    ///     *intensity *= 2;
    ///   });
    ///   assert_eq!(MyPointType(84), storage.get_point::<MyPointType>(0));
    ///   assert_eq!(MyPointType(86), storage.get_point::<MyPointType>(1));
    /// }
    /// ```
    fn transform_attribute<T: PrimitiveType, F: Fn(usize, &mut T) -> ()>(
        &mut self,
        attribute_name: &'static str,
        func: F,
    );
}

impl<B: PointBufferWriteable + ?Sized> PointBufferWriteableExt<B> for B {
    fn set_point<T: PointType>(&mut self, index: usize, point: T) {
        if T::layout() != *self.point_layout() {
            panic!("PointLayout of type T does not match PointLayout of this buffer");
        }
        let point_bytes = unsafe { view_raw_bytes(&point) };
        self.set_raw_point(index, point_bytes);
    }

    fn set_attribute<T: PrimitiveType>(
        &mut self,
        attribute: &PointAttributeDefinition,
        index: usize,
        attribute_value: T,
    ) {
        let attribute_bytes = unsafe { view_raw_bytes(&attribute_value) };
        self.set_raw_attribute(index, attribute, attribute_bytes);
    }

    fn transform_attribute<T: PrimitiveType, F: Fn(usize, &mut T) -> ()>(
        &mut self,
        attribute_name: &'static str,
        func: F,
    ) {
        let source_attribute = PointAttributeDefinition::custom(attribute_name, T::data_type());
        if let Some(target_attribute) = self.point_layout().get_attribute_by_name(attribute_name) {
            // It is important that we use 'set_attribute' and 'get_attribute' here. This is the blanket implementation
            // of 'transform_attribute', so it makes no assumptions on the memory layout of the point data. Since the
            // attribute data might not be properly aligned, we can't get a mutable reference to the attribute value inside
            // the buffer, since reading from such a reference will be UB. Only if the buffer is a PerAttribute buffer could
            // we do this. Unfortunately, we can't implement the trait again for a more specific type, so we would have to
            // do the test if the underlying buffer is PerAttribute here using a virtual call. We could implement this in
            // the future, for now it is left here as a TODO
            if target_attribute.datatype() == source_attribute.datatype() {
                for point_index in 0..self.len() {
                    let mut attribute_value: T = self.get_attribute(&source_attribute, point_index);
                    func(point_index, &mut attribute_value);
                    self.set_attribute(&source_attribute, point_index, attribute_value);
                }
            } else {
                // If the datatypes are different, we have to convert from and to the target datatype
                let target_attribute_definition: PointAttributeDefinition = target_attribute.into();
                if let (Some(convert_to_t), Some(convert_from_t)) = (
                    get_converter_for_attributes(&target_attribute_definition, &source_attribute),
                    get_converter_for_attributes(&source_attribute, &target_attribute_definition),
                ) {
                    // The following code is a bit involved. Assume that 'U' is the type of the attribute stored in
                    // the buffer (but we don't know this type at compile-time). Then the process is as follows:
                    // 1) Get the raw bytes for the value 'U'
                    // 2) Create a temporary variable of type 'T'
                    // 3) Convert from 'U' to 'T'
                    // 4) Apply the transformation function to the 'T' value
                    // 5) Convert from 'T' to 'U'
                    // 6) Set the raw bytes of 'U' in the buffer
                    let mut buffer = vec![0; target_attribute_definition.size() as usize];
                    for point_index in 0..self.len() {
                        self.get_raw_attribute(
                            point_index,
                            &target_attribute_definition,
                            buffer.as_mut_slice(),
                        );

                        let mut as_t = MaybeUninit::<T>::uninit();
                        let mut as_t = unsafe {
                            let tmp_value_bytes = std::slice::from_raw_parts_mut(
                                as_t.as_mut_ptr() as *mut u8,
                                std::mem::size_of::<T>(),
                            );
                            convert_to_t(buffer.as_slice(), tmp_value_bytes);
                            as_t.assume_init()
                        };

                        func(point_index, &mut as_t);

                        unsafe {
                            convert_from_t(view_raw_bytes(&as_t), buffer.as_mut_slice());
                        }

                        self.set_raw_attribute(
                            point_index,
                            &target_attribute_definition,
                            buffer.as_slice(),
                        );
                    }
                } else {
                    panic!("No conversion possible between requested attribute {} and attribute in buffer {}", source_attribute, target_attribute_definition);
                }
            }
        } else {
            panic!("attribute not found in PointLayout of this buffer");
        }
    }
}

/// Extension trait that provides generic methods for accessing point data in an `InterleavedPointBuffer`
pub trait InterleavedPointBufferExt {
    /// Returns a reference to the point at `point_index` from the associated `InterleavedPointBuffer`, strongly typed to the `PointType` `T`
    ///
    /// # Panics
    ///
    /// Panics if `point_index` is >= `self.len()`
    fn get_point_ref<T: PointType>(&self, point_index: usize) -> &T;
    /// Returns a reference to the given `range` of points from the associated `InterleavedPointBuffer`, strongly typed to the `PointType` `T`
    ///
    /// # Panics
    ///
    /// Panics if the start of `range` is greater than the end of `range`, or if the end of `range` is greater than `self.len()`
    fn get_points_ref<T: PointType>(&self, range: Range<usize>) -> &[T];
    /// Returns an iterator over references to all points within the associated `InterleavedPointBuffer`, strongly typed to the `PointType` `T`
    ///
    /// # Panics
    ///
    /// Panics if the associated `InterleavedPointBuffer` does not store points with type `T`
    fn iter_point_ref<T: PointType>(&self) -> PointIteratorByRef<'_, T>;
}

impl<B: InterleavedPointBuffer + ?Sized> InterleavedPointBufferExt for B {
    fn get_point_ref<T: PointType>(&self, point_index: usize) -> &T {
        let raw_point = self.get_raw_point_ref(point_index);
        unsafe {
            let ptr = raw_point.as_ptr() as *const T;
            ptr.as_ref().expect("raw_point pointer was null")
        }
    }

    fn get_points_ref<T: PointType>(&self, range: Range<usize>) -> &[T] {
        let num_points = range.len();
        let raw_points = self.get_raw_points_ref(range);
        unsafe { std::slice::from_raw_parts(raw_points.as_ptr() as *const T, num_points) }
    }

    fn iter_point_ref<T: PointType>(&self) -> PointIteratorByRef<'_, T> {
        PointIteratorByRef::new(self)
    }
}

/// Extension trait that provides generic methods for accessing point data in an `InterleavedPointBufferMut`
pub trait InterleavedPointBufferMutExt {
    /// Returns a mutable reference to the point at `point_index` from the associated `InterleavedPointBuffer`, strongly typed to the `PointType` `T`
    ///
    /// # Panics
    ///
    /// Panics if `point_index` is >= `self.len()`
    fn get_point_mut<T: PointType>(&mut self, point_index: usize) -> &mut T;
    /// Returns a mutable reference to the given `range` of points from the associated `InterleavedPointBuffer`, strongly typed to the `PointType` `T`
    ///
    /// # Panics
    ///
    /// Panics if the start of `range` is greater than the end of `range`, or if the end of `range` is greater than `self.len()`
    fn get_points_mut<T: PointType>(&mut self, range: Range<usize>) -> &mut [T];
    /// Returns an iterator over mutable references to all points within the associated `InterleavedPointBuffer`, strongly typed to the `PointType` `T`
    ///
    /// # Panics
    ///
    /// Panics if the associated `InterleavedPointBuffer` does not store points with type `T`
    fn iter_point_mut<T: PointType>(&mut self) -> PointIteratorByMut<'_, T>;
}

impl<B: InterleavedPointBufferMut + ?Sized> InterleavedPointBufferMutExt for B {
    fn get_point_mut<T: PointType>(&mut self, point_index: usize) -> &mut T {
        let raw_point = self.get_raw_point_mut(point_index);
        unsafe {
            let ptr = raw_point.as_ptr() as *mut T;
            ptr.as_mut().expect("raw_point pointer was null")
        }
    }

    fn get_points_mut<T: PointType>(&mut self, range: Range<usize>) -> &mut [T] {
        let num_points = range.len();
        let raw_points = self.get_raw_points_mut(range);
        unsafe { std::slice::from_raw_parts_mut(raw_points.as_ptr() as *mut T, num_points) }
    }

    fn iter_point_mut<T: PointType>(&mut self) -> PointIteratorByMut<'_, T> {
        PointIteratorByMut::new(self)
    }
}

/// Extension trait that provides generic methods for accessing attribute data in an `PerAttributePointBuffer`
pub trait PerAttributePointBufferExt {
    /// Returns a reference to the given `attribute` of the point at `point_index` in the associated `PerAttributePointBuffer`, strongly typed to the `PrimitiveType` `T`
    ///
    /// # Panics
    ///
    /// Panics if `attribute` is not part of the `PointLayout` of the buffer.
    /// Panics if the `attribute` inside the buffer is not stored as type `T`.
    /// Panics if `point_index` is >= `self.len()`
    fn get_attribute_ref<T: PrimitiveType>(
        &self,
        point_index: usize,
        attribute: &PointAttributeDefinition,
    ) -> &T;
    /// Returns a reference to the `attribute` for the `range` of points in the associated `PerAttributePointBuffer`, strongly typed to the `PrimitiveType` `T`
    ///
    /// # Panics
    ///
    /// Panics if `attribute` is not part of the `PointLayout` of the buffer.
    /// Panics if the `attribute` inside the buffer is not stored as type `T`.
    /// Panics if the start of `range` is greater than the end of `range`, or if the end of `range` is greater than `self.len()`
    fn get_attribute_range_ref<T: PrimitiveType>(
        &self,
        range: Range<usize>,
        attribute: &PointAttributeDefinition,
    ) -> &[T];
    /// Returns an iterator over references to the given `attribute` for all points in the associated `PerAttributePointBuffer`, strongly typed to the `PrimitiveType` `T`
    fn iter_attribute_ref<'a, T: PrimitiveType>(
        &'a self,
        attribute: &'a PointAttributeDefinition,
    ) -> AttributeIteratorByRef<'a, T>;
}

impl<B: PerAttributePointBuffer + ?Sized> PerAttributePointBufferExt for B {
    fn get_attribute_ref<T: PrimitiveType>(
        &self,
        point_index: usize,
        attribute: &PointAttributeDefinition,
    ) -> &T {
        let raw_attribute = self.get_raw_attribute_ref(point_index, attribute);
        unsafe {
            let raw_ptr = raw_attribute.as_ptr() as *const T;
            raw_ptr.as_ref().expect("raw_attribute point was null")
        }
    }

    fn get_attribute_range_ref<T: PrimitiveType>(
        &self,
        range: Range<usize>,
        attribute: &PointAttributeDefinition,
    ) -> &[T] {
        let num_points = range.len();
        let raw_attributes = self.get_raw_attribute_range_ref(range, attribute);
        unsafe { std::slice::from_raw_parts(raw_attributes.as_ptr() as *const T, num_points) }
    }

    fn iter_attribute_ref<'a, T: PrimitiveType>(
        &'a self,
        attribute: &'a PointAttributeDefinition,
    ) -> AttributeIteratorByRef<'a, T> {
        AttributeIteratorByRef::new(self, attribute)
    }
}

/// Extension trait that provides generic methods for accessing attribute data in an `PerAttributePointBufferMut`
pub trait PerAttributePointBufferMutExt {
    /// Returns a mutable reference to the given `attribute` of the point at `point_index` in the associated `PerAttributePointBufferMut`, strongly typed to the `PrimitiveType` `T`
    ///
    /// # Panics
    ///
    /// Panics if `attribute` is not part of the `PointLayout` of the buffer.
    /// Panics if the `attribute` inside the buffer is not stored as type `T`.
    /// Panics if `point_index` is >= `self.len()`
    fn get_attribute_mut<T: PrimitiveType>(
        &mut self,
        point_index: usize,
        attribute: &PointAttributeDefinition,
    ) -> &mut T;
    /// Returns a mutable reference to the `attribute` for the `range` of points in the associated `PerAttributePointBufferMut`, strongly typed to the `PrimitiveType` `T`
    ///
    /// # Panics
    ///
    /// Panics if `attribute` is not part of the `PointLayout` of the buffer.
    /// Panics if the `attribute` inside the buffer is not stored as type `T`.
    /// Panics if the start of `range` is greater than the end of `range`, or if the end of `range` is greater than `self.len()`
    fn get_attribute_range_mut<T: PrimitiveType>(
        &mut self,
        range: Range<usize>,
        attribute: &PointAttributeDefinition,
    ) -> &mut [T];
    /// Returns an iterator over mutable references to the given `attribute` for all points in the associated `PerAttributePointBufferMut`, strongly typed to the `PrimitiveType` `T`
    fn iter_attribute_mut<'a, T: PrimitiveType>(
        &'a mut self,
        attribute: &'a PointAttributeDefinition,
    ) -> AttributeIteratorByMut<'a, T>;
}

impl<'b, B: PerAttributePointBufferMut<'b> + ?Sized> PerAttributePointBufferMutExt for B {
    fn get_attribute_mut<T: PrimitiveType>(
        &mut self,
        point_index: usize,
        attribute: &PointAttributeDefinition,
    ) -> &mut T {
        let raw_attribute = self.get_raw_attribute_mut(point_index, attribute);
        unsafe {
            let raw_ptr = raw_attribute.as_ptr() as *mut T;
            raw_ptr.as_mut().expect("raw_attribute point was null")
        }
    }

    fn get_attribute_range_mut<T: PrimitiveType>(
        &mut self,
        range: Range<usize>,
        attribute: &PointAttributeDefinition,
    ) -> &mut [T] {
        let num_points = range.len();
        let raw_attributes = self.get_raw_attribute_range_mut(range, attribute);
        unsafe { std::slice::from_raw_parts_mut(raw_attributes.as_ptr() as *mut T, num_points) }
    }

    fn iter_attribute_mut<'a, T: PrimitiveType>(
        &'a mut self,
        attribute: &'a PointAttributeDefinition,
    ) -> AttributeIteratorByMut<'a, T> {
        AttributeIteratorByMut::new(self, attribute)
    }
}
