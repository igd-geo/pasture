use std::{
    cmp::Ordering, collections::HashMap, iter::FromIterator, ops::Range, ptr::swap_nonoverlapping,
};

use nalgebra::{Vector3, Vector4};
use permutation::Permutation;

use crate::{
    layout::{PointAttributeDataType, PointAttributeMember, PointLayout, PointType, PrimitiveType},
    util::view_raw_bytes,
};

unsafe fn byte_slice_cast<T>(slice: &[u8]) -> &[T] {
    std::slice::from_raw_parts(
        slice.as_ptr() as *const T,
        slice.len() / std::mem::size_of::<T>(),
    )
}

unsafe fn byte_slice_cast_mut<T>(slice: &mut [u8]) -> &mut [T] {
    std::slice::from_raw_parts_mut(
        slice.as_ptr() as *mut T,
        slice.len() / std::mem::size_of::<T>(),
    )
}

pub trait BufferStorage {
    fn len(&self) -> usize;
    fn get(&self, index: usize, data: &mut [u8]);
    fn get_attribute(&self, attribute: &PointAttributeMember, index: usize, data: &mut [u8]);
}

pub trait BufferStorageMut: BufferStorage {
    fn push(&mut self, point: &[u8]);
    fn push_many(&mut self, points: &[u8]);
    fn swap(&mut self, from_index: usize, to_index: usize);
    // fn append(&mut self, buffer: &mut impl BufferStorage);
    fn clear(&mut self);
    fn resize(&mut self, new_size: usize, value: &[u8]);
}

pub trait BufferStorageContiguous: BufferStorage {
    fn get_ref(&self, index: usize) -> &[u8];
    fn get_range_ref(&self, indices: Range<usize>) -> &[u8];
}

pub trait BufferStorageContiguousMut: BufferStorageContiguous {
    fn get_mut(&mut self, index: usize) -> &mut [u8];
    fn get_range_mut(&mut self, indices: Range<usize>) -> &mut [u8];
    /// Sorts this buffer storage using the given comparator, as if the elements of this buffer were of type `T`
    ///
    /// # safety
    ///
    /// It is only allowed to call this method if the `PointLayout` of this buffer matches the `PointLayout` of type `T`.
    /// Since no `BufferStorage` type is required to store its `PointLayout`, it is the callers responsibility to ensure
    /// that this invariant holds!
    unsafe fn sort_by<C, T: PointType>(&mut self, compare: C)
    where
        C: Fn(&T, &T) -> Ordering;
}

pub trait BufferStorageColumnar: BufferStorage {
    fn get_attribute_ref(&self, attribute: &PointAttributeMember, index: usize) -> &[u8];
    fn get_attribute_range_ref(
        &self,
        attribute: &PointAttributeMember,
        indices: Range<usize>,
    ) -> &[u8];
    fn point_layout(&self) -> &PointLayout;
}

pub trait BufferStorageColumnarMut: BufferStorageColumnar {
    fn get_attribute_mut(&mut self, attribute: &PointAttributeMember, index: usize) -> &mut [u8];
    fn get_attribute_range_mut(
        &mut self,
        attribute: &PointAttributeMember,
        indices: Range<usize>,
    ) -> &mut [u8];
    /// Sorts this buffer storage using the given comparator based on the given attribute, as if the attribute
    /// was of type `T`
    ///
    /// # safety
    ///
    /// It is only allowed to call this method if the datatype of `attribute` is equal to `T`
    unsafe fn sort_by_attribute<C, T: PrimitiveType>(
        &mut self,
        attribute: &PointAttributeMember,
        compare: C,
    ) where
        C: Fn(&T, &T) -> Ordering;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorStorage {
    data: Vec<u8>,
    num_points: usize,
    stride: usize,
}

impl VectorStorage {
    pub fn from_layout(point_layout: &PointLayout) -> Self {
        Self {
            data: vec![],
            num_points: 0,
            stride: point_layout.size_of_point_entry() as usize,
        }
    }

    pub fn from_layout_with_capacity(point_layout: &PointLayout, capacity: usize) -> Self {
        let stride = point_layout.size_of_point_entry() as usize;
        Self {
            data: Vec::with_capacity(capacity * stride),
            num_points: 0,
            stride,
        }
    }
}

impl BufferStorage for VectorStorage {
    fn get(&self, index: usize, data: &mut [u8]) {
        data.copy_from_slice(self.get_ref(index));
    }

    fn get_attribute(&self, attribute: &PointAttributeMember, index: usize, data: &mut [u8]) {
        let point_start = self.stride * index;
        let attribute_start = point_start + attribute.offset() as usize;
        let attribute_end = attribute_start + attribute.size() as usize;
        let attribute_slice = &self.data[attribute_start..attribute_end];
        data.copy_from_slice(attribute_slice);
    }

    fn len(&self) -> usize {
        self.num_points
    }
}

impl BufferStorageMut for VectorStorage {
    fn push(&mut self, point: &[u8]) {
        assert_eq!(point.len(), self.stride);
        self.data.extend_from_slice(point);
        self.num_points += 1;
    }

    fn push_many(&mut self, points: &[u8]) {
        assert_eq!(0, points.len() % self.stride);
        let count = points.len() / self.stride;
        self.data.extend_from_slice(points);
        self.num_points += count;
    }

    fn clear(&mut self) {
        self.data.clear();
        self.num_points = 0;
    }

    fn resize(&mut self, new_size: usize, value: &[u8]) {
        if new_size == self.num_points {
            return;
        }

        if new_size < self.num_points {
            self.data.shrink_to(new_size * self.stride);
        } else {
            let diff = new_size - self.num_points;
            self.data
                .extend(value.iter().copied().cycle().take(diff * self.stride));
        }

        self.num_points = new_size;
    }

    fn swap(&mut self, from_index: usize, to_index: usize) {
        if from_index == to_index {
            return;
        }
        let from_start = from_index * self.stride;
        let to_start = to_index * self.stride;
        // This is safe because the two regions will never overlap
        unsafe {
            let from_ptr = self.data.as_mut_ptr().add(from_start);
            let to_ptr = self.data.as_mut_ptr().add(to_start);
            swap_nonoverlapping(from_ptr, to_ptr, self.stride);
        }
    }
}

impl BufferStorageContiguous for VectorStorage {
    fn get_ref(&self, index: usize) -> &[u8] {
        let point_start = self.stride * index;
        let point_end = self.stride * (index + 1);
        &self.data[point_start..point_end]
    }

    fn get_range_ref(&self, indices: Range<usize>) -> &[u8] {
        let point_start = self.stride * indices.start;
        let point_end = self.stride * indices.end;
        &self.data[point_start..point_end]
    }
}

impl BufferStorageContiguousMut for VectorStorage {
    fn get_mut(&mut self, index: usize) -> &mut [u8] {
        let point_start = self.stride * index;
        let point_end = self.stride * (index + 1);
        &mut self.data[point_start..point_end]
    }

    fn get_range_mut(&mut self, indices: Range<usize>) -> &mut [u8] {
        let point_start = self.stride * indices.start;
        let point_end = self.stride * indices.end;
        &mut self.data[point_start..point_end]
    }

    unsafe fn sort_by<C, T: PointType>(&mut self, compare: C)
    where
        C: Fn(&T, &T) -> Ordering,
    {
        let vec_of_t = unsafe {
            let slice = &mut self.data[..];
            std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut T, self.len())
        };
        vec_of_t.sort_by(compare);
    }
}

impl<P: PointType> FromIterator<P> for VectorStorage {
    fn from_iter<T: IntoIterator<Item = P>>(iter: T) -> Self {
        let mut vec: Vec<P> = iter.into_iter().collect();
        let num_points = vec.len();
        let vec_of_bytes = unsafe {
            let ratio = std::mem::size_of::<P>() / std::mem::size_of::<u8>();

            let length = vec.len() * ratio;
            let capacity = vec.capacity() * ratio;
            let data = vec.as_mut_ptr() as *mut u8;

            std::mem::forget(vec);

            Vec::from_raw_parts(data, length, capacity)
        };
        Self {
            data: vec_of_bytes,
            num_points,
            stride: std::mem::size_of::<P>(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AttributeVector {
    data: Vec<u8>,
    offset_in_point_type: usize,
    size_of_attribute: usize,
}

impl AttributeVector {
    fn from_attribute_with_capacity(attribute: &PointAttributeMember, capacity: usize) -> Self {
        let size_of_attribute = attribute.size() as usize;
        Self {
            data: Vec::with_capacity(capacity * size_of_attribute),
            offset_in_point_type: attribute.offset() as usize,
            size_of_attribute,
        }
    }

    fn get(&self, index: usize, data: &mut [u8]) {
        let slice =
            &self.data[(index * self.size_of_attribute)..((index + 1) * self.size_of_attribute)];
        data.copy_from_slice(slice);
    }

    fn get_ref(&self, range: Range<usize>) -> &[u8] {
        let scaled_range =
            (range.start * self.size_of_attribute)..(range.end * self.size_of_attribute);
        &self.data[scaled_range]
    }

    fn get_mut(&mut self, range: Range<usize>) -> &mut [u8] {
        let scaled_range =
            (range.start * self.size_of_attribute)..(range.end * self.size_of_attribute);
        &mut self.data[scaled_range]
    }

    fn byte_range_within_point(&self) -> Range<usize> {
        self.offset_in_point_type..(self.offset_in_point_type + self.size_of_attribute)
    }

    unsafe fn push_unchecked(&mut self, data: &[u8]) {
        self.data.extend_from_slice(data);
    }

    unsafe fn swap_unchecked(&mut self, from: usize, to: usize) {
        // Caller has to guarantee that from != to
        let from_ptr = self.data.as_mut_ptr().add(from * self.size_of_attribute);
        let to_ptr = self.data.as_mut_ptr().add(to * self.size_of_attribute);
        std::ptr::swap_nonoverlapping(from_ptr, to_ptr, self.size_of_attribute);
    }

    fn len(&self) -> usize {
        self.data.len() / self.size_of_attribute
    }

    fn clear(&mut self) {
        self.data.clear();
    }

    fn grow(&mut self, new_size: usize, value: &[u8]) {
        let points_to_add = new_size - self.len();
        self.data.extend(
            value
                .into_iter()
                .cycle()
                .take(points_to_add * self.size_of_attribute),
        );
    }

    fn truncate(&mut self, new_size: usize) {
        self.data.truncate(new_size * self.size_of_attribute);
    }

    /// View the data for this AttributeVector as a slice of type `T`
    ///
    /// # Safety
    ///
    /// The caller has to make sure that the datatype of this AttributeVector is `T`
    unsafe fn view_as<T: PrimitiveType>(&mut self) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut T, self.len())
    }
}

/// Applies the given permutation to the given range of attribute values as raw bytes. This operation needs to know the datatype of
/// the attribute
///
/// # Safety
///
/// The caller has to make sure that `datatype` actually matches the datatype stored in the attribute_range
unsafe fn apply_permutation_to_attribute_range(
    attribute_range: &mut [u8],
    permutation: &mut Permutation,
    datatype: PointAttributeDataType,
) {
    match datatype {
        PointAttributeDataType::U8 => {
            permutation.apply_slice_in_place(byte_slice_cast_mut::<u8>(attribute_range))
        }
        PointAttributeDataType::I8 => {
            permutation.apply_slice_in_place(byte_slice_cast_mut::<i8>(attribute_range))
        }
        PointAttributeDataType::U16 => {
            permutation.apply_slice_in_place(byte_slice_cast_mut::<u16>(attribute_range))
        }
        PointAttributeDataType::I16 => {
            permutation.apply_slice_in_place(byte_slice_cast_mut::<i16>(attribute_range))
        }
        PointAttributeDataType::U32 => {
            permutation.apply_slice_in_place(byte_slice_cast_mut::<u32>(attribute_range))
        }
        PointAttributeDataType::I32 => {
            permutation.apply_slice_in_place(byte_slice_cast_mut::<i32>(attribute_range))
        }
        PointAttributeDataType::U64 => {
            permutation.apply_slice_in_place(byte_slice_cast_mut::<u64>(attribute_range))
        }
        PointAttributeDataType::I64 => {
            permutation.apply_slice_in_place(byte_slice_cast_mut::<i64>(attribute_range))
        }
        PointAttributeDataType::F32 => {
            permutation.apply_slice_in_place(byte_slice_cast_mut::<f32>(attribute_range))
        }
        PointAttributeDataType::F64 => {
            permutation.apply_slice_in_place(byte_slice_cast_mut::<f64>(attribute_range))
        }
        PointAttributeDataType::Bool => {
            permutation.apply_slice_in_place(byte_slice_cast_mut::<bool>(attribute_range))
        }
        PointAttributeDataType::Vec3u8 => {
            permutation.apply_slice_in_place(byte_slice_cast_mut::<Vector3<u8>>(attribute_range))
        }
        PointAttributeDataType::Vec3u16 => {
            permutation.apply_slice_in_place(byte_slice_cast_mut::<Vector3<u16>>(attribute_range))
        }
        PointAttributeDataType::Vec3f32 => {
            permutation.apply_slice_in_place(byte_slice_cast_mut::<Vector3<f32>>(attribute_range))
        }
        PointAttributeDataType::Vec3f64 => {
            permutation.apply_slice_in_place(byte_slice_cast_mut::<Vector3<f64>>(attribute_range))
        }
        PointAttributeDataType::Vec4u8 => {
            permutation.apply_slice_in_place(byte_slice_cast_mut::<Vector4<u8>>(attribute_range))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnarStorage {
    attributes: HashMap<&'static str, AttributeVector>,
    point_layout: PointLayout,
    num_points: usize,
    size_of_single_point: usize,
}

impl ColumnarStorage {
    pub fn from_layout(point_layout: &PointLayout) -> Self {
        Self::from_layout_with_capacity(point_layout, 0)
    }

    pub fn from_layout_with_capacity(point_layout: &PointLayout, capacity: usize) -> Self {
        Self {
            attributes: point_layout
                .attributes()
                .map(|attribute| {
                    (
                        attribute.name(),
                        AttributeVector::from_attribute_with_capacity(attribute, capacity),
                    )
                })
                .collect(),
            point_layout: point_layout.clone(),
            num_points: 0,
            size_of_single_point: point_layout.size_of_point_entry() as usize,
        }
    }
}

impl BufferStorage for ColumnarStorage {
    fn len(&self) -> usize {
        self.num_points
    }

    fn get(&self, index: usize, data: &mut [u8]) {
        for (_, attribute) in &self.attributes {
            let dst_slice = &mut data[attribute.byte_range_within_point()];
            attribute.get(index, dst_slice);
        }
    }

    fn get_attribute(&self, attribute: &PointAttributeMember, index: usize, data: &mut [u8]) {
        self.attributes
            .get(attribute.name())
            .expect("Attribute not found in this storage")
            .get(index, data);
    }
}

impl BufferStorageMut for ColumnarStorage {
    fn push(&mut self, point: &[u8]) {
        assert_eq!(point.len(), self.size_of_single_point);
        for (_, attribute) in &mut self.attributes {
            let src_slice = &point[attribute.byte_range_within_point()];
            unsafe {
                attribute.push_unchecked(src_slice);
            }
        }
        self.num_points += 1;
    }

    fn push_many(&mut self, points: &[u8]) {
        assert_eq!(0, points.len() % self.size_of_single_point);
        let count = points.len() / self.size_of_single_point;
        for (_, attribute) in &mut self.attributes {
            for idx in 0..count {
                let point_slice = &points[(idx * self.size_of_single_point)..];
                let attribute_slice = &point_slice[attribute.byte_range_within_point()];
                unsafe {
                    attribute.push_unchecked(attribute_slice);
                }
            }
        }
        self.num_points += count;
    }

    fn swap(&mut self, from_index: usize, to_index: usize) {
        if from_index == to_index {
            return;
        }

        // Safe because we check that from_index != to_index
        unsafe {
            self.attributes
                .values_mut()
                .for_each(|attribute| attribute.swap_unchecked(from_index, to_index));
        }
    }

    fn clear(&mut self) {
        self.attributes
            .values_mut()
            .for_each(|attribute| attribute.clear());
        self.num_points = 0;
    }

    fn resize(&mut self, new_size: usize, value: &[u8]) {
        match new_size.cmp(&self.num_points) {
            Ordering::Less => self
                .attributes
                .values_mut()
                .for_each(|attribute| attribute.truncate(new_size)),
            Ordering::Greater => self.attributes.values_mut().for_each(|attribute| {
                let value_slice = &value[attribute.byte_range_within_point()];
                attribute.grow(new_size, value_slice);
            }),
            _ => (),
        }
        self.num_points = new_size;
    }
}

impl BufferStorageColumnar for ColumnarStorage {
    fn get_attribute_ref(&self, attribute: &PointAttributeMember, index: usize) -> &[u8] {
        self.attributes
            .get(attribute.name())
            .expect("Attribute not found in ColumnarStorage")
            .get_ref(index..(index + 1))
    }

    fn get_attribute_range_ref(
        &self,
        attribute: &PointAttributeMember,
        indices: Range<usize>,
    ) -> &[u8] {
        self.attributes
            .get(attribute.name())
            .expect("Attribute not found in ColumnarStorage")
            .get_ref(indices)
    }

    fn point_layout(&self) -> &PointLayout {
        &self.point_layout
    }
}

impl BufferStorageColumnarMut for ColumnarStorage {
    fn get_attribute_mut(&mut self, attribute: &PointAttributeMember, index: usize) -> &mut [u8] {
        self.attributes
            .get_mut(attribute.name())
            .expect("Attribute not found in ColumnarStorage")
            .get_mut(index..(index + 1))
    }

    fn get_attribute_range_mut(
        &mut self,
        attribute: &PointAttributeMember,
        indices: Range<usize>,
    ) -> &mut [u8] {
        self.attributes
            .get_mut(attribute.name())
            .expect("Attribute not found in ColumnarStorage")
            .get_mut(indices)
    }

    unsafe fn sort_by_attribute<C, T: PrimitiveType>(
        &mut self,
        attribute: &PointAttributeMember,
        compare: C,
    ) where
        C: Fn(&T, &T) -> Ordering,
    {
        let attribute_data = self
            .attributes
            .get_mut(attribute.name())
            .expect("Attribute not found in ColumnarStorage");
        let mut permutation = permutation::sort_by(attribute_data.view_as::<T>(), compare);
        let datatypes = self
            .attributes
            .keys()
            .map(|name| {
                self.point_layout
                    .get_attribute_by_name(*name)
                    .expect("Attribute not found in PointLayout")
                    .datatype()
            })
            .collect::<Vec<_>>();
        self.attributes
            .iter_mut()
            .zip(datatypes.into_iter())
            .for_each(|((_, data), datatype)| {
                apply_permutation_to_attribute_range(
                    data.data.as_mut_slice(),
                    &mut permutation,
                    datatype,
                );
            });
    }
}

impl<P: PointType> FromIterator<P> for ColumnarStorage {
    fn from_iter<T: IntoIterator<Item = P>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let capacity = iter.size_hint().1.unwrap_or(0);
        let layout = P::layout();
        let mut storage = ColumnarStorage::from_layout_with_capacity(&layout, capacity);
        // Safe because we know the PointLayout of type P
        unsafe {
            iter.for_each(|point| storage.push(view_raw_bytes(&point)));
        }
        storage
    }
}

pub trait IndexBuffer<'a> {
    type Output: BufferStorage + 'a;

    fn index(&'a self, range: Range<usize>) -> Self::Output;
}

pub trait IndexBufferMut<'a>: IndexBuffer<'a> {
    type OutputMut: BufferStorage + 'a;

    fn index_mut(&'a mut self, range: Range<usize>) -> Self::OutputMut;
}

impl<'a> IndexBuffer<'a> for VectorStorage {
    type Output = SliceStorage<'a, VectorStorage>;

    fn index(&'a self, range: Range<usize>) -> Self::Output {
        SliceStorage::new(self, range)
    }
}

impl<'a> IndexBufferMut<'a> for VectorStorage {
    type OutputMut = SliceStorageMut<'a, VectorStorage>;

    fn index_mut(&'a mut self, range: Range<usize>) -> Self::OutputMut {
        SliceStorageMut::new(self, range)
    }
}

pub struct SliceStorage<'a, S: BufferStorage> {
    storage: &'a S,
    range: Range<usize>,
}

impl<'a, S: BufferStorage + 'a> SliceStorage<'a, S> {
    pub fn new(storage: &'a S, range: Range<usize>) -> Self {
        Self { storage, range }
    }
}

impl<'a, S: BufferStorage> BufferStorage for SliceStorage<'a, S> {
    fn len(&self) -> usize {
        self.range.len()
    }

    fn get(&self, index: usize, data: &mut [u8]) {
        if index >= self.len() {
            panic!("Index out of bounds");
        }
        self.storage.get(index + self.range.start, data);
    }

    fn get_attribute(&self, attribute: &PointAttributeMember, index: usize, data: &mut [u8]) {
        if index >= self.len() {
            panic!("Index out of bounds");
        }
        self.storage
            .get_attribute(attribute, index + self.range.start, data);
    }
}

impl<'a, S: BufferStorageContiguous> BufferStorageContiguous for SliceStorage<'a, S> {
    fn get_ref(&self, index: usize) -> &[u8] {
        if index >= self.len() {
            panic!("Index out of bounds");
        }
        self.storage.get_ref(index + self.range.start)
    }

    fn get_range_ref(&self, indices: Range<usize>) -> &[u8] {
        if indices.end >= self.len() {
            panic!("Index out of bounds");
        }
        self.storage
            .get_range_ref((indices.start + self.range.start)..(indices.end + self.range.start))
    }
}

impl<'a, S: BufferStorageColumnar> BufferStorageColumnar for SliceStorage<'a, S> {
    fn get_attribute_ref(&self, attribute: &PointAttributeMember, index: usize) -> &[u8] {
        if index >= self.len() {
            panic!("Index out of bounds");
        }
        self.storage
            .get_attribute_ref(attribute, index + self.range.start)
    }

    fn get_attribute_range_ref(
        &self,
        attribute: &PointAttributeMember,
        indices: Range<usize>,
    ) -> &[u8] {
        if indices.end >= self.len() {
            panic!("Index out of bounds");
        }
        self.storage.get_attribute_range_ref(
            attribute,
            (indices.start + self.range.start)..(indices.end + self.range.start),
        )
    }

    fn point_layout(&self) -> &PointLayout {
        self.storage.point_layout()
    }
}

impl<'a, S: BufferStorage + 'a> IndexBuffer<'a> for SliceStorage<'a, S> {
    type Output = SliceStorage<'a, S>;

    fn index(&'a self, range: Range<usize>) -> Self::Output {
        if range.start >= self.range.end || range.end > self.range.end {
            panic!("Range is out of bounds");
        }
        let shifted_range = (self.range.start + range.start)..(self.range.start + range.end);
        SliceStorage::new(self.storage, shifted_range)
    }
}

pub struct SliceStorageMut<'a, S: BufferStorage> {
    storage: &'a mut S,
    range: Range<usize>,
}

impl<'a, S: BufferStorage> SliceStorageMut<'a, S> {
    pub fn new(storage: &'a mut S, range: Range<usize>) -> Self {
        Self { storage, range }
    }
}

impl<'a, S: BufferStorage> BufferStorage for SliceStorageMut<'a, S> {
    fn len(&self) -> usize {
        self.range.len()
    }

    fn get(&self, index: usize, data: &mut [u8]) {
        if index >= self.len() {
            panic!("Index out of bounds");
        }
        self.storage.get(index + self.range.start, data);
    }

    fn get_attribute(&self, attribute: &PointAttributeMember, index: usize, data: &mut [u8]) {
        if index >= self.len() {
            panic!("Index out of bounds");
        }
        self.storage
            .get_attribute(attribute, index + self.range.start, data);
    }
}

impl<'a, S: BufferStorageContiguous> BufferStorageContiguous for SliceStorageMut<'a, S> {
    fn get_ref(&self, index: usize) -> &[u8] {
        if index >= self.len() {
            panic!("Index out of bounds");
        }
        self.storage.get_ref(index + self.range.start)
    }

    fn get_range_ref(&self, indices: Range<usize>) -> &[u8] {
        if indices.end >= self.len() {
            panic!("Index out of bounds");
        }
        self.storage
            .get_range_ref((indices.start + self.range.start)..(indices.end + self.range.start))
    }
}

impl<'a, S: BufferStorageContiguousMut> BufferStorageContiguousMut for SliceStorageMut<'a, S> {
    fn get_mut(&mut self, index: usize) -> &mut [u8] {
        if index >= self.len() {
            panic!("Index out of bounds");
        }
        self.storage.get_mut(index + self.range.start)
    }

    fn get_range_mut(&mut self, indices: Range<usize>) -> &mut [u8] {
        if indices.end >= self.len() {
            panic!("Index out of bounds");
        }
        self.storage
            .get_range_mut((indices.start + self.range.start)..(indices.end + self.range.start))
    }

    unsafe fn sort_by<C, T: PointType>(&mut self, compare: C)
    where
        C: Fn(&T, &T) -> Ordering,
    {
        let range_mut_t = unsafe {
            let range_mut = self.get_range_mut(0..self.len());
            std::slice::from_raw_parts_mut(range_mut.as_mut_ptr() as *mut T, self.len())
        };
        range_mut_t.sort_by(compare);
    }
}

impl<'a, S: BufferStorageColumnar> BufferStorageColumnar for SliceStorageMut<'a, S> {
    fn get_attribute_ref(&self, attribute: &PointAttributeMember, index: usize) -> &[u8] {
        if index >= self.len() {
            panic!("Index out of bounds");
        }
        self.storage
            .get_attribute_ref(attribute, index + self.range.start)
    }

    fn get_attribute_range_ref(
        &self,
        attribute: &PointAttributeMember,
        indices: Range<usize>,
    ) -> &[u8] {
        if indices.end >= self.len() {
            panic!("Index out of bounds");
        }
        self.storage.get_attribute_range_ref(
            attribute,
            (indices.start + self.range.start)..(indices.end + self.range.start),
        )
    }

    fn point_layout(&self) -> &PointLayout {
        &self.storage.point_layout()
    }
}

impl<'a, S: BufferStorageColumnarMut> BufferStorageColumnarMut for SliceStorageMut<'a, S> {
    fn get_attribute_mut(&mut self, attribute: &PointAttributeMember, index: usize) -> &mut [u8] {
        if index >= self.len() {
            panic!("Index out of bounds");
        }
        self.storage
            .get_attribute_mut(attribute, index + self.range.start)
    }

    fn get_attribute_range_mut(
        &mut self,
        attribute: &PointAttributeMember,
        indices: Range<usize>,
    ) -> &mut [u8] {
        if indices.end >= self.len() {
            panic!("Index out of bounds");
        }
        self.storage.get_attribute_range_mut(
            attribute,
            (indices.start + self.range.start)..(indices.end + self.range.start),
        )
    }

    unsafe fn sort_by_attribute<C, T: PrimitiveType>(
        &mut self,
        attribute: &PointAttributeMember,
        compare: C,
    ) where
        C: Fn(&T, &T) -> Ordering,
    {
        let attribute_data =
            byte_slice_cast::<T>(self.get_attribute_range_ref(attribute, 0..self.len()));
        let mut permutation = permutation::sort_by(attribute_data, compare);
        self.point_layout()
            .clone() //We have to clone the PointLayout because get_attribute_range_mut requires a borrow to the PointAttributeMember, which comes from self.point_layout()...
            .attributes()
            .for_each(|attribute_in_layout| {
                apply_permutation_to_attribute_range(
                    self.get_attribute_range_mut(attribute_in_layout, 0..self.len()),
                    &mut permutation,
                    attribute_in_layout.datatype(),
                );
            });
    }
}

impl<'a, S: BufferStorage + 'a> IndexBuffer<'a> for SliceStorageMut<'a, S> {
    type Output = SliceStorage<'a, S>;

    fn index(&'a self, range: Range<usize>) -> Self::Output {
        if range.start >= self.range.end || range.end > self.range.end {
            panic!("Range is out of bounds");
        }
        let shifted_range = (self.range.start + range.start)..(self.range.start + range.end);
        SliceStorage::new(self.storage, shifted_range)
    }
}

impl<'a, S: BufferStorage + 'a> IndexBufferMut<'a> for SliceStorageMut<'a, S> {
    type OutputMut = SliceStorageMut<'a, S>;

    fn index_mut(&'a mut self, range: Range<usize>) -> Self::OutputMut {
        if range.start >= self.range.end || range.end > self.range.end {
            panic!("Range is out of bounds");
        }
        let shifted_range = (self.range.start + range.start)..(self.range.start + range.end);
        SliceStorageMut::new(self.storage, shifted_range)
    }
}

#[cfg(test)]
mod tests {
    use pasture_derive::PointType;
    use rand::{distributions::Standard, prelude::Distribution, thread_rng, Rng};

    use crate::{
        containers_v2::BufferViewRef,
        layout::attributes::{GPS_TIME, INTENSITY},
        util::view_raw_bytes,
    };

    use super::*;

    #[derive(PointType, Debug, Copy, Clone, PartialEq)]
    #[repr(C, packed)]
    struct PointType1 {
        #[pasture(BUILTIN_POSITION_3D)]
        position: Vector3<f64>,
        #[pasture(BUILTIN_INTENSITY)]
        intensity: u16,
        #[pasture(BUILTIN_GPS_TIME)]
        gps_time: f64,
    }

    impl PartialOrd for PointType1 {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            let self_intensity = self.intensity;
            let other_intensity = other.intensity;
            self_intensity.partial_cmp(&other_intensity)
        }
    }

    impl Distribution<PointType1> for Standard {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> PointType1 {
            PointType1 {
                position: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
                intensity: rng.gen(),
                gps_time: rng.gen(),
            }
        }
    }

    unsafe fn get_attribute<'a, T: PointType>(
        attribute: &PointAttributeMember,
        point: &'a T,
    ) -> &'a [u8] {
        let offset = attribute.offset() as usize;
        let size = attribute.size() as usize;
        std::slice::from_raw_parts((point as *const T as *const u8).add(offset), size)
    }

    fn assert_sorted_by<T, C: Fn(&T, &T) -> Ordering>(data: &[T], compare: C) {
        for (first, second) in data.iter().zip(data.iter().skip(1)) {
            match compare(first, second) {
                Ordering::Greater => panic!("Range not properly sorted"),
                _ => (),
            }
        }
    }

    fn reference_data<T: PointType>(count: usize) -> Vec<T>
    where
        Standard: Distribution<T>,
    {
        let mut rng = thread_rng();
        (0..count).map(|_| rng.gen::<T>()).collect()
    }

    fn test_buffer_storage<T: BufferStorage, U: PointType + PartialEq + std::fmt::Debug>(
        storage: &T,
        expected_data: &[U],
    ) {
        assert_eq!(storage.len(), expected_data.len());
        let mut actual_point_data: Vec<u8> = vec![0; std::mem::size_of::<U>()];
        for (index, expected_point) in expected_data.iter().enumerate() {
            storage.get(index, &mut actual_point_data);
            let actual_point = unsafe { &byte_slice_cast::<U>(&actual_point_data)[0] };
            assert_eq!(expected_point, actual_point);
        }

        let point_layout = U::layout();
        for attribute in point_layout.attributes() {
            let mut attribute_data: Vec<u8> = vec![0; attribute.size() as usize];
            for (index, expected_point) in expected_data.iter().enumerate() {
                let expected_attribute = unsafe { get_attribute(attribute, expected_point) };
                storage.get_attribute(attribute, index, &mut attribute_data);
                assert_eq!(expected_attribute, attribute_data.as_slice());
            }
        }
    }

    unsafe fn test_buffer_storage_mut<
        T: BufferStorageMut + Clone + PartialEq + std::fmt::Debug,
        U: PointType + Clone + PartialEq + std::fmt::Debug,
    >(
        storage: &T,
        initial_data: &[U],
    ) where
        Standard: Distribution<U>,
    {
        let mut cloned_storage = storage.clone();
        assert_eq!(*storage, cloned_storage);

        let new_data = reference_data::<U>(2);
        cloned_storage.push(view_raw_bytes(&new_data[0]));
        assert_eq!(cloned_storage.len(), initial_data.len() + 1);

        cloned_storage.push_many(std::slice::from_raw_parts(
            new_data.as_ptr() as *const u8,
            new_data.len() * std::mem::size_of::<U>(),
        ));

        let mut expected_data = initial_data.to_vec();
        expected_data.push(new_data[0].clone());
        expected_data.extend(new_data.iter().cloned());

        let mut actual_data = BufferViewRef::from_storage_and_layout(&cloned_storage, &U::layout())
            .into_iter()
            .collect::<Vec<U>>();
        assert_eq!(expected_data, actual_data);

        cloned_storage.swap(0, cloned_storage.len() - 1);
        let expected_last_idx = expected_data.len() - 1;
        expected_data.swap(0, expected_last_idx);
        actual_data = BufferViewRef::from_storage_and_layout(&cloned_storage, &U::layout())
            .into_iter()
            .collect::<Vec<U>>();
        assert_eq!(expected_data, actual_data);

        let points_to_add = 4;
        cloned_storage.resize(
            cloned_storage.len() + points_to_add,
            view_raw_bytes(&new_data[0]),
        );

        expected_data.extend(std::iter::repeat(new_data[0].clone()).take(points_to_add));
        actual_data = BufferViewRef::from_storage_and_layout(&cloned_storage, &U::layout())
            .into_iter()
            .collect::<Vec<U>>();
        assert_eq!(expected_data, actual_data);

        // Value with which we resize doesn't matter because we know that this resize call will shrink!
        cloned_storage.resize(1, &[]);
        expected_data.resize(1, new_data[0].clone());
        actual_data = BufferViewRef::from_storage_and_layout(&cloned_storage, &U::layout())
            .into_iter()
            .collect::<Vec<U>>();
        assert_eq!(expected_data, actual_data);

        cloned_storage.clear();
        assert_eq!(0, cloned_storage.len());
        actual_data = BufferViewRef::from_storage_and_layout(&cloned_storage, &U::layout())
            .into_iter()
            .collect::<Vec<U>>();
        assert_eq!(Vec::<U>::default(), actual_data);
    }

    unsafe fn test_buffer_storage_contiguous<
        T: BufferStorageContiguous,
        U: PointType + Clone + PartialEq + std::fmt::Debug,
    >(
        storage: &T,
        expected_data: &[U],
    ) {
        for (index, expected_point) in expected_data.iter().enumerate() {
            let actual_point = &byte_slice_cast::<U>(storage.get_ref(index))[0];
            assert_eq!(expected_point, actual_point);
        }

        let actual_data = byte_slice_cast::<U>(storage.get_range_ref(0..storage.len()));
        assert_eq!(expected_data, actual_data);
    }

    unsafe fn test_buffer_storage_contiguous_mut<
        T: BufferStorageContiguousMut + Clone + PartialEq + std::fmt::Debug,
        U: PointType + Clone + PartialEq + PartialOrd + std::fmt::Debug,
    >(
        storage: &T,
    ) where
        Standard: Distribution<U>,
    {
        let mut cloned_storage = storage.clone();
        assert_eq!(*storage, cloned_storage);

        let two_new_points = reference_data::<U>(2);
        for index in 0..storage.len() {
            let point = &mut byte_slice_cast_mut::<U>(cloned_storage.get_mut(index))[0];
            *point = two_new_points[0].clone();
        }

        let mut expected_data = std::iter::repeat(two_new_points[0].clone())
            .take(storage.len())
            .collect::<Vec<_>>();
        let mut actual_data = BufferViewRef::from_storage_and_layout(&cloned_storage, &U::layout())
            .into_iter()
            .collect::<Vec<U>>();
        assert_eq!(expected_data, actual_data);

        let points_mut = byte_slice_cast_mut::<U>(cloned_storage.get_range_mut(0..storage.len()));
        for point in points_mut {
            *point = two_new_points[1].clone();
        }

        expected_data = std::iter::repeat(two_new_points[1].clone())
            .take(storage.len())
            .collect::<Vec<_>>();
        actual_data = BufferViewRef::from_storage_and_layout(&cloned_storage, &U::layout())
            .into_iter()
            .collect::<Vec<U>>();
        assert_eq!(expected_data, actual_data);

        cloned_storage.sort_by(|a: &U, b: &U| a.partial_cmp(b).unwrap());
        actual_data = BufferViewRef::from_storage_and_layout(&cloned_storage, &U::layout())
            .into_iter()
            .collect::<Vec<U>>();
        assert_sorted_by(&actual_data, |a, b| a.partial_cmp(b).unwrap());
    }

    unsafe fn test_buffer_storage_columnar<T: BufferStorageColumnar, U: PointType>(
        storage: &T,
        expected_data: &[U],
    ) {
        let layout = U::layout();
        for attribute in layout.attributes() {
            let expected_attribute_data = expected_data
                .iter()
                .map(|point| get_attribute(attribute, point))
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            let actual_attribute_data = (0..storage.len())
                .map(|index| storage.get_attribute_ref(attribute, index))
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            let actual_attribute_data_ref =
                storage.get_attribute_range_ref(attribute, 0..storage.len());

            assert_eq!(expected_attribute_data, actual_attribute_data);
            assert_eq!(
                expected_attribute_data.as_slice(),
                actual_attribute_data_ref
            );
        }
    }

    unsafe fn test_buffer_storage_columnar_mut<
        T: BufferStorageColumnarMut + Clone + PartialEq + std::fmt::Debug,
        U: PointType + Clone + PartialEq + std::fmt::Debug,
    >(
        storage: &T,
    ) where
        Standard: Distribution<U>,
    {
        let mut cloned_storage = storage.clone();
        assert_eq!(*storage, cloned_storage);

        // Overwrite all attributes of each point with new data
        let new_data = reference_data::<U>(cloned_storage.len());
        let layout = U::layout();
        for attribute in layout.attributes() {
            for index in 0..cloned_storage.len() {
                let new_attribute_data = get_attribute(attribute, &new_data[index]);
                let attribute_data_in_storage = cloned_storage.get_attribute_mut(attribute, index);
                attribute_data_in_storage.copy_from_slice(new_attribute_data);
            }
        }

        let actual_data = BufferViewRef::from_storage_and_layout(&cloned_storage, &layout)
            .into_iter()
            .collect::<Vec<U>>();
        assert_eq!(new_data, actual_data);

        // Back to old data but with get_attribute_range_mut
        for attribute in layout.attributes() {
            let old_attribute_data = storage.get_attribute_range_ref(attribute, 0..storage.len());
            let attribute_data_in_storage =
                cloned_storage.get_attribute_range_mut(attribute, 0..cloned_storage.len());
            attribute_data_in_storage.copy_from_slice(old_attribute_data);
        }

        let expected_data = BufferViewRef::from_storage_and_layout(storage, &layout)
            .into_iter()
            .collect::<Vec<U>>();
        let actual_data = BufferViewRef::from_storage_and_layout(&cloned_storage, &layout)
            .into_iter()
            .collect::<Vec<U>>();
        assert_eq!(expected_data, actual_data);
    }

    unsafe fn test_vector_storage_with_type<
        T: PointType + Clone + PartialEq + PartialOrd + std::fmt::Debug,
    >()
    where
        Standard: Distribution<T>,
    {
        {
            let empty_storage = VectorStorage::from_layout(&T::layout());
            let empty_data: &[T] = &[];
            test_buffer_storage(&empty_storage, empty_data);
            test_buffer_storage_mut(&empty_storage, empty_data);
            test_buffer_storage_contiguous(&empty_storage, empty_data);
            test_buffer_storage_contiguous_mut(&empty_storage);
        }

        {
            const COUNT: usize = 16;
            let expected_data = reference_data::<T>(COUNT);
            let storage = expected_data.iter().cloned().collect::<VectorStorage>();
            test_buffer_storage(&storage, &expected_data);
            test_buffer_storage_mut(&storage, &expected_data);
            test_buffer_storage_contiguous(&storage, &expected_data);
            test_buffer_storage_contiguous_mut(&storage);
        }
    }

    unsafe fn test_columnar_storage_with_type<
        T: PointType + Clone + PartialEq + PartialOrd + std::fmt::Debug,
    >()
    where
        Standard: Distribution<T>,
    {
        {
            let empty_storage = ColumnarStorage::from_layout(&T::layout());
            let empty_data: &[T] = &[];
            test_buffer_storage(&empty_storage, empty_data);
            test_buffer_storage_mut(&empty_storage, empty_data);
            test_buffer_storage_columnar(&empty_storage, empty_data);
            test_buffer_storage_columnar_mut(&empty_storage);
        }

        {
            const COUNT: usize = 16;
            let expected_data = reference_data::<T>(COUNT);
            let storage = expected_data.iter().cloned().collect::<ColumnarStorage>();
            test_buffer_storage(&storage, &expected_data);
            test_buffer_storage_mut(&storage, &expected_data);
            test_buffer_storage_columnar(&storage, &expected_data);
            test_buffer_storage_columnar_mut(&storage);
        }
    }

    #[test]
    fn test_buffers() {
        unsafe {
            test_vector_storage_with_type::<PointType1>();
            test_columnar_storage_with_type::<PointType1>();
        }
    }

    #[test]
    fn test_columnar_storage_sort_by_attribute() {
        const COUNT: usize = 16;
        let reference_data = reference_data::<PointType1>(COUNT);
        let mut storage = reference_data.iter().copied().collect::<ColumnarStorage>();

        let layout = PointType1::layout();
        unsafe {
            storage.sort_by_attribute(
                layout
                    .get_attribute(&INTENSITY)
                    .expect("Intensity attribute not found"),
                |a: &u16, b: &u16| a.cmp(b),
            );
        }
        let mut actual_data = BufferViewRef::from_storage_and_layout(&storage, &layout)
            .into_iter()
            .collect::<Vec<PointType1>>();
        assert_sorted_by(&actual_data, |a, b| {
            let intensity_a = a.intensity;
            let intensity_b = b.intensity;
            intensity_a.cmp(&intensity_b)
        });

        unsafe {
            storage.sort_by_attribute(
                layout
                    .get_attribute(&GPS_TIME)
                    .expect("GPS_TIME attribute not found"),
                |a: &f64, b: &f64| a.total_cmp(b),
            );
        }

        actual_data = BufferViewRef::from_storage_and_layout(&storage, &layout)
            .into_iter()
            .collect::<Vec<PointType1>>();
        assert_sorted_by(&actual_data, |a, b| {
            let time_a = a.gps_time;
            let time_b = b.gps_time;
            time_a.total_cmp(&time_b)
        });
    }
}
