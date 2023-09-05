use crate::{
    containers::{
        BorrowedBuffer, BorrowedMutBuffer, ColumnarBuffer, ColumnarBufferMut, MakeBufferFromLayout,
        OwningBuffer,
    },
    layout::{PointAttributeDefinition, PointAttributeMember, PointLayout, PrimitiveType},
};

use super::{get_generic_converter, AttributeConversionFn};

/// Function that transform a single point attribute in its raw, untyped form
type AttributeTransformFn = Box<dyn Fn(&mut [u8])>;

/// Converts the given transform function into an untyped version that works on raw attribute memory
fn to_untyped_transform_fn<T: PrimitiveType, F: Fn(T) -> T + 'static>(
    transform_fn: F,
) -> AttributeTransformFn {
    let untyped_transform_fn = move |attribute_memory: &mut [u8]| {
        let attribute_ptr_typed = attribute_memory.as_mut_ptr() as *mut T;
        // Is safe as long as the caller ensures that `attribute_memory` contains valid memory for `T`
        // The `BufferLayoutConverter` ensures this because we perform a attribute data type check above
        unsafe {
            let attribute_value = attribute_ptr_typed.read_unaligned();
            let transformed_value = transform_fn(attribute_value);
            attribute_ptr_typed.write_unaligned(transformed_value);
        }
    };
    Box::new(untyped_transform_fn)
}

pub struct AttributeMapping<'a> {
    target_attribute: &'a PointAttributeMember,
    source_attribute: &'a PointAttributeMember,
    converter: Option<AttributeConversionFn>,
    transformation: Option<AttributeTransformFn>,
}

impl<'a> AttributeMapping<'a> {
    /// Returns the maximum number of bytes required to convert the data for this mapping
    pub(crate) fn required_buffer_size(&self) -> usize {
        self.source_attribute
            .size()
            .max(self.target_attribute.size()) as usize
    }
}

/// A converter that can convert a point buffer from one `PointLayout` into another
pub struct BufferLayoutConverter<'a> {
    from_layout: &'a PointLayout,
    to_layout: &'a PointLayout,
    mappings: Vec<AttributeMapping<'a>>,
}

impl<'a> BufferLayoutConverter<'a> {
    /// Creates a new `BufferLayoutConverter` from `from_layout` into `to_layout`. This generates default mappings for each
    /// attribute in `to_layout` following the rules of [`crate::layout::conversion::get_converter_for_attributes`]
    ///
    /// # Panics
    ///
    /// If any `PointAttributeMember` in `to_layout` is not present (by name) in `from_layout`. If you want to allow missing
    /// attributes in `from_layout` and fill their values with default values, use [`Self::for_layouts_with_default`] instead!
    pub fn for_layouts(from_layout: &'a PointLayout, to_layout: &'a PointLayout) -> Self {
        let default_mappings = to_layout.attributes().map(|to_attribute| {
            let from_attribute = from_layout.get_attribute(to_attribute.attribute_definition()).expect("Attribute not found in `from_layout`! When calling `BufferLayoutConverter::for_layouts`, the source PointLayout must contain all attributes from the target PointLayout. If you want to use default values for attributes that are not present in the source layout, use `BufferLayoutConverter::for_layouts_with_default` instead!");
            Self::make_default_mapping(from_attribute, to_attribute)
        }).collect();
        Self {
            from_layout,
            to_layout,
            mappings: default_mappings,
        }
    }

    /// Like [`Self::for_layouts`], but if an attribute from `to_layout` is not present in `from_layout`, the conversion still
    /// happens and will fill the target buffer with default values for this attribute instead!
    pub fn for_layouts_with_default(
        from_layout: &'a PointLayout,
        to_layout: &'a PointLayout,
    ) -> Self {
        let default_mappings = to_layout
            .attributes()
            .filter_map(|to_attribute| {
                from_layout
                    .get_attribute(to_attribute.attribute_definition())
                    .map(|from_attribute| Self::make_default_mapping(from_attribute, to_attribute))
            })
            .collect();
        Self {
            from_layout,
            to_layout,
            mappings: default_mappings,
        }
    }

    /// Sets a custom mapping from `from_attribute` to `to_attribute`. This overrides the default mapping function for
    /// `to_attribute`. It allows mapping attributes with different names to each other and in particular allows the
    /// conversion to be surjective: Multiple attributes in `to_attribute` can be converted from the same attribute in
    /// `from_attribute`. An example where this is useful are bit-flag attributes, such as in the LAS file format, which
    /// semantically represent multiple attributes, but are represented as a single attribute in pasture.
    ///
    ///
    /// # Panics
    ///
    /// If `from_attribute` is not part of the source `PointLayout`.
    /// If `to_attribute` is not part of the target `PointLayout`.
    pub fn set_custom_mapping(
        &mut self,
        from_attribute: &PointAttributeDefinition,
        to_attribute: &PointAttributeDefinition,
    ) {
        let from_attribute_member = self
            .from_layout
            .get_attribute(from_attribute)
            .expect("from_attribute not found in source PointLayout");
        let to_attribute_member = self
            .to_layout
            .get_attribute(to_attribute)
            .expect("to_attribute not found in target PointLayout");

        if let Some(previous_mapping) = self
            .mappings
            .iter_mut()
            .find(|mapping| mapping.target_attribute.attribute_definition() == to_attribute)
        {
            *previous_mapping =
                Self::make_default_mapping(from_attribute_member, to_attribute_member);
        } else {
            self.mappings.push(Self::make_default_mapping(
                from_attribute_member,
                to_attribute_member,
            ));
        }
    }

    /// Like [`Self::set_custom_mapping`], but transforms the target attribute value using the `transform_fn`
    ///
    /// # Panics
    ///
    /// If `from_attribute` is not part of the source `PointLayout`.
    /// If `to_attribute` is not part of the target `PointLayout`.
    /// If `T::data_type()` does not match `to_attribute.datatype()`.
    pub fn set_custom_mapping_with_transformation<T: PrimitiveType, F: Fn(T) -> T>(
        &mut self,
        from_attribute: &PointAttributeDefinition,
        to_attribute: &PointAttributeDefinition,
        transform_fn: F,
    ) where
        F: 'static,
    {
        let from_attribute_member = self
            .from_layout
            .get_attribute(from_attribute)
            .expect("from_attribute not found in source PointLayout");
        let to_attribute_member = self
            .to_layout
            .get_attribute(to_attribute)
            .expect("to_attribute not found in target PointLayout");
        assert_eq!(T::data_type(), to_attribute_member.datatype());

        if let Some(previous_mapping) = self
            .mappings
            .iter_mut()
            .find(|mapping| mapping.target_attribute.attribute_definition() == to_attribute)
        {
            *previous_mapping = Self::make_transformed_mapping(
                from_attribute_member,
                to_attribute_member,
                transform_fn,
            );
        } else {
            self.mappings.push(Self::make_transformed_mapping(
                from_attribute_member,
                to_attribute_member,
                transform_fn,
            ));
        }
    }

    /// Convert the `source_buffer` into the target `PointLayout` and return its data as a new buffer of
    /// type `OutBuffer`
    ///
    /// # Panics
    ///
    /// If `source_buffer.point_layout()` does not match the source `PointLayout` used to construct this `BufferLayoutConverter`
    pub fn convert<
        'b,
        'c,
        'd,
        OutBuffer: OwningBuffer<'c> + MakeBufferFromLayout<'c> + 'c,
        InBuffer: BorrowedBuffer<'b>,
    >(
        &self,
        source_buffer: &'d InBuffer,
    ) -> OutBuffer
    where
        'b: 'd,
    {
        let mut target_buffer = OutBuffer::new_from_layout(self.to_layout.clone());
        target_buffer.resize(source_buffer.len());
        self.convert_into(source_buffer, &mut target_buffer);
        target_buffer
    }

    /// Like [`convert`], but converts into an existing buffer instead of allocating a new buffer
    ///
    /// # Panics
    ///
    /// If `source_buffer.point_layout()` does not match the source `PointLayout` used to construct this `BufferLayoutConverter`
    /// If `target_buffer.point_layout()` does not match the target `PointLayout` used to construct this `BufferLayoutConverter`
    /// If `target_buffer.len()` is less than `source_buffer.len()`
    pub fn convert_into<'b, 'c, 'd, 'e>(
        &self,
        source_buffer: &'c impl BorrowedBuffer<'b>,
        target_buffer: &'e mut impl BorrowedMutBuffer<'d>,
    ) where
        'b: 'c,
        'd: 'e,
    {
        assert_eq!(source_buffer.point_layout(), self.from_layout);
        assert_eq!(target_buffer.point_layout(), self.to_layout);
        assert!(target_buffer.len() >= source_buffer.len());

        let max_attribute_size = self
            .mappings
            .iter()
            .map(|mapping| mapping.required_buffer_size())
            .max();
        if let Some(max_attribute_size) = max_attribute_size {
            // TODO Implement four cases:
            // 1) Source and target are both columnar -> can get attribute ranges for source and target, easy
            // 2) Source is columnar, target not -> can at least get attribute ranges for source
            // 3) Target is columnar, source not -> can at least get attribute ranges for target
            // 4) Neither is columnar -> Use the base-case implementation already present here!
            match (source_buffer.as_columnar(), target_buffer.as_columnar_mut()) {
                (Some(source_buffer), Some(target_buffer)) => {
                    self.convert_columnar_to_columnar(source_buffer, target_buffer);
                }
                (Some(source_buffer), None) => {
                    self.convert_columnar_to_general(
                        source_buffer,
                        target_buffer,
                        max_attribute_size,
                    );
                }
                (None, Some(target_buffer)) => {
                    self.convert_general_to_columnar(
                        source_buffer,
                        target_buffer,
                        max_attribute_size,
                    );
                }
                (None, None) => self.convert_general_to_general(
                    source_buffer,
                    target_buffer,
                    max_attribute_size,
                ),
            }
        }
    }

    /// Make a default (i.e. untransformed) mapping from `from_attribute` to `to_attribute`. This allows
    /// that the two attributes have different names, so long as there is a valid conversion from the
    /// datatype of `from_attribute` to the datatype of `to_attribute`
    ///
    /// # Panics
    ///
    /// If there is no possible conversion from the datatype of `from_attribute` into the datatype of `to_attribute`
    fn make_default_mapping(
        from_attribute: &'a PointAttributeMember,
        to_attribute: &'a PointAttributeMember,
    ) -> AttributeMapping<'a> {
        if from_attribute.datatype() == to_attribute.datatype() {
            AttributeMapping {
                target_attribute: to_attribute,
                source_attribute: from_attribute,
                converter: None,
                transformation: None,
            }
        } else {
            let from_datatype = from_attribute.datatype();
            let to_datatype = to_attribute.datatype();
            let converter =
                get_generic_converter(from_datatype, to_datatype).unwrap_or_else(|| {
                    panic!(
                        "No conversion from {} to {} possible",
                        from_datatype, to_datatype
                    )
                });
            AttributeMapping {
                target_attribute: to_attribute,
                source_attribute: from_attribute,
                converter: Some(converter),
                transformation: None,
            }
        }
    }

    /// Like `make_transformed_mapping`, but with an additional transformation function that gets applied
    /// to the attribute values during conversion
    ///
    /// # Panics
    ///
    /// If `T::data_type()` does not match `to_attribute.datatype()`
    fn make_transformed_mapping<T: PrimitiveType>(
        from_attribute: &'a PointAttributeMember,
        to_attribute: &'a PointAttributeMember,
        transform_fn: impl Fn(T) -> T + 'static,
    ) -> AttributeMapping<'a> {
        let mut mapping = Self::make_default_mapping(from_attribute, to_attribute);
        mapping.transformation = Some(to_untyped_transform_fn(transform_fn));
        mapping
    }

    fn convert_columnar_to_columnar(
        &self,
        source_buffer: &dyn ColumnarBuffer,
        target_buffer: &mut dyn ColumnarBufferMut,
    ) {
        let num_points = source_buffer.len();
        for mapping in &self.mappings {
            let source_attribute_data = source_buffer.get_attribute_range_ref(
                mapping.source_attribute.attribute_definition(),
                0..num_points,
            );
            if let Some(converter) = mapping.converter {
                let target_attribute_data = target_buffer.get_attribute_range_mut(
                    mapping.target_attribute.attribute_definition(),
                    0..num_points,
                );
                let source_attribute_size = mapping.source_attribute.size() as usize;
                let target_attribute_size = mapping.target_attribute.size() as usize;
                for (source_chunk, target_chunk) in source_attribute_data
                    .chunks_exact(source_attribute_size)
                    .zip(target_attribute_data.chunks_exact_mut(target_attribute_size))
                {
                    // Safe because we guarantee that the slice sizes match and their data comes from the
                    // correct attribute
                    unsafe {
                        converter(source_chunk, target_chunk);
                        if let Some(transformation) = mapping.transformation.as_deref() {
                            transformation(target_chunk);
                        }
                    }
                }
            } else {
                // Safe because if the source and target attributes would not be equal, there would be a
                // converter
                unsafe {
                    target_buffer.set_attribute_range(
                        mapping.target_attribute.attribute_definition(),
                        0..num_points,
                        source_attribute_data,
                    );
                }
                if let Some(transformation) = mapping.transformation.as_deref() {
                    let target_attribute_range = target_buffer.get_attribute_range_mut(
                        mapping.target_attribute.attribute_definition(),
                        0..num_points,
                    );
                    let target_attribute_size = mapping.target_attribute.size() as usize;
                    for target_chunk in
                        target_attribute_range.chunks_exact_mut(target_attribute_size)
                    {
                        transformation(target_chunk);
                    }
                }
            }
        }
    }

    fn convert_columnar_to_general<'b, B: BorrowedMutBuffer<'b>>(
        &self,
        source_buffer: &dyn ColumnarBuffer,
        target_buffer: &mut B,
        max_attribute_size: usize,
    ) {
        let num_points = source_buffer.len();
        let mut convert_buffer = vec![0; max_attribute_size];
        let mut transform_buffer = vec![0; max_attribute_size];

        for mapping in &self.mappings {
            let source_attribute_data = source_buffer.get_attribute_range_ref(
                mapping.source_attribute.attribute_definition(),
                0..num_points,
            );
            let source_attribute_size = mapping.source_attribute.size() as usize;
            let target_attribute_size = mapping.target_attribute.size() as usize;

            if let Some(converter) = mapping.converter {
                let target_attribute_size = mapping.target_attribute.size() as usize;
                let target_attribute_chunk = &mut convert_buffer[..target_attribute_size];
                for (index, source_chunk) in source_attribute_data
                    .chunks_exact(source_attribute_size)
                    .enumerate()
                {
                    // Safe because we guarantee that the slice sizes match and their data comes from the
                    // correct attribute
                    unsafe {
                        converter(source_chunk, target_attribute_chunk);
                        if let Some(transformation) = mapping.transformation.as_deref() {
                            transformation(target_attribute_chunk);
                        }
                        target_buffer.set_attribute(
                            mapping.target_attribute.attribute_definition(),
                            index,
                            target_attribute_chunk,
                        );
                    }
                }
            } else {
                for (index, attribute_data) in source_attribute_data
                    .chunks_exact(source_attribute_size)
                    .enumerate()
                {
                    let attribute_data =
                        if let Some(transformation) = mapping.transformation.as_deref() {
                            let transform_slice = &mut transform_buffer[..target_attribute_size];
                            transform_slice.copy_from_slice(attribute_data);
                            transformation(transform_slice);
                            transform_slice
                        } else {
                            attribute_data
                        };

                    // Safe because size matches and the source and target attributes match, otherwise there
                    // would be a converter
                    unsafe {
                        target_buffer.set_attribute(
                            mapping.target_attribute.attribute_definition(),
                            index,
                            attribute_data,
                        );
                    }
                }
            }
        }
    }

    fn convert_general_to_columnar<'b, B: BorrowedBuffer<'b>>(
        &self,
        source_buffer: &B,
        target_buffer: &mut dyn ColumnarBufferMut,
        max_attribute_size: usize,
    ) {
        let mut buffer: Vec<u8> = vec![0; max_attribute_size];
        let num_points = source_buffer.len();

        for mapping in &self.mappings {
            let target_attribute_range = target_buffer.get_attribute_range_mut(
                mapping.target_attribute.attribute_definition(),
                0..num_points,
            );
            let target_attribute_size = mapping.target_attribute.size() as usize;

            for (point_index, target_attribute_chunk) in target_attribute_range
                .chunks_exact_mut(target_attribute_size)
                .enumerate()
            {
                let buf = &mut buffer[..mapping.source_attribute.size() as usize];
                // Safe because we check that the point layouts match
                unsafe {
                    source_buffer.get_attribute_unchecked(
                        mapping.source_attribute,
                        point_index,
                        buf,
                    );
                }

                if let Some(converter) = mapping.converter {
                    // Safety: converter came from the source and target PointLayouts
                    // buffer sizes are correct because they come from the PointAttributeMembers
                    // set_attribute is correct for the same reasons
                    unsafe {
                        converter(buf, target_attribute_chunk);
                    }
                    if let Some(transformation) = mapping.transformation.as_deref() {
                        transformation(target_attribute_chunk);
                    }
                } else {
                    if let Some(transformation) = mapping.transformation.as_deref() {
                        transformation(buf);
                    }
                    target_attribute_chunk.copy_from_slice(buf);
                }
            }
        }
    }

    fn convert_general_to_general<
        'b,
        'c,
        InBuffer: BorrowedBuffer<'b>,
        OutBuffer: BorrowedMutBuffer<'c>,
    >(
        &self,
        source_buffer: &InBuffer,
        target_buffer: &mut OutBuffer,
        max_attribute_size: usize,
    ) {
        let mut buffer: Vec<u8> = vec![0; max_attribute_size];
        let mut converter_buffer: Vec<u8> = vec![0; max_attribute_size];

        for mapping in &self.mappings {
            for point_index in 0..source_buffer.len() {
                let buf = &mut buffer[..mapping.source_attribute.size() as usize];
                // Safe because we check that the point layouts match
                unsafe {
                    source_buffer.get_attribute_unchecked(
                        mapping.source_attribute,
                        point_index,
                        buf,
                    );
                }

                if let Some(converter) = mapping.converter {
                    let target_buf =
                        &mut converter_buffer[..mapping.target_attribute.size() as usize];
                    // Safety: converter came from the source and target PointLayouts
                    // buffer sizes are correct because they come from the PointAttributeMembers
                    // set_attribute is correct for the same reasons
                    unsafe {
                        converter(buf, target_buf);
                        if let Some(transformation) = mapping.transformation.as_deref() {
                            transformation(target_buf);
                        }
                        target_buffer.set_attribute(
                            mapping.target_attribute.attribute_definition(),
                            point_index,
                            target_buf,
                        );
                    }
                } else {
                    if let Some(transformation) = mapping.transformation.as_deref() {
                        transformation(buf);
                    }
                    // Safety: buf has the correct size because otherwise we would have a converter
                    unsafe {
                        target_buffer.set_attribute(
                            mapping.target_attribute.attribute_definition(),
                            point_index,
                            buf,
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter::FromIterator;

    use itertools::Itertools;
    use nalgebra::Vector3;
    use rand::{thread_rng, Rng};

    use crate::{
        containers::{HashMapBuffer, VectorBuffer},
        layout::{
            attributes::{CLASSIFICATION, POSITION_3D, RETURN_NUMBER},
            PointType,
        },
        test_utils::{CustomPointTypeBig, CustomPointTypeSmall, DefaultPointDistribution},
    };

    use super::*;

    fn buffer_converter_default_generic<
        TFrom: for<'a> BorrowedBuffer<'a> + FromIterator<CustomPointTypeBig>,
        TTo: for<'a> OwningBuffer<'a> + for<'a> MakeBufferFromLayout<'a>,
    >() {
        let rng = thread_rng();
        let source_points = rng
            .sample_iter::<CustomPointTypeBig, _>(DefaultPointDistribution)
            .take(16)
            .collect::<TFrom>();

        // `CustomPointTypeSmall` is a subset of `CustomPointTypeBig`, so this conversion should work!
        let target_layout = CustomPointTypeSmall::layout();
        let converter =
            BufferLayoutConverter::for_layouts(source_points.point_layout(), &target_layout);

        let converted_points = converter.convert::<TTo, _>(&source_points);

        assert_eq!(target_layout, *converted_points.point_layout());

        let expected_positions = source_points
            .view_attribute::<Vector3<f64>>(&POSITION_3D)
            .into_iter()
            .collect_vec();
        let actual_positions = converted_points
            .view_attribute::<Vector3<f64>>(&POSITION_3D)
            .into_iter()
            .collect_vec();
        assert_eq!(expected_positions, actual_positions);

        let expected_classifications = source_points
            .view_attribute::<u8>(&CLASSIFICATION)
            .into_iter()
            .collect_vec();
        let actual_classifications = converted_points
            .view_attribute::<u8>(&CLASSIFICATION)
            .into_iter()
            .collect_vec();
        assert_eq!(expected_classifications, actual_classifications);
    }

    fn buffer_converter_multiple_attributes_from_one_generic<
        TFrom: for<'a> BorrowedBuffer<'a> + FromIterator<CustomPointTypeBig>,
        TTo: for<'a> OwningBuffer<'a> + for<'a> MakeBufferFromLayout<'a>,
    >() {
        let rng = thread_rng();
        let source_points = rng
            .sample_iter::<CustomPointTypeBig, _>(DefaultPointDistribution)
            .take(16)
            .collect::<TFrom>();

        // Convert a single source attribute (CLASSIFICATION) into two attributes that both have `u8` as their type
        let custom_layout = PointLayout::from_attributes(&[CLASSIFICATION, RETURN_NUMBER]);

        let mut converter = BufferLayoutConverter::for_layouts_with_default(
            source_points.point_layout(),
            &custom_layout,
        );
        converter.set_custom_mapping(&CLASSIFICATION, &RETURN_NUMBER);

        let converted_points = converter.convert::<TTo, _>(&source_points);
        assert_eq!(custom_layout, *converted_points.point_layout());

        let expected_classifications = source_points
            .view_attribute::<u8>(&CLASSIFICATION)
            .into_iter()
            .collect_vec();
        let actual_classifications = converted_points
            .view_attribute::<u8>(&CLASSIFICATION)
            .into_iter()
            .collect_vec();
        let actual_return_numbers = converted_points
            .view_attribute::<u8>(&RETURN_NUMBER)
            .into_iter()
            .collect_vec();

        assert_eq!(expected_classifications, actual_classifications);
        // Yes classifications, this test checks that we can convert two attributes from the same source attribute!
        assert_eq!(expected_classifications, actual_return_numbers);
    }

    fn buffer_converter_transformed_attribute_generic<
        TFrom: for<'a> BorrowedBuffer<'a> + FromIterator<CustomPointTypeBig>,
        TTo: for<'a> OwningBuffer<'a> + for<'a> MakeBufferFromLayout<'a>,
    >() {
        let rng = thread_rng();
        let source_points = rng
            .sample_iter::<CustomPointTypeBig, _>(DefaultPointDistribution)
            .take(16)
            .collect::<TFrom>();

        let custom_layout = PointLayout::from_attributes(&[POSITION_3D]);

        let mut converter = BufferLayoutConverter::for_layouts_with_default(
            source_points.point_layout(),
            &custom_layout,
        );
        const OFFSET: f64 = 42.0;
        let transform_positions_fn =
            |source_position: Vector3<f64>| -> Vector3<f64> { source_position.add_scalar(OFFSET) };
        converter.set_custom_mapping_with_transformation(
            &POSITION_3D,
            &POSITION_3D,
            transform_positions_fn,
        );

        let converted_points = converter.convert::<TTo, _>(&source_points);
        assert_eq!(custom_layout, *converted_points.point_layout());

        let expected_positions = source_points
            .view_attribute::<Vector3<f64>>(&POSITION_3D)
            .into_iter()
            .map(transform_positions_fn)
            .collect_vec();
        let actual_positions = converted_points
            .view_attribute::<Vector3<f64>>(&POSITION_3D)
            .into_iter()
            .collect_vec();

        assert_eq!(expected_positions, actual_positions);
    }

    fn buffer_converter_identity_generic<
        TFrom: for<'a> BorrowedBuffer<'a> + FromIterator<CustomPointTypeBig>,
        TTo: for<'a> OwningBuffer<'a> + for<'a> MakeBufferFromLayout<'a>,
    >() {
        let rng = thread_rng();
        let source_points = rng
            .sample_iter::<CustomPointTypeBig, _>(DefaultPointDistribution)
            .take(16)
            .collect::<TFrom>();

        let converter = BufferLayoutConverter::for_layouts_with_default(
            source_points.point_layout(),
            source_points.point_layout(),
        );
        let converted_points = converter.convert::<TTo, _>(&source_points);

        let expected_points = source_points
            .view::<CustomPointTypeBig>()
            .into_iter()
            .collect_vec();
        let actual_points = converted_points
            .view::<CustomPointTypeBig>()
            .into_iter()
            .collect_vec();
        assert_eq!(expected_points, actual_points);
    }

    #[test]
    fn test_buffer_converter_default() {
        buffer_converter_default_generic::<VectorBuffer, VectorBuffer>();
        buffer_converter_default_generic::<VectorBuffer, HashMapBuffer>();
        buffer_converter_default_generic::<HashMapBuffer, VectorBuffer>();
        buffer_converter_default_generic::<HashMapBuffer, HashMapBuffer>();
    }

    #[test]
    fn test_buffer_converter_multiple_attributes_from_one() {
        buffer_converter_multiple_attributes_from_one_generic::<VectorBuffer, VectorBuffer>();
        buffer_converter_multiple_attributes_from_one_generic::<VectorBuffer, HashMapBuffer>();
        buffer_converter_multiple_attributes_from_one_generic::<HashMapBuffer, VectorBuffer>();
        buffer_converter_multiple_attributes_from_one_generic::<HashMapBuffer, HashMapBuffer>();
    }

    #[test]
    fn test_buffer_converter_transformed_attribute() {
        buffer_converter_transformed_attribute_generic::<VectorBuffer, VectorBuffer>();
        buffer_converter_transformed_attribute_generic::<VectorBuffer, HashMapBuffer>();
        buffer_converter_transformed_attribute_generic::<HashMapBuffer, VectorBuffer>();
        buffer_converter_transformed_attribute_generic::<HashMapBuffer, HashMapBuffer>();
    }

    #[test]
    fn test_buffer_converter_identity() {
        buffer_converter_identity_generic::<VectorBuffer, VectorBuffer>();
        buffer_converter_identity_generic::<VectorBuffer, HashMapBuffer>();
        buffer_converter_identity_generic::<HashMapBuffer, VectorBuffer>();
        buffer_converter_identity_generic::<HashMapBuffer, HashMapBuffer>();
    }

    #[test]
    #[should_panic]
    fn test_buffer_converter_mismatched_len() {
        const COUNT: usize = 16;
        let rng = thread_rng();
        let source_points = rng
            .sample_iter::<CustomPointTypeBig, _>(DefaultPointDistribution)
            .take(COUNT)
            .collect::<VectorBuffer>();

        let mut target_buffer =
            VectorBuffer::with_capacity(COUNT / 2, source_points.point_layout().clone());

        let converter: BufferLayoutConverter<'_> = BufferLayoutConverter::for_layouts_with_default(
            source_points.point_layout(),
            source_points.point_layout(),
        );
        converter.convert_into(&source_points, &mut target_buffer);
    }
}
