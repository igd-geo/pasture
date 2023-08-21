use std::{collections::HashMap, ops::Range};

use anyhow::{anyhow, bail, Context, Result};
use bitfield::BitRange;
use bytemuck::Pod;
use num_traits::AsPrimitive;
use pasture_core::{
    layout::{
        attributes::{
            CLASSIFICATION, CLASSIFICATION_FLAGS, COLOR_RGB, EDGE_OF_FLIGHT_LINE, GPS_TIME,
            INTENSITY, NIR, NUMBER_OF_RETURNS, POINT_SOURCE_ID, POSITION_3D, RETURN_NUMBER,
            RETURN_POINT_WAVEFORM_LOCATION, SCANNER_CHANNEL, SCAN_ANGLE, SCAN_ANGLE_RANK,
            SCAN_DIRECTION_FLAG, USER_DATA, WAVEFORM_DATA_OFFSET, WAVEFORM_PACKET_SIZE,
            WAVEFORM_PARAMETERS, WAVE_PACKET_DESCRIPTOR_INDEX,
        },
        conversion::{get_converter_for_attributes, get_generic_converter, AttributeConversionFn},
        PointAttributeDataType, PointAttributeMember, PointLayout,
    },
    nalgebra::Vector3,
};

use crate::las::{point_layout_from_las_metadata, ATTRIBUTE_BASIC_FLAGS, ATTRIBUTE_EXTENDED_FLAGS};

use super::{ExtraBytesDataType, LASMetadata};

/**
 * Parsing LAS into pasture types is nasty, because there are so many combinations:
 * - A destination point attribute with a different type
 * - A destination point attribute with scaling and an offset
 * - Unpacking of packed values, e.g. the LAS flag attributes `ReturnNumber`, `Number of returns` etc.
 * - A combination of both
 * - A missing attribute in the destination `PointType`
 *
 * Since this makes the code very complicated, the idea is to build a Parser made up of several small functions
 * in a single step when creating an LAS reader. The functions themselves are responsible for parsing a single
 * attribute and only get memory ranges as their input
 */

pub(crate) type ExtractSourceValueFn = fn(source_bytes: &[u8], destination_bytes: &mut [u8]);
pub(crate) type ApplyOffsetAndScaleFn =
    fn(source_bytes: &[u8], destination_bytes: &mut [u8], offset_bytes: &[u8], scale_bytes: &[u8]);

#[derive(Debug)]
pub(crate) enum AttributeParseFn {
    /// Parsing is equal to a `memcpy`. This is for attributes that directly match to a pasture `PointAttributeDefinition`.
    /// An example for this is the `INTENSITY` attribute, which is a single `u16` in the LAS file and (by default) a single
    /// `u16` value in pasture
    Memcpy {
        source_bytes: Range<usize>,
        destination_bytes: Range<usize>,
    },
    /// Parsing of bitfield values which take less memory than a single byte within the LAS file. An example for this is
    /// the `RETURN_VALUE` attribute, which takes either 3 or 4 bits in the LAS file, but is represented by a `u8` value
    /// in pasture. pasture does not support `PointAttributeDefinition`s that are smaller than a single byte, which on the
    /// one hand means that every single attribute has to be parsed individually, but allows accessing packed bitfield
    /// values separately, instead of through a combined `LAS_BIT_VALUES` attribute
    Bitfield {
        extract_source_value_fn: ExtractSourceValueFn,
        source_bytes: Range<usize>,
        destination_bytes: Range<usize>,
    },
    /// Parsing a value from type `A` into type `B` using runtime type conversion. This is used whenever the destination
    /// `PointLayout` of parsing has a known LAS attribute, but with a different type than the default type in the LAS file
    TypeConverted {
        extract_source_value_fn: ExtractSourceValueFn,
        source_bytes: Range<usize>,
        destination_bytes: Range<usize>,
        conversion_buffer: Vec<u8>,
        type_converter: AttributeConversionFn,
    },
    /// Parsing a value and applying an offset and/or scaling value to it
    Scaled {
        apply_offset_and_scale_fn: ApplyOffsetAndScaleFn,
        source_bytes: Range<usize>,
        destination_bytes: Range<usize>,
        offset_values_bytes: Vec<u8>,
        scale_value_bytes: Vec<u8>,
    },
    /// A combination of `Scaled` value parsing with runtime type conversion. This requires a temporary buffer for storing the
    /// result of applying the scaling prior to type conversion
    ScaledAndTypeConverted {
        apply_offset_and_scale_fn: ApplyOffsetAndScaleFn,
        source_bytes: Range<usize>,
        destination_bytes: Range<usize>,
        scaling_buffer: Vec<u8>,
        offset_values_bytes: Vec<u8>,
        scale_value_bytes: Vec<u8>,
        type_converter: AttributeConversionFn,
    },
}

impl AttributeParseFn {
    /// Apply the attribute parsing function to the given source and destination point memory ranges
    ///
    /// # Safety
    ///
    /// Calling this function is only safe if `self` was constructed correctly, and both `source_point`
    /// and `destination_point` point to the correct memory ranges. `source_point` is assumed to point
    /// to the whole memory of a single LAS point corresponding to the `PointParser` that this parsing
    /// function came from. `destination_point` is assumed to point to the whole memory of a pasture
    /// `PointType`, also corresponding to the `PointParser` for this parsing function
    pub unsafe fn apply(&mut self, source_point: &[u8], destination_point: &mut [u8]) {
        match self {
            AttributeParseFn::Memcpy {
                source_bytes,
                destination_bytes,
            } => Self::memcpy(
                source_point,
                destination_point,
                source_bytes.clone(),
                destination_bytes.clone(),
            ),
            AttributeParseFn::Bitfield {
                extract_source_value_fn,
                source_bytes,
                destination_bytes,
            } => Self::bitfield(
                source_point,
                destination_point,
                *extract_source_value_fn,
                source_bytes.clone(),
                destination_bytes.clone(),
            ),
            AttributeParseFn::TypeConverted {
                extract_source_value_fn,
                source_bytes,
                destination_bytes,
                type_converter,
                ref mut conversion_buffer,
            } => Self::type_converted(
                source_point,
                destination_point,
                *extract_source_value_fn,
                source_bytes.clone(),
                destination_bytes.clone(),
                conversion_buffer,
                *type_converter,
            ),
            AttributeParseFn::Scaled {
                apply_offset_and_scale_fn,
                source_bytes,
                destination_bytes,
                offset_values_bytes,
                scale_value_bytes,
            } => Self::scaled(
                source_point,
                destination_point,
                *apply_offset_and_scale_fn,
                source_bytes.clone(),
                destination_bytes.clone(),
                offset_values_bytes,
                scale_value_bytes,
            ),
            AttributeParseFn::ScaledAndTypeConverted {
                apply_offset_and_scale_fn,
                source_bytes,
                destination_bytes,
                ref mut scaling_buffer,
                offset_values_bytes,
                scale_value_bytes,
                type_converter,
            } => Self::type_converted_and_scaled(
                source_point,
                destination_point,
                *apply_offset_and_scale_fn,
                source_bytes.clone(),
                destination_bytes.clone(),
                scaling_buffer,
                offset_values_bytes,
                scale_value_bytes,
                *type_converter,
            ),
        }
    }

    #[inline(always)]
    unsafe fn memcpy(
        source_point: &[u8],
        destination_point: &mut [u8],
        source_bytes: Range<usize>,
        destination_bytes: Range<usize>,
    ) {
        destination_point[destination_bytes].copy_from_slice(&source_point[source_bytes]);
    }

    #[inline(always)]
    unsafe fn bitfield(
        source_point: &[u8],
        destination_point: &mut [u8],
        extract_source_value_fn: fn(&[u8], &mut [u8]),
        source_bytes: Range<usize>,
        destination_bytes: Range<usize>,
    ) {
        let source_slice = &source_point[source_bytes];
        let destination_slice = &mut destination_point[destination_bytes];
        extract_source_value_fn(source_slice, destination_slice);
    }

    #[inline(always)]
    unsafe fn type_converted(
        source_point: &[u8],
        destination_point: &mut [u8],
        extract_source_value_fn: fn(&[u8], &mut [u8]),
        source_bytes: Range<usize>,
        destination_bytes: Range<usize>,
        buffer: &mut [u8],
        type_converter: AttributeConversionFn,
    ) {
        // Read the value in its source type from the input data
        let source_slice = &source_point[source_bytes];
        let source_value_buffer = &mut buffer[..source_slice.len()];
        extract_source_value_fn(source_slice, source_value_buffer);

        // Convert and write the value in its output format to the output data
        let destination_slice = &mut destination_point[destination_bytes];
        type_converter(source_value_buffer, destination_slice);
    }

    #[inline(always)]
    unsafe fn scaled(
        source_point: &[u8],
        destination_point: &mut [u8],
        apply_offset_and_scale_fn: fn(&[u8], &mut [u8], offset_bytes: &[u8], scale_bytes: &[u8]),
        source_bytes: Range<usize>,
        destination_bytes: Range<usize>,
        offset_values_bytes: &[u8],
        scale_value_bytes: &[u8],
    ) {
        // Apply offset and scaling parameters to the value in the output data in-place!
        let source_slice = &source_point[source_bytes];
        let output_slice = &mut destination_point[destination_bytes];
        apply_offset_and_scale_fn(
            source_slice,
            output_slice,
            offset_values_bytes,
            scale_value_bytes,
        );
    }

    #[inline(always)]
    unsafe fn type_converted_and_scaled(
        source_point: &[u8],
        destination_point: &mut [u8],
        apply_offset_and_scale_fn: fn(&[u8], &mut [u8], offset_bytes: &[u8], scale_bytes: &[u8]),
        source_bytes: Range<usize>,
        destination_bytes: Range<usize>,
        scaling_buffer: &mut [u8],
        offset_values_bytes: &[u8],
        scale_value_bytes: &[u8],
        type_converter: AttributeConversionFn,
    ) {
        // Apply offset and scaling parameters to the value in the output data. We can't do it in-place because
        // the offset and scale might be in a different type (e.g. f64 for i32 positional values)
        let source_slice = &source_point[source_bytes];
        let intermediate_size = std::mem::size_of::<Vector3<f64>>();
        let intermediate_slice = &mut scaling_buffer[..intermediate_size];

        apply_offset_and_scale_fn(
            source_slice,
            intermediate_slice,
            offset_values_bytes,
            scale_value_bytes,
        );

        // Perform type conversion
        let destination_slice = &mut destination_point[destination_bytes];
        type_converter(intermediate_slice, destination_slice);
    }
}

/// Custom parser for a single LAS point. The Parser takes the raw memory of a single LAS point
/// as input, and the raw memory of a single pasture point as output and performs the parsing
/// process
#[derive(Debug)]
pub(crate) struct PointParser {
    attribute_parsing_functions: Vec<AttributeParseFn>,
}

impl PointParser {
    /// Build a parser for the given LAS file into the given destination PointLayout
    pub(crate) fn build(
        las_metadata: &LASMetadata,
        destination_point_layout: &PointLayout,
    ) -> Result<Self> {
        let source_point_layout = point_layout_from_las_metadata(las_metadata, true)
            .context("Could not create a matching PointLayout for the LAS file")?;

        // The parser only works if the source PointLayout EXACTLY matches the bits of the LAS point record!
        // This might be problematic for the packed fields such as return number, number of returns etc. because
        // they might share the same byte. This means that we can't use `pasture_derive` for getting the PointLayout,
        // AND it means that the PointLayout that this function uses will be different from the default PointLayout
        // of the LASReader!!
        assert_eq!(
            source_point_layout.size_of_point_entry() as usize,
            las_metadata.point_format().len() as usize
        );

        // TODO: If both point layouts are equal, parsing is a single memcpy (assuming that the source_point_layout EXACTLY
        // matches the LAS memory layout
        // if source_point_layout == *destination_point_layout {
        //     let source_size = source_point_layout.size_of_point_entry() as usize;
        //     let destination_size = destination_point_layout.size_of_point_entry() as usize;
        //     return Ok(Self {
        //         attribute_parsing_function: vec![AttributeParseFn::Memcpy {
        //             source_bytes: 0..source_size,
        //             destination_bytes: 0..destination_size,
        //         }],
        //     });
        // }

        // The neat thing with the new parser: We only store parsing functions for the attributes that actually
        // exist within the output PointLayout

        let attributes_in_both_layouts =
            destination_point_layout
                .attributes()
                .filter_map(|destination_attribute| {
                    source_point_layout
                        .get_attribute_by_name(destination_attribute.name())
                        .map(|source_attribute| (source_attribute, destination_attribute))
                });

        let known_las_attributes_lut: HashMap<&str, BuildParserFn> = vec![
            (POSITION_3D.name(), build_position_parser as BuildParserFn),
            (INTENSITY.name(), build_intensity_parser),
            (RETURN_NUMBER.name(), build_return_number_parser),
            (NUMBER_OF_RETURNS.name(), build_number_of_returns_parser),
            (SCAN_DIRECTION_FLAG.name(), build_scan_direction_flag_parser),
            (EDGE_OF_FLIGHT_LINE.name(), build_edge_of_flight_line_parser),
            (CLASSIFICATION.name(), build_classification_parser),
            (SCAN_ANGLE_RANK.name(), build_scan_angle_rank_parser),
            (USER_DATA.name(), build_user_data_parser),
            (POINT_SOURCE_ID.name(), build_point_source_id_parser),
            (GPS_TIME.name(), build_gps_time_parser),
            (COLOR_RGB.name(), build_color_rgb_parser),
            (
                WAVE_PACKET_DESCRIPTOR_INDEX.name(),
                build_wave_packet_index_parser,
            ),
            (
                WAVEFORM_DATA_OFFSET.name(),
                build_waveform_byte_offset_parser,
            ),
            (
                WAVEFORM_PACKET_SIZE.name(),
                build_waveform_packet_size_parser,
            ),
            (
                RETURN_POINT_WAVEFORM_LOCATION.name(),
                build_waveform_return_point_parser,
            ),
            (WAVEFORM_PARAMETERS.name(), build_waveform_parameters_parser),
            (
                CLASSIFICATION_FLAGS.name(),
                build_classification_flags_parser,
            ),
            (SCAN_ANGLE.name(), build_scan_angle_parser),
            (NIR.name(), build_nir_parser),
        ]
        .into_iter()
        .collect::<HashMap<_, _>>();

        let mut parse_functions = attributes_in_both_layouts
            .map(
                |(source_attribute, destination_attribute)| -> Result<AttributeParseFn> {
                    if let Some(known_parser_factory) =
                        known_las_attributes_lut.get(source_attribute.name())
                    {
                        known_parser_factory(source_attribute, destination_attribute, las_metadata)
                    } else {
                        // This is an extra bytes attribute!
                        build_extra_bytes_attribute(
                            source_attribute,
                            destination_attribute,
                            las_metadata,
                        )
                    }
                },
            )
            .collect::<Result<Vec<_>, _>>()
            .context("Failed to build attribute parsing function")?;

        if let Some(attribute_flags) =
            source_point_layout.get_attribute_by_name(ATTRIBUTE_BASIC_FLAGS.name())
        {
            parse_functions.append(
                &mut Self::build_basic_flags_parsers(
                    attribute_flags,
                    destination_point_layout,
                    las_metadata,
                )
                .context("Failed to build flags attribute parsers")?,
            );
        } else if let Some(extended_attribute_flags) =
            source_point_layout.get_attribute_by_name(ATTRIBUTE_EXTENDED_FLAGS.name())
        {
            parse_functions.append(
                &mut Self::build_extended_flags_parsers(
                    extended_attribute_flags,
                    destination_point_layout,
                    las_metadata,
                )
                .context("Failed to build flags attribute parsers")?,
            );
        }

        Ok(Self {
            attribute_parsing_functions: parse_functions,
        })
    }

    /// Parse a single point
    pub(crate) unsafe fn parse_one(&mut self, input_point: &[u8], output_point: &mut [u8]) {
        for parsing_fn in &mut self.attribute_parsing_functions {
            parsing_fn.apply(input_point, output_point);
        }
    }

    fn build_basic_flags_parsers(
        source_flags_attribute: &PointAttributeMember,
        destination_point_layout: &PointLayout,
        las_metadata: &LASMetadata,
    ) -> Result<Vec<AttributeParseFn>> {
        let mut ret: Vec<_> = Default::default();
        if let Some(return_number_attribute) =
            destination_point_layout.get_attribute_by_name(RETURN_NUMBER.name())
        {
            ret.push(
                build_return_number_parser(
                    source_flags_attribute,
                    return_number_attribute,
                    las_metadata,
                )
                .context("Failed to build parser for RETURN_NUMBER attribute")?,
            );
        }
        if let Some(number_of_returns_attribute) =
            destination_point_layout.get_attribute_by_name(NUMBER_OF_RETURNS.name())
        {
            ret.push(
                build_number_of_returns_parser(
                    source_flags_attribute,
                    number_of_returns_attribute,
                    las_metadata,
                )
                .context("Failed to build parser for NUMBER_OF_RETURNS attribute")?,
            );
        }
        if let Some(scan_direction_flag_attribute) =
            destination_point_layout.get_attribute_by_name(SCAN_DIRECTION_FLAG.name())
        {
            ret.push(
                build_scan_direction_flag_parser(
                    source_flags_attribute,
                    scan_direction_flag_attribute,
                    las_metadata,
                )
                .context("Failed to build parser for SCAN_DIRECTION_FLAG attribute")?,
            );
        }
        if let Some(edge_of_flight_line_attribute) =
            destination_point_layout.get_attribute_by_name(EDGE_OF_FLIGHT_LINE.name())
        {
            ret.push(
                build_edge_of_flight_line_parser(
                    source_flags_attribute,
                    edge_of_flight_line_attribute,
                    las_metadata,
                )
                .context("Failed to build parser for EDGE_OF_FLIGHT_LINE attribute")?,
            );
        }
        Ok(ret)
    }

    fn build_extended_flags_parsers(
        source_flags_attribute: &PointAttributeMember,
        destination_point_layout: &PointLayout,
        las_metadata: &LASMetadata,
    ) -> Result<Vec<AttributeParseFn>> {
        let mut ret: Vec<_> = Default::default();
        if let Some(return_number_attribute) =
            destination_point_layout.get_attribute_by_name(RETURN_NUMBER.name())
        {
            ret.push(
                build_return_number_parser(
                    source_flags_attribute,
                    return_number_attribute,
                    las_metadata,
                )
                .context("Failed to build parser for RETURN_NUMBER attribute")?,
            );
        }
        if let Some(number_of_returns_attribute) =
            destination_point_layout.get_attribute_by_name(NUMBER_OF_RETURNS.name())
        {
            ret.push(
                build_number_of_returns_parser(
                    source_flags_attribute,
                    number_of_returns_attribute,
                    las_metadata,
                )
                .context("Failed to build parser for NUMBER_OF_RETURNS attribute")?,
            );
        }
        if let Some(classification_flags_attribute) =
            destination_point_layout.get_attribute_by_name(CLASSIFICATION_FLAGS.name())
        {
            ret.push(
                build_classification_flags_parser(
                    source_flags_attribute,
                    classification_flags_attribute,
                    las_metadata,
                )
                .context("Failed to build parser for CLASSIFICATION_FLAGS attribute")?,
            );
        }
        if let Some(scanner_channel_attribute) =
            destination_point_layout.get_attribute_by_name(SCANNER_CHANNEL.name())
        {
            ret.push(
                build_scanner_channel_parser(
                    source_flags_attribute,
                    scanner_channel_attribute,
                    las_metadata,
                )
                .context("Failed to build parser for SCANNER_CHANNEL attribute")?,
            );
        }
        if let Some(scan_direction_flag_attribute) =
            destination_point_layout.get_attribute_by_name(SCAN_DIRECTION_FLAG.name())
        {
            ret.push(
                build_scan_direction_flag_parser(
                    source_flags_attribute,
                    scan_direction_flag_attribute,
                    las_metadata,
                )
                .context("Failed to build parser for SCAN_DIRECTION_FLAG attribute")?,
            );
        }
        if let Some(edge_of_flight_line_attribute) =
            destination_point_layout.get_attribute_by_name(EDGE_OF_FLIGHT_LINE.name())
        {
            ret.push(
                build_edge_of_flight_line_parser(
                    source_flags_attribute,
                    edge_of_flight_line_attribute,
                    las_metadata,
                )
                .context("Failed to build parser for EDGE_OF_FLIGHT_LINE attribute")?,
            );
        }
        Ok(ret)
    }
}

/// Extracts the range of bits from FromBit to ToBit from `source_bytes` and puts them into `destination_bytes`
fn extract_bit_range<const FROM_BIT: usize, const TO_BIT: usize>(
    source_bytes: &[u8],
    destination_bytes: &mut [u8],
) {
    if TO_BIT >= 8 {
        let source_bytes_u16 = source_bytes.as_ptr() as *const u16;
        unsafe {
            destination_bytes[0] = source_bytes_u16
                .read_unaligned()
                .bit_range(TO_BIT, FROM_BIT);
        }
    } else {
        destination_bytes[0] = source_bytes[0].bit_range(TO_BIT, FROM_BIT);
    }
}

type BuildParserFn =
    fn(&PointAttributeMember, &PointAttributeMember, &LASMetadata) -> Result<AttributeParseFn>;

/// Build a parser for a generic attribute that can be memcpy'd into the destination buffer. This is for all attributes
/// where the type in the LAS point record matches the default type in pasture, e.g. intensities, classifications etc.
fn build_generic_memcpyable_parser(
    source_attribute: &PointAttributeMember,
    destination_attribute: &PointAttributeMember,
) -> Result<AttributeParseFn> {
    if source_attribute.datatype() != destination_attribute.datatype() {
        let extract_source_value_fn = |source_bytes: &[u8], destination_bytes: &mut [u8]| {
            destination_bytes.copy_from_slice(source_bytes);
        };
        let type_converter =
            get_converter_for_attributes(&source_attribute.into(), &destination_attribute.into())
                .ok_or(anyhow!(
                "No attribute type conversion from type {} to type {} possible",
                source_attribute.name(),
                destination_attribute.name()
            ))?;
        let conversion_buffer = vec![0; source_attribute.size() as usize];
        Ok(AttributeParseFn::TypeConverted {
            extract_source_value_fn,
            source_bytes: source_attribute.byte_range_within_point(),
            destination_bytes: destination_attribute.byte_range_within_point(),
            conversion_buffer,
            type_converter,
        })
    } else {
        Ok(AttributeParseFn::Memcpy {
            source_bytes: source_attribute.byte_range_within_point(),
            destination_bytes: destination_attribute.byte_range_within_point(),
        })
    }
}

fn build_position_parser(
    source_attribute: &PointAttributeMember,
    destination_attribute: &PointAttributeMember,
    las_metadata: &LASMetadata,
) -> Result<AttributeParseFn> {
    let apply_offset_and_scale_fn = |source_bytes: &[u8],
                                     destination_bytes: &mut [u8],
                                     offset_bytes: &[u8],
                                     scale_bytes: &[u8]| {
        let offset: &[f64] = bytemuck::cast_slice(offset_bytes);
        let scale: &[f64] = bytemuck::cast_slice(scale_bytes);

        let source_position = source_bytes.as_ptr() as *mut i32;
        let destination_position = destination_bytes.as_mut_ptr() as *mut f64;

        // Pointer can never be null and we use write_unaligned
        unsafe {
            let x = (source_position.read_unaligned() as f64 * scale[0]) + offset[0];
            let y = (source_position.add(1).read_unaligned() as f64 * scale[1]) + offset[1];
            let z = (source_position.add(2).read_unaligned() as f64 * scale[2]) + offset[2];

            destination_position.write_unaligned(x);
            destination_position.add(1).write_unaligned(y);
            destination_position.add(2).write_unaligned(z);
        }
    };

    let transforms = las_metadata
        .raw_las_header()
        .ok_or(anyhow!("No LAS header information present"))?
        .transforms();
    let offsets = [
        transforms.x.offset,
        transforms.y.offset,
        transforms.z.offset,
    ];
    let scales = [transforms.x.scale, transforms.y.scale, transforms.z.scale];

    // Positions are always scaled (pasture currently has no support for reading LAS positions as raw i32 values)
    // but are they also in a different type?
    // Since we always scale, the 'source' datatype prior to conversion is Vec3f64, NOT the source_attribute (which has
    // type Vec3i32)
    if destination_attribute.datatype() != PointAttributeDataType::Vec3f64 {
        let converter = get_converter_for_attributes(&POSITION_3D, &destination_attribute.into())
            .ok_or(anyhow!(
            "No attribute type conversion from type {} to type {} possible",
            source_attribute.name(),
            destination_attribute.name()
        ))?;

        let scaling_buffer = vec![0; std::mem::size_of::<[f64; 3]>()];

        Ok(AttributeParseFn::ScaledAndTypeConverted {
            apply_offset_and_scale_fn,
            source_bytes: source_attribute.byte_range_within_point(),
            destination_bytes: destination_attribute.byte_range_within_point(),
            offset_values_bytes: bytemuck::cast_slice(&offsets).to_owned(),
            scale_value_bytes: bytemuck::cast_slice(&scales).to_owned(),
            scaling_buffer,
            type_converter: converter,
        })
    } else {
        Ok(AttributeParseFn::Scaled {
            apply_offset_and_scale_fn,
            source_bytes: source_attribute.byte_range_within_point(),
            destination_bytes: destination_attribute.byte_range_within_point(),
            offset_values_bytes: bytemuck::cast_slice(&offsets).to_owned(),
            scale_value_bytes: bytemuck::cast_slice(&scales).to_owned(),
        })
    }
}

fn build_intensity_parser(
    source_attribute: &PointAttributeMember,
    destination_attribute: &PointAttributeMember,
    _las_metadata: &LASMetadata,
) -> Result<AttributeParseFn> {
    build_generic_memcpyable_parser(source_attribute, destination_attribute)
}

fn build_bit_attribute_parser(
    source_attribute: &PointAttributeMember,
    destination_attribute: &PointAttributeMember,
    default_destination_datatype: PointAttributeDataType,
    bit_extract_fn: fn(&[u8], &mut [u8]) -> (),
) -> Result<AttributeParseFn> {
    if destination_attribute.datatype() != default_destination_datatype {
        let type_converter = get_generic_converter(
            default_destination_datatype,
            destination_attribute.datatype(),
        )
        .ok_or(anyhow!(
            "No attribute type conversion from type {} to type {} possible",
            default_destination_datatype,
            destination_attribute.datatype()
        ))?;
        let conversion_buffer = vec![0; default_destination_datatype.size() as usize];
        Ok(AttributeParseFn::TypeConverted {
            extract_source_value_fn: bit_extract_fn,
            source_bytes: source_attribute.byte_range_within_point(),
            destination_bytes: destination_attribute.byte_range_within_point(),
            conversion_buffer,
            type_converter,
        })
    } else {
        Ok(AttributeParseFn::Bitfield {
            extract_source_value_fn: bit_extract_fn,
            source_bytes: source_attribute.byte_range_within_point(),
            destination_bytes: destination_attribute.byte_range_within_point(),
        })
    }
}

fn build_return_number_parser(
    source_attribute: &PointAttributeMember,
    destination_attribute: &PointAttributeMember,
    las_metadata: &LASMetadata,
) -> Result<AttributeParseFn> {
    // LAS formats 0-5 use 3 bits for return values, formats 6-10 (the extended formats) use 4 bits
    let extract_source_value_fn = if las_metadata.point_format().is_extended {
        extract_bit_range::<0, 3>
    } else {
        extract_bit_range::<0, 2>
    };

    build_bit_attribute_parser(
        source_attribute,
        destination_attribute,
        PointAttributeDataType::U8,
        extract_source_value_fn,
    )
}

fn build_number_of_returns_parser(
    source_attribute: &PointAttributeMember,
    destination_attribute: &PointAttributeMember,
    las_metadata: &LASMetadata,
) -> Result<AttributeParseFn> {
    // LAS formats 0-5 use 3 bits for number of returns, formats 6-10 (the extended formats) use 4 bits
    let extract_source_value_fn = if las_metadata.point_format().is_extended {
        extract_bit_range::<4, 7>
    } else {
        extract_bit_range::<3, 5>
    };

    build_bit_attribute_parser(
        source_attribute,
        destination_attribute,
        PointAttributeDataType::U8,
        extract_source_value_fn,
    )
}

fn build_scan_direction_flag_parser(
    source_attribute: &PointAttributeMember,
    destination_attribute: &PointAttributeMember,
    las_metadata: &LASMetadata,
) -> Result<AttributeParseFn> {
    let extract_source_value_fn = if las_metadata.point_format().is_extended {
        extract_bit_range::<14, 14>
    } else {
        extract_bit_range::<6, 6>
    };

    build_bit_attribute_parser(
        source_attribute,
        destination_attribute,
        PointAttributeDataType::Bool,
        extract_source_value_fn,
    )
}

fn build_edge_of_flight_line_parser(
    source_attribute: &PointAttributeMember,
    destination_attribute: &PointAttributeMember,
    las_metadata: &LASMetadata,
) -> Result<AttributeParseFn> {
    let extract_source_value_fn = if las_metadata.point_format().is_extended {
        extract_bit_range::<15, 15>
    } else {
        extract_bit_range::<7, 7>
    };

    build_bit_attribute_parser(
        source_attribute,
        destination_attribute,
        PointAttributeDataType::Bool,
        extract_source_value_fn,
    )
}

fn build_classification_parser(
    source_attribute: &PointAttributeMember,
    destination_attribute: &PointAttributeMember,
    _las_metadata: &LASMetadata,
) -> Result<AttributeParseFn> {
    build_generic_memcpyable_parser(source_attribute, destination_attribute)
}

fn build_scan_angle_rank_parser(
    source_attribute: &PointAttributeMember,
    destination_attribute: &PointAttributeMember,
    _las_metadata: &LASMetadata,
) -> Result<AttributeParseFn> {
    build_generic_memcpyable_parser(source_attribute, destination_attribute)
}

fn build_user_data_parser(
    source_attribute: &PointAttributeMember,
    destination_attribute: &PointAttributeMember,
    _las_metadata: &LASMetadata,
) -> Result<AttributeParseFn> {
    build_generic_memcpyable_parser(source_attribute, destination_attribute)
}

fn build_point_source_id_parser(
    source_attribute: &PointAttributeMember,
    destination_attribute: &PointAttributeMember,
    _las_metadata: &LASMetadata,
) -> Result<AttributeParseFn> {
    build_generic_memcpyable_parser(source_attribute, destination_attribute)
}

fn build_color_rgb_parser(
    source_attribute: &PointAttributeMember,
    destination_attribute: &PointAttributeMember,
    _las_metadata: &LASMetadata,
) -> Result<AttributeParseFn> {
    build_generic_memcpyable_parser(source_attribute, destination_attribute)
}

fn build_gps_time_parser(
    source_attribute: &PointAttributeMember,
    destination_attribute: &PointAttributeMember,
    _las_metadata: &LASMetadata,
) -> Result<AttributeParseFn> {
    build_generic_memcpyable_parser(source_attribute, destination_attribute)
}

fn build_classification_flags_parser(
    source_attribute: &PointAttributeMember,
    destination_attribute: &PointAttributeMember,
    las_metadata: &LASMetadata,
) -> Result<AttributeParseFn> {
    if !las_metadata.point_format().is_extended {
        bail!("Classification flags are only present in extended LAS formats!");
    }

    build_bit_attribute_parser(
        source_attribute,
        destination_attribute,
        PointAttributeDataType::U8,
        extract_bit_range::<8, 11>,
    )
}

fn build_scanner_channel_parser(
    source_attribute: &PointAttributeMember,
    destination_attribute: &PointAttributeMember,
    las_metadata: &LASMetadata,
) -> Result<AttributeParseFn> {
    if !las_metadata.point_format().is_extended {
        bail!("Scanner channel information is only present in extended LAS formats!");
    }

    build_bit_attribute_parser(
        source_attribute,
        destination_attribute,
        PointAttributeDataType::U8,
        extract_bit_range::<12, 13>,
    )
}

fn build_nir_parser(
    source_attribute: &PointAttributeMember,
    destination_attribute: &PointAttributeMember,
    _las_metadata: &LASMetadata,
) -> Result<AttributeParseFn> {
    build_generic_memcpyable_parser(source_attribute, destination_attribute)
}

fn build_wave_packet_index_parser(
    source_attribute: &PointAttributeMember,
    destination_attribute: &PointAttributeMember,
    _las_metadata: &LASMetadata,
) -> Result<AttributeParseFn> {
    build_generic_memcpyable_parser(source_attribute, destination_attribute)
}

fn build_waveform_byte_offset_parser(
    source_attribute: &PointAttributeMember,
    destination_attribute: &PointAttributeMember,
    _las_metadata: &LASMetadata,
) -> Result<AttributeParseFn> {
    build_generic_memcpyable_parser(source_attribute, destination_attribute)
}

fn build_waveform_packet_size_parser(
    source_attribute: &PointAttributeMember,
    destination_attribute: &PointAttributeMember,
    _las_metadata: &LASMetadata,
) -> Result<AttributeParseFn> {
    build_generic_memcpyable_parser(source_attribute, destination_attribute)
}

fn build_waveform_return_point_parser(
    source_attribute: &PointAttributeMember,
    destination_attribute: &PointAttributeMember,
    _las_metadata: &LASMetadata,
) -> Result<AttributeParseFn> {
    build_generic_memcpyable_parser(source_attribute, destination_attribute)
}

fn build_waveform_parameters_parser(
    source_attribute: &PointAttributeMember,
    destination_attribute: &PointAttributeMember,
    _las_metadata: &LASMetadata,
) -> Result<AttributeParseFn> {
    build_generic_memcpyable_parser(source_attribute, destination_attribute)
}

fn build_scan_angle_parser(
    source_attribute: &PointAttributeMember,
    destination_attribute: &PointAttributeMember,
    _las_metadata: &LASMetadata,
) -> Result<AttributeParseFn> {
    build_generic_memcpyable_parser(source_attribute, destination_attribute)
}

fn offset_scale_function<T: Pod + AsPrimitive<f64>>(
    in_bytes: &[u8],
    out_bytes: &mut [u8],
    offset_bytes: &[u8],
    scale_bytes: &[u8],
) where
    f64: AsPrimitive<T>,
{
    let offset: f64 = bytemuck::cast_slice(offset_bytes)[0];
    let scale: f64 = bytemuck::cast_slice(scale_bytes)[0];
    let in_bytes: &[T] = bytemuck::cast_slice(in_bytes);
    let out_bytes: &mut [T] = bytemuck::cast_slice_mut(out_bytes);
    let unscaled_value: f64 = in_bytes[0].as_();
    let scaled_value = (unscaled_value * scale) + offset;
    out_bytes[0] = scaled_value.as_();
}

fn build_generic_offset_scale_function(
    datatype: ExtraBytesDataType,
) -> Result<ApplyOffsetAndScaleFn> {
    // TODO Maybe it would make more sense to have the output format of scaled/offset extra bytes always
    // be `f64`? Otherwise we lose a lot of precision by first offsetting/scaling and then truncating back
    // to the original data type (e.g. `u8`)
    let function = match datatype {
        ExtraBytesDataType::U8 => offset_scale_function::<u8>,
        ExtraBytesDataType::I8 => offset_scale_function::<i8>,
        ExtraBytesDataType::U16 => offset_scale_function::<u16>,
        ExtraBytesDataType::I16 => offset_scale_function::<i16>,
        ExtraBytesDataType::U32 => offset_scale_function::<u32>,
        ExtraBytesDataType::I32 => offset_scale_function::<i32>,
        ExtraBytesDataType::U64 => offset_scale_function::<u64>,
        ExtraBytesDataType::I64 => offset_scale_function::<i64>,
        ExtraBytesDataType::F32 => offset_scale_function::<f32>,
        ExtraBytesDataType::F64 => offset_scale_function::<f64>,
        _ => bail!("Unsupported extra bytes datatype {datatype}"),
    };
    Ok(function)
}

fn build_extra_bytes_attribute(
    source_attribute: &PointAttributeMember,
    destination_attribute: &PointAttributeMember,
    las_metadata: &LASMetadata,
) -> Result<AttributeParseFn> {
    if let Some(extra_bytes_description) = las_metadata.extra_bytes_vlr().and_then(|vlr| {
        vlr.entries()
            .iter()
            .find(|entry| entry.name() == source_attribute.name())
    }) {
        if extra_bytes_description.options().use_offset()
            || extra_bytes_description.options().use_scale()
        {
            // Build a corresponding offset/scale function based on the actual type of the extra bytes
            let apply_offset_and_scale_fn =
                build_generic_offset_scale_function(extra_bytes_description.data_type())
                    .context("Can't build offset and scale function")?;

            let offset = extra_bytes_description.offset().unwrap_or_default();
            let scale = extra_bytes_description.scale().unwrap_or(1.0);

            if source_attribute.datatype() != destination_attribute.datatype() {
                let converter = get_converter_for_attributes(
                    &source_attribute.into(),
                    &destination_attribute.into(),
                )
                .ok_or(anyhow!(
                    "No attribute type conversion from type {} to type {} possible",
                    source_attribute.name(),
                    destination_attribute.name()
                ))?;

                let scaling_buffer = vec![
                    0;
                    extra_bytes_description
                        .data_type()
                        .size()
                        .ok_or(anyhow!("Invalid extra bytes datatype"))?
                ];

                Ok(AttributeParseFn::ScaledAndTypeConverted {
                    apply_offset_and_scale_fn,
                    source_bytes: source_attribute.byte_range_within_point(),
                    destination_bytes: destination_attribute.byte_range_within_point(),
                    offset_values_bytes: bytemuck::bytes_of(&offset).to_owned(),
                    scale_value_bytes: bytemuck::bytes_of(&scale).to_owned(),
                    scaling_buffer,
                    type_converter: converter,
                })
            } else {
                Ok(AttributeParseFn::Scaled {
                    apply_offset_and_scale_fn,
                    source_bytes: source_attribute.byte_range_within_point(),
                    destination_bytes: destination_attribute.byte_range_within_point(),
                    offset_values_bytes: bytemuck::bytes_of(&offset).to_owned(),
                    scale_value_bytes: bytemuck::bytes_of(&scale).to_owned(),
                })
            }
        } else {
            build_generic_memcpyable_parser(source_attribute, destination_attribute)
        }
    } else {
        // unexplained extra bytes entry can be memcpied
        build_generic_memcpyable_parser(source_attribute, destination_attribute)
    }
}
