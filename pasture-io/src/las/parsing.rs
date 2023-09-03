#![allow(clippy::too_many_arguments)]
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
        // let source_value_buffer = &mut buffer[..source_slice.len()];
        extract_source_value_fn(source_slice, &mut buffer[..]);

        // Convert and write the value in its output format to the output data
        let destination_slice = &mut destination_point[destination_bytes];
        type_converter(buffer, destination_slice);
    }

    #[inline(always)]
    unsafe fn scaled(
        source_point: &[u8],
        destination_point: &mut [u8],
        apply_offset_and_scale_fn: ApplyOffsetAndScaleFn,
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
        apply_offset_and_scale_fn: ApplyOffsetAndScaleFn,
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
        let type_converter = get_converter_for_attributes(
            source_attribute.attribute_definition(),
            destination_attribute.attribute_definition(),
        )
        .ok_or(anyhow!(
            "No attribute type conversion from type {} to type {} possible",
            source_attribute,
            destination_attribute
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
        let converter = get_converter_for_attributes(
            &POSITION_3D,
            destination_attribute.attribute_definition(),
        )
        .ok_or(anyhow!(
            "No attribute type conversion from type {} to type {} possible for attribute {}",
            PointAttributeDataType::Vec3f64,
            destination_attribute.datatype(),
            destination_attribute.name(),
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
            "No attribute type conversion from type {} to type {} possible for attribute {}",
            default_destination_datatype,
            destination_attribute.datatype(),
            destination_attribute.name()
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
        PointAttributeDataType::U8,
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
        PointAttributeDataType::U8,
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
    let in_bytes = in_bytes.as_ptr() as *const T;
    let out_bytes = out_bytes.as_mut_ptr() as *mut T;
    // Pointers are never null and we know data is properly initialized because
    // it comes from a binary file
    unsafe {
        let unscaled_value: f64 = in_bytes.read_unaligned().as_();
        let scaled_value = (unscaled_value * scale) + offset;
        out_bytes.write_unaligned(scaled_value.as_());
    }
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
                    source_attribute.attribute_definition(),
                    destination_attribute.attribute_definition(),
                )
                .ok_or(anyhow!(
                    "No attribute type conversion from type {} to type {} possible",
                    source_attribute,
                    destination_attribute
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

#[cfg(test)]
mod tests {
    use std::convert::TryInto;

    use super::*;
    use crate::las::{LasPointFormat10, LasPointFormat5};

    use bitfield::{bitfield_bitrange, bitfield_fields};
    use las_rs::{point::Format, Builder, Transform, Vector, Version};
    use pasture_core::layout::{PointAttributeDefinition, PointType, PrimitiveType};
    use static_assertions::const_assert_eq;

    #[derive(Debug, Copy, Clone, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
    #[repr(C)]
    pub struct BasicFlags(u8);
    bitfield_bitrange! {struct BasicFlags(u8)}

    impl BasicFlags {
        bitfield_fields! {
            u8;
            pub return_number, set_return_number: 2, 0;
            pub number_of_returns, set_number_of_returns: 5, 3;
            pub scan_direction_flag, set_scan_direction_flag: 6;
            pub edge_of_flight_line, set_edge_of_flight_line: 7;
        }
    }

    #[derive(Debug, Copy, Clone, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
    #[repr(C)]
    pub struct ExtendedFlags(u16);
    bitfield_bitrange! {struct ExtendedFlags(u16)}

    impl ExtendedFlags {
        bitfield_fields! {
            u16;
            pub return_number, set_return_number: 3, 0;
            pub number_of_returns, set_number_of_returns: 7, 4;
            pub classification_flags, set_classification_flags: 11, 8;
            pub scanner_channel, set_scanner_channel: 13, 12;
            pub scan_direction_flag, set_scan_direction_flag: 14;
            pub edge_of_flight_line, set_edge_of_flight_line: 15;
        }
    }

    #[derive(Debug, Copy, Clone, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
    #[repr(C, packed)]
    struct RawLASPointFormat5 {
        x: i32,
        y: i32,
        z: i32,
        intensity: u16,
        flags: BasicFlags,
        classification: u8,
        scan_angle_rank: i8,
        user_data: u8,
        point_source_id: u16,
        gps_time: f64,
        red: u16,
        green: u16,
        blue: u16,
        wave_packet_desriptor_index: u8,
        byte_offset_to_waveform_data: u64,
        waveform_packet_size: u32,
        return_point_waveform_location: f32,
        dx: f32,
        dy: f32,
        dz: f32,
    }
    const_assert_eq!(63, std::mem::size_of::<RawLASPointFormat5>());

    #[derive(Debug, Copy, Clone, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
    #[repr(C, packed)]
    struct RawLASPointFormat10 {
        x: i32,
        y: i32,
        z: i32,
        intensity: u16,
        flags: ExtendedFlags,
        classification: u8,
        user_data: u8,
        scan_angle: i16,
        point_source_id: u16,
        gps_time: f64,
        red: u16,
        green: u16,
        blue: u16,
        nir: u16,
        wave_packet_desriptor_index: u8,
        byte_offset_to_waveform_data: u64,
        waveform_packet_size: u32,
        return_point_waveform_location: f32,
        dx: f32,
        dy: f32,
        dz: f32,
    }
    const_assert_eq!(67, std::mem::size_of::<RawLASPointFormat10>());

    fn get_test_las_metadata(point_format: Format) -> Result<LASMetadata> {
        let mut builder = Builder::default();
        builder.version = Version::new(1, 4);
        builder.transforms = Vector {
            x: Transform {
                offset: 0.0,
                scale: 1.0,
            },
            y: Transform {
                offset: 0.0,
                scale: 1.0,
            },
            z: Transform {
                offset: 0.0,
                scale: 1.0,
            },
        };
        builder.point_format = point_format;
        let header = builder.into_header()?;
        header.try_into()
    }

    fn get_test_point_las_format5() -> (RawLASPointFormat5, LasPointFormat5) {
        let raw_point = RawLASPointFormat5 {
            x: 123,
            y: 456,
            z: 789,
            intensity: 0x4123,
            flags: BasicFlags(0xFF),
            classification: 4,
            scan_angle_rank: -65,
            user_data: 128,
            point_source_id: 0xAB10,
            gps_time: 1337.0,
            red: 0xaaaa,
            green: 0xbbbb,
            blue: 0xcccc,
            wave_packet_desriptor_index: 10,
            byte_offset_to_waveform_data: 9876,
            waveform_packet_size: 8765,
            return_point_waveform_location: 0.1,
            dx: 0.2,
            dy: 0.3,
            dz: 0.4,
        };

        let expected_point = LasPointFormat5 {
            byte_offset_to_waveform_data: raw_point.byte_offset_to_waveform_data,
            classification: raw_point.classification,
            color_rgb: Vector3::new(raw_point.red, raw_point.green, raw_point.blue),
            edge_of_flight_line: raw_point.flags.edge_of_flight_line() as u8,
            gps_time: raw_point.gps_time,
            intensity: raw_point.intensity,
            number_of_returns: raw_point.flags.number_of_returns(),
            point_source_id: raw_point.point_source_id,
            position: Vector3::new(raw_point.x as f64, raw_point.y as f64, raw_point.z as f64),
            return_number: raw_point.flags.return_number(),
            return_point_waveform_location: raw_point.return_point_waveform_location,
            scan_angle_rank: raw_point.scan_angle_rank,
            scan_direction_flag: raw_point.flags.scan_direction_flag() as u8,
            user_data: raw_point.user_data,
            wave_packet_descriptor_index: raw_point.wave_packet_desriptor_index,
            waveform_packet_size: raw_point.waveform_packet_size,
            waveform_parameters: Vector3::new(raw_point.dx, raw_point.dy, raw_point.dz),
        };

        (raw_point, expected_point)
    }

    fn get_test_point_las_format10() -> (RawLASPointFormat10, LasPointFormat10) {
        let raw_point = RawLASPointFormat10 {
            x: 123,
            y: 456,
            z: 789,
            intensity: 0x4123,
            flags: ExtendedFlags(0xFFFF),
            classification: 4,
            scan_angle: -1234,
            user_data: 128,
            point_source_id: 0xAB10,
            gps_time: 1337.0,
            red: 0xaaaa,
            green: 0xbbbb,
            blue: 0xcccc,
            nir: 0xdddd,
            wave_packet_desriptor_index: 10,
            byte_offset_to_waveform_data: 9876,
            waveform_packet_size: 8765,
            return_point_waveform_location: 0.1,
            dx: 0.2,
            dy: 0.3,
            dz: 0.4,
        };

        let flags = raw_point.flags;
        let expected_point = LasPointFormat10 {
            byte_offset_to_waveform_data: raw_point.byte_offset_to_waveform_data,
            classification: raw_point.classification,
            classification_flags: flags.classification_flags() as u8,
            color_rgb: Vector3::new(raw_point.red, raw_point.green, raw_point.blue),
            edge_of_flight_line: flags.edge_of_flight_line() as u8,
            gps_time: raw_point.gps_time,
            intensity: raw_point.intensity,
            nir: raw_point.nir,
            number_of_returns: flags.number_of_returns() as u8,
            point_source_id: raw_point.point_source_id,
            position: Vector3::new(raw_point.x as f64, raw_point.y as f64, raw_point.z as f64),
            return_number: flags.return_number() as u8,
            return_point_waveform_location: raw_point.return_point_waveform_location,
            scan_angle: raw_point.scan_angle,
            scan_direction_flag: flags.scan_direction_flag() as u8,
            scanner_channel: flags.scanner_channel() as u8,
            user_data: raw_point.user_data,
            wave_packet_descriptor_index: raw_point.wave_packet_desriptor_index,
            waveform_packet_size: raw_point.waveform_packet_size,
            waveform_parameters: Vector3::new(raw_point.dx, raw_point.dy, raw_point.dz),
        };

        (raw_point, expected_point)
    }

    unsafe fn parse_vector3_attribute_in_format<T>(
        input_point: &[u8],
        las_metadata: &LASMetadata,
        expected_value: Vector3<T>,
        attribute_definition: &PointAttributeDefinition,
    ) -> Result<()>
    where
        T: std::fmt::Debug + 'static,
        Vector3<T>: PrimitiveType + Default + PartialEq,
    {
        let mut actual_value = Vector3::<T>::default();
        let layout = PointLayout::from_attributes(&[
            attribute_definition.with_custom_datatype(Vector3::<T>::data_type())
        ]);

        let mut parser = PointParser::build(las_metadata, &layout)?;

        unsafe {
            let destination_data = bytemuck::bytes_of_mut(&mut actual_value);
            parser.parse_one(input_point, destination_data);
        }

        assert_eq!(
            expected_value,
            actual_value,
            "Parsed vector attribute {} with type {} is wrong",
            attribute_definition.name(),
            Vector3::<T>::data_type()
        );

        Ok(())
    }

    unsafe fn parse_scalar_attribute_in_format<T>(
        input_point: &[u8],
        las_metadata: &LASMetadata,
        expected_value: T,
        attribute_definition: &PointAttributeDefinition,
    ) -> Result<()>
    where
        T: Default + PrimitiveType + PartialEq + std::fmt::Debug,
    {
        let mut actual_value = T::default();
        let layout = PointLayout::from_attributes(&[
            attribute_definition.with_custom_datatype(T::data_type())
        ]);

        let mut parser = PointParser::build(las_metadata, &layout)?;

        unsafe {
            let destination_data = bytemuck::bytes_of_mut(&mut actual_value);
            parser.parse_one(input_point, destination_data);
        }

        assert_eq!(
            expected_value,
            actual_value,
            "Parsed scalar attribute {} with type {} is wrong",
            attribute_definition.name(),
            T::data_type()
        );

        Ok(())
    }

    /// Parse the given Vector3 attribute in all possible formats (i.e. all known Vector3 variants that
    /// pasture knows as a primitive type). This tests all possible Vector3 conversions
    fn parse_vector3_attribute_in_various_formats<T>(
        input_point: &[u8],
        las_metadata: &LASMetadata,
        expected_value: Vector3<T>,
        attribute_definition: &PointAttributeDefinition,
    ) -> Result<()>
    where
        T: Copy
            + AsPrimitive<u8>
            + AsPrimitive<u16>
            + AsPrimitive<i32>
            + AsPrimitive<f32>
            + AsPrimitive<f64>,
    {
        unsafe {
            parse_vector3_attribute_in_format::<f64>(
                input_point,
                las_metadata,
                Vector3::new(
                    expected_value[0].as_(),
                    expected_value[1].as_(),
                    expected_value[2].as_(),
                ),
                &attribute_definition.with_custom_datatype(PointAttributeDataType::Vec3f64),
            )?;
            parse_vector3_attribute_in_format::<f32>(
                input_point,
                las_metadata,
                Vector3::new(
                    expected_value[0].as_(),
                    expected_value[1].as_(),
                    expected_value[2].as_(),
                ),
                &attribute_definition.with_custom_datatype(PointAttributeDataType::Vec3f32),
            )?;
            parse_vector3_attribute_in_format::<i32>(
                input_point,
                las_metadata,
                Vector3::new(
                    expected_value[0].as_(),
                    expected_value[1].as_(),
                    expected_value[2].as_(),
                ),
                &attribute_definition.with_custom_datatype(PointAttributeDataType::Vec3i32),
            )?;
            parse_vector3_attribute_in_format::<u16>(
                input_point,
                las_metadata,
                Vector3::new(
                    expected_value[0].as_(),
                    expected_value[1].as_(),
                    expected_value[2].as_(),
                ),
                &attribute_definition.with_custom_datatype(PointAttributeDataType::Vec3u16),
            )?;
            parse_vector3_attribute_in_format::<u8>(
                input_point,
                las_metadata,
                Vector3::new(
                    expected_value[0].as_(),
                    expected_value[1].as_(),
                    expected_value[2].as_(),
                ),
                &attribute_definition.with_custom_datatype(PointAttributeDataType::Vec3u8),
            )?;
        }

        Ok(())
    }

    fn parse_scalar_attribute_in_various_formats<T>(
        input_point: &[u8],
        las_metadata: &LASMetadata,
        expected_value: T,
        attribute_definition: &PointAttributeDefinition,
    ) -> Result<()>
    where
        T: Copy
            + AsPrimitive<f64>
            + AsPrimitive<f32>
            + AsPrimitive<u8>
            + AsPrimitive<u16>
            + AsPrimitive<u32>
            + AsPrimitive<u64>
            + AsPrimitive<i8>
            + AsPrimitive<i16>
            + AsPrimitive<i32>
            + AsPrimitive<i64>,
    {
        unsafe {
            parse_scalar_attribute_in_format::<f64>(
                input_point,
                las_metadata,
                expected_value.as_(),
                &attribute_definition.with_custom_datatype(PointAttributeDataType::F64),
            )?;
            parse_scalar_attribute_in_format::<f32>(
                input_point,
                las_metadata,
                expected_value.as_(),
                &attribute_definition.with_custom_datatype(PointAttributeDataType::F32),
            )?;
            parse_scalar_attribute_in_format::<i8>(
                input_point,
                las_metadata,
                expected_value.as_(),
                &attribute_definition.with_custom_datatype(PointAttributeDataType::I8),
            )?;
            parse_scalar_attribute_in_format::<i16>(
                input_point,
                las_metadata,
                expected_value.as_(),
                &attribute_definition.with_custom_datatype(PointAttributeDataType::I16),
            )?;
            parse_scalar_attribute_in_format::<i32>(
                input_point,
                las_metadata,
                expected_value.as_(),
                &attribute_definition.with_custom_datatype(PointAttributeDataType::I32),
            )?;
            parse_scalar_attribute_in_format::<i64>(
                input_point,
                las_metadata,
                expected_value.as_(),
                &attribute_definition.with_custom_datatype(PointAttributeDataType::I64),
            )?;

            parse_scalar_attribute_in_format::<u8>(
                input_point,
                las_metadata,
                expected_value.as_(),
                &attribute_definition.with_custom_datatype(PointAttributeDataType::U8),
            )?;
            parse_scalar_attribute_in_format::<u16>(
                input_point,
                las_metadata,
                expected_value.as_(),
                &attribute_definition.with_custom_datatype(PointAttributeDataType::U16),
            )?;
            parse_scalar_attribute_in_format::<u32>(
                input_point,
                las_metadata,
                expected_value.as_(),
                &attribute_definition.with_custom_datatype(PointAttributeDataType::U32),
            )?;
            parse_scalar_attribute_in_format::<u64>(
                input_point,
                las_metadata,
                expected_value.as_(),
                &attribute_definition.with_custom_datatype(PointAttributeDataType::U64),
            )?;

            parse_scalar_attribute_in_format::<u8>(
                input_point,
                las_metadata,
                expected_value.as_(),
                &attribute_definition.with_custom_datatype(PointAttributeDataType::U8),
            )?;
        }

        Ok(())
    }

    #[test]
    fn parse_las_format5_default() -> Result<()> {
        let (test_point, expected_point) = get_test_point_las_format5();
        let target_point_layout = LasPointFormat5::layout();
        let las_metadata = get_test_las_metadata(Format::new(5)?)?;

        let mut parsed_point = LasPointFormat5::default();

        let mut parser = PointParser::build(&las_metadata, &target_point_layout)?;
        unsafe {
            let source_data = bytemuck::bytes_of(&test_point);
            let destination_data = bytemuck::bytes_of_mut(&mut parsed_point);
            parser.parse_one(source_data, destination_data);
        }

        assert_eq!(expected_point, parsed_point);

        Ok(())
    }

    #[test]
    fn parse_las_format5_individual_attributes() -> Result<()> {
        let (test_point, expected_point) = get_test_point_las_format5();
        // let target_point_layout = LasPointFormat5::layout();
        let las_metadata = get_test_las_metadata(Format::new(5)?)?;

        let input_point_data = bytemuck::bytes_of(&test_point);

        // Parse all attributes individually
        parse_vector3_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.position,
            &POSITION_3D,
        )
        .context("Parsing position failed")?;
        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.intensity,
            &INTENSITY,
        )
        .context("Parsing intensity failed")?;
        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.return_number,
            &RETURN_NUMBER,
        )
        .context("Parsing return number failed")?;
        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.number_of_returns,
            &NUMBER_OF_RETURNS,
        )
        .context("Parsing number of returns failed")?;

        // bool attributes can only be parsed as bools, pasture does not support bool-to-numeric upcasting!
        unsafe {
            parse_scalar_attribute_in_format(
                input_point_data,
                &las_metadata,
                expected_point.scan_direction_flag,
                &SCAN_DIRECTION_FLAG,
            )
            .context("Parsing scan direction flag failed")?;
            parse_scalar_attribute_in_format(
                input_point_data,
                &las_metadata,
                expected_point.edge_of_flight_line,
                &EDGE_OF_FLIGHT_LINE,
            )
            .context("Parsing edge of flight line failed")?;
        }

        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.classification,
            &CLASSIFICATION,
        )
        .context("Parsing classification failed")?;
        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.scan_angle_rank,
            &SCAN_ANGLE_RANK,
        )
        .context("Parsing scan angle rank failed")?;
        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.user_data,
            &USER_DATA,
        )
        .context("Parsing user data failed")?;
        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.point_source_id,
            &POINT_SOURCE_ID,
        )
        .context("Parsing point source ID failed")?;
        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.gps_time,
            &GPS_TIME,
        )
        .context("Parsing gps time failed")?;
        let expected_color = expected_point.color_rgb;
        parse_vector3_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_color,
            &COLOR_RGB,
        )
        .context("Parsing color failed")?;
        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.wave_packet_descriptor_index,
            &WAVE_PACKET_DESCRIPTOR_INDEX,
        )
        .context("Parsing wave packet descriptor index failed")?;
        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.byte_offset_to_waveform_data,
            &WAVEFORM_DATA_OFFSET,
        )
        .context("Parsing waveform data offset failed")?;
        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.waveform_packet_size,
            &WAVEFORM_PACKET_SIZE,
        )
        .context("Parsing waveform packet size failed")?;
        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.return_point_waveform_location,
            &RETURN_POINT_WAVEFORM_LOCATION,
        )
        .context("Parsing return point waveform location failed")?;
        parse_vector3_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.waveform_parameters,
            &WAVEFORM_PARAMETERS,
        )
        .context("Parsing waveform parameters failed")?;

        Ok(())
    }

    #[test]
    fn parse_las_format10_default() -> Result<()> {
        let (test_point, expected_point) = get_test_point_las_format10();
        let target_point_layout = LasPointFormat10::layout();
        let las_metadata = get_test_las_metadata(Format::new(10)?)?;

        let mut parsed_point = LasPointFormat10::default();

        let mut parser = PointParser::build(&las_metadata, &target_point_layout)?;
        unsafe {
            let source_data = bytemuck::bytes_of(&test_point);
            let destination_data = bytemuck::bytes_of_mut(&mut parsed_point);
            parser.parse_one(source_data, destination_data);
        }

        assert_eq!(expected_point, parsed_point);

        Ok(())
    }

    #[test]
    fn parse_las_format10_individual_attributes() -> Result<()> {
        let (test_point, expected_point) = get_test_point_las_format10();
        // let target_point_layout = LasPointFormat5::layout();
        let las_metadata = get_test_las_metadata(Format::new(10)?)?;

        let input_point_data = bytemuck::bytes_of(&test_point);

        // Parse all attributes individually
        parse_vector3_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.position,
            &POSITION_3D,
        )
        .context("Parsing position failed")?;
        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.intensity,
            &INTENSITY,
        )
        .context("Parsing intensity failed")?;
        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.return_number,
            &RETURN_NUMBER,
        )
        .context("Parsing return number failed")?;
        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.number_of_returns,
            &NUMBER_OF_RETURNS,
        )
        .context("Parsing number of returns failed")?;
        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.classification_flags,
            &CLASSIFICATION_FLAGS,
        )
        .context("Parsing classification flags failed")?;
        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.scanner_channel,
            &SCANNER_CHANNEL,
        )
        .context("Parsing scanner channel failed")?;

        // bool attributes can only be parsed as bools, pasture does not support bool-to-numeric upcasting!
        unsafe {
            parse_scalar_attribute_in_format(
                input_point_data,
                &las_metadata,
                expected_point.scan_direction_flag,
                &SCAN_DIRECTION_FLAG,
            )
            .context("Parsing scan direction flag failed")?;
            parse_scalar_attribute_in_format(
                input_point_data,
                &las_metadata,
                expected_point.edge_of_flight_line,
                &EDGE_OF_FLIGHT_LINE,
            )
            .context("Parsing edge of flight line failed")?;
        }

        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.classification,
            &CLASSIFICATION,
        )
        .context("Parsing classification failed")?;
        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.scan_angle,
            &SCAN_ANGLE,
        )
        .context("Parsing scan angle failed")?;
        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.user_data,
            &USER_DATA,
        )
        .context("Parsing user data failed")?;
        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.point_source_id,
            &POINT_SOURCE_ID,
        )
        .context("Parsing point source ID failed")?;
        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.gps_time,
            &GPS_TIME,
        )
        .context("Parsing gps time failed")?;
        let expected_color = expected_point.color_rgb;
        parse_vector3_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_color,
            &COLOR_RGB,
        )
        .context("Parsing color failed")?;
        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.nir,
            &NIR,
        )
        .context("Parsing nir failed")?;
        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.wave_packet_descriptor_index,
            &WAVE_PACKET_DESCRIPTOR_INDEX,
        )
        .context("Parsing wave packet descriptor index failed")?;
        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.byte_offset_to_waveform_data,
            &WAVEFORM_DATA_OFFSET,
        )
        .context("Parsing waveform data offset failed")?;
        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.waveform_packet_size,
            &WAVEFORM_PACKET_SIZE,
        )
        .context("Parsing waveform packet size failed")?;
        parse_scalar_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.return_point_waveform_location,
            &RETURN_POINT_WAVEFORM_LOCATION,
        )
        .context("Parsing return point waveform location failed")?;
        parse_vector3_attribute_in_various_formats(
            input_point_data,
            &las_metadata,
            expected_point.waveform_parameters,
            &WAVEFORM_PARAMETERS,
        )
        .context("Parsing waveform parameters failed")?;

        Ok(())
    }
}
