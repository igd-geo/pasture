use std::io::Cursor;

use anyhow::Result;
use byteorder::{NativeEndian, ReadBytesExt};
use pasture_core::{
    layout::attributes,
    layout::conversion::get_converter_for_attributes,
    layout::{
        conversion::AttributeConversionFn, PointAttributeDefinition, PointLayout, PrimitiveType,
    },
    nalgebra::Vector3,
    util::view_raw_bytes_mut,
};

/// ReaderFn is a helper function that allows reading a single value of a specific point attribute from an arbitrary
/// buffer, applying all necessary conversions or falling back to default values if required. This abstraction is
/// necessary to deal with the general case of an arbitrary source point layout in the LASWriter that has to be
/// converted into a specific LAS point layout. For any mandatory LAS attribute, there are three cases:
/// 1) Source layout contains default PointAttributeDefinition for attribute
/// 2) Source layout contains PointAttributeDefinition with a different data type
/// 3) Source layout does not contain PointAttributeDefinition for attribute
/// Depending on the scenario, this requires either a regular read, a converted read, or no read at all. To prevent
/// that we have to handle the three scenarios at every place where we write LAS data, the `ReaderFn` abstraction
/// is introduced
pub(crate) type ReaderFn<T> = Box<dyn Fn(usize, &mut Cursor<Vec<u8>>) -> Result<T>>;

fn read_attribute_in_custom_layout<T: PrimitiveType + Default>(
    attribute_def: &PointAttributeDefinition,
    attribute_offset: usize,
    current_point_index: usize,
    size_of_single_point: usize,
    converter: AttributeConversionFn,
    point_read: &mut Cursor<Vec<u8>>,
) -> Result<T> {
    let attribute_size = attribute_def.size() as usize;
    let attribute_start = (current_point_index * size_of_single_point) + attribute_offset;
    let attribute_slice =
        &point_read.get_ref()[attribute_start..(attribute_start + attribute_size)];

    let mut ret: T = Default::default();
    let ret_slice_mut = unsafe { view_raw_bytes_mut(&mut ret) };

    unsafe {
        converter(attribute_slice, ret_slice_mut);
    }
    Ok(ret)
}

fn read_position_in_default_layout(
    point_read: &mut Cursor<Vec<u8>>,
    attribute_offset: usize,
    current_point_index: usize,
    size_of_single_point: usize,
) -> Result<Vector3<f64>> {
    let attribute_start_pos =
        ((current_point_index * size_of_single_point) + attribute_offset) as u64;
    point_read.set_position(attribute_start_pos);
    let global_x = point_read.read_f64::<NativeEndian>()?;
    let global_y = point_read.read_f64::<NativeEndian>()?;
    let global_z = point_read.read_f64::<NativeEndian>()?;
    Ok(Vector3::new(global_x, global_y, global_z))
}

fn read_intensity_in_default_layout(
    point_read: &mut Cursor<Vec<u8>>,
    attribute_offset: usize,
    current_point_index: usize,
    size_of_single_point: usize,
) -> Result<u16> {
    let attribute_start_pos =
        ((current_point_index * size_of_single_point) + attribute_offset) as u64;
    point_read.set_position(attribute_start_pos);
    Ok(point_read.read_u16::<NativeEndian>()?)
}

fn read_return_number_in_default_layout(
    point_read: &mut Cursor<Vec<u8>>,
    attribute_offset: usize,
    current_point_index: usize,
    size_of_single_point: usize,
) -> Result<u8> {
    let attribute_start_pos =
        ((current_point_index * size_of_single_point) + attribute_offset) as u64;
    point_read.set_position(attribute_start_pos);
    Ok(point_read.read_u8()?)
}

fn read_number_of_returns_in_default_layout(
    point_read: &mut Cursor<Vec<u8>>,
    attribute_offset: usize,
    current_point_index: usize,
    size_of_single_point: usize,
) -> Result<u8> {
    let attribute_start_pos =
        ((current_point_index * size_of_single_point) + attribute_offset) as u64;
    point_read.set_position(attribute_start_pos);
    Ok(point_read.read_u8()?)
}

fn read_classification_flags_in_default_layout(
    point_read: &mut Cursor<Vec<u8>>,
    attribute_offset: usize,
    current_point_index: usize,
    size_of_single_point: usize,
) -> Result<u8> {
    let attribute_start_pos =
        ((current_point_index * size_of_single_point) + attribute_offset) as u64;
    point_read.set_position(attribute_start_pos);
    Ok(point_read.read_u8()?)
}

fn read_scanner_channel_in_default_layout(
    point_read: &mut Cursor<Vec<u8>>,
    attribute_offset: usize,
    current_point_index: usize,
    size_of_single_point: usize,
) -> Result<u8> {
    let attribute_start_pos =
        ((current_point_index * size_of_single_point) + attribute_offset) as u64;
    point_read.set_position(attribute_start_pos);
    Ok(point_read.read_u8()?)
}

fn read_scan_direction_flag_in_default_layout(
    point_read: &mut Cursor<Vec<u8>>,
    attribute_offset: usize,
    current_point_index: usize,
    size_of_single_point: usize,
) -> Result<bool> {
    let attribute_start_pos =
        ((current_point_index * size_of_single_point) + attribute_offset) as u64;
    point_read.set_position(attribute_start_pos);
    Ok(point_read.read_u8()? > 0)
}

fn read_edge_of_flight_line_in_default_layout(
    point_read: &mut Cursor<Vec<u8>>,
    attribute_offset: usize,
    current_point_index: usize,
    size_of_single_point: usize,
) -> Result<bool> {
    let attribute_start_pos =
        ((current_point_index * size_of_single_point) + attribute_offset) as u64;
    point_read.set_position(attribute_start_pos);
    Ok(point_read.read_u8()? > 0)
}

fn read_classification_in_default_layout(
    point_read: &mut Cursor<Vec<u8>>,
    attribute_offset: usize,
    current_point_index: usize,
    size_of_single_point: usize,
) -> Result<u8> {
    let attribute_start_pos =
        ((current_point_index * size_of_single_point) + attribute_offset) as u64;
    point_read.set_position(attribute_start_pos);
    Ok(point_read.read_u8()?)
}

fn read_user_data_in_default_layout(
    point_read: &mut Cursor<Vec<u8>>,
    attribute_offset: usize,
    current_point_index: usize,
    size_of_single_point: usize,
) -> Result<u8> {
    let attribute_start_pos =
        ((current_point_index * size_of_single_point) + attribute_offset) as u64;
    point_read.set_position(attribute_start_pos);
    Ok(point_read.read_u8()?)
}

fn read_scan_angle_rank_in_default_layout(
    point_read: &mut Cursor<Vec<u8>>,
    attribute_offset: usize,
    current_point_index: usize,
    size_of_single_point: usize,
) -> Result<i8> {
    let attribute_start_pos =
        ((current_point_index * size_of_single_point) + attribute_offset) as u64;
    point_read.set_position(attribute_start_pos);
    Ok(point_read.read_i8()?)
}

fn read_extended_scan_angle_rank_in_default_layout(
    point_read: &mut Cursor<Vec<u8>>,
    attribute_offset: usize,
    current_point_index: usize,
    size_of_single_point: usize,
) -> Result<i16> {
    let attribute_start_pos =
        ((current_point_index * size_of_single_point) + attribute_offset) as u64;
    point_read.set_position(attribute_start_pos);
    Ok(point_read.read_i16::<NativeEndian>()?)
}

fn read_point_source_id_in_default_layout(
    point_read: &mut Cursor<Vec<u8>>,
    attribute_offset: usize,
    current_point_index: usize,
    size_of_single_point: usize,
) -> Result<u16> {
    let attribute_start_pos =
        ((current_point_index * size_of_single_point) + attribute_offset) as u64;
    point_read.set_position(attribute_start_pos);
    Ok(point_read.read_u16::<NativeEndian>()?)
}

fn read_gps_time_in_default_layout(
    point_read: &mut Cursor<Vec<u8>>,
    attribute_offset: usize,
    current_point_index: usize,
    size_of_single_point: usize,
) -> Result<f64> {
    let attribute_start_pos =
        ((current_point_index * size_of_single_point) + attribute_offset) as u64;
    point_read.set_position(attribute_start_pos);
    Ok(point_read.read_f64::<NativeEndian>()?)
}

fn read_color_rgb_in_default_layout(
    point_read: &mut Cursor<Vec<u8>>,
    attribute_offset: usize,
    current_point_index: usize,
    size_of_single_point: usize,
) -> Result<Vector3<u16>> {
    let attribute_start_pos =
        ((current_point_index * size_of_single_point) + attribute_offset) as u64;
    point_read.set_position(attribute_start_pos);
    let r = point_read.read_u16::<NativeEndian>()?;
    let g = point_read.read_u16::<NativeEndian>()?;
    let b = point_read.read_u16::<NativeEndian>()?;
    Ok(Vector3::new(r, g, b))
}

fn read_nir_in_default_layout(
    point_read: &mut Cursor<Vec<u8>>,
    attribute_offset: usize,
    current_point_index: usize,
    size_of_single_point: usize,
) -> Result<u16> {
    let attribute_start_pos =
        ((current_point_index * size_of_single_point) + attribute_offset) as u64;
    point_read.set_position(attribute_start_pos);
    Ok(point_read.read_u16::<NativeEndian>()?)
}

fn read_wave_packet_descriptor_index_in_default_layout(
    point_read: &mut Cursor<Vec<u8>>,
    attribute_offset: usize,
    current_point_index: usize,
    size_of_single_point: usize,
) -> Result<u8> {
    let attribute_start_pos =
        ((current_point_index * size_of_single_point) + attribute_offset) as u64;
    point_read.set_position(attribute_start_pos);
    Ok(point_read.read_u8()?)
}

fn read_byte_offset_to_waveform_data_in_default_layout(
    point_read: &mut Cursor<Vec<u8>>,
    attribute_offset: usize,
    current_point_index: usize,
    size_of_single_point: usize,
) -> Result<u64> {
    let attribute_start_pos =
        ((current_point_index * size_of_single_point) + attribute_offset) as u64;
    point_read.set_position(attribute_start_pos);
    Ok(point_read.read_u64::<NativeEndian>()?)
}

fn read_waveform_packet_size_in_default_layout(
    point_read: &mut Cursor<Vec<u8>>,
    attribute_offset: usize,
    current_point_index: usize,
    size_of_single_point: usize,
) -> Result<u32> {
    let attribute_start_pos =
        ((current_point_index * size_of_single_point) + attribute_offset) as u64;
    point_read.set_position(attribute_start_pos);
    Ok(point_read.read_u32::<NativeEndian>()?)
}

fn read_return_point_waveform_location_in_default_layout(
    point_read: &mut Cursor<Vec<u8>>,
    attribute_offset: usize,
    current_point_index: usize,
    size_of_single_point: usize,
) -> Result<f32> {
    let attribute_start_pos =
        ((current_point_index * size_of_single_point) + attribute_offset) as u64;
    point_read.set_position(attribute_start_pos);
    Ok(point_read.read_f32::<NativeEndian>()?)
}

fn read_waveform_parameters_in_default_layout(
    point_read: &mut Cursor<Vec<u8>>,
    attribute_offset: usize,
    current_point_index: usize,
    size_of_single_point: usize,
) -> Result<Vector3<f32>> {
    let attribute_start_pos =
        ((current_point_index * size_of_single_point) + attribute_offset) as u64;
    point_read.set_position(attribute_start_pos);
    let dx = point_read.read_f32::<NativeEndian>()?;
    let dy = point_read.read_f32::<NativeEndian>()?;
    let dz = point_read.read_f32::<NativeEndian>()?;
    Ok(Vector3::new(dx, dy, dz))
}

macro_rules! make_get_reader_fn {
    ($name:ident, $type:ty, $attribute:ident, $read_default_fn:ident) => {
        pub(crate) fn $name(source_layout: &PointLayout) -> ReaderFn<$type> {
            let default_attribute = attributes::$attribute;
            let source_attribute = source_layout.get_attribute_by_name(default_attribute.name());

            match source_attribute {
                None => Box::new(|_, _| -> Result<$type> { Ok(Default::default()) }),
                Some(attribute) => {
                    if attribute.datatype() == default_attribute.datatype() {
                        let offset_in_point = source_layout
                            .offset_of(attribute)
                            .expect("Attribute offset not found")
                            as usize;
                        let size_of_single_point = source_layout.size_of_point_entry() as usize;
                        Box::new(move |current_point_index, point_read| {
                            $read_default_fn(
                                point_read,
                                offset_in_point,
                                current_point_index,
                                size_of_single_point,
                            )
                        })
                    } else {
                        let attribute_clone = attribute.clone();
                        let offset_in_point = source_layout
                            .offset_of(attribute)
                            .expect("Attribute offset not found")
                            as usize;
                        let size_of_single_point = source_layout.size_of_point_entry() as usize;
                        let converter = get_converter_for_attributes(attribute, &default_attribute)
                            .expect("No converter for attribute found");
                        Box::new(move |current_point_index, point_read| {
                            read_attribute_in_custom_layout::<$type>(
                                &attribute_clone,
                                offset_in_point,
                                current_point_index,
                                size_of_single_point,
                                converter,
                                point_read,
                            )
                        })
                    }
                }
            }
        }
    };
}

make_get_reader_fn!(
    get_position_reader,
    Vector3<f64>,
    POSITION_3D,
    read_position_in_default_layout
);

make_get_reader_fn!(
    get_intensity_reader,
    u16,
    INTENSITY,
    read_intensity_in_default_layout
);

make_get_reader_fn!(
    get_return_number_reader,
    u8,
    RETURN_NUMBER,
    read_return_number_in_default_layout
);

make_get_reader_fn!(
    get_number_of_returns_reader,
    u8,
    NUMBER_OF_RETURNS,
    read_number_of_returns_in_default_layout
);

make_get_reader_fn!(
    get_classification_flags_reader,
    u8,
    CLASSIFICATION_FLAGS,
    read_classification_flags_in_default_layout
);

make_get_reader_fn!(
    get_scanner_channel_reader,
    u8,
    SCANNER_CHANNEL,
    read_scanner_channel_in_default_layout
);

make_get_reader_fn!(
    get_scan_direction_flag_reader,
    bool,
    SCAN_DIRECTION_FLAG,
    read_scan_direction_flag_in_default_layout
);

make_get_reader_fn!(
    get_edge_of_flight_line_reader,
    bool,
    EDGE_OF_FLIGHT_LINE,
    read_edge_of_flight_line_in_default_layout
);

make_get_reader_fn!(
    get_classification_reader,
    u8,
    CLASSIFICATION,
    read_classification_in_default_layout
);

make_get_reader_fn!(
    get_user_data_reader,
    u8,
    USER_DATA,
    read_user_data_in_default_layout
);

make_get_reader_fn!(
    get_scan_angle_rank_reader,
    i8,
    SCAN_ANGLE_RANK,
    read_scan_angle_rank_in_default_layout
);

make_get_reader_fn!(
    get_extended_scan_angle_rank_reader,
    i16,
    SCAN_ANGLE_RANK_EXTENDED,
    read_extended_scan_angle_rank_in_default_layout
);

make_get_reader_fn!(
    get_point_source_id_reader,
    u16,
    POINT_SOURCE_ID,
    read_point_source_id_in_default_layout
);

make_get_reader_fn!(
    get_gps_time_reader,
    f64,
    GPS_TIME,
    read_gps_time_in_default_layout
);

make_get_reader_fn!(
    get_color_reader,
    Vector3<u16>,
    COLOR_RGB,
    read_color_rgb_in_default_layout
);

make_get_reader_fn!(get_nir_reader, u16, NIR, read_nir_in_default_layout);

make_get_reader_fn!(
    get_wave_packet_descriptor_index_reader,
    u8,
    WAVE_PACKET_DESCRIPTOR_INDEX,
    read_wave_packet_descriptor_index_in_default_layout
);

make_get_reader_fn!(
    get_waveform_data_offset_reader,
    u64,
    WAVEFORM_DATA_OFFSET,
    read_byte_offset_to_waveform_data_in_default_layout
);

make_get_reader_fn!(
    get_waveform_packet_size_reader,
    u32,
    WAVEFORM_PACKET_SIZE,
    read_waveform_packet_size_in_default_layout
);

make_get_reader_fn!(
    get_return_point_waveform_location_reader,
    f32,
    RETURN_POINT_WAVEFORM_LOCATION,
    read_return_point_waveform_location_in_default_layout
);

make_get_reader_fn!(
    get_waveform_parameters_reader,
    Vector3<f32>,
    WAVEFORM_PARAMETERS,
    read_waveform_parameters_in_default_layout
);
