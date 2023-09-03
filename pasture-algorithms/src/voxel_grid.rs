use std::{collections::HashMap, u16};

use pasture_core::{
    containers::{BorrowedBuffer, OwningBuffer, UntypedPoint, UntypedPointBuffer},
    layout::{
        attributes::{self, POSITION_3D},
        PointAttributeDataType, PointAttributeDefinition, PointLayout,
    },
    nalgebra::Vector3,
};

use crate::bounds::calculate_bounds;

pub struct Voxel {
    pos: (usize, usize, usize),
    points: Vec<usize>,
}

/// finds leaf of point p by iterating over the marked axis
fn find_leaf(
    p: Vector3<f64>,
    markers_x: &Vec<f64>,
    markers_y: &Vec<f64>,
    markers_z: &Vec<f64>,
) -> (usize, usize, usize) {
    let mut index_x = 0;
    let mut index_y = 0;
    let mut index_z = 0;
    while !markers_x.is_empty() && markers_x[index_x] < p.x {
        index_x += 1;
    }
    while !markers_y.is_empty() && markers_y[index_y] < p.y {
        index_y += 1;
    }
    while !markers_z.is_empty() && markers_z[index_z] < p.z {
        index_z += 1;
    }
    // clamp values to the better fitting marker: [i] or [i-1]
    if index_x > 0 && p.x - markers_x[index_x - 1] < markers_x[index_x] - p.x {
        index_x -= 1;
    }
    if index_y > 0 && p.y - markers_y[index_y - 1] < markers_y[index_y] - p.y {
        index_y -= 1;
    }
    if index_z > 0 && p.z - markers_z[index_z - 1] < markers_z[index_z] - p.z {
        index_z -= 1;
    }
    (index_x, index_y, index_z)
}

/// marks the axis with leafsize between markers
fn create_markers_for_axis(
    aabb: pasture_core::math::AABB<f64>,
    leafsize_x: f64,
    leafsize_y: f64,
    leafsize_z: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut markers_x = vec![];
    let mut markers_y = vec![];
    let mut markers_z = vec![];
    let mut curr_x = aabb.min().x;
    let mut curr_y = aabb.min().y;
    let mut curr_z = aabb.min().z;
    while curr_x < aabb.max().x {
        curr_x += leafsize_x;
        markers_x.push(curr_x);
    }
    while curr_y < aabb.max().y {
        curr_y += leafsize_y;
        markers_y.push(curr_y);
    }
    while curr_z < aabb.max().z {
        curr_z += leafsize_z;
        markers_z.push(curr_z);
    }
    (markers_x, markers_y, markers_z)
}

/// Downsamples `buffer` by applying a voxelgrid-filter.
/// Currently only works for point_layouts which only contain the builtin-types.
/// Writes results into `filtered_buffer`.
///
/// # Examples
/// ```
/// # use pasture_algorithms::voxel_grid::voxelgrid_filter;
/// # use pasture_core::{containers::*, layout::{attributes, PointType}, nalgebra::{Scalar, Vector3}};
/// # use pasture_derive::PointType;
/// #[repr(C)]
/// #[derive(PointType, Debug, Clone, Copy, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
/// # struct SimplePoint {
/// #    #[pasture(BUILTIN_POSITION_3D)]
/// #   pub position: Vector3<f64>,
/// # }
/// let mut points = vec![];
/// // generate some points
/// for i in 0..100{
///     for j in 0..100{
///         points.push(SimplePoint{position: Vector3::new(0.0, f64::from(i), f64::from(j))});
///     }
/// }
/// let buffer = points.into_iter().collect::<HashMapBuffer>();
/// let mut filtered = HashMapBuffer::new_from_layout(buffer.point_layout().clone());
/// voxelgrid_filter(&buffer, 1.5, 1.5, 1.5, &mut filtered);
/// // filtered now has fewer points than buffer
/// assert!(filtered.len() < buffer.len() / 2);
/// ```
pub fn voxelgrid_filter<'a, 'b, PB: BorrowedBuffer<'a>, PBW: OwningBuffer<'b>>(
    buffer: &'a PB,
    leafsize_x: f64,
    leafsize_y: f64,
    leafsize_z: f64,
    filtered_buffer: &'b mut PBW,
) {
    if !buffer
        .point_layout()
        .has_attribute(&attributes::POSITION_3D)
    {
        panic!("The PointBuffer does not have the attribute attributes::POSITION_3D which is needed for the creation of the voxel grid.");
    }

    // get the bounding box of the pointcloud
    let aabb = calculate_bounds(buffer).unwrap();

    // create arrays with all markers of one axis
    let (markers_x, markers_y, markers_z) =
        create_markers_for_axis(aabb, leafsize_x, leafsize_y, leafsize_z);

    let mut voxels: Vec<Voxel> = Vec::with_capacity(buffer.len());

    // create the VoxelGrid
    for (i, p) in buffer
        .view_attribute::<Vector3<f64>>(&POSITION_3D)
        .into_iter()
        .enumerate()
    {
        // get position of p's leaf
        let pos = find_leaf(p, &markers_x, &markers_y, &markers_z);
        // associate the right voxel with the point
        match voxels.binary_search_by_key(&pos, |v| v.pos) {
            // voxel already exists -> push p to points
            Ok(index) => voxels[index].points.push(i),
            // voxel not existing -> create new voxel
            Err(index) => voxels.insert(
                index,
                Voxel {
                    pos,
                    points: vec![i],
                },
            ),
        }
    }
    // approximate the centroid point in the voxel for all attributes in the given buffer
    for v in &mut voxels {
        let layout = filtered_buffer.point_layout().clone();
        // using untyped point for now -> in future maybe different
        let mut centroid = UntypedPointBuffer::new(&layout);
        set_all_attributes(&layout, &mut centroid, v, buffer);
        // This is safe because the `centroid` has the same `PointLayout` as `filtered_buffer`
        unsafe {
            filtered_buffer.push_points(centroid.get_cursor().into_inner());
        }
    }
}

/// calculates the attribute attribute_definition via max-pooling
fn centroid_max_pool<'a, T: BorrowedBuffer<'a>>(
    v: &Voxel,
    buffer: &'a T,
    attribute_definition: &PointAttributeDefinition,
    point_type: PointAttributeDataType,
) -> f64 {
    let mut curr_max = 0.0;
    let mut value;
    for p in &v.points {
        match point_type {
            PointAttributeDataType::U8 => {
                value = buffer.view_attribute::<u8>(attribute_definition).at(*p) as f64;
            }
            PointAttributeDataType::I8 => {
                value = buffer.view_attribute::<i8>(attribute_definition).at(*p) as f64;
            }
            PointAttributeDataType::U16 => {
                value = buffer.view_attribute::<u16>(attribute_definition).at(*p) as f64;
            }
            PointAttributeDataType::I16 => {
                value = buffer.view_attribute::<i16>(attribute_definition).at(*p) as f64;
            }
            PointAttributeDataType::U32 => {
                value = buffer.view_attribute::<u32>(attribute_definition).at(*p) as f64;
            }
            PointAttributeDataType::I32 => {
                value = buffer.view_attribute::<i32>(attribute_definition).at(*p) as f64;
            }
            PointAttributeDataType::U64 => {
                value = buffer.view_attribute::<u64>(attribute_definition).at(*p) as f64;
            }
            PointAttributeDataType::I64 => {
                value = buffer.view_attribute::<i64>(attribute_definition).at(*p) as f64;
            }
            PointAttributeDataType::F32 => {
                value = buffer.view_attribute::<f32>(attribute_definition).at(*p) as f64;
            }
            PointAttributeDataType::F64 => {
                value = buffer.view_attribute::<f64>(attribute_definition).at(*p);
            }
            _ => unimplemented!(),
        }
        if value > curr_max {
            curr_max = value;
        }
    }
    curr_max
}

/// returns the most common value in the voxel for attribute_definition
fn centroid_most_common<'a, T: BorrowedBuffer<'a>>(
    v: &Voxel,
    buffer: &'a T,
    attribute_definition: &PointAttributeDefinition,
    point_type: PointAttributeDataType,
) -> isize {
    let mut map = HashMap::new();
    for p in &v.points {
        match point_type {
            PointAttributeDataType::U8 => {
                *map.entry(
                    buffer
                        .view_attribute::<u8>(attribute_definition)
                        .at(*p)
                        .to_string(),
                )
                .or_insert(0) += 1
            }
            PointAttributeDataType::I8 => {
                *map.entry(
                    buffer
                        .view_attribute::<i8>(attribute_definition)
                        .at(*p)
                        .to_string(),
                )
                .or_insert(0) += 1
            }
            PointAttributeDataType::U16 => {
                *map.entry(
                    buffer
                        .view_attribute::<u16>(attribute_definition)
                        .at(*p)
                        .to_string(),
                )
                .or_insert(0) += 1
            }
            PointAttributeDataType::I16 => {
                *map.entry(
                    buffer
                        .view_attribute::<i16>(attribute_definition)
                        .at(*p)
                        .to_string(),
                )
                .or_insert(0) += 1
            }
            PointAttributeDataType::U32 => {
                *map.entry(
                    buffer
                        .view_attribute::<u32>(attribute_definition)
                        .at(*p)
                        .to_string(),
                )
                .or_insert(0) += 1
            }
            PointAttributeDataType::I32 => {
                *map.entry(
                    buffer
                        .view_attribute::<i32>(attribute_definition)
                        .at(*p)
                        .to_string(),
                )
                .or_insert(0) += 1
            }
            PointAttributeDataType::U64 => {
                *map.entry(
                    buffer
                        .view_attribute::<u64>(attribute_definition)
                        .at(*p)
                        .to_string(),
                )
                .or_insert(0) += 1
            }
            PointAttributeDataType::I64 => {
                *map.entry(
                    buffer
                        .view_attribute::<i64>(attribute_definition)
                        .at(*p)
                        .to_string(),
                )
                .or_insert(0) += 1
            }
            PointAttributeDataType::F32 => {
                *map.entry(
                    buffer
                        .view_attribute::<f32>(attribute_definition)
                        .at(*p)
                        .to_string(),
                )
                .or_insert(0) += 1
            }
            PointAttributeDataType::F64 => {
                *map.entry(
                    buffer
                        .view_attribute::<f64>(attribute_definition)
                        .at(*p)
                        .to_string(),
                )
                .or_insert(0) += 1
            }
            _ => unimplemented!(),
        }
    }
    let mut highest_count = 0;
    let mut curr_key: String = "".to_string();
    for (key, value) in map {
        if value > highest_count {
            highest_count = value;
            curr_key = key;
        }
    }
    curr_key.parse::<isize>().unwrap()
}

/// returns the average value in the voxel for attribute_definition
/// vector types only
fn centroid_average_vec<'a, T: BorrowedBuffer<'a>>(
    v: &Voxel,
    buffer: &'a T,
    attribute_definition: &PointAttributeDefinition,
    point_type: PointAttributeDataType,
) -> Vector3<f64> {
    let mut x_sum = 0.0;
    let mut y_sum = 0.0;
    let mut z_sum = 0.0;
    for p in &v.points {
        match point_type {
            PointAttributeDataType::Vec3u8 => {
                let vec = buffer
                    .view_attribute::<Vector3<u8>>(attribute_definition)
                    .at(*p);
                x_sum += vec.x as f64;
                y_sum += vec.y as f64;
                z_sum += vec.z as f64;
            }
            PointAttributeDataType::Vec3u16 => {
                let vec = buffer
                    .view_attribute::<Vector3<u16>>(attribute_definition)
                    .at(*p);
                x_sum += vec.x as f64;
                y_sum += vec.y as f64;
                z_sum += vec.z as f64;
            }
            PointAttributeDataType::Vec3f32 => {
                let vec = buffer
                    .view_attribute::<Vector3<f32>>(attribute_definition)
                    .at(*p);
                x_sum += vec.x as f64;
                y_sum += vec.y as f64;
                z_sum += vec.z as f64;
            }
            PointAttributeDataType::Vec3f64 => {
                let vec = buffer
                    .view_attribute::<Vector3<f64>>(attribute_definition)
                    .at(*p);
                x_sum += vec.x;
                y_sum += vec.y;
                z_sum += vec.z;
            }
            PointAttributeDataType::Vec4u8 => unimplemented!(),
            _ => panic!("Invalid data type for centroid_average_vec"),
        }
    }

    // average over all values of one axis is the centroid's axis point
    let num_of_points = v.points.len() as f64;
    let centroid_x = x_sum / num_of_points;
    let centroid_y = y_sum / num_of_points;
    let centroid_z = z_sum / num_of_points;
    Vector3::new(centroid_x, centroid_y, centroid_z)
}

/// returns the average value in the voxel for attribute_definition
/// numeric types only
fn centroid_average_num<'a, PB: BorrowedBuffer<'a>>(
    v: &mut Voxel,
    buffer: &'a PB,
    attribute_definition: &PointAttributeDefinition,
    point_type: PointAttributeDataType,
) -> f64 {
    let mut sum = 0.0;
    for p in &v.points {
        match point_type {
            PointAttributeDataType::U8 => {
                sum += buffer.view_attribute::<u8>(attribute_definition).at(*p) as f64
            }
            PointAttributeDataType::I8 => {
                sum += buffer.view_attribute::<i8>(attribute_definition).at(*p) as f64
            }
            PointAttributeDataType::U16 => {
                sum += buffer.view_attribute::<u16>(attribute_definition).at(*p) as f64
            }
            PointAttributeDataType::I16 => {
                sum += buffer.view_attribute::<i16>(attribute_definition).at(*p) as f64
            }
            PointAttributeDataType::U32 => {
                sum += buffer.view_attribute::<u32>(attribute_definition).at(*p) as f64
            }
            PointAttributeDataType::I32 => {
                sum += buffer.view_attribute::<i32>(attribute_definition).at(*p) as f64
            }
            PointAttributeDataType::U64 => {
                sum += buffer.view_attribute::<u64>(attribute_definition).at(*p) as f64
            }
            PointAttributeDataType::I64 => {
                sum += buffer.view_attribute::<i64>(attribute_definition).at(*p) as f64
            }
            PointAttributeDataType::F32 => {
                sum += buffer.view_attribute::<f32>(attribute_definition).at(*p) as f64
            }
            PointAttributeDataType::F64 => {
                sum += buffer.view_attribute::<f64>(attribute_definition).at(*p)
            }
            PointAttributeDataType::Vec3u8 => panic!("For vector types use centroid_average_vec."),
            PointAttributeDataType::Vec3u16 => panic!("For vector types use centroid_average_vec."),
            PointAttributeDataType::Vec3f32 => panic!("For vector types use centroid_average_vec."),
            PointAttributeDataType::Vec3f64 => panic!("For vector types use centroid_average_vec."),
            PointAttributeDataType::Vec4u8 => panic!("For vector types use centroid_average_vec."),
            _ => unimplemented!(),
        }
    }
    sum / v.points.len() as f64
}

/// sets all attributes of the point-buffer for the centroid
/// currently, only standard builtin types work.
fn set_all_attributes<'a, PB: BorrowedBuffer<'a>>(
    target_layout: &PointLayout,
    centroid: &mut UntypedPointBuffer,
    v: &mut Voxel,
    buffer: &'a PB,
) {
    //TODO: for now we just check that the layout of the filtered_buffer does not contain any waveform values.
    // A future version of this algorithm should take a separate object that contains a mapping between point attributes and the desired type of 'reduction function'.
    // This way we could store defaults in this object but have the users overwrite the defaults with whatever they want.
    if target_layout.has_attribute(&attributes::WAVEFORM_DATA_OFFSET)
        || target_layout.has_attribute(&attributes::WAVEFORM_PACKET_SIZE)
        || target_layout.has_attribute(&attributes::WAVEFORM_PARAMETERS)
        || target_layout.has_attribute(&attributes::WAVE_PACKET_DESCRIPTOR_INDEX)
        || target_layout.has_attribute(&attributes::RETURN_POINT_WAVEFORM_LOCATION)
    {
        panic!("Waveform data currently not supported!");
    }

    for a in target_layout.attributes() {
        if a.name() == attributes::POSITION_3D.name()
            && a.datatype() == attributes::POSITION_3D.datatype()
        {
            let position = centroid_average_vec(v, buffer, &attributes::POSITION_3D, a.datatype());
            let pos_slice = bytemuck::bytes_of(&position);
            centroid
                .set_raw_attribute(&attributes::POSITION_3D, pos_slice)
                .unwrap();
        } else if a.name() == attributes::INTENSITY.name()
            && a.datatype() == attributes::INTENSITY.datatype()
        {
            let average =
                centroid_average_num::<PB>(v, buffer, &attributes::INTENSITY, a.datatype()) as u16;
            let avg_slice = bytemuck::bytes_of(&average);
            centroid
                .set_raw_attribute(&attributes::INTENSITY, avg_slice)
                .unwrap();
        } else if a.name() == attributes::RETURN_NUMBER.name()
            && a.datatype() == attributes::RETURN_NUMBER.datatype()
        {
            let most_common =
                centroid_most_common::<PB>(v, buffer, &attributes::RETURN_NUMBER, a.datatype())
                    as u8;
            let mc_slice = bytemuck::bytes_of(&most_common);
            centroid
                .set_raw_attribute(&attributes::RETURN_NUMBER, mc_slice)
                .unwrap();
        } else if a.name() == attributes::NUMBER_OF_RETURNS.name()
            && a.datatype() == attributes::NUMBER_OF_RETURNS.datatype()
        {
            let most_common =
                centroid_most_common::<PB>(v, buffer, &attributes::NUMBER_OF_RETURNS, a.datatype())
                    as u8;
            let mc_slice = bytemuck::bytes_of(&most_common);
            centroid
                .set_raw_attribute(&attributes::NUMBER_OF_RETURNS, mc_slice)
                .unwrap();
        } else if a.name() == attributes::CLASSIFICATION_FLAGS.name()
            && a.datatype() == attributes::CLASSIFICATION_FLAGS.datatype()
        {
            let max =
                centroid_max_pool::<PB>(v, buffer, &attributes::CLASSIFICATION_FLAGS, a.datatype())
                    as u8;
            let m_slice = bytemuck::bytes_of(&max);
            centroid
                .set_raw_attribute(&attributes::CLASSIFICATION_FLAGS, m_slice)
                .unwrap();
        } else if a.name() == attributes::SCANNER_CHANNEL.name()
            && a.datatype() == attributes::SCANNER_CHANNEL.datatype()
        {
            let most_common =
                centroid_most_common::<PB>(v, buffer, &attributes::SCANNER_CHANNEL, a.datatype())
                    as u8;
            let mc_slice = bytemuck::bytes_of(&most_common);
            centroid
                .set_raw_attribute(&attributes::SCANNER_CHANNEL, mc_slice)
                .unwrap();
        } else if a.name() == attributes::SCAN_DIRECTION_FLAG.name()
            && a.datatype() == attributes::SCAN_DIRECTION_FLAG.datatype()
        {
            let most_common = centroid_most_common::<PB>(
                v,
                buffer,
                &attributes::SCAN_DIRECTION_FLAG,
                a.datatype(),
            ) != 0;
            let mc_slice = bytemuck::bytes_of(&most_common);
            centroid
                .set_raw_attribute(&attributes::SCAN_DIRECTION_FLAG, mc_slice)
                .unwrap();
        } else if a.name() == attributes::EDGE_OF_FLIGHT_LINE.name()
            && a.datatype() == attributes::EDGE_OF_FLIGHT_LINE.datatype()
        {
            let most_common = centroid_most_common::<PB>(
                v,
                buffer,
                &attributes::EDGE_OF_FLIGHT_LINE,
                a.datatype(),
            ) != 0;
            let mc_slice = bytemuck::bytes_of(&most_common);
            centroid
                .set_raw_attribute(&attributes::EDGE_OF_FLIGHT_LINE, mc_slice)
                .unwrap();
        } else if a.name() == attributes::CLASSIFICATION.name()
            && a.datatype() == attributes::CLASSIFICATION.datatype()
        {
            let most_common =
                centroid_most_common::<PB>(v, buffer, &attributes::CLASSIFICATION, a.datatype())
                    as u8;
            let mc_slice = bytemuck::bytes_of(&most_common);
            centroid
                .set_raw_attribute(&attributes::CLASSIFICATION, mc_slice)
                .unwrap();
        } else if a.name() == attributes::SCAN_ANGLE_RANK.name()
            && a.datatype() == attributes::SCAN_ANGLE_RANK.datatype()
        {
            let most_common =
                centroid_most_common::<PB>(v, buffer, &attributes::SCAN_ANGLE_RANK, a.datatype())
                    as i8;
            let mc_slice = bytemuck::bytes_of(&most_common);
            centroid
                .set_raw_attribute(&attributes::SCAN_ANGLE_RANK, mc_slice)
                .unwrap();
        } else if a.name() == attributes::SCAN_ANGLE.name()
            && a.datatype() == attributes::SCAN_ANGLE.datatype()
        {
            let most_common =
                centroid_most_common::<PB>(v, buffer, &attributes::SCAN_ANGLE, a.datatype()) as i16;
            let mc_slice = bytemuck::bytes_of(&most_common);
            centroid
                .set_raw_attribute(&attributes::SCAN_ANGLE, mc_slice)
                .unwrap();
        } else if a.name() == attributes::USER_DATA.name()
            && a.datatype() == attributes::USER_DATA.datatype()
        {
            let most_common =
                centroid_most_common::<PB>(v, buffer, &attributes::USER_DATA, a.datatype()) as u8;
            let mc_slice = bytemuck::bytes_of(&most_common);
            centroid
                .set_raw_attribute(&attributes::USER_DATA, mc_slice)
                .unwrap();
        } else if a.name() == attributes::POINT_SOURCE_ID.name()
            && a.datatype() == attributes::POINT_SOURCE_ID.datatype()
        {
            let most_common =
                centroid_most_common::<PB>(v, buffer, &attributes::POINT_SOURCE_ID, a.datatype())
                    as u16;
            let mc_slice = bytemuck::bytes_of(&most_common);
            centroid
                .set_raw_attribute(&attributes::POINT_SOURCE_ID, mc_slice)
                .unwrap();
        } else if a.name() == attributes::COLOR_RGB.name()
            && a.datatype() == attributes::COLOR_RGB.datatype()
        {
            let color = centroid_average_vec(v, buffer, &attributes::COLOR_RGB, a.datatype());
            let color_u16 = Vector3::new(color.x as u16, color.y as u16, color.z as u16);
            let col_slice = bytemuck::bytes_of(&color_u16);
            centroid
                .set_raw_attribute(&attributes::COLOR_RGB, col_slice)
                .unwrap();
        } else if a.name() == attributes::GPS_TIME.name()
            && a.datatype() == attributes::GPS_TIME.datatype()
        {
            let max = centroid_max_pool::<PB>(v, buffer, &attributes::GPS_TIME, a.datatype());
            let m_slice = bytemuck::bytes_of(&max);
            centroid
                .set_raw_attribute(&attributes::GPS_TIME, m_slice)
                .unwrap();
        } else if a.name() == attributes::NIR.name() && a.datatype() == attributes::NIR.datatype() {
            let average =
                centroid_average_num::<PB>(v, buffer, &attributes::NIR, a.datatype()) as u16;
            let avg_slice = bytemuck::bytes_of(&average);
            centroid
                .set_raw_attribute(&attributes::NIR, avg_slice)
                .unwrap();

        // The waveform-data attributes are implemented in the following lines, but there is no use-case for it currently.
        /* } else if &a.name() == &attributes::WAVE_PACKET_DESCRIPTOR_INDEX.name() {
            let most_common = centroid_most_common::<PB>(
                v,
                buffer,
                &attributes::WAVE_PACKET_DESCRIPTOR_INDEX,
                a.datatype(),
            ) as u8;
            let mc_slice = unsafe { view_raw_bytes(&most_common) };
            &centroid.set_raw_attribute(&attributes::WAVE_PACKET_DESCRIPTOR_INDEX, mc_slice);
        } else if &a.name() == &attributes::WAVEFORM_DATA_OFFSET.name() {
            let most_common = centroid_most_common::<PB>(
                v,
                buffer,
                &attributes::WAVEFORM_DATA_OFFSET,
                a.datatype(),
            ) as u64;
            let mc_slice = unsafe { view_raw_bytes(&most_common) };
            &centroid.set_raw_attribute(&attributes::WAVEFORM_DATA_OFFSET, mc_slice);
        } else if &a.name() == &attributes::WAVEFORM_PACKET_SIZE.name() {
            let most_common = centroid_most_common::<PB>(
                v,
                buffer,
                &attributes::WAVEFORM_PACKET_SIZE,
                a.datatype(),
            ) as u32;
            let mc_slice = unsafe { view_raw_bytes(&most_common) };
            &centroid.set_raw_attribute(&attributes::WAVEFORM_PACKET_SIZE, mc_slice);
        } else if &a.name() == &attributes::RETURN_POINT_WAVEFORM_LOCATION.name() {
            let average = centroid_average_num::<PB>(
                v,
                buffer,
                &attributes::RETURN_POINT_WAVEFORM_LOCATION,
                a.datatype(),
            ) as f32;
            let avg_slice = unsafe { view_raw_bytes(&average) };
            &centroid.set_raw_attribute(&attributes::RETURN_POINT_WAVEFORM_LOCATION, avg_slice);
        } else if &a.name() == &attributes::WAVEFORM_PARAMETERS.name() {
            let params =
                centroid_average_vec(v, buffer, &attributes::WAVEFORM_PARAMETERS, a.datatype());
            let params_f32 = Vector3::new(params.x as f32, params.y as f32, params.z as f32);
            let par_slice = unsafe { view_raw_bytes(&params_f32) };
            &centroid.set_raw_attribute(&attributes::WAVEFORM_PARAMETERS, par_slice); */
        } else if a.name() == attributes::POINT_ID.name()
            && a.datatype() == attributes::POINT_ID.datatype()
        {
            let max =
                centroid_max_pool::<PB>(v, buffer, &attributes::POINT_ID, a.datatype()) as u64;
            let m_slice = bytemuck::bytes_of(&max);
            centroid
                .set_raw_attribute(&attributes::POINT_ID, m_slice)
                .unwrap();
        } else if a.name() == attributes::NORMAL.name()
            && a.datatype() == attributes::NORMAL.datatype()
        {
            let normal = centroid_average_vec(v, buffer, &attributes::NORMAL, a.datatype());
            let normal_f32 = Vector3::new(normal.x as f32, normal.y as f32, normal.z as f32);
            let nor_slice = bytemuck::bytes_of(&normal_f32);
            centroid
                .set_raw_attribute(&attributes::NORMAL, nor_slice)
                .unwrap();
        }
        // we have a non-standard attribute -> use max-pooling for numbers and average for vec
        // currently, only f64 and Vec3f64 is supported
        else {
            panic!(
                "attribute is non-standard which is not supported currently: {:?}",
                a
            );
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::voxel_grid::voxelgrid_filter;
    use pasture_core::{
        containers::{BorrowedBuffer, HashMapBuffer, MakeBufferFromLayout},
        layout::attributes,
        nalgebra::Vector3,
    };
    use pasture_derive::PointType;
    use rand::{prelude::ThreadRng, Rng};

    #[repr(C, packed)]
    #[derive(PointType, Debug, Copy, Clone, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
    pub struct CompletePoint {
        #[pasture(BUILTIN_POSITION_3D)]
        pub position: Vector3<f64>,
        #[pasture(BUILTIN_INTENSITY)]
        pub intensity: u16,
        #[pasture(BUILTIN_RETURN_NUMBER)]
        pub return_number: u8,
        #[pasture(BUILTIN_NUMBER_OF_RETURNS)]
        pub num_of_returns: u8,
        #[pasture(BUILTIN_CLASSIFICATION_FLAGS)]
        pub classification_flags: u8,
        #[pasture(BUILTIN_SCANNER_CHANNEL)]
        pub scanner_channel: u8,
        #[pasture(BUILTIN_SCAN_DIRECTION_FLAG)]
        pub scan_dir_flag: u8,
        #[pasture(BUILTIN_EDGE_OF_FLIGHT_LINE)]
        pub edge_of_flight_line: u8,
        #[pasture(BUILTIN_CLASSIFICATION)]
        pub classification: u8,
        #[pasture(BUILTIN_SCAN_ANGLE_RANK)]
        pub scan_angle_rank: i8,
        #[pasture(BUILTIN_SCAN_ANGLE)]
        pub scan_angle: i16,
        #[pasture(BUILTIN_USER_DATA)]
        pub user_data: u8,
        #[pasture(BUILTIN_POINT_SOURCE_ID)]
        pub point_source_id: u16,
        #[pasture(BUILTIN_COLOR_RGB)]
        pub color_rgb: Vector3<u16>,
        #[pasture(BUILTIN_GPS_TIME)]
        pub gps_time: f64,
        #[pasture(BUILTIN_NIR)]
        pub nir: u16,
        // #[pasture(BUILTIN_WAVE_PACKET_DESCRIPTOR_INDEX)]
        // pub wave_packet_descriptor_index: u8,
        // #[pasture(BUILTIN_WAVEFORM_DATA_OFFSET)]
        // pub waveform_data_offset: u64,
        // #[pasture(BUILTIN_WAVEFORM_PACKET_SIZE)]
        // pub waveform_packet_size: u32,
        // #[pasture(BUILTIN_RETURN_POINT_WAVEFORM_LOCATION)]
        // pub return_point_waveform_location: f32,
        // #[pasture(BUILTIN_WAVEFORM_PARAMETERS)]
        // pub waveform_parameters: Vector3<f32>,
    }

    fn _generate_vec3f32(rng: &mut ThreadRng) -> Vector3<f32> {
        Vector3::new(rng.gen_range(-30.0..10.0), rng.gen_range(-11.1..10.0), 31.0)
    }
    fn generate_vec3u16(rng: &mut ThreadRng) -> Vector3<u16> {
        Vector3::new(rng.gen_range(11..120), rng.gen_range(11..120), 42)
    }

    fn setup_point_cloud() -> HashMapBuffer {
        let mut rng = rand::thread_rng();
        let mut points = vec![];
        // create vertices between 0.0 and 10.0
        points.push(CompletePoint {
            position: Vector3::new(0.0, 0.0, 0.0),
            intensity: rng.gen_range(200..800),
            return_number: rng.gen_range(20..80),
            num_of_returns: rng.gen_range(20..80),
            classification_flags: rng.gen_range(7..20),
            scanner_channel: rng.gen_range(7..20),
            scan_dir_flag: rng.gen_range(0..47),
            edge_of_flight_line: rng.gen_range(0..81),
            classification: rng.gen_range(121..200),
            scan_angle_rank: rng.gen_range(-121..20),
            scan_angle: rng.gen_range(-21..8),
            user_data: rng.gen_range(1..8),
            point_source_id: rng.gen_range(9..89),
            color_rgb: generate_vec3u16(&mut rng),
            gps_time: rng.gen_range(-22.4..81.3),
            nir: rng.gen_range(4..82),
            // wave_packet_descriptor_index: rng.gen_range(2..42),
            // waveform_data_offset: rng.gen_range(1..31),
            // waveform_packet_size: rng.gen_range(32..64),
            // return_point_waveform_location: rng.gen_range(-32.2..64.1),
            // waveform_parameters: generate_vec3f32(&mut rng),
        });
        points.push(CompletePoint {
            position: Vector3::new(10.0, 10.0, 10.0),
            intensity: rng.gen_range(200..800),
            return_number: rng.gen_range(20..80),
            num_of_returns: rng.gen_range(20..80),
            classification_flags: rng.gen_range(7..20),
            scanner_channel: rng.gen_range(7..20),
            scan_dir_flag: rng.gen_range(0..47),
            edge_of_flight_line: rng.gen_range(0..81),
            classification: rng.gen_range(121..200),
            scan_angle_rank: rng.gen_range(-121..20),
            scan_angle: rng.gen_range(-21..8),
            user_data: rng.gen_range(1..8),
            point_source_id: rng.gen_range(9..89),
            color_rgb: generate_vec3u16(&mut rng),
            gps_time: rng.gen_range(-22.4..81.3),
            nir: rng.gen_range(4..82),
            // wave_packet_descriptor_index: rng.gen_range(2..42),
            // waveform_data_offset: rng.gen_range(1..31),
            // waveform_packet_size: rng.gen_range(32..64),
            // return_point_waveform_location: rng.gen_range(-32.2..64.1),
            // waveform_parameters: generate_vec3f32(&mut rng),
        });
        // generate 3000 points
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    //1.5, 2.5, ...
                    points.push(CompletePoint {
                        // average_vec
                        position: Vector3::new(
                            f64::from(i) + 0.5,
                            f64::from(j) + 0.5,
                            f64::from(k) + 0.5,
                        ),
                        // average_num
                        intensity: 2,
                        // most_common num
                        return_number: 32,
                        num_of_returns: rng.gen_range(20..80),
                        // max_pool
                        classification_flags: 3,
                        scanner_channel: rng.gen_range(7..20),
                        // most_common bool
                        scan_dir_flag: 0,
                        edge_of_flight_line: rng.gen_range(0..81),
                        classification: rng.gen_range(121..200),
                        scan_angle_rank: rng.gen_range(-121..20),
                        scan_angle: rng.gen_range(-21..8),
                        user_data: rng.gen_range(1..8),
                        point_source_id: rng.gen_range(9..89),
                        color_rgb: generate_vec3u16(&mut rng),
                        gps_time: rng.gen_range(-22.4..81.3),
                        nir: rng.gen_range(4..82),
                        // wave_packet_descriptor_index: rng.gen_range(2..42),
                        // waveform_data_offset: rng.gen_range(1..31),
                        // waveform_packet_size: rng.gen_range(32..64),
                        // return_point_waveform_location: rng.gen_range(-32.2..64.1),
                        // waveform_parameters: generate_vec3f32(&mut rng),
                    });
                    // 1.6, 2.6, ...
                    points.push(CompletePoint {
                        position: Vector3::new(
                            f64::from(i) + 0.6,
                            f64::from(j) + 0.6,
                            f64::from(k) + 0.6,
                        ),
                        intensity: 4,
                        return_number: 42,
                        num_of_returns: rng.gen_range(20..80),
                        classification_flags: 7,
                        scanner_channel: rng.gen_range(7..20),
                        scan_dir_flag: 0,
                        edge_of_flight_line: rng.gen_range(0..81),
                        classification: rng.gen_range(121..200),
                        scan_angle_rank: rng.gen_range(-121..20),
                        scan_angle: rng.gen_range(-21..8),
                        user_data: rng.gen_range(1..8),
                        point_source_id: rng.gen_range(9..89),
                        color_rgb: generate_vec3u16(&mut rng),
                        gps_time: rng.gen_range(-22.4..81.3),
                        nir: rng.gen_range(4..82),
                        // wave_packet_descriptor_index: rng.gen_range(2..42),
                        // waveform_data_offset: rng.gen_range(1..31),
                        // waveform_packet_size: rng.gen_range(32..64),
                        // return_point_waveform_location: rng.gen_range(-32.2..64.1),
                        // waveform_parameters: generate_vec3f32(&mut rng),
                    });
                    // 1.7, 2.7, ...
                    points.push(CompletePoint {
                        position: Vector3::new(
                            f64::from(i) + 0.7,
                            f64::from(j) + 0.7,
                            f64::from(k) + 0.7,
                        ),
                        intensity: 6,
                        return_number: 42,
                        num_of_returns: rng.gen_range(20..80),
                        classification_flags: 133,
                        scanner_channel: rng.gen_range(7..20),
                        scan_dir_flag: 1,
                        edge_of_flight_line: rng.gen_range(0..81),
                        classification: rng.gen_range(121..200),
                        scan_angle_rank: rng.gen_range(-121..20),
                        scan_angle: rng.gen_range(-21..8),
                        user_data: rng.gen_range(1..8),
                        point_source_id: rng.gen_range(9..89),
                        color_rgb: generate_vec3u16(&mut rng),
                        gps_time: rng.gen_range(-22.4..81.3),
                        nir: rng.gen_range(4..82),
                        // wave_packet_descriptor_index: rng.gen_range(2..42),
                        // waveform_data_offset: rng.gen_range(1..31),
                        // waveform_packet_size: rng.gen_range(32..64),
                        // return_point_waveform_location: rng.gen_range(-32.2..64.1),
                        // waveform_parameters: generate_vec3f32(&mut rng),
                    });
                }
            }
        }
        points.into_iter().collect()
    }

    #[test]
    fn test_voxel_grid_filter() {
        let buffer = setup_point_cloud();
        assert!(buffer.len() == 3002);
        let mut filtered = HashMapBuffer::new_from_layout(buffer.point_layout().clone());
        voxelgrid_filter(&buffer, 1.0, 1.0, 1.0, &mut filtered);
        // filtered now has only 1000 points
        assert!(filtered.len() == 1000);

        // average_vec
        let first_pos = filtered
            .view_attribute::<Vector3<f64>>(&attributes::POSITION_3D)
            .at(1);
        assert!(first_pos.x > 0.59 && first_pos.x < 0.61);
        assert!(first_pos.y > 0.59 && first_pos.y < 0.61);
        assert!(first_pos.z > 1.59 && first_pos.z < 1.61);
        // average_num
        assert!(filtered.view_attribute::<u16>(&attributes::INTENSITY).at(1) == 4);
        // most_common num
        assert!(
            filtered
                .view_attribute::<u8>(&attributes::RETURN_NUMBER)
                .at(1)
                == 42
        );
        // max_pool
        assert!(
            filtered
                .view_attribute::<u8>(&attributes::CLASSIFICATION_FLAGS)
                .at(1)
                == 133
        );
    }
}
