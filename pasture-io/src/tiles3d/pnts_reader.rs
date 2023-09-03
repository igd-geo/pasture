use std::{
    collections::HashMap,
    convert::TryInto,
    fs::File,
    io::{BufRead, BufReader, Seek, SeekFrom},
    path::Path,
};

use anyhow::{anyhow, bail, Context, Result};
use pasture_core::{
    containers::OwningBuffer,
    layout::{
        attributes::{COLOR_RGB, NORMAL, POSITION_3D},
        conversion::get_converter_for_attributes,
        FieldAlignment, PointAttributeDataType, PointLayout,
    },
    meta::Metadata,
    nalgebra::{clamp, Vector3},
};

use crate::tiles3d::{deser_feature_table_header, FeatureTableValue, PntsHeader};
use crate::{
    base::{PointReader, SeekToPoint},
    tiles3d::{attributes::COLOR_RGBA, json_arr_to_vec3f32, json_arr_to_vec4u8},
};

use super::{json_arr_to_vec3f64, PntsMetadata};

/// Defines how the `PntsReader` reads positions if the `RTC_CENTER` point semantic is present
#[derive(Copy, Clone, Debug)]
pub enum PntsReadPositionsMode {
    /// Reads points relative to center. If the position `(10, 10, 10)` is stored in the PNTS file and
    /// `RTC_CENTER` is `(20, 20, 20)`, calling `PntsReader::read` will return the position `(10, 10, 10)`
    RelativeToCenter,
    /// Reads points in absolute coordinates. If the position `(10, 10, 10)` is stored in the PNTS file and
    /// `RTC_CENTER` is `(20, 20, 20)`, calling `PntsReader::read` will return the position `(30, 30, 30)`
    Absolute,
}

/// A reader for points in the 3D Tiles PNTS format
pub struct PntsReader<R: BufRead + Seek> {
    reader: R,
    metadata: PntsMetadata,
    layout: PointLayout,
    current_point_index: usize,
    attribute_offsets: HashMap<String, u64>,
    read_positions_mode: PntsReadPositionsMode,
}

impl<R: BufRead + Seek> PntsReader<R> {
    pub fn from_read(mut read: R) -> Result<PntsReader<R>> {
        // PNTS is little-endian, this is the default of bincode
        let header: PntsHeader = bincode::deserialize_from(&mut read)
            .context("Could not deserialize PNTS header from reader")?;
        header.verify_magic()?;
        let position_after_header = read.stream_position()? as usize;
        assert_eq!(position_after_header, PntsHeader::BYTE_LENGTH);

        // The 3D Tiles spec is ambiguous when it comes to the exact values in the header. It says that 'featureTableJSONByteLength'
        // contains 'The length of the Feature Table JSON section in bytes.', and this section should END on an 8-byte boundary
        // within the file. However the example PNTS file that you can download from [here](https://github.com/CesiumGS/3d-tiles-samples)
        // does not match this definition!! There, the header size + featureTableJSONByteLength does NOT sum up to a multiple of 8...
        // The code is thus written in a way to automatically read padding bytes so that after deser_feature_table_header, we are at
        // the start of the FeatureTable body

        let mut feature_table_header = deser_feature_table_header(
            &mut read,
            header.feature_table_json_byte_length as usize,
            position_after_header,
        )?;
        // TODO BatchTable support

        // The following functions mutate the feature table header HashMap and remove the entries that
        // are relevant. This is done because both point semantics and global semantics are stored in the
        // same header, so this makes parsing easier
        let (layout, mut attribute_offsets) =
            Self::layout_from_feature_table_header(&mut feature_table_header)?;
        let metadata = Self::metadata_from_feature_table_header(&mut feature_table_header)?;

        // TODO Log all parameters that could not be parsed. This requires logging support for pasture

        let feature_table_binary_offset = read.stream_position()?;
        // Convert offsets in binary body to offsets within whole file
        for offset in attribute_offsets.values_mut() {
            *offset += feature_table_binary_offset;
        }

        Ok(Self {
            reader: read,
            metadata,
            layout,
            current_point_index: 0,
            attribute_offsets,
            read_positions_mode: PntsReadPositionsMode::Absolute,
        })
    }

    /// Sets the `PntsReadPositionsMode` for this `PntsReader`
    pub fn set_read_positions_mode(&mut self, read_mode: PntsReadPositionsMode) {
        self.read_positions_mode = read_mode;
    }

    /// Returns the `PntsReadPositionsMode` for this `PntsReader`. The default value is always `PntsReadPositionsMode::Absolute`.
    pub fn read_positions_mode(&self) -> PntsReadPositionsMode {
        self.read_positions_mode
    }

    /// Creates a `PointLayout` from the given FeatureTable header. Since 3D Tiles PNTS stores point attributes in per-attribute
    /// format, there is no 'right' way to create a `PointLayout`, since the order of the of the attributes is arbitrary. We could
    /// use the order in which they are defined in the header, however we are using a HashMap for easy lookup, so we don't have the
    /// order at this point. Instead, we check all supported attributes ('point semantics' in 3D Tiles jargon) in exactly the order
    /// that they are defined in [here](https://github.com/CesiumGS/3d-tiles/blob/master/specification/TileFormats/PointCloud/README.md#semantics).
    fn layout_from_feature_table_header(
        header: &mut HashMap<String, FeatureTableValue>,
    ) -> Result<(PointLayout, HashMap<String, u64>)> {
        // 3D Tiles .pnts has very few supported point attributes, so we can just enumerate them by hand
        let mut layout: PointLayout = Default::default();
        let mut attribute_offsets = HashMap::new();
        if header.contains_key("POSITION") {
            let pos_attribute = &header["POSITION"];
            match pos_attribute {
                FeatureTableValue::DataReference(reference) => {
                    attribute_offsets.insert(POSITION_3D.name().to_owned(), reference.byte_offset as u64);
                    layout.add_attribute(POSITION_3D.with_custom_datatype(PointAttributeDataType::Vec3f32), FieldAlignment::Packed(1));
                },
                _ => bail!("Found PNTS attribute POSITION ({:?}) but it was not a reference to the feature table binary!", pos_attribute),
            }
            header.remove("POSITION");
        }

        // TODO Quantized positions, which probably require an option to de-quantize during reading (similar to LAS/LAZ)

        if header.contains_key("RGBA") {
            let color_attribute = &header["RGBA"];
            match color_attribute {
                FeatureTableValue::DataReference(reference) => {
                    attribute_offsets.insert(COLOR_RGBA.name().to_owned(), reference.byte_offset as u64);
                    layout.add_attribute(COLOR_RGBA, FieldAlignment::Packed(1));
                },
                _ => bail!("Found PNTS attribute RGBA ({:?}) but it was not a reference to the feature table binary!", color_attribute),
            }
            header.remove("RGBA");
        }

        if header.contains_key("RGB") {
            let color_attribute = &header["RGB"];
            match color_attribute {
                FeatureTableValue::DataReference(reference) => {
                    attribute_offsets.insert(COLOR_RGB.name().to_owned(), reference.byte_offset as u64);
                    layout.add_attribute(COLOR_RGB.with_custom_datatype(PointAttributeDataType::Vec3u8), FieldAlignment::Packed(1));
                },
                _ => bail!("Found PNTS attribute RGB ({:?}) but it was not a reference to the feature table binary!", color_attribute),
            }
            header.remove("RGB");
        }

        // TOOD RGB565

        if header.contains_key("NORMAL") {
            let normal_attribute = &header["NORMAL"];
            match normal_attribute {
                FeatureTableValue::DataReference(reference) => {
                    attribute_offsets.insert(NORMAL.name().to_owned(), reference.byte_offset as u64);
                    layout.add_attribute(NORMAL ,FieldAlignment::Packed(1));
                },
                _ => bail!("Found PNTS attribute NORMAL ({:?}) but it was not a reference to the feature table binary!", normal_attribute),
            }
            header.remove("NORMAL");
        }

        // Normal oct16p

        // Batch ID

        Ok((layout, attribute_offsets))
    }

    fn metadata_from_feature_table_header(
        header: &mut HashMap<String, FeatureTableValue>,
    ) -> Result<PntsMetadata> {
        let num_points = header
            .get("POINTS_LENGTH")
            .map(|entry| match entry {
                FeatureTableValue::SingleValue(v) => v
                    .as_u64()
                    .map(|val| val as usize)
                    .ok_or(anyhow!("POINTS_LENGTH value vas no integer number")),
                _ => Err(anyhow!("POINTS_LENGTH value was no single value entry")),
            })
            .ok_or(anyhow!(
                "Mandatory value POINTS_LENGTH not found in feature table header"
            ))??;

        let rtc_center = header
            .get("RTC_CENTER")
            .map(|entry| match entry {
                FeatureTableValue::Array(array) => json_arr_to_vec3f64(array),
                _ => Err(anyhow!("RTC_CENTER value was no array entry")),
            })
            .transpose()?;

        let quantized_volume_offset = header
            .get("QUANTIZED_VOLUME_OFFSET")
            .map(|entry| match entry {
                FeatureTableValue::Array(array) => json_arr_to_vec3f32(array),
                _ => Err(anyhow!("QUANTIZED_VOLUME_OFFSET value was no array entry")),
            })
            .transpose()?;

        let quantized_volume_scale = header
            .get("QUANTIZED_VOLUME_SCALE")
            .map(|entry| match entry {
                FeatureTableValue::Array(array) => json_arr_to_vec3f32(array),
                _ => Err(anyhow!("QUANTIZED_VOLUME_SCALE value was no array entry")),
            })
            .transpose()?;

        let constant_rgba = header
            .get("CONSTANT_RGBA")
            .map(|entry| match entry {
                FeatureTableValue::Array(array) => json_arr_to_vec4u8(array),
                _ => Err(anyhow!("CONSTANT_RGBA value was no array entry")),
            })
            .transpose()?;

        let batch_length = header
            .get("BATCH_LENGTH")
            .map(|entry| match entry {
                FeatureTableValue::SingleValue(v) => v
                    .as_u64()
                    .map(|val| val as usize)
                    .ok_or(anyhow!("BATCH_LENGTH value was no integer number")),
                _ => Err(anyhow!("BATCH_LENGTH value was no single value entry")),
            })
            .transpose()?;

        Ok(PntsMetadata::new(
            num_points,
            rtc_center,
            quantized_volume_offset,
            quantized_volume_scale,
            constant_rgba,
            batch_length,
        ))
    }

    fn apply_rtc_center_offset<'a, 'b, B: OwningBuffer<'a>>(
        &self,
        point_buffer: &'b mut B,
    ) -> Result<()>
    where
        'a: 'b,
    {
        let maybe_position = point_buffer
            .point_layout()
            .get_attribute_by_name(POSITION_3D.name());
        if let (Some(rtc_center), Some(position_attribute)) =
            (self.metadata.rtc_center(), maybe_position)
        {
            // The default datatype for positions in the PNTS format is Vec3f32, so we try to apply the offset
            // first on this datatype. If the datatype does not match, we can only use Vec3f64, other types are
            // not supported at the moment
            let position_attribute = position_attribute.clone();
            match position_attribute.datatype() {
                PointAttributeDataType::Vec3f32 => point_buffer.transform_attribute(
                    position_attribute.attribute_definition(),
                    |_, position: Vector3<f32>| -> Vector3<f32> {
                        Vector3::new(
                            (position.x as f64 + rtc_center.x) as f32,
                            (position.y as f64 + rtc_center.y) as f32,
                            (position.z as f64 + rtc_center.z) as f32,
                        )
                    },
                ),
                PointAttributeDataType::Vec3f64 => point_buffer.transform_attribute(
                    position_attribute.attribute_definition(),
                    |_, position: Vector3<f64>| -> Vector3<f64> { position + rtc_center },
                ),
                other => bail!("Unsupported datatype {other} for POSITION_3D attribute"),
            }
        }
        Ok(())
    }
}

impl PntsReader<BufReader<File>> {
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<PntsReader<BufReader<File>>> {
        let reader = BufReader::new(File::open(path)?);
        PntsReader::<BufReader<File>>::from_read(reader)
    }
}

impl<R: BufRead + Seek> PointReader for PntsReader<R> {
    fn read_into<'a, 'b, B: OwningBuffer<'a>>(
        &mut self,
        point_buffer: &'b mut B,
        count: usize,
    ) -> Result<usize>
    where
        'a: 'b,
    {
        let remaining_points = self.metadata.points_length() - self.current_point_index;
        let num_to_read = usize::min(remaining_points, count);
        if num_to_read == 0 {
            bail!("No points remaining in PNTS file")
        }

        let target_layout = point_buffer.point_layout().clone();
        point_buffer.resize(num_to_read);
        for attribute in self.layout.attributes() {
            // Try to read this attribute only if it exists in the target buffer's PointLayout
            if let Some(target_attribute) = target_layout.get_attribute_by_name(attribute.name()) {
                let attribute_stride = attribute.size();
                let offset_to_first_point_of_attribute =
                    *self.attribute_offsets.get(attribute.name()).unwrap();
                let offset_to_current_point_of_attribute = offset_to_first_point_of_attribute
                    + (self.current_point_index as u64 * attribute_stride);

                self.reader
                    .seek(SeekFrom::Start(offset_to_current_point_of_attribute))?;

                // Maybe we have to convert the datatype?
                let converter = get_converter_for_attributes(
                    attribute.attribute_definition(),
                    target_attribute.attribute_definition(),
                );
                if let Some(conversion_fn) = converter {
                    let mut src_buf: Vec<u8> = vec![0; attribute.size() as usize];
                    let mut dst_buf: Vec<u8> = vec![0; target_attribute.size() as usize];
                    let target_attribute_def = target_attribute.attribute_definition();
                    for point_index in 0..num_to_read {
                        self.reader.read_exact(src_buf.as_mut_slice())?;
                        unsafe {
                            conversion_fn(src_buf.as_slice(), dst_buf.as_mut_slice());
                            point_buffer.set_attribute(
                                target_attribute_def,
                                point_index,
                                dst_buf.as_slice(),
                            );
                        }
                    }
                } else {
                    let mut buf: Vec<u8> = vec![0; attribute.size() as usize];
                    let target_attribute_def = target_attribute.attribute_definition();
                    for point_index in 0..num_to_read {
                        self.reader.read_exact(buf.as_mut_slice())?;
                        unsafe {
                            point_buffer.set_attribute(
                                target_attribute_def,
                                point_index,
                                buf.as_slice(),
                            );
                        }
                    }
                }
            }
        }

        self.current_point_index += num_to_read;

        if let PntsReadPositionsMode::Absolute = self.read_positions_mode {
            self.apply_rtc_center_offset(point_buffer)
                .context("Failed to apply RTC_CENTER offset")?;
        }

        Ok(num_to_read)
    }

    fn get_metadata(&self) -> &dyn Metadata {
        &self.metadata
    }

    fn get_default_point_layout(&self) -> &PointLayout {
        &self.layout
    }
}

impl<R: BufRead + Seek> SeekToPoint for PntsReader<R> {
    fn seek_point(&mut self, position: std::io::SeekFrom) -> Result<usize> {
        let new_point_idx: u64 = match position {
            SeekFrom::Start(offset) => offset,
            SeekFrom::End(offset) => {
                let max_point_as_i64: i64 = self.metadata.points_length().try_into()?;
                (max_point_as_i64 + offset).try_into()?
            }
            SeekFrom::Current(offset) => {
                let cur_point_asi64: i64 = self.current_point_index.try_into()?;
                (cur_point_asi64 + offset).try_into()?
            }
        };
        let new_point_idx_clamped = clamp(new_point_idx, 0, self.metadata.points_length() as u64);
        self.current_point_index = new_point_idx_clamped.try_into()?;
        Ok(self.current_point_index)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use crate::{base::PointWriter, tiles3d::PntsWriter};

    use super::*;
    use pasture_core::{
        containers::{BorrowedBuffer, HashMapBuffer, VectorBuffer},
        layout::PointType,
    };
    use pasture_derive::PointType;

    #[repr(C, packed)]
    #[derive(
        Copy, Clone, PartialEq, PointType, Debug, bytemuck::AnyBitPattern, bytemuck::NoUninit,
    )]
    struct TestPoint(#[pasture(BUILTIN_POSITION_3D)] Vector3<f32>);

    #[test]
    fn test_pnts_reader_read_modes() {
        let test_points = vec![
            TestPoint(Vector3::new(10.0_f32, 10.0_f32, 10.0_f32)),
            TestPoint(Vector3::new(20.0_f32, 20.0_f32, 20.0_f32)),
        ];
        let test_points_global = vec![
            TestPoint(Vector3::new(20.0_f32, 20.0_f32, 20.0_f32)),
            TestPoint(Vector3::new(30.0_f32, 30.0_f32, 30.0_f32)),
        ];

        let mut cursor = Cursor::new(Vec::<u8>::new());

        {
            let points = test_points.iter().copied().collect::<HashMapBuffer>();
            let mut writer = PntsWriter::from_write_and_layout(&mut cursor, TestPoint::layout());
            writer.set_rtc_center(Vector3::new(10.0, 10.0, 10.0));
            writer
                .write(&points)
                .expect("Could not write points in PNTS format");
        }

        cursor.seek(SeekFrom::Start(0)).unwrap();

        // Read once with Absolute positioning and once with RelativeToCenter positioning
        {
            let mut reader = PntsReader::from_read(&mut cursor).expect("Could not open PntsReader");
            let points = reader
                .read::<VectorBuffer>(reader.get_metadata().number_of_points().unwrap())
                .expect("Could not read points in PNTS format");

            let actual_points = points.view::<TestPoint>().into_iter().collect::<Vec<_>>();
            assert_eq!(test_points_global, actual_points);
        }

        cursor.seek(SeekFrom::Start(0)).unwrap();

        {
            let mut reader = PntsReader::from_read(&mut cursor).expect("Could not open PntsReader");
            reader.set_read_positions_mode(PntsReadPositionsMode::RelativeToCenter);
            let points = reader
                .read::<VectorBuffer>(reader.get_metadata().number_of_points().unwrap())
                .expect("Could not read points in PNTS format");

            let actual_points = points.view::<TestPoint>().into_iter().collect::<Vec<_>>();
            assert_eq!(test_points, actual_points);
        }
    }
}
