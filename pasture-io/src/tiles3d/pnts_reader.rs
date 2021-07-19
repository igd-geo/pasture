use core::num;
use std::{
    collections::HashMap,
    convert::TryInto,
    fs::File,
    io::{BufRead, BufReader, Read, Seek, SeekFrom},
    path::Path,
};

use anyhow::{anyhow, bail, Result};
use pasture_core::{
    containers::{
        PerAttributePointBufferMut, PerAttributeVecPointStorage, PointBuffer, PointBufferWriteable,
    },
    layout::{
        attributes::{COLOR_RGB, NORMAL, POSITION_3D},
        conversion::get_converter_for_attributes,
        FieldAlignment, PointAttributeDataType, PointAttributeDefinition, PointLayout,
    },
    meta::Metadata,
    nalgebra::clamp,
};

use crate::{
    base::{PointReader, SeekToPoint},
    tiles3d::{attributes::COLOR_RGBA, json_arr_to_vec3f32, json_arr_to_vec4u8},
};
use crate::{
    las::LASMetadata,
    tiles3d::{deser_feature_table_header, FeatureTableValue, PntsHeader},
};

use super::PntsMetadata;

pub struct PntsReader<R: BufRead + Seek> {
    reader: R,
    metadata: PntsMetadata,
    layout: PointLayout,
    current_point_index: usize,
    attribute_offsets: HashMap<String, u64>,
}

impl<R: BufRead + Seek> PntsReader<R> {
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<PntsReader<BufReader<File>>> {
        let reader = BufReader::new(File::open(path)?);
        PntsReader::<BufReader<File>>::from_read(reader)
    }

    pub fn from_read(mut read: R) -> Result<PntsReader<R>> {
        let _header: PntsHeader = bincode::deserialize_from(&mut read)?;
        // TODO BatchTable support
        let mut feature_table_header = deser_feature_table_header(&mut read)?;

        // The following functions mutate the feature table header HashMap and remove the entries that
        // are relevant. This is done because both point semantics and global semantics are stored in the
        // same header, so this makes parsing easier
        let (layout, mut attribute_offsets) =
            Self::layout_from_feature_table_header(&mut feature_table_header)?;
        let metadata = Self::metadata_from_feature_table_header(&mut feature_table_header)?;

        // TODO Log all parameters that could not be parsed. This requires logging support for pasture

        let feature_table_binary_offset: u64 = (PntsHeader::BYTE_LENGTH
            + _header.feature_table_json_byte_length as usize)
            .try_into()
            .unwrap();
        // Ensure that reader is at correct position according to parameters in header
        let current_read_position = read.seek(SeekFrom::Current(0))?;
        if current_read_position != feature_table_binary_offset {
            bail!("FeatureTable header does not match size defined in PNTS header. Expected FeatureTable body to start at offset {} but it starts at offset {}!", feature_table_binary_offset, current_read_position);
        }

        // Convert offsets in binary body to offsets within whole file
        for (_, offset) in &mut attribute_offsets {
            *offset += feature_table_binary_offset;
        }

        Ok(Self {
            reader: read,
            metadata,
            layout,
            current_point_index: 0,
            attribute_offsets,
        })
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
                FeatureTableValue::Array(array) => json_arr_to_vec3f32(&array),
                _ => Err(anyhow!("RTC_CENTER value was no array entry")),
            })
            .transpose()?;

        let quantized_volume_offset = header
            .get("QUANTIZED_VOLUME_OFFSET")
            .map(|entry| match entry {
                FeatureTableValue::Array(array) => json_arr_to_vec3f32(&array),
                _ => Err(anyhow!("QUANTIZED_VOLUME_OFFSET value was no array entry")),
            })
            .transpose()?;

        let quantized_volume_scale = header
            .get("QUANTIZED_VOLUME_SCALE")
            .map(|entry| match entry {
                FeatureTableValue::Array(array) => json_arr_to_vec3f32(&array),
                _ => Err(anyhow!("QUANTIZED_VOLUME_SCALE value was no array entry")),
            })
            .transpose()?;

        let constant_rgba = header
            .get("CONSTANT_RGBA")
            .map(|entry| match entry {
                FeatureTableValue::Array(array) => json_arr_to_vec4u8(&array),
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
}

impl<R: BufRead + Seek> PointReader for PntsReader<R> {
    fn read(&mut self, count: usize) -> Result<Box<dyn PointBuffer>> {
        let remaining_points = self.metadata.points_length() - self.current_point_index;
        let num_to_read = usize::min(remaining_points, count);
        if num_to_read == 0 {
            bail!("No points remaining in PNTS file")
        }

        let mut buffer = PerAttributeVecPointStorage::new(self.layout.clone());
        buffer.resize(num_to_read);
        for attribute in self.layout.attributes() {
            let attribute_stride = attribute.size();
            let offset_to_first_point_of_attribute =
                *self.attribute_offsets.get(attribute.name()).unwrap();
            let offset_to_current_point_of_attribute = offset_to_first_point_of_attribute
                + (self.current_point_index as u64 * attribute_stride);

            self.reader
                .seek(SeekFrom::Start(offset_to_current_point_of_attribute))?;
            let target_buffer =
                buffer.get_raw_attribute_range_mut(0..num_to_read, &attribute.into());
            self.reader.read_exact(target_buffer)?;
        }

        self.current_point_index += num_to_read;

        Ok(Box::new(buffer))
    }

    fn read_into(
        &mut self,
        point_buffer: &mut dyn PointBufferWriteable,
        count: usize,
    ) -> Result<usize> {
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
                let converter =
                    get_converter_for_attributes(&attribute.into(), &target_attribute.into());
                if let Some(conversion_fn) = converter {
                    let mut src_buf: Vec<u8> = vec![0; attribute.size() as usize];
                    let mut dst_buf: Vec<u8> = vec![0; attribute.size() as usize];
                    let target_attribute_def: PointAttributeDefinition = target_attribute.into();
                    for point_index in 0..num_to_read {
                        self.reader.read_exact(src_buf.as_mut_slice())?;
                        unsafe {
                            conversion_fn(src_buf.as_slice(), dst_buf.as_mut_slice());
                        }
                        point_buffer.set_raw_attribute(
                            point_index,
                            &target_attribute_def,
                            dst_buf.as_slice(),
                        );
                    }
                } else {
                    let mut buf: Vec<u8> = vec![0; attribute.size() as usize];
                    let target_attribute_def: PointAttributeDefinition = target_attribute.into();
                    for point_index in 0..num_to_read {
                        self.reader.read_exact(buf.as_mut_slice())?;
                        point_buffer.set_raw_attribute(
                            point_index,
                            &target_attribute_def,
                            buf.as_slice(),
                        );
                    }
                }
            }
        }

        self.current_point_index += num_to_read;
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
    use pasture_core::{containers::PointBufferExt, nalgebra::Vector3};

    use super::*;

    #[test]
    fn test_pnts_reader() -> Result<()> {
        let mut reader =
            PntsReader::<BufReader<File>>::from_path("/Users/pbormann/Downloads/points.pnts")?;
        let metadata = reader.get_metadata();
        let num_points = metadata.number_of_points().unwrap();
        let points = reader.read(8000)?;
        let positions = points
            .iter_attribute::<Vector3<f32>>(
                &POSITION_3D.with_custom_datatype(PointAttributeDataType::Vec3f32),
            )
            .collect::<Vec<_>>();
        Ok(())
    }
}
