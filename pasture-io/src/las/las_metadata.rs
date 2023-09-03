use std::{
    any::Any,
    borrow::Cow,
    convert::{TryFrom, TryInto},
    fmt::Display,
    iter::FromIterator,
    path::Path,
};

use anyhow::{anyhow, bail, Context, Result};
use bitfield::bitfield;
use chrono::Datelike;
use las::{Bounds, Header};
use las_rs::{point::Format, raw::vlr::RecordLength, Vector, Vlr};
use pasture_core::{
    layout::{PointAttributeDataType, PointAttributeDefinition},
    math::AABB,
    meta::Metadata,
    nalgebra::Point3,
};
use static_assertions::const_assert_eq;

use super::{las_string_to_rust_string, write_rust_string_into_las_ascii_array};

/// Contains constants for possible named fields in a `LASMetadata` structure
pub mod named_fields {
    /// File source ID as per the LAS 1.4 specification
    pub const FILE_SOURCE_I_D: &str = "LASFIELD_FileSourceID";
    /// LAS file version
    pub const VERSION: &str = "LASFIELD_Version";
    /// System identifier as per the LAS 1.4 specification
    pub const SYSTEM_IDENTIFIER: &str = "LASFIELD_SystemIdentifier";
    /// Information about the generating software
    pub const GENERATING_SOFTWARE: &str = "LASFIELD_GeneratingSoftware";
    /// Day of year on which the file was created as per the LAS 1.4 specification
    pub const FILE_CREATION_DAY_OF_YEAR: &str = "LASFIELD_FileCreationDayOfYear";
    /// Year in which the file was created
    pub const FILE_CREATION_YEAR: &str = "LASFIELD_FileCreationYear";

    //TODO More fields
}

/// Converts a las-rs `Bounds` type into a pasture-core bounding box (`AABB<f64>`)
pub fn las_bounds_to_pasture_bounds(las_bounds: Bounds) -> AABB<f64> {
    let min_point = Point3::new(las_bounds.min.x, las_bounds.min.y, las_bounds.min.z);
    let max_point = Point3::new(las_bounds.max.x, las_bounds.max.y, las_bounds.max.z);
    AABB::from_min_max_unchecked(min_point, max_point)
}

/// Converts a pasture-core bounding box (`AABB<f64>`) into a las-rs `Bounds` type
pub fn pasture_bounds_to_las_bounds(bounds: &AABB<f64>) -> Bounds {
    Bounds {
        min: Vector {
            x: bounds.min().x,
            y: bounds.min().y,
            z: bounds.min().z,
        },
        max: Vector {
            x: bounds.max().x,
            y: bounds.max().y,
            z: bounds.max().z,
        },
    }
}

/// Tries to determine whether the given `path` represents a compressed LAZ file or an uncompressed LAS file
pub fn path_is_compressed_las_file<P: AsRef<Path>>(path: P) -> Result<bool> {
    path.as_ref()
        .extension()
        .map(|extension| extension == "laz")
        .ok_or(anyhow!(
            "Could not determine file extension of file {}",
            path.as_ref().display()
        ))
}

const KNOWN_VLR_USER_ID: &str = "LASF_Spec";

#[derive(Clone, Debug, Default)]
pub struct ClassificationLookupEntry {
    pub classification: u8,
    pub description: String,
}

/// VLR that describes the classification names
#[derive(Clone, Debug)]
pub struct ClassificationLookup {
    entries: [ClassificationLookupEntry; 256],
}

impl ClassificationLookup {
    pub const RECORD_ID: u16 = 0;

    pub fn entries(&self) -> &[ClassificationLookupEntry] {
        &self.entries
    }
}

impl TryFrom<&Vlr> for ClassificationLookup {
    type Error = anyhow::Error;

    fn try_from(value: &Vlr) -> std::result::Result<Self, Self::Error> {
        if value.user_id != KNOWN_VLR_USER_ID {
            return Err(anyhow!(
                "Expected user_id {KNOWN_VLR_USER_ID} but got {}",
                value.user_id
            ));
        }
        if value.record_id != Self::RECORD_ID {
            return Err(anyhow!(
                "Expected record ID {} but got {}",
                Self::RECORD_ID,
                value.record_id
            ));
        }
        let expected_length: usize = 256 * 16;
        if value.data.len() != expected_length {
            return Err(anyhow!("Classification lookup VLR is defined to have a size of {expected_length} bytes, but got {} bytes instead", value.data.len()));
        }

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct RawEntry {
            class_number: u8,
            description: [u8; 15],
        }
        const_assert_eq!(16, std::mem::size_of::<RawEntry>());
        let data_as_raw_entries: &[RawEntry] = bytemuck::cast_slice(&value.data);
        let mut ret = Self {
            entries: array_init::array_init(|_| Default::default()),
        };
        for (index, raw_entry) in data_as_raw_entries.iter().enumerate().take(256) {
            ret.entries[index] = ClassificationLookupEntry {
                classification: raw_entry.class_number,
                description: String::from_utf8_lossy(&raw_entry.description).into_owned(),
            };
        }
        Ok(ret)
    }
}

impl Display for ClassificationLookup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Classification Lookup")?;
        for entry in &self.entries {
            writeln!(f, "{:3}: {}", entry.classification, entry.description)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct TextAreaDescription {
    text: String,
}

impl TextAreaDescription {
    pub const RECORD_ID: u16 = 3;

    pub fn text(&self) -> &str {
        &self.text
    }
}

impl TryFrom<&'_ Vlr> for TextAreaDescription {
    type Error = anyhow::Error;

    fn try_from(value: &Vlr) -> std::result::Result<Self, Self::Error> {
        if value.user_id != KNOWN_VLR_USER_ID {
            return Err(anyhow!(
                "Expected user_id {KNOWN_VLR_USER_ID} but got {}",
                value.user_id
            ));
        }
        if value.record_id != Self::RECORD_ID {
            return Err(anyhow!(
                "Expected record ID {} but got {}",
                Self::RECORD_ID,
                value.record_id
            ));
        }

        let text = String::from_utf8(value.data.clone())
            .context("Text data of Text Area Description VLR contains invalid utf8")?;
        Ok(Self { text })
    }
}

impl Display for TextAreaDescription {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Text Area Description\n{}", self.text)
    }
}

/// Data type of VLR extra bytes record
#[derive(Copy, Clone, Debug)]
pub enum ExtraBytesDataType {
    Undocumented,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F32,
    F64,
    Deprecated(u8),
    Reserved(u8),
}

impl ExtraBytesDataType {
    /// Returns the byte size of a single value of this `ExtraBytesDataType`. This value might be unspecified
    /// if the data type is `Undocumented`, `Deprecated`, or `Reserved`
    pub fn size(&self) -> Option<usize> {
        match self {
            ExtraBytesDataType::U8 => Some(1),
            ExtraBytesDataType::I8 => Some(1),
            ExtraBytesDataType::U16 => Some(2),
            ExtraBytesDataType::I16 => Some(2),
            ExtraBytesDataType::U32 => Some(4),
            ExtraBytesDataType::I32 => Some(4),
            ExtraBytesDataType::U64 => Some(8),
            ExtraBytesDataType::I64 => Some(8),
            ExtraBytesDataType::F32 => Some(4),
            ExtraBytesDataType::F64 => Some(8),
            ExtraBytesDataType::Undocumented
            | ExtraBytesDataType::Deprecated(_)
            | ExtraBytesDataType::Reserved(_) => None,
        }
    }
    /// Does this data type represent an unsigned integer?
    pub fn is_unsigned(&self) -> bool {
        matches!(self, Self::U8 | Self::U16 | Self::U32 | Self::U64)
    }

    /// Does this data type represent a signed integer?
    pub fn is_signed(&self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }

    /// Does this data type represent a floating point value?
    pub fn is_floating_point(&self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }
}

impl From<u8> for ExtraBytesDataType {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Undocumented,
            1 => Self::U8,
            2 => Self::I8,
            3 => Self::U16,
            4 => Self::I16,
            5 => Self::U32,
            6 => Self::I32,
            7 => Self::U64,
            8 => Self::I64,
            9 => Self::F32,
            10 => Self::F64,
            11..=30 => Self::Deprecated(value),
            31.. => Self::Reserved(value),
        }
    }
}

impl From<ExtraBytesDataType> for u8 {
    fn from(val: ExtraBytesDataType) -> Self {
        match val {
            ExtraBytesDataType::Undocumented => 0,
            ExtraBytesDataType::U8 => 1,
            ExtraBytesDataType::I8 => 2,
            ExtraBytesDataType::U16 => 3,
            ExtraBytesDataType::I16 => 4,
            ExtraBytesDataType::U32 => 5,
            ExtraBytesDataType::I32 => 6,
            ExtraBytesDataType::U64 => 7,
            ExtraBytesDataType::I64 => 8,
            ExtraBytesDataType::F32 => 9,
            ExtraBytesDataType::F64 => 10,
            ExtraBytesDataType::Deprecated(value) => value,
            ExtraBytesDataType::Reserved(value) => value,
        }
    }
}

impl TryFrom<ExtraBytesDataType> for PointAttributeDataType {
    type Error = anyhow::Error;

    fn try_from(value: ExtraBytesDataType) -> std::result::Result<Self, Self::Error> {
        match value {
            ExtraBytesDataType::U8 => Ok(Self::U8),
            ExtraBytesDataType::I8 => Ok(Self::I8),
            ExtraBytesDataType::U16 => Ok(Self::U16),
            ExtraBytesDataType::I16 => Ok(Self::I16),
            ExtraBytesDataType::U32 => Ok(Self::U32),
            ExtraBytesDataType::I32 => Ok(Self::I32),
            ExtraBytesDataType::U64 => Ok(Self::U64),
            ExtraBytesDataType::I64 => Ok(Self::I64),
            ExtraBytesDataType::F32 => Ok(Self::F32),
            ExtraBytesDataType::F64 => Ok(Self::F64),
            ExtraBytesDataType::Undocumented => {
                bail!("Extra bytes of type 'undocumented' are currently unsupported in pasture")
            }
            ExtraBytesDataType::Deprecated(_) => {
                bail!("Extra bytes of type 'deprecated' are currently unsupported in pasture")
            }
            ExtraBytesDataType::Reserved(_) => {
                bail!("Extra bytes of type 'reserved' are currently unsupported in pasture")
            }
        }
    }
}

impl Display for ExtraBytesDataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self)
    }
}

bitfield! {
    #[derive(Default, Copy, Clone)]
    pub struct ExtraBytesOptions(u8);
    impl Debug;
    pub no_data_is_relevant, set_no_data_is_relevant: 0;
    pub min_is_relevant, set_min_is_relevant: 1;
    pub max_is_relevant, set_max_is_relevant: 2;
    pub use_scale, set_use_scale: 3;
    pub use_offset, set_use_offset: 4;
}

/// Raw binary representation of an entry in a 'Extra bytes' VLR
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub(crate) struct RawExtraBytesEntry {
    reserved: [u8; 2],
    data_type: u8,
    options: u8,
    name: [u8; 32],
    unused: [u8; 4],
    no_data: [u8; 8],
    deprecated_1: [u8; 16],
    min: [u8; 8],
    deprecated_2: [u8; 16],
    max: [u8; 8],
    deprecated_3: [u8; 16],
    scale: f64,
    deprecated_4: [u8; 16],
    offset: f64,
    deprecated_5: [u8; 16],
    description: [u8; 32],
}
const RAW_EXTRA_BYTES_ENTRY_SIZE: usize = 192;
const_assert_eq!(
    RAW_EXTRA_BYTES_ENTRY_SIZE,
    std::mem::size_of::<RawExtraBytesEntry>()
);

/// Entry within the 'Extra bytes' LAS VLR
#[derive(Clone, Debug)]
pub struct ExtraBytesEntry {
    data_type: ExtraBytesDataType,
    options: ExtraBytesOptions,
    name: String,
    scale: f64,
    offset: f64,
    description: String,
    min_value: [u8; 8],
    max_value: [u8; 8],
    no_data_value: [u8; 8],
}

impl ExtraBytesEntry {
    /// The data type of the extra bytes
    pub fn data_type(&self) -> ExtraBytesDataType {
        self.data_type
    }

    /// Options of the extra bytes
    pub fn options(&self) -> ExtraBytesOptions {
        self.options
    }

    /// The name of the extra bytes
    pub fn name(&self) -> &str {
        &self.name
    }

    /// A description of the extra bytes
    pub fn description(&self) -> &str {
        &self.description
    }

    /// An optional scaling parameter to be used during parsing of the extra bytes
    pub fn scale(&self) -> Option<f64> {
        if self.options.use_scale() {
            Some(self.scale)
        } else {
            None
        }
    }

    /// An optional offset parameter to be used during parsing of the extra bytes
    pub fn offset(&self) -> Option<f64> {
        if self.options.use_offset() {
            Some(self.offset)
        } else {
            None
        }
    }

    pub fn min_value_raw(&self) -> [u8; 8] {
        self.min_value
    }

    pub fn min_as_unsigned(&self) -> Result<u64> {
        if !self.data_type.is_unsigned() {
            bail!("Extra bytes datatype is not an unsigned integer type");
        }

        let as_u64 = u64::from_le_bytes(self.min_value);
        Ok(as_u64)
    }

    pub fn min_as_signed(&self) -> Result<i64> {
        if !self.data_type.is_signed() {
            bail!("Extra bytes datatype is not a signed integer type");
        }

        let as_i64 = i64::from_le_bytes(self.min_value);
        Ok(as_i64)
    }

    pub fn min_as_float(&self) -> Result<f64> {
        if !self.data_type.is_unsigned() {
            bail!("Extra bytes datatype is not a floating point type");
        }

        let as_f64 = f64::from_le_bytes(self.min_value);
        Ok(as_f64)
    }

    pub fn max_value_raw(&self) -> [u8; 8] {
        self.max_value
    }

    pub fn max_as_unsigned(&self) -> Result<u64> {
        if !self.data_type.is_unsigned() {
            bail!("Extra bytes datatype is not an unsigned integer type");
        }

        let as_u64 = u64::from_le_bytes(self.max_value);
        Ok(as_u64)
    }

    pub fn max_as_signed(&self) -> Result<i64> {
        if !self.data_type.is_signed() {
            bail!("Extra bytes datatype is not a signed integer type");
        }

        let as_i64 = i64::from_le_bytes(self.max_value);
        Ok(as_i64)
    }

    pub fn max_as_float(&self) -> Result<f64> {
        if !self.data_type.is_unsigned() {
            bail!("Extra bytes datatype is not a floating point type");
        }

        let as_f64 = f64::from_le_bytes(self.max_value);
        Ok(as_f64)
    }

    pub fn no_data_value_raw(&self) -> [u8; 8] {
        self.no_data_value
    }

    pub fn no_data_value_as_unsigned(&self) -> Result<u64> {
        if !self.data_type.is_unsigned() {
            bail!("Extra bytes datatype is not an unsigned integer type");
        }

        let as_u64 = u64::from_le_bytes(self.no_data_value);
        Ok(as_u64)
    }

    pub fn no_data_value_as_signed(&self) -> Result<i64> {
        if !self.data_type.is_signed() {
            bail!("Extra bytes datatype is not a signed integer type");
        }

        let as_i64 = i64::from_le_bytes(self.no_data_value);
        Ok(as_i64)
    }

    pub fn no_data_value_as_float(&self) -> Result<f64> {
        if !self.data_type.is_unsigned() {
            bail!("Extra bytes datatype is not a floating point type");
        }

        let as_f64 = f64::from_le_bytes(self.no_data_value);
        Ok(as_f64)
    }

    /// Returns a matching `PointAttributeDefinition` for the extra bytes described by this `ExtraBytesEntry`
    pub fn get_point_attribute(&self) -> Result<PointAttributeDefinition> {
        let pasture_datatype: PointAttributeDataType =
            self.data_type.try_into().context("Invalid data type")?;
        Ok(PointAttributeDefinition::custom(
            Cow::Owned(self.name.clone()),
            pasture_datatype,
        ))
    }
}

impl Display for ExtraBytesEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let scale_str = self.scale().map(|scale| format!("Scale: {scale}"));
        let offset_str = self.offset().map(|offset| format!("Offset: {offset}"));
        let min_str = if !self.options.min_is_relevant() {
            None
        } else if self.data_type.is_floating_point() {
            Some(format!("Min: {}", self.min_as_float().unwrap()))
        } else if self.data_type.is_signed() {
            Some(format!("Min: {}", self.min_as_signed().unwrap()))
        } else if self.data_type.is_unsigned() {
            Some(format!("Min: {}", self.min_as_unsigned().unwrap()))
        } else {
            Some(format!("Min (raw bytes): {:#?}", self.min_value))
        };
        let max_str = if !self.options.max_is_relevant() {
            None
        } else if self.data_type.is_floating_point() {
            Some(format!("Max: {}", self.max_as_float().unwrap()))
        } else if self.data_type.is_signed() {
            Some(format!("Max: {}", self.max_as_signed().unwrap()))
        } else if self.data_type.is_unsigned() {
            Some(format!("Max: {}", self.max_as_unsigned().unwrap()))
        } else {
            Some(format!("Max (raw bytes): {:#?}", self.max_value))
        };
        let no_value_str = if !self.options.no_data_is_relevant() {
            None
        } else if self.data_type.is_floating_point() {
            Some(format!(
                "No-data value: {}",
                self.no_data_value_as_float().unwrap()
            ))
        } else if self.data_type.is_signed() {
            Some(format!(
                "No-data value: {}",
                self.no_data_value_as_signed().unwrap()
            ))
        } else if self.data_type.is_unsigned() {
            Some(format!(
                "No-data value: {}",
                self.no_data_value_as_unsigned().unwrap()
            ))
        } else {
            Some(format!("No-data value (raw bytes): {:#?}", self.min_value))
        };
        let helper_array = [scale_str, offset_str, min_str, max_str, no_value_str];
        let combined_strings = helper_array
            .iter()
            .filter_map(|s| s.as_ref().map(|s| s.to_owned()));
        let combined_strings =
            itertools::Itertools::intersperse_with(combined_strings, || " ".into())
                .collect::<String>();

        write!(f, "{} {} {}", self.name, self.data_type, combined_strings)
    }
}

impl TryFrom<&RawExtraBytesEntry> for ExtraBytesEntry {
    type Error = anyhow::Error;
    fn try_from(raw_entry: &RawExtraBytesEntry) -> Result<Self> {
        Ok(Self {
            data_type: raw_entry.data_type.into(),
            description: las_string_to_rust_string(&raw_entry.description)
                .context("Description is invalid string")?,
            name: las_string_to_rust_string(&raw_entry.name).context("Name is invalid string")?,
            offset: raw_entry.offset,
            max_value: raw_entry.max,
            min_value: raw_entry.min,
            no_data_value: raw_entry.no_data,
            options: ExtraBytesOptions(raw_entry.options),
            scale: raw_entry.scale,
        })
    }
}

impl From<&ExtraBytesEntry> for RawExtraBytesEntry {
    fn from(val: &ExtraBytesEntry) -> Self {
        let mut ret = RawExtraBytesEntry {
            data_type: val.data_type.into(),
            options: val.options.0,
            no_data: val.no_data_value,
            min: val.min_value,
            max: val.max_value,
            scale: val.scale,
            offset: val.offset,
            ..Default::default()
        };
        write_rust_string_into_las_ascii_array(&val.name, &mut ret.name);
        write_rust_string_into_las_ascii_array(&val.description, &mut ret.description);
        ret
    }
}

/// Build to create `ExtraBytesEntry` values
#[derive(Clone, Debug)]
pub struct ExtraBytesEntryBuilder {
    extra_bytes_entry: ExtraBytesEntry,
}

impl ExtraBytesEntryBuilder {
    pub fn new(data_type: ExtraBytesDataType, name: String, description: String) -> Self {
        Self {
            extra_bytes_entry: ExtraBytesEntry {
                data_type,
                name,
                description,
                max_value: Default::default(),
                min_value: Default::default(),
                no_data_value: Default::default(),
                offset: Default::default(),
                options: Default::default(),
                scale: Default::default(),
            },
        }
    }

    pub fn with_offset(mut self, offset: f64) -> Self {
        self.extra_bytes_entry.offset = offset;
        self.extra_bytes_entry.options.set_use_offset(true);
        self
    }

    pub fn with_scale(mut self, scale: f64) -> Self {
        self.extra_bytes_entry.scale = scale;
        self.extra_bytes_entry.options.set_use_scale(true);
        self
    }

    pub fn min_data_value(mut self, min_data_value: [u8; 8]) -> Self {
        self.extra_bytes_entry.min_value = min_data_value;
        self.extra_bytes_entry.options.set_min_is_relevant(true);
        self
    }

    pub fn max_data_value(mut self, max_data_value: [u8; 8]) -> Self {
        self.extra_bytes_entry.max_value = max_data_value;
        self.extra_bytes_entry.options.set_max_is_relevant(true);
        self
    }

    pub fn no_data_value(mut self, no_data_value: [u8; 8]) -> Self {
        self.extra_bytes_entry.no_data_value = no_data_value;
        self.extra_bytes_entry.options.set_no_data_is_relevant(true);
        self
    }

    pub fn build(self) -> ExtraBytesEntry {
        self.extra_bytes_entry
    }
}

// LAS VLR describing the meaning of extra bytes within a point record
#[derive(Clone, Debug)]
pub struct ExtraBytesVlr {
    entries: Vec<ExtraBytesEntry>,
}

impl ExtraBytesVlr {
    pub const RECORD_ID: u16 = 4;

    pub fn entries(&self) -> &[ExtraBytesEntry] {
        &self.entries
    }
}

impl FromIterator<ExtraBytesEntry> for ExtraBytesVlr {
    fn from_iter<T: IntoIterator<Item = ExtraBytesEntry>>(iter: T) -> Self {
        Self {
            entries: iter.into_iter().collect(),
        }
    }
}

impl TryFrom<&'_ Vlr> for ExtraBytesVlr {
    type Error = anyhow::Error;

    fn try_from(value: &'_ Vlr) -> std::result::Result<Self, Self::Error> {
        if value.user_id != KNOWN_VLR_USER_ID {
            return Err(anyhow!(
                "Expected user_id {KNOWN_VLR_USER_ID} but got {}",
                value.user_id
            ));
        }
        if value.record_id != Self::RECORD_ID {
            return Err(anyhow!(
                "Expected record ID {} but got {}",
                Self::RECORD_ID,
                value.record_id
            ));
        }

        if value.data.len() % RAW_EXTRA_BYTES_ENTRY_SIZE != 0 {
            bail!("VLR data size ({} bytes) is not a multiple of the size of an EXTRA_BYTES entry ({} bytes)", value.data.len(), RAW_EXTRA_BYTES_ENTRY_SIZE);
        }

        let raw_entries: &[RawExtraBytesEntry] = bytemuck::cast_slice(&value.data);
        let entries = raw_entries
            .iter()
            .map(|raw_entry| raw_entry.try_into())
            .collect::<Result<_, _>>()?;

        Ok(Self { entries })
    }
}

impl TryInto<Vlr> for &ExtraBytesVlr {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<Vlr> {
        let entries: Vec<u8> = self
            .entries
            .iter()
            .flat_map(|entry| {
                let raw_entry: RawExtraBytesEntry = entry.into();
                bytemuck::bytes_of(&raw_entry).to_owned()
            })
            .collect::<Vec<_>>();
        assert!(entries.len() % RAW_EXTRA_BYTES_ENTRY_SIZE == 0);

        let mut raw_vlr = las_rs::raw::Vlr::default();
        write_rust_string_into_las_ascii_array("LASF_Spec", &mut raw_vlr.user_id);
        raw_vlr.record_id = ExtraBytesVlr::RECORD_ID;
        raw_vlr.record_length_after_header = RecordLength::Vlr(
            entries
                .len()
                .try_into()
                .context("Length of Extra Bytes VLR entries exceeds capacity of u16 value")?,
        );
        raw_vlr.data = entries;

        Ok(Vlr::new(raw_vlr))
    }
}

impl Display for ExtraBytesVlr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Extra Bytes:")?;
        for entry in &self.entries {
            writeln!(f, "\t{}", entry)?;
        }
        Ok(())
    }
}

fn display_generic_vlr(vlr: &Vlr, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    writeln!(f, "\t{}", vlr.description)?;
    writeln!(f, "\t\tUser:      {}", vlr.user_id)?;
    writeln!(f, "\t\tRecord:    {}", vlr.record_id)?;

    if let Ok(vlr_as_str) = String::from_utf8(vlr.data.clone()) {
        writeln!(f, "\t\tData:      {vlr_as_str}")
    } else if vlr.data.len() > 32 {
        writeln!(f, "\t\tData:      {:?}...", &vlr.data[0..32])
    } else {
        writeln!(f, "\t\tData:      {:?}", vlr.data)
    }
}

/// `Metadata` implementation for LAS/LAZ files
#[derive(Debug, Clone)]
pub struct LASMetadata {
    bounds: AABB<f64>,
    point_count: usize,
    point_format: Format,
    classification_lookup_vlr: Option<Box<ClassificationLookup>>, //Boxed because it is large
    text_area_description_vlr: Option<TextAreaDescription>,
    extra_bytes_vlr: Option<ExtraBytesVlr>,
    raw_las_header: Option<Header>,
}

impl LASMetadata {
    /// Creates a new `LASMetadata` from the given parameters
    /// ```
    /// # use pasture_io::las::LASMetadata;
    /// # use pasture_core::math::AABB;
    /// # use pasture_core::nalgebra::Point3;
    ///
    /// let min = Point3::new(0.0, 0.0, 0.0);
    /// let max = Point3::new(1.0, 1.0, 1.0);
    /// let format = pasture_io::las_rs::point::Format::new(0).unwrap();
    /// let metadata = LASMetadata::new(AABB::from_min_max(min, max), 1024, format);
    /// ```
    pub fn new(bounds: AABB<f64>, point_count: usize, point_format: Format) -> Self {
        Self {
            bounds,
            point_count,
            point_format,
            raw_las_header: None,
            classification_lookup_vlr: None,
            extra_bytes_vlr: None,
            text_area_description_vlr: None,
        }
    }

    /// Returns the number of points for the associated `LASMetadata`
    pub fn point_count(&self) -> usize {
        self.point_count
    }

    /// Returns the LAS point format for the associated `LASMetadata`
    pub fn point_format(&self) -> Format {
        self.point_format
    }

    /// Returns the raw LAS header for the associated `LASMetadata`. This value is only present if the
    /// associated `LASMetadata` was created from a raw LAS header
    pub fn raw_las_header(&self) -> Option<&Header> {
        self.raw_las_header.as_ref()
    }

    /// Returns the Classification Lookup VLR, if it exists
    pub fn classification_lookup_vlr(&self) -> Option<&ClassificationLookup> {
        self.classification_lookup_vlr.as_deref()
    }

    /// Returns the Text Area Description VLR, if it exists
    pub fn text_area_description_vlr(&self) -> Option<&TextAreaDescription> {
        self.text_area_description_vlr.as_ref()
    }

    /// Returns the Extra Bytes VLR, if it exists
    pub fn extra_bytes_vlr(&self) -> Option<&ExtraBytesVlr> {
        self.extra_bytes_vlr.as_ref()
    }
}

impl Display for LASMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Bounds (min):                {}", self.bounds.min())?;
        writeln!(f, "Bounds (max):                {}", self.bounds.max())?;
        writeln!(f, "Number of point records:     {}", self.point_count)?;
        writeln!(f, "Point record format:         {}", self.point_format)?;

        if let Some(classification_vlr) = &self.classification_lookup_vlr {
            write!(f, "{}", classification_vlr)?;
        }
        if let Some(text_area_description_vlr) = &self.text_area_description_vlr {
            write!(f, "{}", text_area_description_vlr)?;
        }
        if let Some(extra_bytes) = &self.extra_bytes_vlr {
            write!(f, "{}", extra_bytes)?;
        }

        if let Some(las_header) = &self.raw_las_header {
            writeln!(f, "Raw LAS header entries:")?;
            //writeln!(f, "\tFile signature:          {}", las_header.si)
            writeln!(
                f,
                "\tFile source ID:              {}",
                las_header.file_source_id()
            )?;
            //writeln!(f, "\tGlobal encoding:         {}", las_header.);
            writeln!(f, "\tGUID:                        {}", las_header.guid())?;
            writeln!(f, "\tVersion:                     {}", las_header.version())?;
            writeln!(
                f,
                "\tSystem identifier:           {}",
                las_header.system_identifier()
            )?;
            writeln!(
                f,
                "\tGenerating software:         {}",
                las_header.generating_software()
            )?;
            writeln!(
                f,
                "\tFile creation date:          {}",
                las_header
                    .date()
                    .map(|date| date.to_string())
                    .unwrap_or("N/A".into())
            )?;
            writeln!(
                f,
                "\tPoint data format:           {}",
                las_header.point_format(),
            )?;

            let regular_bytes =
                las_header.point_format().len() - las_header.point_format().extra_bytes;
            writeln!(
                f,
                "\tPoint record size:           {} ({} + {} extra bytes)",
                las_header.point_format().len(),
                regular_bytes,
                las_header.point_format().extra_bytes
            )?;

            writeln!(
                f,
                "\tNumber of point records:     {}",
                las_header.number_of_points()
            )?;
            writeln!(
                f,
                "\tNumber of points by return:  {} {} {} {} {}",
                las_header.number_of_points_by_return(1).unwrap_or_default(),
                las_header.number_of_points_by_return(2).unwrap_or_default(),
                las_header.number_of_points_by_return(3).unwrap_or_default(),
                las_header.number_of_points_by_return(4).unwrap_or_default(),
                las_header.number_of_points_by_return(5).unwrap_or_default()
            )?;
            writeln!(
                f,
                "\tScale (x y z):               {} {} {}",
                las_header.transforms().x.scale,
                las_header.transforms().y.scale,
                las_header.transforms().z.scale
            )?;
            writeln!(
                f,
                "\tOffset (x y z):              {} {} {}",
                las_header.transforms().x.offset,
                las_header.transforms().y.offset,
                las_header.transforms().z.offset
            )?;
            writeln!(
                f,
                "\tMin (x y z):                 {} {} {}",
                las_header.bounds().min.x,
                las_header.bounds().min.y,
                las_header.bounds().min.z,
            )?;
            writeln!(
                f,
                "\tMax (x y z):                 {} {} {}",
                las_header.bounds().max.x,
                las_header.bounds().max.y,
                las_header.bounds().max.z,
            )?;

            if las_header.vlrs().is_empty() {
                writeln!(f, "No VLRs")?;
            } else {
                writeln!(f, "VLRs")?;
                for vlr in las_header.vlrs() {
                    display_generic_vlr(vlr, f)?;
                }
            }

            if las_header.evlrs().is_empty() {
                writeln!(f, "No extended VLRs")?;
            } else {
                writeln!(f, "Extended VLRs")?;
                for evlr in las_header.evlrs() {
                    display_generic_vlr(evlr, f)?;
                }
            }
        }

        Ok(())
    }
}

impl Metadata for LASMetadata {
    fn bounds(&self) -> Option<AABB<f64>> {
        Some(self.bounds)
    }

    fn number_of_points(&self) -> Option<usize> {
        Some(self.point_count)
    }

    fn get_named_field(&self, field_name: &str) -> Option<Box<dyn Any>> {
        match field_name {
            named_fields::FILE_CREATION_DAY_OF_YEAR => self
                .raw_las_header
                .as_ref()
                .and_then(|header| header.date())
                .map(|date| -> Box<dyn Any> {
                    let day_of_year: u16 = date.ordinal().try_into().unwrap();
                    Box::new(day_of_year)
                }),
            named_fields::FILE_CREATION_YEAR => self
                .raw_las_header
                .as_ref()
                .and_then(|header| header.date())
                .map(|date| -> Box<dyn Any> {
                    let year: u16 = date.year().try_into().unwrap();
                    Box::new(year)
                }),
            named_fields::FILE_SOURCE_I_D => self
                .raw_las_header
                .as_ref()
                .map(|header| -> Box<dyn Any> { Box::new(header.file_source_id()) }),
            named_fields::GENERATING_SOFTWARE => {
                self.raw_las_header.as_ref().map(|header| -> Box<dyn Any> {
                    Box::new(header.generating_software().to_owned())
                })
            }
            named_fields::SYSTEM_IDENTIFIER => self
                .raw_las_header
                .as_ref()
                .map(|header| -> Box<dyn Any> { Box::new(header.system_identifier().to_owned()) }),
            named_fields::VERSION => self
                .raw_las_header
                .as_ref()
                .map(|header| -> Box<dyn Any> { Box::new(header.version().to_string()) }),
            _ => None,
        }
    }

    fn clone_into_box(&self) -> Box<dyn Metadata> {
        Box::new(self.clone())
    }
}

impl TryFrom<&las::Header> for LASMetadata {
    type Error = anyhow::Error;

    fn try_from(header: &las::Header) -> std::result::Result<Self, Self::Error> {
        let classification_lookup_vlr = header
            .vlrs()
            .iter()
            .find(|vlr| {
                vlr.user_id == KNOWN_VLR_USER_ID && vlr.record_id == ClassificationLookup::RECORD_ID
            })
            .map(ClassificationLookup::try_from)
            .transpose()
            .context("Could not parse Classification Lookup VLR")?;

        let text_area_description_vlr = header
            .vlrs()
            .iter()
            .find(|vlr| {
                vlr.user_id == KNOWN_VLR_USER_ID && vlr.record_id == TextAreaDescription::RECORD_ID
            })
            .map(TextAreaDescription::try_from)
            .transpose()
            .context("Could not parse Text Area Description VLR")?;

        let extra_bytes_vlr = header
            .vlrs()
            .iter()
            .find(|vlr| {
                vlr.user_id == KNOWN_VLR_USER_ID && vlr.record_id == ExtraBytesVlr::RECORD_ID
            })
            .map(ExtraBytesVlr::try_from)
            .transpose()
            .context("Could not parse Extra Bytes VLR")?;

        Ok(Self {
            bounds: las_bounds_to_pasture_bounds(header.bounds()),
            point_count: header.number_of_points() as usize,
            point_format: *header.point_format(),
            raw_las_header: Some(header.clone()),
            classification_lookup_vlr: classification_lookup_vlr.map(Box::new),
            extra_bytes_vlr,
            text_area_description_vlr,
        })
    }
}

impl TryFrom<las::Header> for LASMetadata {
    type Error = anyhow::Error;

    fn try_from(value: las::Header) -> std::result::Result<Self, Self::Error> {
        (&value).try_into()
    }
}
