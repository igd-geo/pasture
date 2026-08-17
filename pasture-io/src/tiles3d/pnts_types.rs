use std::io::{Read, Write};

use anyhow::{Result, bail};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use serde::{Deserialize, Serialize};

pub mod attributes {
    use std::borrow::Cow;

    use pasture_core::layout::{PointAttributeDataType, PointAttributeDefinition};

    /// Attribute definition for an RGBA color in the 3D Tiles format
    pub const COLOR_RGBA: PointAttributeDefinition = PointAttributeDefinition::custom(
        Cow::Borrowed("ColorRGBA"),
        PointAttributeDataType::Vec4u8,
    );
}

/// Header of .pnts files
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PntsHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub byte_length: u32,
    pub feature_table_json_byte_length: u32,
    pub feature_table_binary_byte_length: u32,
    pub batch_table_json_byte_length: u32,
    pub batch_table_binary_byte_length: u32,
}

impl PntsHeader {
    /// Length of a .pnts header in bytes
    /// (Header has 7 fields, each field has a length of 4 bytes -> 7*4=28.)
    pub const BYTE_LENGTH: usize = 28;

    pub fn new(
        version: u32,
        byte_length: u32,
        feature_table_json_byte_length: u32,
        feature_table_binary_byte_length: u32,
        batch_table_json_byte_length: u32,
        batch_table_binary_byte_length: u32,
    ) -> Self {
        Self {
            magic: *b"pnts",
            version,
            byte_length,
            feature_table_json_byte_length,
            feature_table_binary_byte_length,
            batch_table_json_byte_length,
            batch_table_binary_byte_length,
        }
    }

    /// Returns an Err if the magic bytes in this header are not correct
    pub fn verify_magic(&self) -> Result<()> {
        if self.magic != *b"pnts" {
            bail!(
                "No valid PNTS file, expected first four bytes to be equal to 'pnts', but was '{:?}' instead",
                self.magic
            );
        }
        Ok(())
    }

    pub fn write_to(&self, write: &mut impl Write) -> Result<()> {
        write.write_all(&self.magic)?;
        write.write_u32::<LittleEndian>(self.version)?;
        write.write_u32::<LittleEndian>(self.byte_length)?;
        write.write_u32::<LittleEndian>(self.feature_table_json_byte_length)?;
        write.write_u32::<LittleEndian>(self.feature_table_binary_byte_length)?;
        write.write_u32::<LittleEndian>(self.batch_table_json_byte_length)?;
        write.write_u32::<LittleEndian>(self.batch_table_binary_byte_length)?;
        Ok(())
    }

    pub fn read_from(read: &mut impl Read) -> Result<Self> {
        let mut magic: [u8; 4] = [0; 4];
        read.read_exact(&mut magic)?;
        let version = read.read_u32::<LittleEndian>()?;
        let byte_length = read.read_u32::<LittleEndian>()?;
        let feature_table_json_byte_length = read.read_u32::<LittleEndian>()?;
        let feature_table_binary_byte_length = read.read_u32::<LittleEndian>()?;
        let batch_table_json_byte_length = read.read_u32::<LittleEndian>()?;
        let batch_table_binary_byte_length = read.read_u32::<LittleEndian>()?;

        Ok(PntsHeader {
            magic,
            version,
            byte_length,
            feature_table_binary_byte_length,
            feature_table_json_byte_length,
            batch_table_json_byte_length,
            batch_table_binary_byte_length,
        })
    }
}
