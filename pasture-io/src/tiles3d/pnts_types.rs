use anyhow::{bail, Result};
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
            magic: [b'p', b'n', b't', b's'],
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
        if self.magic != [b'p', b'n', b't', b's'] {
            bail!("No valid PNTS file, expected first four bytes to be equal to 'pnts', but was '{:?}' instead", self.magic);
        }
        Ok(())
    }
}
