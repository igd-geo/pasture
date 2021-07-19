use serde::{Deserialize, Serialize};
use static_assertions::const_assert;

pub mod attributes {
    use pasture_core::layout::{PointAttributeDataType, PointAttributeDefinition};

    /// Attribute definition for an RGBA color in the 3D Tiles format
    pub const COLOR_RGBA: PointAttributeDefinition =
        PointAttributeDefinition::custom("ColorRGBA", PointAttributeDataType::Vec4u8);
}

/// Header of .pnts files
#[repr(packed)]
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
}

const_assert!(PntsHeader::BYTE_LENGTH == std::mem::size_of::<PntsHeader>());
