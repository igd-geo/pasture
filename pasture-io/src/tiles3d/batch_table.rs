use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::{
    collections::HashMap,
    convert::TryFrom,
    convert::TryInto,
    io::{BufRead, Seek, Write},
};

use super::{read_json_header, write_json_header};

/// A reference to data inside a BatchTable binary body
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct BatchTableDataReference {
    #[serde(rename = "byteOffset")]
    pub byte_offset: usize,
    #[serde(rename = "componentType")]
    pub component_type: String,
    #[serde(rename = "type")]
    pub scalar_or_vector_type: String,
}

/// An entry inside a BatchTable
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum BatchTableEntry {
    /// An entry containing array data
    ArrayData(Vec<Value>),
    /// An entry refering to the binary body of the BatchTable
    DataReference(BatchTableDataReference),
}

impl TryFrom<Value> for BatchTableEntry {
    type Error = anyhow::Error;

    fn try_from(val: Value) -> Result<Self> {
        if val.is_array() {
            let val_as_array = val.as_array().unwrap();
            return Ok(BatchTableEntry::ArrayData(val_as_array.clone()));
        }
        if val.is_object() {
            let data_reference = serde_json::from_value::<BatchTableDataReference>(val)?;
            return Ok(BatchTableEntry::DataReference(data_reference));
        }

        bail!("JSON value cannot be converted to BatchTableEntry because it is neither an array nor an object")
    }
}

impl TryFrom<&Value> for BatchTableEntry {
    type Error = anyhow::Error;

    fn try_from(val: &Value) -> Result<Self> {
        if val.is_array() {
            let val_as_array = val.as_array().unwrap();
            return Ok(BatchTableEntry::ArrayData(val_as_array.clone()));
        }
        if val.is_object() {
            let data_reference = serde_json::from_value::<BatchTableDataReference>(val.clone())?;
            return Ok(BatchTableEntry::DataReference(data_reference));
        }

        bail!("JSON value cannot be converted to BatchTableEntry because it is neither an array nor an object")
    }
}

impl Into<Value> for BatchTableEntry {
    fn into(self) -> Value {
        match self {
            BatchTableEntry::ArrayData(array) => Value::Array(array),
            BatchTableEntry::DataReference(data_reference) => serde_json::to_value(data_reference)
                .expect("Could not convert BatchTableEntry to JSON Value"),
        }
    }
}

impl Into<Value> for &BatchTableEntry {
    fn into(self) -> Value {
        match self {
            BatchTableEntry::ArrayData(array) => Value::Array(array.clone()),
            BatchTableEntry::DataReference(data_reference) => serde_json::to_value(data_reference)
                .expect("Could not convert BatchTableEntry to JSON Value"),
        }
    }
}

/// A 3D Tiles BatchTable header, which is a collection of BatchTableEntries
pub type BatchTableHeader = HashMap<String, BatchTableEntry>;

/// Deserialize a `BatchTableHeader` from the given `reader`. If successful, returns the serialized header and the
/// `reader` will be at the start of the binary body of the 3D Tiles BatchTable. See the [3D Tiles documentation](https://github.com/CesiumGS/3d-tiles/blob/master/specification/TileFormats/BatchTable/README.md)
/// for more information. If this operation fails, the reader will be in an undefined state.
pub fn deser_batch_table_header<R: BufRead + Seek>(mut reader: R) -> Result<BatchTableHeader> {
    let batch_table_header_json = read_json_header(&mut reader)?;
    let batch_table_json_obj = batch_table_header_json
        .as_object()
        .ok_or(anyhow!("BatchTable JSON header was no JSON object"))?;
    // Convert JSON object to `BatchTableHeader`
    Ok(batch_table_json_obj
        .iter()
        .map(|(k, v)| -> Result<(String, BatchTableEntry)> {
            let batch_table_entry: BatchTableEntry = v.try_into()?;
            Ok((k.clone(), batch_table_entry))
        })
        .collect::<Result<HashMap<_, _>, _>>()?)
}

/// Serializes the given `BatchTableHeader` to the given `writer`. If successful, the `writer` will be at the appropriate
/// position for writing the BatchTable body (i.e. required padding spaces have been written as per the [3D Tiles documentation](https://github.com/CesiumGS/3d-tiles/blob/master/specification/TileFormats/BatchTable/README.md)).
pub fn ser_batch_table_header<W: Write>(
    mut writer: W,
    batch_table_header: &BatchTableHeader,
) -> Result<()> {
    let header_as_map = batch_table_header
        .iter()
        .map(|(k, v)| -> (String, Value) { (k.clone(), v.into()) })
        .collect::<Map<_, _>>();
    let header_json_obj = Value::Object(header_as_map);

    write_json_header(&mut writer, &header_json_obj)
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json::json;
    use std::io::{BufReader, BufWriter, Cursor, SeekFrom};

    fn dummy_batch_table_header() -> BatchTableHeader {
        let mut header = BatchTableHeader::new();
        header.insert(
            "ARRAY_FIELD".into(),
            BatchTableEntry::ArrayData(vec![json!(1), json!(2), json!(3)]),
        );
        header.insert(
            "REFERENCE_FIELD".into(),
            BatchTableEntry::DataReference(BatchTableDataReference {
                byte_offset: 42,
                component_type: "FLOAT".into(),
                scalar_or_vector_type: "SCALAR".into(),
            }),
        );
        header
    }

    #[test]
    fn test_3dtiles_batch_table_io() -> Result<()> {
        let expected_header = dummy_batch_table_header();

        let mut writer = BufWriter::new(Cursor::new(vec![]));
        ser_batch_table_header(&mut writer, &expected_header)?;

        // Make sure that the header is written with padding bytes so that we are at an 8-byte boundary
        let stream_pos = writer.seek(SeekFrom::Current(0))?;
        assert_eq!(stream_pos % 8, 0);

        let mut cursor = writer.into_inner()?;
        cursor.seek(SeekFrom::Start(0))?;

        let mut reader = BufReader::new(cursor);
        let actual_header = deser_batch_table_header(&mut reader)?;

        assert_eq!(expected_header, actual_header);

        Ok(())
    }
}
