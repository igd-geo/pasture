use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::{
    collections::HashMap,
    convert::TryFrom,
    convert::TryInto,
    ffi::CStr,
    ffi::CString,
    io::{BufRead, Read, Seek, SeekFrom, Write},
};

/// A reference to data inside a FeatureTable binary body
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct FeatureTableDataReference {
    #[serde(rename = "byteOffset")]
    pub byte_offset: usize,
    #[serde(rename = "componentType", skip_serializing_if = "Option::is_none")]
    pub component_type: Option<String>,
}

/// Different possible values for an entry in a 3D Tiles FeatureTable
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum FeatureTableValue {
    SingleValue(serde_json::Value),
    Array(Vec<serde_json::Value>),
    DataReference(FeatureTableDataReference),
}

impl TryFrom<Value> for FeatureTableValue {
    type Error = anyhow::Error;

    fn try_from(val: Value) -> Result<Self> {
        if val.is_array() {
            let as_array = val.as_array().unwrap();
            return Ok(FeatureTableValue::Array(as_array.clone()));
        }

        if val.is_object() {
            let as_obj = val.as_object().unwrap();
            // Object can mean single-value entry OR reference to binary data. The latter is identified by the
            // presence of a 'byteOffset' key
            if as_obj.contains_key("byteOffset") {
                let data_reference = serde_json::from_value::<FeatureTableDataReference>(val)?;
                return Ok(FeatureTableValue::DataReference(data_reference));
            } else {
                return Ok(FeatureTableValue::SingleValue(val));
            }
        }

        Ok(FeatureTableValue::SingleValue(val))
    }
}

impl TryFrom<&Value> for FeatureTableValue {
    type Error = anyhow::Error;

    fn try_from(val: &Value) -> Result<Self> {
        if val.is_array() {
            let as_array = val.as_array().unwrap();
            return Ok(FeatureTableValue::Array(as_array.clone()));
        }

        if val.is_object() {
            let as_obj = val.as_object().unwrap();
            // Object can mean single-value entry OR reference to binary data. The latter is identified by the
            // presence of a 'byteOffset' key
            if as_obj.contains_key("byteOffset") {
                let data_reference =
                    serde_json::from_value::<FeatureTableDataReference>(val.clone())?;
                return Ok(FeatureTableValue::DataReference(data_reference));
            } else {
                return Ok(FeatureTableValue::SingleValue(val.clone()));
            }
        }

        Ok(FeatureTableValue::SingleValue(val.clone()))
    }
}

impl Into<Value> for FeatureTableValue {
    fn into(self) -> Value {
        match self {
            FeatureTableValue::SingleValue(val) => val,
            FeatureTableValue::Array(arr) => Value::Array(arr),
            FeatureTableValue::DataReference(data_reference) => {
                serde_json::to_value(data_reference)
                    .expect("Could not convert FeatureTableDataReference to JSON Value")
            }
        }
    }
}

impl Into<Value> for &FeatureTableValue {
    fn into(self) -> Value {
        match self {
            FeatureTableValue::SingleValue(val) => val.clone(),
            FeatureTableValue::Array(arr) => Value::Array(arr.clone()),
            FeatureTableValue::DataReference(data_reference) => {
                serde_json::to_value(data_reference)
                    .expect("Could not convert FeatureTableDataReference to JSON Value")
            }
        }
    }
}

/// 3D Tiles feature table structure
pub type FeatureTableHeader = HashMap<String, FeatureTableValue>;

/// Deserialize a `FeatureTableHeader` from the given `reader`. If successful, returns the serialized header and the
/// `reader` will be at the start of the binary body of the 3D Tiles FeatureTable. See the [3D Tiles documentation](https://github.com/CesiumGS/3d-tiles/blob/master/specification/TileFormats/FeatureTable/README.md)
/// for more information. If this operation fails, the reader will be in an undefined state.
pub fn deser_feature_table_header<R: BufRead + Seek>(mut reader: R) -> Result<FeatureTableHeader> {
    let mut str_buf = vec![];
    reader.read_until(0, &mut str_buf)?;

    // Move reader to 8-byte boundary as this is required by the 3D Tiles spec
    {
        let where_are_we = reader.seek(SeekFrom::Current(0))?;
        let next_8_byte_boundary = ((where_are_we + 7) / 8) * 8;
        reader.seek(SeekFrom::Start(next_8_byte_boundary))?;
    }

    let feature_table_json_str = CStr::from_bytes_with_nul(str_buf.as_slice())?;
    let feature_table_json: Value = serde_json::from_str(feature_table_json_str.to_str()?)?;
    let feature_table_obj = feature_table_json
        .as_object()
        .ok_or(anyhow!("FeatureTable JSON header was no JSON object"))?;
    // Convert the object to our `FeatureTableHeader` type
    Ok(feature_table_obj
        .iter()
        .map(|(k, v)| -> Result<(String, FeatureTableValue)> {
            let feature_table_value: FeatureTableValue = v.try_into()?;
            Ok((k.clone(), feature_table_value))
        })
        .collect::<Result<HashMap<_, _>, _>>()?)
}

/// Serializes the given `FeatureTableHeader` to the given `writer`. If successful, the `writer` will be at the appropriate
/// position for writing the FeatureTable body (i.e. required padding spaces have been written as per the [3D Tiles documentation](https://github.com/CesiumGS/3d-tiles/blob/master/specification/TileFormats/FeatureTable/README.md)).
pub fn ser_feature_table_header<W: Write>(
    mut writer: W,
    feature_table_header: &FeatureTableHeader,
) -> Result<()> {
    let header_as_map = feature_table_header
        .iter()
        .map(|(k, v)| -> (String, Value) { (k.clone(), v.into()) })
        .collect::<Map<_, _>>();
    let header_json = serde_json::to_string(&Value::Object(header_as_map))?;
    let header_json_cstr = CString::new(header_json)?;

    writer.write(header_json_cstr.as_bytes_with_nul())?;

    let header_json_len = header_json_cstr.as_bytes_with_nul().len();
    let next_8_byte_boundary = ((header_json_len + 7) / 8) * 8;
    let num_padding_bytes = next_8_byte_boundary - header_json_len;

    if num_padding_bytes > 0 {
        writer.write(&vec![0x20; num_padding_bytes])?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::{BufReader, BufWriter, Cursor};

    use super::*;
    use serde_json::json;

    fn dummy_feature_table_header() -> FeatureTableHeader {
        let mut header = FeatureTableHeader::new();
        header.insert(
            "SINGLE_FIELD".into(),
            FeatureTableValue::SingleValue(json!(23)),
        );
        header.insert(
            "ARRAY_FIELD".into(),
            FeatureTableValue::Array(vec![json!(1), json!(2), json!(3)]),
        );
        header.insert(
            "REFERENCE_FIELD".into(),
            FeatureTableValue::DataReference(FeatureTableDataReference {
                byte_offset: 42,
                component_type: Some("FLOAT".into()),
            }),
        );
        header
    }

    #[test]
    fn test_3dtiles_feature_table_io() -> Result<()> {
        let expected_header = dummy_feature_table_header();

        let mut writer = BufWriter::new(Cursor::new(vec![]));
        ser_feature_table_header(&mut writer, &expected_header)?;

        // Make sure that the header is written with padding bytes so that we are at an 8-byte boundary
        let stream_pos = writer.seek(SeekFrom::Current(0))?;
        assert_eq!(stream_pos % 8, 0);

        let mut cursor = writer.into_inner()?;
        cursor.seek(SeekFrom::Start(0))?;

        let mut reader = BufReader::new(cursor);
        let actual_header = deser_feature_table_header(&mut reader)?;

        assert_eq!(expected_header, actual_header);

        Ok(())
    }
}
