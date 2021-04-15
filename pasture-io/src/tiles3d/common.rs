use anyhow::Result;
use serde_json::Value;
use std::{
    ffi::{CStr, CString},
    io::{BufRead, Seek, SeekFrom, Write},
};

/// Reads a JSON header (e.g. FeatureTable or BatchTable header) from the given `reader`. This assumes that the JSON header
/// is 8-byte aligned and moves the `reader` accordingly
pub fn read_json_header<R: BufRead + Seek>(mut reader: R) -> Result<Value> {
    let mut str_buf = vec![];
    reader.read_until(0, &mut str_buf)?;

    // Move reader to 8-byte boundary as this is required by the 3D Tiles spec
    {
        let where_are_we = reader.seek(SeekFrom::Current(0))?;
        let next_8_byte_boundary = ((where_are_we + 7) / 8) * 8;
        reader.seek(SeekFrom::Start(next_8_byte_boundary))?;
    }

    let json_str = CStr::from_bytes_with_nul(str_buf.as_slice())?;
    let json: Value = serde_json::from_str(json_str.to_str()?)?;

    Ok(json)
}

/// Writes a JSON header (e.g. FeatureTable or BatchTable header) to the given `writer`. This pads the header with trailing
/// spaces to an 8-byte boundary
pub fn write_json_header<W: Write>(mut writer: W, json_header: &Value) -> Result<()> {
    // Convert to CString, then fill with padding bytes if required
    let header_json = serde_json::to_string(json_header)?;
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
