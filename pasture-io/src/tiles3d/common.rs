use anyhow::{anyhow, bail, Context, Result};
use pasture_core::{
    math::Alignable,
    nalgebra::{Vector3, Vector4},
};
use serde_json::Value;
use std::{
    convert::TryInto,
    io::{BufRead, Seek, Write},
};

/// Reads a JSON header (e.g. FeatureTable or BatchTable header) from the given `reader` according to the given
/// `header_size`
pub fn read_json_header<R: BufRead + Seek>(mut reader: R, header_size: usize) -> Result<Value> {
    let mut str_buf = vec![0; header_size];
    reader
        .read_exact(str_buf.as_mut_slice())
        .context("JSON header size is larger than the remaining number of bytes in the reader")?;
    let header_text = String::from_utf8(str_buf).context("JSON header is no valid UTF-8 text")?;
    let json: Value =
        serde_json::from_str(header_text.as_str()).context("Could not parse JSON header")?;

    Ok(json)
}

/// Writes a JSON header (e.g. FeatureTable or BatchTable header) to the given `writer`. This pads the header with trailing
/// spaces to an 8-byte boundary based on the given `position_in_file`. We specifically DON'T use the position in the `writer``
/// because the `writer` might be a temporary writer and not the final file writer!
pub fn write_json_header<W: Write>(
    mut writer: W,
    json_header: &Value,
    position_in_file: usize,
) -> Result<()> {
    // Convert to CString, then fill with padding bytes if required
    let header_json =
        serde_json::to_string(json_header).context("Could not convert JSON header to string")?;

    writer
        .write_all(header_json.as_bytes())
        .context("Could not write JSON header to writer")?;

    let current_position_in_file = position_in_file + header_json.as_bytes().len();

    let next_8_byte_boundary = current_position_in_file.align_to(8);
    let num_padding_bytes = next_8_byte_boundary - current_position_in_file;

    if num_padding_bytes > 0 {
        writer
            .write(&vec![0x20; num_padding_bytes as usize])
            .context("Could not write padding bytes of JSON header to writer")?;
    }

    Ok(())
}

/// Converts an array of JSON Values into a Vector3<f32>
pub fn json_arr_to_vec3f32(json_arr: &[Value]) -> Result<Vector3<f32>> {
    if json_arr.len() != 3 {
        bail!(
            "JSON array must have length 3 to convert to Vector3<f32> (but has length {})",
            json_arr.len()
        )
    }
    let vals = json_arr
        .iter()
        .map(|v| v.as_f64().ok_or(anyhow!("Can't convert JSON value to f64")))
        .collect::<Result<Vec<_>, _>>()?;

    let x = vals[0] as f32;
    let y = vals[1] as f32;
    let z = vals[2] as f32;

    Ok(Vector3::new(x, y, z))
}

/// Converts an array of JSON Values into a Vector3<f64>
pub fn json_arr_to_vec3f64(json_arr: &[Value]) -> Result<Vector3<f64>> {
    if json_arr.len() != 3 {
        bail!(
            "JSON array must have length 3 to convert to Vector3<f32> (but has length {})",
            json_arr.len()
        )
    }
    let vals = json_arr
        .iter()
        .map(|v| v.as_f64().ok_or(anyhow!("Can't convert JSON value to f64")))
        .collect::<Result<Vec<_>, _>>()?;

    let x = vals[0];
    let y = vals[1];
    let z = vals[2];

    Ok(Vector3::new(x, y, z))
}

pub fn json_arr_to_vec4u8(json_arr: &[Value]) -> Result<Vector4<u8>> {
    if json_arr.len() != 4 {
        bail!(
            "JSON array must have length 4 to convert to Vector4<u8> (but has length {})",
            json_arr.len()
        )
    }
    let vals = json_arr
        .iter()
        .map(|v| v.as_u64().ok_or(anyhow!("Can't convert JSON value to u64")))
        .collect::<Result<Vec<_>, _>>()?;

    let r: u8 = vals[0].try_into()?;
    let g: u8 = vals[1].try_into()?;
    let b: u8 = vals[2].try_into()?;
    let a: u8 = vals[3].try_into()?;

    Ok(Vector4::new(r, g, b, a))
}
