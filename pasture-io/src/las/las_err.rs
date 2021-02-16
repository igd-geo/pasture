use anyhow::anyhow;

/// Maps the internal error type of the laz-rs crate to an `anyhow::Error`. Unfortunately, the laz-rs error type
/// does not implement the `Error` trait :(
pub(crate) fn map_laz_err(laz_err: laz::LasZipError) -> anyhow::Error {
    anyhow!("LasZip error: {}", laz_err.to_string())
}
