use pasture_core::meta::Metadata;
use std::fmt::Display;

/// `Metadata` implementation for ascii files
/// In general there is no metadata in ascii files.
#[derive(Debug, Clone, Default)]
pub struct AsciiMetadata;

impl Display for AsciiMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Ascii Metadata")?;
        Ok(())
    }
}

impl Metadata for AsciiMetadata {
    fn bounds(&self) -> Option<pasture_core::math::AABB<f64>> {
        None
    }

    fn number_of_points(&self) -> Option<usize> {
        None
    }

    fn get_named_field(&self, _field_name: &str) -> Option<Box<dyn std::any::Any>> {
        None
    }

    fn clone_into_box(&self) -> Box<dyn Metadata> {
        Box::new(self.clone())
    }
}
