use std::{any::Any, fmt::Display};

use pasture_core::{
    math::AABB,
    meta::Metadata,
    nalgebra::{Vector3, Vector4},
};

/// Metadata for .pnts files. Contains the PNTS global semantics
#[derive(Clone, Debug)]
pub struct PntsMetadata {
    points_length: usize,
    rtc_center: Option<Vector3<f32>>,
    quantized_volume_offset: Option<Vector3<f32>>,
    quantized_volume_scale: Option<Vector3<f32>>,
    constant_rgba: Option<Vector4<u8>>,
    batch_length: Option<usize>,
}

impl PntsMetadata {
    pub fn new(
        points_length: usize,
        rtc_center: Option<Vector3<f32>>,
        quantized_volume_offset: Option<Vector3<f32>>,
        quantized_volume_scale: Option<Vector3<f32>>,
        constant_rgba: Option<Vector4<u8>>,
        batch_length: Option<usize>,
    ) -> Self {
        Self {
            points_length,
            rtc_center,
            quantized_volume_offset,
            quantized_volume_scale,
            constant_rgba,
            batch_length,
        }
    }

    pub fn points_length(&self) -> usize {
        self.points_length
    }
}

impl Metadata for PntsMetadata {
    fn bounds(&self) -> Option<AABB<f64>> {
        None
    }

    fn number_of_points(&self) -> Option<usize> {
        Some(self.points_length)
    }

    fn get_named_field(&self, field_name: &str) -> Option<Box<dyn std::any::Any>> {
        match field_name {
            "RTC_CENTER" => self
                .rtc_center
                .map(|v| -> Box<dyn Any> { Box::new(v.clone()) }),
            "QUANTIZED_VOLUME_OFFSET" => self
                .quantized_volume_offset
                .map(|v| -> Box<dyn Any> { Box::new(v.clone()) }),
            "QUANTIZED_VOLUME_SCALE" => self
                .quantized_volume_scale
                .map(|v| -> Box<dyn Any> { Box::new(v.clone()) }),
            "CONSTANT_RGBA" => self
                .constant_rgba
                .map(|v| -> Box<dyn Any> { Box::new(v.clone()) }),
            "BATCH_LENGTH" => Some(Box::new(self.batch_length)),
            _ => None,
        }
    }

    fn clone_into_box(&self) -> Box<dyn Metadata> {
        Box::new(self.clone())
    }
}

impl Display for PntsMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PntsMetadata {{\n")?;
        write!(f, "\t\"points_length\": {}\n", self.points_length)?;
        if let Some(rtc_center) = &self.rtc_center {
            write!(f, "\t\"rtc_center\": {}\n", rtc_center)?;
        }
        if let Some(v) = &self.quantized_volume_offset {
            write!(f, "\t\"quantized_volume_offset\": {}\n", v)?;
        }
        if let Some(v) = &self.quantized_volume_scale {
            write!(f, "\t\"quantized_volume_scale\": {}\n", v)?;
        }
        if let Some(v) = &self.constant_rgba {
            write!(f, "\t\"constant_rgba\": {}\n", v)?;
        }
        if let Some(v) = &self.batch_length {
            write!(f, "\t\"batch_length\": {}\n", v)?;
        }
        Ok(())
    }
}
