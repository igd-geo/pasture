use las::{Bounds, Header};
use las_rs::Vector;
use pasture_core::{math::AABB, meta::Metadata, nalgebra::Point3};

/// Converts a las-rs `Bounds` type into a pasture-core bounding box (`AABB<f64>`)
pub fn las_bounds_to_pasture_bounds(las_bounds: Bounds) -> AABB<f64> {
    let min_point = Point3::new(las_bounds.min.x, las_bounds.min.y, las_bounds.min.z);
    let max_point = Point3::new(las_bounds.max.x, las_bounds.max.y, las_bounds.max.z);
    AABB::from_min_max_unchecked(min_point, max_point)
}

/// Converts a pasture-core bounding box (`AABB<f64>`) into a las-rs `Bounds` type
pub fn pasture_bounds_to_las_bounds(bounds: &AABB<f64>) -> Bounds {
    Bounds {
        min: Vector {
            x: bounds.min().x,
            y: bounds.min().y,
            z: bounds.min().z,
        },
        max: Vector {
            x: bounds.max().x,
            y: bounds.max().y,
            z: bounds.max().z,
        },
    }
}

/// `Metadata` implementation for LAS/LAZ files
#[derive(Debug, Clone)]
pub struct LASMetadata {
    bounds: AABB<f64>,
    point_count: usize,
    point_format: u8,
    raw_las_header: Option<Header>,
}

impl LASMetadata {
    /// Creates a new `LASMetadata` from the given parameters
    /// ```
    /// # use pasture_io::las::LASMetadata;
    /// # use pasture_core::math::AABB;
    /// # use pasture_core::nalgebra::Point3;
    ///
    /// let min = Point3::new(0.0, 0.0, 0.0);
    /// let max = Point3::new(1.0, 1.0, 1.0);
    /// let metadata = LASMetadata::new(AABB::from_min_max(min, max), 1024, 0);
    /// ```
    pub fn new(bounds: AABB<f64>, point_count: usize, point_format: u8) -> Self {
        Self {
            bounds,
            point_count,
            point_format,
            raw_las_header: None,
        }
    }

    /// Returns the number of points for the associated `LASMetadata`
    pub fn point_count(&self) -> usize {
        self.point_count
    }

    /// Returns the LAS point format for the associated `LASMetadata`
    pub fn point_format(&self) -> u8 {
        self.point_format
    }
}

impl Metadata for LASMetadata {
    fn bounds(&self) -> Option<AABB<f64>> {
        Some(self.bounds)
    }

    fn get_named_field(&self, _field_name: &str) -> Option<&dyn std::any::Any> {
        todo!()
    }
}

impl From<&las::Header> for LASMetadata {
    fn from(header: &las::Header) -> Self {
        Self {
            bounds: las_bounds_to_pasture_bounds(header.bounds()),
            point_count: header.number_of_points() as usize,
            point_format: header
                .point_format()
                .to_u8()
                .expect("Invalid LAS point format"),
            raw_las_header: Some(header.clone()),
        }
    }
}

impl From<las::Header> for LASMetadata {
    fn from(header: las::Header) -> Self {
        (&header).into()
    }
}
