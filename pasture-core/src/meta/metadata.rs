use crate::math::AABB;

use std::{any::Any, fmt::Display};

/// Trait that represents metadata of a point cloud. Metadata is a very loose term that represents
/// everything that is not the point data itself. There are some common accessors in this trait for
/// things like a bounding box, but also generic accessors for named parameters that depend on the
/// actual type of point cloud.
pub trait Metadata: Display {
    /// Returns the bounding box of the associated `Metadata`. Not every point cloud `Metadata` will have
    /// bounding box information readily available, in which case `None` is returned.
    fn bounds(&self) -> Option<AABB<f64>>;
    /// Returns the value of the metadata field named `field_name`, if it exists.
    fn get_named_field(&self, field_name: &str) -> Option<Box<dyn Any>>;
    /// Clone the associated `Metadata` and put it into a `Box`
    fn clone_into_box(&self) -> Box<dyn Metadata>;
}
