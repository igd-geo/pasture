use super::point_layout::*;

/// Trait that marks a Rust type for usage as a type in which point data can be stored.
/// This trait allows the mapping between Rust types at compile time and the dynamic PointLayout
/// type.
///
/// TODO Write a more in-depth explanation
pub trait PointType {
    fn layout() -> PointLayout;
}

/// Returns the corresponding PointLayout for the given PointType T
pub fn get_point_layout<T: PointType>() -> PointLayout {
    T::layout()
}
