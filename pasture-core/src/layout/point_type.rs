use super::point_layout::*;

/// Trait that marks a Rust type for usage as a type in which point data can be stored.
/// This trait allows the mapping between Rust types at compile time and the dynamic `PointLayout`
/// type.
/// **You will almost never want to implement `PointType` manually! Prefer to use the `#[derive(PointType)]` procedural macro!**
pub trait PointType {
    /// Returns the associated `PointLayout` that describes the type implementing this trait.
    ///
    /// *Note:* This
    /// returns the `PointLayout` by value, even though it is a 'One instance per type' kind of object. There
    /// is an interesting discussion regarding this topic here: https://internals.rust-lang.org/t/per-type-static-variables-take-2/11551
    /// The essence seems to be that per-type static variables are not supported because of potential issues
    /// with dll linking on Windows. So for now we stick to returning the `PointLayout` by value, instead of
    /// a potentially more efficient `&'static PointLayout`
    fn layout() -> PointLayout;
}

/// Returns the corresponding PointLayout for the given PointType T
pub fn get_point_layout<T: PointType>() -> PointLayout {
    T::layout()
}
