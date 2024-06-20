use pasture_core::{
    containers::{BorrowedBuffer, BorrowedBufferExt},
    layout::{PointAttributeDefinition, PrimitiveType},
    math::MinMax,
};

/// Returns the minimum and maximum value of the given point `attribute` within `buffer`. Returns `None` if `buffer` contains no points. For
/// vector `PrimitiveType`s such as `Vector3<f64>`, the component-wise minimum and maximum is applied.
///
/// # Panics
///
/// If `attribute` is not part of the point layout of `buffer`, or the attribute within `buffer` is not of type `T`
pub fn minmax_attribute<'a, T: PrimitiveType + MinMax + Copy, B: BorrowedBuffer<'a>>(
    buffer: &'a B,
    attribute: &PointAttributeDefinition,
) -> Option<(T, T)> {
    if !buffer
        .point_layout()
        .has_attribute_with_name(attribute.name())
    {
        panic!(
            "Attribute {} not contained in PointLayout buffer ({})",
            attribute,
            buffer.point_layout()
        );
    }

    let mut minmax = None;

    if T::data_type() == attribute.datatype() {
        for val in buffer.view_attribute::<T>(attribute) {
            match minmax {
                None => minmax = Some((val, val)),
                Some((old_min, old_max)) => {
                    minmax = Some((val.infimum(&old_min), val.supremum(&old_max)));
                }
            }
        }
    } else {
        for val in buffer.view_attribute_with_conversion::<T>(attribute).ok()? {
            match minmax {
                None => minmax = Some((val, val)),
                Some((old_min, old_max)) => {
                    minmax = Some((val.infimum(&old_min), val.supremum(&old_max)));
                }
            }
        }
    }

    minmax
}
