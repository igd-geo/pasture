use std::cmp;

use nalgebra::{Scalar, Vector3};

/// Helper trait for computing minimum and maximum values for types. This is used in conjunction
/// with `PrimitiveType` to enable min/max computations even for vector types
pub trait MinMax {
    /// Computes the infimum of this value and `other`. For scalar types, the infimum is simply the
    /// minimum of the two types (as defined by `PartialOrd`), for vector types, this is the component-wise
    /// minimum
    ///
    /// # Example
    /// ```
    /// use pasture_core::math::MinMax;
    /// # use pasture_core::nalgebra::Vector3;
    ///
    /// assert_eq!(5i32.infimum(3i32), 3i32);
    /// assert_eq!(Vector3::new(1.0, 2.0, 3.0).infimum(Vector3::new(2.0, 1.0, 0.0)), Vector3::new(1.0, 1.0, 0.0));
    /// ```
    fn infimum(&self, other: &Self) -> Self;
    /// Computes the supremum of this value and `other`. For scalar types, the infimum is simply the
    /// maximum of the two types (as defined by `PartialOrd`), for vector types, this is the component-wise
    /// maximum
    ///
    /// # Example
    /// ```
    /// use pasture_core::math::MinMax;
    /// # use pasture_core::nalgebra::Vector3;
    ///
    /// assert_eq!(5i32.supremum(3i32), 5i32);
    /// assert_eq!(Vector3::new(1.0, 2.0, 3.0).supremum(Vector3::new(2.0, 1.0, 4.0)), Vector3::new(2.0, 2.0, 4.0));
    /// ```
    fn supremum(&self, other: &Self) -> Self;
}

macro_rules! impl_minmax_for_primitive_type {
    ($type:tt) => {
        impl MinMax for $type {
            fn infimum(&self, other: &Self) -> Self {
                cmp::min(*self, *other)
            }

            fn supremum(&self, other: &Self) -> Self {
                cmp::max(*self, *other)
            }
        }
    };
}

impl_minmax_for_primitive_type! {u8}
impl_minmax_for_primitive_type! {u16}
impl_minmax_for_primitive_type! {u32}
impl_minmax_for_primitive_type! {u64}
impl_minmax_for_primitive_type! {i8}
impl_minmax_for_primitive_type! {i16}
impl_minmax_for_primitive_type! {i32}
impl_minmax_for_primitive_type! {i64}
impl_minmax_for_primitive_type! {bool}

impl MinMax for f32 {
    fn infimum(&self, other: &Self) -> Self {
        if *self < *other {
            *self
        } else {
            *other
        }
    }

    fn supremum(&self, other: &Self) -> Self {
        if *self > *other {
            *self
        } else {
            *other
        }
    }
}

impl MinMax for f64 {
    fn infimum(&self, other: &Self) -> Self {
        if *self < *other {
            *self
        } else {
            *other
        }
    }

    fn supremum(&self, other: &Self) -> Self {
        if *self > *other {
            *self
        } else {
            *other
        }
    }
}

impl<T: MinMax + Scalar> MinMax for Vector3<T> {
    fn infimum(&self, other: &Self) -> Self {
        Vector3::new(
            self.x.infimum(&other.x),
            self.y.infimum(&other.y),
            self.z.infimum(&other.z),
        )
    }

    fn supremum(&self, other: &Self) -> Self {
        Vector3::new(
            self.x.supremum(&other.x),
            self.y.supremum(&other.y),
            self.z.supremum(&other.z),
        )
    }
}
