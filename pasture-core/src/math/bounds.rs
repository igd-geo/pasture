use std::iter::FromIterator;

use float_ord::FloatOrd;
use nalgebra::{ClosedSub, Point3, Scalar, Vector3};

/// 3D axis-aligned bounding box
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AABB<T: Scalar + PartialOrd> {
    min: Point3<T>,
    max: Point3<T>,
}

impl<T: Scalar + ClosedSub + PartialOrd + Copy> AABB<T> {
    /// Creates a new AABB from the given minimum and maximum coordinates. Panics if the minimum position is
    /// not less than or equal to the maximum position
    /// ```
    /// # use pasture_core::math::AABB;
    /// let bounds = AABB::from_min_max(nalgebra::Point3::new(0.0, 0.0, 0.0), nalgebra::Point3::new(1.0, 1.0, 1.0));
    /// ```
    pub fn from_min_max(min: Point3<T>, max: Point3<T>) -> Self {
        if min.x > max.x || min.y > max.y || min.z > max.z {
            panic!("AABB::from_min_max: Minimum position must be <= maximum position!");
        }
        Self { min, max }
    }

    /// Creates a new AABB from the given minimum and maximum coordinates. Similar to [from_min_max](AABB::from_min_max)
    /// but performs no checks that min <= max. If you know that min <= max, prefer this function over [from_min_max](AABB::from_min_max)
    /// ```
    /// # use pasture_core::math::AABB;
    /// let bounds = AABB::from_min_max_unchecked(nalgebra::Point3::new(0.0, 0.0, 0.0), nalgebra::Point3::new(1.0, 1.0, 1.0));
    /// ```
    pub fn from_min_max_unchecked(min: Point3<T>, max: Point3<T>) -> Self {
        Self { min, max }
    }

    /// Returns the minimum point of this AABB
    /// ```
    /// # use pasture_core::math::AABB;
    /// let bounds = AABB::from_min_max_unchecked(nalgebra::Point3::new(-1.0, -1.0, -1.0), nalgebra::Point3::new(1.0, 1.0, 1.0));
    /// assert_eq!(*bounds.min(), nalgebra::Point3::new(-1.0, -1.0, -1.0));
    /// ```
    pub fn min(&self) -> &Point3<T> {
        &self.min
    }

    /// Returns the maximum point of this AABB
    /// ```
    /// # use pasture_core::math::AABB;
    /// let bounds = AABB::from_min_max_unchecked(nalgebra::Point3::new(-1.0, -1.0, -1.0), nalgebra::Point3::new(1.0, 1.0, 1.0));
    /// assert_eq!(*bounds.max(), nalgebra::Point3::new(1.0, 1.0, 1.0));
    /// ```
    pub fn max(&self) -> &Point3<T> {
        &self.max
    }

    /// Returns the extent of this AABB. The extent is the size between the minimum and maximum position of this AABB
    /// ```
    /// # use pasture_core::math::AABB;
    /// let bounds = AABB::from_min_max_unchecked(nalgebra::Point3::new(0.0, 0.0, 0.0), nalgebra::Point3::new(1.0, 1.0, 1.0));
    /// assert_eq!(bounds.extent(), nalgebra::Vector3::new(1.0, 1.0, 1.0));
    /// ```
    pub fn extent(&self) -> Vector3<T> {
        self.max - self.min
    }

    /// Performs an intersection test between this AABB and the given AABB. Returns true if the two
    /// bounding boxes intersect. If one of the boxes is fully contained within the other, this also
    /// counts as an intersection
    /// ```
    /// # use pasture_core::math::AABB;
    /// let bounds_a = AABB::from_min_max_unchecked(nalgebra::Point3::new(0.0, 0.0, 0.0), nalgebra::Point3::new(1.0, 1.0, 1.0));
    /// let bounds_b = AABB::from_min_max_unchecked(nalgebra::Point3::new(0.5, 0.5, 0.5), nalgebra::Point3::new(1.5, 1.5, 1.5));
    /// assert!(bounds_a.intersects(&bounds_b));
    /// ```
    pub fn intersects(&self, other: &AABB<T>) -> bool {
        (self.min.x <= other.max.x && self.max.x >= other.min.x)
            && (self.min.y <= other.max.y && self.max.y >= other.min.y)
            && (self.min.z <= other.max.z && self.max.z >= other.min.z)
    }

    /// Returns true if the given point is contained within this AABB. Points right on the boundary
    /// of this AABB (e.g. point.x == self.max.x or self.min.x) will return true as well.
    /// ```
    /// # use pasture_core::math::AABB;
    /// let bounds = AABB::from_min_max_unchecked(nalgebra::Point3::new(0.0, 0.0, 0.0), nalgebra::Point3::new(1.0, 1.0, 1.0));
    /// assert!(bounds.contains(&nalgebra::Point3::new(0.5, 0.5, 0.5)));
    /// ```
    pub fn contains(&self, point: &Point3<T>) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// Computes the union of the given bounding boxes. The union of two bounding boxes a and b is defined as the
    /// smallest AABB that fully contains both a and b.
    /// ```
    /// # use pasture_core::math::AABB;
    /// let bounds_a = AABB::from_min_max_unchecked(nalgebra::Point3::new(0.0, 0.0, 0.0), nalgebra::Point3::new(1.0, 1.0, 1.0));
    /// let bounds_b = AABB::from_min_max_unchecked(nalgebra::Point3::new(2.0, 2.0, 2.0), nalgebra::Point3::new(3.0, 3.0, 3.0));
    /// let merged_bounds = AABB::union(&bounds_a, &bounds_b);
    /// assert_eq!(*merged_bounds.min(), nalgebra::Point3::new(0.0, 0.0, 0.0));
    /// assert_eq!(*merged_bounds.max(), nalgebra::Point3::new(3.0, 3.0, 3.0));
    /// ```
    pub fn union(a: &AABB<T>, b: &AABB<T>) -> Self {
        let min_x = if a.min.x < b.min.x { a.min.x } else { b.min.x };
        let min_y = if a.min.y < b.min.y { a.min.y } else { b.min.y };
        let min_z = if a.min.z < b.min.z { a.min.z } else { b.min.z };

        let max_x = if a.max.x > b.max.x { a.max.x } else { b.max.x };
        let max_y = if a.max.y > b.max.y { a.max.y } else { b.max.y };
        let max_z = if a.max.z > b.max.z { a.max.z } else { b.max.z };

        Self {
            min: Point3::new(min_x, min_y, min_z),
            max: Point3::new(max_x, max_y, max_z),
        }
    }

    /// Extends the given AABB so that it contains the given point.
    /// ```
    /// # use pasture_core::math::AABB;
    /// let bounds = AABB::from_min_max_unchecked(nalgebra::Point3::new(0.0, 0.0, 0.0), nalgebra::Point3::new(1.0, 1.0, 1.0));
    /// let extended_bounds = AABB::extend_with_point(&bounds, &nalgebra::Point3::new(2.0, 2.0, 2.0));
    /// assert_eq!(*extended_bounds.min(), nalgebra::Point3::new(0.0, 0.0, 0.0));
    /// assert_eq!(*extended_bounds.max(), nalgebra::Point3::new(2.0, 2.0, 2.0));
    /// ```
    pub fn extend_with_point(bounds: &AABB<T>, point: &Point3<T>) -> AABB<T> {
        let min_x = if bounds.min.x < point.x {
            bounds.min.x
        } else {
            point.x
        };
        let min_y = if bounds.min.y < point.y {
            bounds.min.y
        } else {
            point.y
        };
        let min_z = if bounds.min.z < point.z {
            bounds.min.z
        } else {
            point.z
        };

        let max_x = if bounds.max.x > point.x {
            bounds.max.x
        } else {
            point.x
        };
        let max_y = if bounds.max.y > point.y {
            bounds.max.y
        } else {
            point.y
        };
        let max_z = if bounds.max.z > point.z {
            bounds.max.z
        } else {
            point.z
        };

        Self {
            min: Point3::new(min_x, min_y, min_z),
            max: Point3::new(max_x, max_y, max_z),
        }
    }
}

impl AABB<f32> {
    /// Returns the center point of this AABB.
    /// ```
    /// # use pasture_core::math::AABB;
    /// let bounds = AABB::<f32>::from_min_max_unchecked(nalgebra::Point3::new(0.0, 0.0, 0.0), nalgebra::Point3::new(1.0, 2.0, 3.0));
    /// assert_eq!(bounds.center(), nalgebra::Point3::new(0.5, 1.0, 1.5));
    /// ```
    pub fn center(&self) -> Point3<f32> {
        Point3::new(
            (self.min.x + self.max.x) / 2.0,
            (self.min.y + self.max.y) / 2.0,
            (self.min.z + self.max.z) / 2.0,
        )
    }

    /// Returns a cubic version of the associated `AABB`. For this, the shortest two axes of the bounds
    /// are elongated symmetrically from the center of the bounds so that all axis are of equal length
    /// ```
    /// # use pasture_core::math::AABB;
    /// let bounds = AABB::<f32>::from_min_max_unchecked(nalgebra::Point3::new(0.0, 0.0, 0.0), nalgebra::Point3::new(1.0, 2.0, 4.0));
    /// let cubic_bounds = AABB::<f32>::from_min_max_unchecked(nalgebra::Point3::new(-1.5, -1.0, 0.0), nalgebra::Point3::new(2.5, 3.0, 4.0));
    /// assert_eq!(bounds.as_cubic().min(), cubic_bounds.min());
    /// assert_eq!(bounds.as_cubic().max(), cubic_bounds.max());
    /// ```
    pub fn as_cubic(&self) -> AABB<f32> {
        let extent = self.extent();
        let max_axis = std::cmp::max(
            FloatOrd(extent.x),
            std::cmp::max(FloatOrd(extent.y), FloatOrd(extent.z)),
        )
        .0;
        let max_axis_half = max_axis / 2.0;
        let center = self.center();
        Self {
            min: center - Vector3::new(max_axis_half, max_axis_half, max_axis_half),
            max: center + Vector3::new(max_axis_half, max_axis_half, max_axis_half),
        }
    }
}

impl AABB<f64> {
    /// Returns the center point of this AABB.
    /// ```
    /// # use pasture_core::math::AABB;
    /// let bounds = AABB::<f64>::from_min_max_unchecked(nalgebra::Point3::new(0.0, 0.0, 0.0), nalgebra::Point3::new(1.0, 2.0, 3.0));
    /// assert_eq!(bounds.center(), nalgebra::Point3::new(0.5, 1.0, 1.5));
    /// ```
    pub fn center(&self) -> Point3<f64> {
        Point3::new(
            (self.min.x + self.max.x) / 2.0,
            (self.min.y + self.max.y) / 2.0,
            (self.min.z + self.max.z) / 2.0,
        )
    }

    /// Like `contains`, but performs epsilon comparison on floating point values using the given `epsilon` value
    pub fn contains_approx(&self, point: &Point3<f64>, epsilon: f64) -> bool {
        let dx_min = point.x - self.min.x;
        let dx_max = point.x - self.max.x;
        let dy_min = point.y - self.min.y;
        let dy_max = point.y - self.max.y;
        let dz_min = point.z - self.min.z;
        let dz_max = point.z - self.max.z;

        !(dx_min < -epsilon
            || dx_max > epsilon
            || dy_min < -epsilon
            || dy_max > epsilon
            || dz_min < -epsilon
            || dz_max > epsilon)
    }

    /// Returns a cubic version of the associated `AABB`. For this, the shortest two axes of the bounds
    /// are elongated symmetrically from the center of the bounds so that all axis are of equal length
    /// ```
    /// # use pasture_core::math::AABB;
    /// let bounds = AABB::<f64>::from_min_max_unchecked(nalgebra::Point3::new(0.0, 0.0, 0.0), nalgebra::Point3::new(1.0, 2.0, 4.0));
    /// let cubic_bounds = AABB::<f64>::from_min_max_unchecked(nalgebra::Point3::new(-1.5, -1.0, 0.0), nalgebra::Point3::new(2.5, 3.0, 4.0));
    /// assert_eq!(bounds.as_cubic().min(), cubic_bounds.min());
    /// assert_eq!(bounds.as_cubic().max(), cubic_bounds.max());
    /// ```
    pub fn as_cubic(&self) -> AABB<f64> {
        let extent = self.extent();
        let max_axis = std::cmp::max(
            FloatOrd(extent.x),
            std::cmp::max(FloatOrd(extent.y), FloatOrd(extent.z)),
        )
        .0;
        let max_axis_half = max_axis / 2.0;
        let center = self.center();
        Self {
            min: center - Vector3::new(max_axis_half, max_axis_half, max_axis_half),
            max: center + Vector3::new(max_axis_half, max_axis_half, max_axis_half),
        }
    }
}

impl<U: Into<Point3<f64>>> FromIterator<U> for AABB<f64> {
    fn from_iter<V: IntoIterator<Item = U>>(iter: V) -> Self {
        let mut min = Point3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut max = Point3::new(f64::MIN, f64::MIN, f64::MIN);
        for point in iter.into_iter() {
            let point: Point3<f64> = point.into();
            min.x = if min.x < point.x { min.x } else { point.x };
            min.y = if min.y < point.y { min.y } else { point.y };
            min.z = if min.z < point.z { min.z } else { point.z };

            max.x = if max.x > point.x { max.x } else { point.x };
            max.y = if max.y > point.y { max.y } else { point.y };
            max.z = if max.z > point.z { max.z } else { point.z };
        }
        Self::from_min_max(min, max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_contains_approx() {
        let bounds =
            AABB::from_min_max_unchecked(Point3::new(1.0, 1.0, 1.0), Point3::new(2.0, 2.0, 2.0));
        let p1 = Point3::new(0.99, 0.99, 0.99);
        let p2 = Point3::new(2.001, 1.999, 2.0);

        assert!(bounds.contains_approx(&p1, 0.015));
        assert!(!bounds.contains_approx(&p1, 0.001));
        assert!(bounds.contains_approx(&p2, 0.0015));
        assert!(!bounds.contains_approx(&p2, 0.0001));
    }

    #[test]
    fn aabb_from_iter() {
        let points = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(-1.0, -1.0, -1.0),
        ];
        let bounds: AABB<f64> = points.into_iter().collect();
        let expected_bounds =
            AABB::from_min_max(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        assert_eq!(expected_bounds, bounds);
    }
}
