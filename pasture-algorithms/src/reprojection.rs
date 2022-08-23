use std::ffi::CString;

use anyhow::Result;
use pasture_core::containers::{PointBufferWriteable, PointBufferWriteableExt};
use pasture_core::math::AABB;
use pasture_core::nalgebra::{Point3, Vector3};

use pasture_core::containers::{PointBuffer, PointBufferExt};
use pasture_core::layout::attributes::POSITION_3D;
/// Wrapper around the proj types from the proj_sys crate. Supports transformations (the Rust proj bindings don't support this)
pub struct Projection {
    proj_context: *mut proj_sys::pj_ctx,
    projection: *mut proj_sys::PJconsts,
}

impl Projection {
    pub fn new(source_crs: &str, target_crs: &str) -> Result<Self> {
        let src_cstr = CString::new(source_crs)?;
        let target_cstr = CString::new(target_crs)?;

        unsafe {
            let proj_context = proj_sys::proj_context_create();

            let projection = proj_sys::proj_create_crs_to_crs(
                proj_context,
                src_cstr.as_ptr(),
                target_cstr.as_ptr(),
                std::ptr::null_mut(),
            );

            Ok(Self {
                proj_context,
                projection,
            })
        }
    }

    /// Performs a transformation of the given position
    pub fn transform(&self, position: Vector3<f64>) -> Vector3<f64> {
        unsafe {
            let coord = proj_sys::proj_coord(position.x, position.y, position.z, 0.0);
            let target_coords =
                proj_sys::proj_trans(self.projection, proj_sys::PJ_DIRECTION_PJ_FWD, coord);
            Vector3::new(target_coords.v[0], target_coords.v[1], target_coords.v[2])
        }
    }

    /// Transforms the given bounding box using the associated `Projection`. This keeps the bounding box axis-aligned,
    /// the size might change though
    pub fn transform_bounds(&self, bounds: &AABB<f64>) -> AABB<f64> {
        let transformed_min =
            self.transform(Vector3::new(bounds.min().x, bounds.min().y, bounds.min().z));
        let transformed_max =
            self.transform(Vector3::new(bounds.max().x, bounds.max().y, bounds.max().z));

        let bounds: AABB<f64> = AABB::from_min_max_unchecked(
            Point3::from(transformed_min),
            Point3::from(transformed_min),
        );
        AABB::extend_with_point(&bounds, &Point3::from(transformed_max))
    }
}

impl Drop for Projection {
    fn drop(&mut self) {
        unsafe {
            proj_sys::proj_destroy(self.projection);
            proj_sys::proj_context_destroy(self.proj_context);
        }
    }
}

/// Reprojection Algorithm
/// Rewrites the 3D coordinates from the given point cloud to the given target coordinate reference system.
/// It iterates over all points in the given point cloud.
/// Make sure that source_crs and target_crs are valid coordinate reference systems.
///
/// # Panics
///
/// Panics if the PointLayout of this buffer does not contain the given attribute.
///
/// # Examples
///
/// ```
/// # use pasture_algorithms::reprojection::reproject_point_cloud_within;
/// # use pasture_core::containers::*;
/// # use pasture_core::layout::PointType;
/// # use pasture_core::nalgebra::Vector3;
/// # use pasture_derive::PointType;

/// #[repr(C)]
/// #[derive(PointType, Debug, Clone, Copy)]
/// struct SimplePoint {
///     #[pasture(BUILTIN_POSITION_3D)]
///     pub position: Vector3<f64>,
///     #[pasture(BUILTIN_INTENSITY)]
///     pub intensity: u16,
/// }

/// fn main() {
///     let points = vec![
///         SimplePoint {
///             position: Vector3::new(1.0, 22.0, 0.0),
///             intensity: 42,
///         },
///         SimplePoint {
///             position: Vector3::new(12.0, 23.0, 0.0),
///             intensity: 84,
///         },
///         SimplePoint {
///             position: Vector3::new(10.0, 8.0, 2.0),
///             intensity: 84,
///         },
///         SimplePoint {
///             position: Vector3::new(10.0, 0.0, 1.0),
///             intensity: 84,
///         },
///     ];

///     let mut interleaved = InterleavedVecPointStorage::new(SimplePoint::layout());

///     interleaved.push_points(points.as_slice());
///     let points = vec![
///         SimplePoint {
///             position: Vector3::new(1.0, 22.0, 0.0),
///             intensity: 42,
///         },
///         SimplePoint {
///             position: Vector3::new(12.0, 23.0, 0.0),
///             intensity: 84,
///         },
///         SimplePoint {
///             position: Vector3::new(10.0, 8.0, 2.0),
///             intensity: 84,
///         },
///         SimplePoint {
///             position: Vector3::new(10.0, 0.0, 1.0),
///             intensity: 84,
///         },
///     ];

///     let mut interleaved = InterleavedVecPointStorage::new(SimplePoint::layout());

///     interleaved.push_points(points.as_slice());

///     reproject_point_cloud_within::<InterleavedVecPointStorage>(
///         &mut interleaved,
///         "EPSG:4326",
///         "EPSG:3309",
///     );

///     for point in interleaved.iter_point::<SimplePoint>() {
///         println!("{:?}", point);
///     }
/// }
/// ```
pub fn reproject_point_cloud_within<T: PointBuffer + PointBufferWriteable>(
    point_cloud: &mut T,
    source_crs: &str,
    target_crs: &str,
) {
    let proj = Projection::new(source_crs, target_crs).unwrap();

    for index in 0..point_cloud.len() {
        let point = point_cloud.get_attribute(&POSITION_3D, index);
        let reproj = proj.transform(point);
        point_cloud.set_attribute(&POSITION_3D, index, reproj);
    }
}

/// Reprojection Algorithm
/// Rewrites the 3D coordinates from the given point cloud to the given target coordinate reference system.
/// It iterates over all points in the given point cloud.
/// Make sure that source_crs and target_crs are valid coordinate reference systems.
///
/// # Panics
///
/// Panics if the PointLayout of this buffer does not contain the given attribute.
///
/// # Examples
///
/// ```
/// # use pasture_algorithms::reprojection::reproject_point_cloud_between;
/// # use pasture_core::containers::*;
/// # use pasture_core::layout::PointType;
/// # use pasture_core::nalgebra::Vector3;
/// # use pasture_derive::PointType;

/// #[repr(C)]
/// #[derive(PointType, Debug, Clone, Copy)]
/// struct SimplePoint {
///     #[pasture(BUILTIN_POSITION_3D)]
///     pub position: Vector3<f64>,
///     #[pasture(BUILTIN_INTENSITY)]
///     pub intensity: u16,
/// }

/// fn main() {
///     let points = vec![
///         SimplePoint {
///             position: Vector3::new(1.0, 22.0, 0.0),
///             intensity: 42,
///         },
///         SimplePoint {
///             position: Vector3::new(12.0, 23.0, 0.0),
///             intensity: 84,
///         },
///         SimplePoint {
///             position: Vector3::new(10.0, 8.0, 2.0),
///             intensity: 84,
///         },
///         SimplePoint {
///             position: Vector3::new(10.0, 0.0, 1.0),
///             intensity: 84,
///         },
///     ];

///     let mut interleaved = InterleavedVecPointStorage::new(SimplePoint::layout());

///     interleaved.push_points(points.as_slice());
///     let points = vec![
///         SimplePoint {
///             position: Vector3::new(1.0, 22.0, 0.0),
///             intensity: 42,
///         },
///         SimplePoint {
///             position: Vector3::new(12.0, 23.0, 0.0),
///             intensity: 84,
///         },
///         SimplePoint {
///             position: Vector3::new(10.0, 8.0, 2.0),
///             intensity: 84,
///         },
///         SimplePoint {
///             position: Vector3::new(10.0, 0.0, 1.0),
///             intensity: 84,
///         },
///     ];

///     let mut interleaved = InterleavedVecPointStorage::new(SimplePoint::layout());

///     interleaved.push_points(points.as_slice());

///     let mut attribute = PerAttributeVecPointStorage::with_capacity(interleaved.len(), SimplePoint::layout());

///     attribute.resize(interleaved.len());

///     reproject_point_cloud_between(&mut interleaved, &mut attribute, "EPSG:4326", "EPSG:3309");

///     for point in attribute.iter_point::<SimplePoint>() {
///         println!("{:?}", point);
///     }
/// }
/// ```
pub fn reproject_point_cloud_between<
    T1: PointBuffer + PointBufferWriteable,
    T2: PointBuffer + PointBufferWriteable,
>(
    source_point_cloud: &mut T1,
    target_point_cloud: &mut T2,
    source_crs: &str,
    target_crs: &str,
) {
    if source_point_cloud.len() != target_point_cloud.len() {
        panic!("The point clouds don't have the same size!");
    }

    let proj = Projection::new(source_crs, target_crs).unwrap();

    for (index, point) in source_point_cloud
        .iter_attribute::<Vector3<f64>>(&POSITION_3D)
        .enumerate()
    {
        let reproj = proj.transform(point);
        target_point_cloud.set_attribute(&POSITION_3D, index, reproj);
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use pasture_core::{
        containers::{InterleavedVecPointStorage, PerAttributeVecPointStorage, OwningPointBuffer},
        layout::PointType,
        nalgebra::Vector3,
    };
    use pasture_derive::PointType;

    use super::*;

    #[repr(C)]
    #[derive(PointType, Debug, Clone, Copy)]
    pub struct SimplePoint {
        #[pasture(BUILTIN_POSITION_3D)]
        pub position: Vector3<f64>,
        #[pasture(BUILTIN_INTENSITY)]
        pub intensity: u16,
    }

    #[test]
    fn reproject_epsg4326_epsg3309_within() {
        let points = vec![
            SimplePoint {
                position: Vector3::new(1.0, 22.0, 0.0),
                intensity: 42,
            },
            SimplePoint {
                position: Vector3::new(12.0, 23.0, 0.0),
                intensity: 84,
            },
            SimplePoint {
                position: Vector3::new(10.0, 8.0, 2.0),
                intensity: 84,
            },
            SimplePoint {
                position: Vector3::new(10.0, 0.0, 1.0),
                intensity: 84,
            },
        ];

        let mut interleaved = InterleavedVecPointStorage::new(SimplePoint::layout());

        interleaved.push_points(points.as_slice());

        reproject_point_cloud_within(&mut interleaved, "EPSG:4326", "EPSG:3309");

        let results = vec![
            Vector3::new(12185139.590523569, 7420953.944297638, 0.0),
            Vector3::new(11104667.534080556, 7617693.973680517, 0.0),
            Vector3::new(11055663.927418157, 5832081.512011217, 2.0),
            Vector3::new(10807262.110686881, 4909128.916889962, 1.0),
        ];

        for (index, coord) in interleaved
            .iter_attribute::<Vector3<f64>>(&POSITION_3D)
            .enumerate()
        {
            assert_approx_eq!(coord[0], results[index][0], 0.0001);
            assert_approx_eq!(coord[1], results[index][1], 0.0001);
            assert_approx_eq!(coord[2], results[index][2], 0.0001);
        }
    }
    #[test]
    fn reproject_epsg4326_epsg3309_between() {
        let points = vec![
            SimplePoint {
                position: Vector3::new(1.0, 22.0, 0.0),
                intensity: 42,
            },
            SimplePoint {
                position: Vector3::new(12.0, 23.0, 0.0),
                intensity: 84,
            },
            SimplePoint {
                position: Vector3::new(10.0, 8.0, 2.0),
                intensity: 84,
            },
            SimplePoint {
                position: Vector3::new(10.0, 0.0, 1.0),
                intensity: 84,
            },
        ];

        let mut interleaved = InterleavedVecPointStorage::new(SimplePoint::layout());

        interleaved.push_points(points.as_slice());

        let mut attribute =
            PerAttributeVecPointStorage::with_capacity(interleaved.len(), SimplePoint::layout());

        attribute.resize(interleaved.len());

        reproject_point_cloud_between(&mut interleaved, &mut attribute, "EPSG:4326", "EPSG:3309");

        let results = vec![
            Vector3::new(12185139.590523569, 7420953.944297638, 0.0),
            Vector3::new(11104667.534080556, 7617693.973680517, 0.0),
            Vector3::new(11055663.927418157, 5832081.512011217, 2.0),
            Vector3::new(10807262.110686881, 4909128.916889962, 1.0),
        ];

        for (index, coord) in attribute
            .iter_attribute::<Vector3<f64>>(&POSITION_3D)
            .enumerate()
        {
            assert_approx_eq!(coord[0], results[index][0], 0.0001);
            assert_approx_eq!(coord[1], results[index][1], 0.0001);
            assert_approx_eq!(coord[2], results[index][2], 0.0001);
        }
    }
    #[test]
    #[should_panic(expected = "The point clouds don't have the same size!")]
    fn reproject_epsg4326_epsg3309_between_error() {
        let points = vec![
            SimplePoint {
                position: Vector3::new(1.0, 22.0, 0.0),
                intensity: 42,
            },
            SimplePoint {
                position: Vector3::new(12.0, 23.0, 0.0),
                intensity: 84,
            },
            SimplePoint {
                position: Vector3::new(10.0, 8.0, 2.0),
                intensity: 84,
            },
            SimplePoint {
                position: Vector3::new(10.0, 0.0, 1.0),
                intensity: 84,
            },
        ];

        let mut interleaved = InterleavedVecPointStorage::new(SimplePoint::layout());

        interleaved.push_points(points.as_slice());

        let mut attribute = PerAttributeVecPointStorage::with_capacity(2, SimplePoint::layout());

        attribute.resize(2);

        reproject_point_cloud_between(&mut interleaved, &mut attribute, "EPSG:4326", "EPSG:3309");
    }
}
