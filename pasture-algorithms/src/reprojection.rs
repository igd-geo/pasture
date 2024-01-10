use std::ffi::CString;

use anyhow::Result;
use pasture_core::containers::{BorrowedBufferExt, BorrowedMutBuffer, BorrowedMutBufferExt};
use pasture_core::math::AABB;
use pasture_core::nalgebra::{Point3, Vector3};

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

/// #[repr(C, packed)]
/// #[derive(PointType, Debug, Clone, Copy, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
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

///     let mut interleaved = points.into_iter().collect::<VectorBuffer>();

///     reproject_point_cloud_within(
///         &mut interleaved,
///         "EPSG:4326",
///         "EPSG:3309",
///     );

///     for point in interleaved.view::<SimplePoint>() {
///         println!("{:?}", point);
///     }
/// }
/// ```
pub fn reproject_point_cloud_within<'a, T: BorrowedMutBuffer<'a>>(
    point_cloud: &'a mut T,
    source_crs: &str,
    target_crs: &str,
) {
    let proj = Projection::new(source_crs, target_crs).unwrap();

    let num_points = point_cloud.len();
    let mut positions_view = point_cloud.view_attribute_mut(&POSITION_3D);
    for index in 0..num_points {
        let point = positions_view.at(index);
        let reproj = proj.transform(point);
        positions_view.set_at(index, reproj);
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
/// #[repr(C, packed)]
/// #[derive(PointType, Debug, Clone, Copy, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
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
///     let mut interleaved = points.into_iter().collect::<VectorBuffer>();
///     let mut attribute = HashMapBuffer::with_capacity(interleaved.len(), SimplePoint::layout());
///     attribute.resize(interleaved.len());
///     reproject_point_cloud_between(&mut interleaved, &mut attribute, "EPSG:4326", "EPSG:3309");
///     for point in attribute.view::<SimplePoint>() {
///         println!("{:?}", point);
///     }
/// }
/// ```
pub fn reproject_point_cloud_between<
    'a,
    'b,
    T1: BorrowedMutBuffer<'a>,
    T2: BorrowedMutBuffer<'b>,
>(
    source_point_cloud: &'a mut T1,
    target_point_cloud: &'b mut T2,
    source_crs: &str,
    target_crs: &str,
) {
    if source_point_cloud.len() != target_point_cloud.len() {
        panic!("The point clouds don't have the same size!");
    }

    let proj = Projection::new(source_crs, target_crs).unwrap();

    let mut target_positions = target_point_cloud.view_attribute_mut(&POSITION_3D);
    for (index, point) in source_point_cloud
        .view_attribute::<Vector3<f64>>(&POSITION_3D)
        .into_iter()
        .enumerate()
    {
        let reproj = proj.transform(point);
        target_positions.set_at(index, reproj);
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use pasture_core::{
        containers::{BorrowedBuffer, HashMapBuffer, OwningBuffer, VectorBuffer},
        layout::PointType,
        nalgebra::Vector3,
    };
    use pasture_derive::PointType;

    use super::*;

    #[repr(C, packed)]
    #[derive(PointType, Debug, Clone, Copy, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
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

        let mut interleaved = points.into_iter().collect::<VectorBuffer>();

        reproject_point_cloud_within(&mut interleaved, "EPSG:4326", "EPSG:3309");

        let results = vec![
            Vector3::new(12185139.590523569, 7420953.944297638, 0.0),
            Vector3::new(11104667.534080556, 7617693.973680517, 0.0),
            Vector3::new(11055663.927418157, 5832081.512011217, 2.0),
            Vector3::new(10807262.110686881, 4909128.916889962, 1.0),
        ];

        for (index, coord) in interleaved
            .view_attribute::<Vector3<f64>>(&POSITION_3D)
            .into_iter()
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

        let mut interleaved = points.into_iter().collect::<VectorBuffer>();

        let mut attribute = HashMapBuffer::with_capacity(interleaved.len(), SimplePoint::layout());

        attribute.resize(interleaved.len());

        reproject_point_cloud_between(&mut interleaved, &mut attribute, "EPSG:4326", "EPSG:3309");

        let results = vec![
            Vector3::new(12185139.590523569, 7420953.944297638, 0.0),
            Vector3::new(11104667.534080556, 7617693.973680517, 0.0),
            Vector3::new(11055663.927418157, 5832081.512011217, 2.0),
            Vector3::new(10807262.110686881, 4909128.916889962, 1.0),
        ];

        for (index, coord) in attribute
            .view_attribute::<Vector3<f64>>(&POSITION_3D)
            .into_iter()
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

        let mut interleaved = points.into_iter().collect::<VectorBuffer>();

        let mut attribute = HashMapBuffer::with_capacity(2, SimplePoint::layout());

        attribute.resize(2);

        reproject_point_cloud_between(&mut interleaved, &mut attribute, "EPSG:4326", "EPSG:3309");
    }
}
