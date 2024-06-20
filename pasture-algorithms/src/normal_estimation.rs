// The normal estimation algorithm is inspired by the PCL library (https://pointclouds.org/)
use core::panic;
use kd_tree::{self, KdPoint, KdTree};
use num_traits::{self};
use pasture_core::containers::{
    BorrowedBuffer, BorrowedBufferExt, BorrowedMutBufferExt, HashMapBuffer, OwningBuffer,
};
use pasture_core::layout::{attributes::POSITION_3D, PointType};
use pasture_core::nalgebra::{DMatrix, Vector3};
use std::result::Result;

/// Normal Estimation Algorithm
/// returns a vector of quintuplets where each quintuplet has the following values: (current point, normal_vector, curvature).
/// It iterates over all points in the buffer and constructs new point buffers of size k_nn.
/// Make sure that knn >= 3 holds as it is not possible to construct a plane over less than three point
///
/// # Panics
///
/// Panics if the number of k nearest neighbors is less than 3 or the point cloud has less than 3 elements.
///
/// # Examples
///
/// ```
/// # use kd_tree::{self, KdPoint};
/// # use pasture_core::nalgebra::Vector3;
/// # use pasture_core::{containers::*, layout::PointType};
/// # use pasture_algorithms::normal_estimation::compute_normals;
/// # use pasture_derive::PointType;
/// # use typenum;
///
/// #[repr(C, packed)]
/// #[derive(PointType, Debug, Clone, Copy, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
/// struct SimplePoint {
///     #[pasture(BUILTIN_POSITION_3D)]
///     pub position: Vector3<f64>,
///     #[pasture(BUILTIN_INTENSITY)]
///     pub intensity: u16,
/// }
/// impl KdPoint for SimplePoint {
///     type Scalar = f64;
///     type Dim = typenum::U3;
///     fn at(&self, k: usize) -> f64 {
///         let position = self.position;
///         position[k]
///     }
/// }
/// fn main() {
///     let points = vec![
///         SimplePoint {
///             position: Vector3::new(11.0, 22.0, 154.0),
///             intensity: 42,
///         },
///         SimplePoint {
///             position: Vector3::new(12.0, 23.0, 0.0),
///             intensity: 84,
///         },
///         SimplePoint {
///             position: Vector3::new(103.0, 84.0, 2.0),
///             intensity: 84,
///         },
///         SimplePoint {
///             position: Vector3::new(101.0, 0.0, 1.0),
///             intensity: 84,
///         },
///     ];
///
///     let interleaved = points.into_iter().collect::<VectorBuffer>();
///
///     let solution_vec = compute_normals::<VectorBuffer, SimplePoint>(&interleaved, 4);
///     for solution in solution_vec {
///    println!(
///        "Point: {:?}, n_x: {}, n_y: {}, n_z: {}, curvature: {}",
///        solution.0, solution.0[0], solution.0[1], solution.0[2], solution.1
///    );
/// }
/// }
/// ```

pub fn compute_normals<'a, T: BorrowedBuffer<'a>, P: PointType + KdPoint + Copy>(
    point_cloud: &'a T,
    k_nn: usize,
) -> Vec<(Vector3<f64>, f64)>
where
    P::Scalar: num_traits::Float,
{
    if point_cloud.len() < 3 {
        panic!("The point cloud is too small. Please use a point cloud that has 3 or more points!");
    }
    if k_nn < 3 {
        panic!("The k nearest neigbors attribute is too small!");
    }

    // this is the solution that will be returned
    let mut points_with_normals_curvature = vec![];

    // transform point cloud in vector of points
    let mut points: Vec<[f64; 3]> = vec![];
    for point in point_cloud.view_attribute::<Vector3<f64>>(&POSITION_3D) {
        points.push(*point.as_ref());
    }

    // construct kd tree over the vector of points.
    let cloud_as_kd_tree = KdTree::build_by_ordered_float(points);

    // iterate over all points in the point cloud and and calculate the k nearest neighbors with the constructed kd tree
    for point in point_cloud.view_attribute::<Vector3<f64>>(&POSITION_3D) {
        let r: &[f64; 3] = point.as_ref();
        let nearest_points = cloud_as_kd_tree.nearests(r, k_nn);

        // stores the k nearest neighbors in a PointStorage
        let mut k_nn_buffer =
            HashMapBuffer::with_capacity(nearest_points.len(), point_cloud.point_layout().clone());
        k_nn_buffer.resize(nearest_points.len());

        for (index, point) in nearest_points.iter().enumerate() {
            k_nn_buffer.view_attribute_mut(&POSITION_3D).set_at(
                index,
                Vector3::new(point.item[0], point.item[1], point.item[2]),
            );
        }

        // coordinates of the surface normal and curvature
        let (surface_normal, curvature) = normal_estimation(&k_nn_buffer);

        // solution vector
        points_with_normals_curvature.push((surface_normal, curvature));
    }

    points_with_normals_curvature
}

/// checks whether a given point cloud has points with coordinates that are Not a Number
fn is_dense<'a, T: BorrowedBuffer<'a>>(point_cloud: &'a T) -> bool {
    for point in point_cloud.view_attribute::<Vector3<f64>>(&POSITION_3D) {
        if point.x.is_nan() || point.y.is_nan() || point.z.is_nan() {
            return false;
        }
    }
    true
}

/// checks whether a given point has finite coordinates
fn is_finite(point: &Vector3<f64>) -> bool {
    if point.x.is_finite() && point.y.is_finite() && point.z.is_finite() {
        return true;
    }
    false
}

/// Computes the centroid for a given point cloud.
/// The centroid is the point that has the same distance to all other points in the point cloud.
/// Iterates over all points in the 'point_cloud'.
///
/// # Panics
///
/// If the point_cloud is empty
///
/// # Examples
///
/// ```
/// # use pasture_core::{containers::*, layout::PointType};
/// # use pasture_core::nalgebra::Vector3;
/// # use pasture_derive::PointType;
/// # use pasture_algorithms::normal_estimation::compute_centroid;
///
/// #[repr(C, packed)]
/// #[derive(PointType, Debug, Clone, Copy, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
/// struct SimplePoint {
///     #[pasture(BUILTIN_POSITION_3D)]
///     pub position: Vector3<f64>,
///     #[pasture(BUILTIN_INTENSITY)]
///     pub intensity: u16,
/// }
/// let points: Vec<SimplePoint> = vec![
///     SimplePoint {
///         position: Vector3::new(1.0, 0.0, 0.0),
///         intensity: 42,
///     },
///     SimplePoint {
///         position: Vector3::new(0.0, 1.0, 0.0),
///         intensity: 84,
///     },
///     SimplePoint {
///         position: Vector3::new(1.0, 1.0, 0.0),
///         intensity: 84,
///     },
///     SimplePoint {
///         position: Vector3::new(-1.0, 0.0, 0.0),
///         intensity: 84,
///     },
/// ];

/// let interleaved = points.into_iter().collect::<VectorBuffer>();

/// let centroid = compute_centroid(&interleaved);
///
/// ```
pub fn compute_centroid<'a, T: BorrowedBuffer<'a>>(point_cloud: &'a T) -> Vector3<f64> {
    if point_cloud.is_empty() {
        panic!("The point cloud is empty!");
    }

    let mut centroid = Vector3::<f64>::zeros();
    let mut temp_centroid = Vector3::<f64>::zeros();

    if is_dense(point_cloud) {
        // add all points up
        for point in point_cloud.view_attribute::<Vector3<f64>>(&POSITION_3D) {
            temp_centroid[0] += point.x;
            temp_centroid[1] += point.y;
            temp_centroid[2] += point.z;
        }

        //normalize over all points
        centroid[0] = temp_centroid[0] / point_cloud.len() as f64;
        centroid[1] = temp_centroid[1] / point_cloud.len() as f64;
        centroid[2] = temp_centroid[2] / point_cloud.len() as f64;
    } else {
        let mut points_in_cloud = 0;
        for point in point_cloud.view_attribute::<Vector3<f64>>(&POSITION_3D) {
            if is_finite(&point) {
                // add all points up
                temp_centroid[0] += point.x;
                temp_centroid[1] += point.y;
                temp_centroid[2] += point.z;
                points_in_cloud += 1;
            }
        }

        // normalize over all points
        centroid[0] = temp_centroid[0] / points_in_cloud as f64;
        centroid[1] = temp_centroid[1] / points_in_cloud as f64;
        centroid[2] = temp_centroid[2] / points_in_cloud as f64;
    }

    centroid
}

/// compute the covariance matrix for a given point cloud which is a measure of spread out the points are
fn compute_covariance_matrix<'a, T: BorrowedBuffer<'a>>(
    point_cloud: &'a T,
) -> Result<DMatrix<f64>, &'static str> {
    let mut covariance_matrix = DMatrix::<f64>::zeros(3, 3);
    let mut point_count = 0;

    // compute the centroid of the point cloud
    let centroid = compute_centroid(point_cloud);

    if is_dense(point_cloud) {
        point_count = point_cloud.len();
        let mut diff_mean = Vector3::<f64>::zeros();
        for point in point_cloud.view_attribute::<Vector3<f64>>(&POSITION_3D) {
            // calculate difference from the centroid for each point
            diff_mean[0] = point.x - centroid[0];
            diff_mean[1] = point.y - centroid[1];
            diff_mean[2] = point.z - centroid[2];

            covariance_matrix[(1, 1)] += diff_mean[1] * diff_mean[1];
            covariance_matrix[(1, 2)] += diff_mean[1] * diff_mean[2];
            covariance_matrix[(2, 2)] += diff_mean[2] * diff_mean[2];

            let diff_x = diff_mean[0];
            diff_mean.iter_mut().for_each(|x| *x *= diff_x);

            covariance_matrix[(0, 0)] += diff_mean[0];
            covariance_matrix[(0, 1)] += diff_mean[1];
            covariance_matrix[(0, 2)] += diff_mean[2];
        }
    } else {
        // in this case we dont know the number of points in the point cloud that are finite
        for point in point_cloud.view_attribute::<Vector3<f64>>(&POSITION_3D) {
            if !is_finite(&point) {
                continue;
            }
            // only compute the covariance matrix for finite points
            let mut diff_mean = Vector3::<f64>::zeros();
            // calculate difference from the centroid for each point
            diff_mean[0] = point.x - centroid[0];
            diff_mean[1] = point.y - centroid[1];
            diff_mean[2] = point.z - centroid[2];

            covariance_matrix[(1, 1)] += diff_mean[1] * diff_mean[1];
            covariance_matrix[(1, 2)] += diff_mean[1] * diff_mean[2];
            covariance_matrix[(2, 2)] += diff_mean[2] * diff_mean[2];

            let diff_x = diff_mean[0];
            diff_mean.iter_mut().for_each(|x| *x *= diff_x);

            covariance_matrix[(0, 0)] += diff_mean[0];
            covariance_matrix[(0, 1)] += diff_mean[1];
            covariance_matrix[(0, 2)] += diff_mean[2];
            point_count += 1;
        }
    }

    if point_count < 3 {
        return Err("The number of valid (finite and non-NaN values) points in a k nearest neighborhood is not enough to span a plane!");
    }

    covariance_matrix[(1, 0)] = covariance_matrix[(0, 1)];
    covariance_matrix[(2, 0)] = covariance_matrix[(0, 2)];
    covariance_matrix[(2, 1)] = covariance_matrix[(1, 2)];

    Ok(covariance_matrix)
}

/// find the eigen value solution if the highest degree of the polynomial is 2
fn solve_polynomial_quadratic(coefficient_2: f64, coefficient_1: f64) -> Vector3<f64> {
    let mut eigen_values = Vector3::<f64>::zeros();

    eigen_values[0] = 0.0;

    let mut delta = coefficient_2 * coefficient_2 - 4.0 * coefficient_1;

    if delta < 0.0 {
        delta = 0.0;
    }

    let sqrt_delta = f64::sqrt(delta);

    eigen_values[2] = 0.5 * (coefficient_2 + sqrt_delta);
    eigen_values[1] = 0.5 * (coefficient_2 - sqrt_delta);

    eigen_values
}

/// solve the polynomial to find the eigen values for a given covariance matrix
fn solve_polynomial(covariance_matrix: &DMatrix<f64>) -> Vector3<f64> {
    let coefficient_0 = covariance_matrix[(0, 0)]
        * covariance_matrix[(1, 1)]
        * covariance_matrix[(2, 2)]
        + 2.0 * covariance_matrix[(0, 1)] * covariance_matrix[(0, 2)] * covariance_matrix[(1, 2)]
        - covariance_matrix[(0, 0)] * covariance_matrix[(1, 2)] * covariance_matrix[(1, 2)]
        - covariance_matrix[(1, 1)] * covariance_matrix[(0, 2)] * covariance_matrix[(0, 2)]
        - covariance_matrix[(2, 2)] * covariance_matrix[(0, 1)] * covariance_matrix[(0, 1)];
    let coefficient_1 = covariance_matrix[(0, 0)] * covariance_matrix[(1, 1)]
        - covariance_matrix[(0, 1)] * covariance_matrix[(0, 1)]
        + covariance_matrix[(0, 0)] * covariance_matrix[(2, 2)]
        - covariance_matrix[(0, 2)] * covariance_matrix[(0, 2)]
        + covariance_matrix[(1, 1)] * covariance_matrix[(2, 2)]
        - covariance_matrix[(1, 2)] * covariance_matrix[(1, 2)];
    let coefficient_2 =
        covariance_matrix[(0, 0)] + covariance_matrix[(1, 1)] + covariance_matrix[(2, 2)];

    // check if one eigen value solution is zero
    if coefficient_0.abs() < std::f64::EPSILON {
        solve_polynomial_quadratic(coefficient_2, coefficient_1)
    } else {
        let mut eigen_values = Vector3::<f64>::zeros();

        let one_third = 1.0 / 3.0;
        let sqrt_3 = f64::sqrt(3.0);

        let coefficient_2_third = coefficient_2 * one_third;
        let mut alpha_third = (coefficient_1 - coefficient_2 * coefficient_2_third) * one_third;
        if alpha_third > 0.0 {
            alpha_third = 0.0;
        }

        let half_beta = 0.5
            * (coefficient_0
                + coefficient_2_third
                    * (2.0 * coefficient_2_third * coefficient_2_third - coefficient_1));

        let mut q = half_beta * half_beta + alpha_third * alpha_third * alpha_third;
        if q > 0.0 {
            q = 0.0;
        }

        // calculate eigen values
        let rho = f64::sqrt(-alpha_third);
        let theta = f64::atan2(f64::sqrt(-q), half_beta) * one_third;
        let cosine_of_theta = f64::cos(theta);
        let sine_of_theta = f64::sin(theta);

        eigen_values[0] = coefficient_2_third + 2.0 * rho * cosine_of_theta;
        eigen_values[1] = coefficient_2_third - rho * (cosine_of_theta + sqrt_3 * sine_of_theta);
        eigen_values[2] = coefficient_2_third - rho * (cosine_of_theta - sqrt_3 * sine_of_theta);

        // sort increasing so that eigen_values[0] is the smallest eigen value
        eigen_values
            .as_mut_slice()
            .sort_by(|a, b| a.partial_cmp(b).unwrap());

        // if the smallest eigen value is zero or less the solution is a quadratic
        if eigen_values[0] <= 0.0 {
            solve_polynomial_quadratic(coefficient_2, coefficient_1)
        } else {
            eigen_values
        }
    }
}

/// calculates the largest eigen vector for a given matrix
fn get_largest_eigen_vector(scaled_matrix: &DMatrix<f64>) -> Vector3<f64> {
    let rows = vec![
        scaled_matrix.row(0).cross(&scaled_matrix.row(1)),
        scaled_matrix.row(0).cross(&scaled_matrix.row(2)),
        scaled_matrix.row(1).cross(&scaled_matrix.row(2)),
    ];

    let mut cross_product = DMatrix::<f64>::zeros(3, 3);

    // write rows of cross product
    for it in cross_product.row_iter_mut().zip(rows) {
        let (mut cross_row, row) = it;
        // row from rows vector is written to the row of the cross product
        cross_row.copy_from(&row);
    }

    // find largest eigen vector
    let mut largest_eigen_vec = cross_product.row(0);
    for row in cross_product.row_iter() {
        if row.norm() > largest_eigen_vec.norm() {
            largest_eigen_vec = row;
        }
    }

    // set eigen vector to largest vector
    let mut eigen_vector = Vector3::<f64>::zeros();
    for i in 0..eigen_vector.len() {
        eigen_vector[i] = largest_eigen_vec[i];
    }

    eigen_vector
}

/// for a given 3x3 matrix the functions calculates the eigen vector of the smallest eigen value that can be found
fn eigen_3x3(covariance_matrix: &DMatrix<f64>) -> (f64, Vector3<f64>) {
    let scale = covariance_matrix.abs().max();
    let mut covariance_matrix_scaled = DMatrix::<f64>::zeros(3, 3);
    for i in 0..covariance_matrix.len() {
        covariance_matrix_scaled[i] = covariance_matrix[i] / scale;
    }

    // scale the matrix down
    for (index, value) in covariance_matrix.iter().enumerate() {
        covariance_matrix_scaled[index] = value / scale;
    }

    let eigen_values = solve_polynomial(covariance_matrix);
    // undo scale for smallest eigen vector
    let eigen_value = eigen_values[0] * scale;

    // subtract the smallest eigen value from the diagonal of the matrix
    covariance_matrix_scaled
        .diagonal()
        .iter_mut()
        .for_each(|x| *x -= eigen_values[0]);

    let eigen_vector = get_largest_eigen_vector(&covariance_matrix_scaled);
    (eigen_value, eigen_vector)
}

/// calculates the orientation of the surface as a normal vector with components n_x, n_y and n_z as well as the curvature of the surface for a given covariance matrix
fn solve_plane_parameter(covariance_matrix: &DMatrix<f64>) -> (Vector3<f64>, f64) {
    let (eigen_value, eigen_vector) = eigen_3x3(covariance_matrix);

    let eigen_sum = covariance_matrix[0] + covariance_matrix[4] + covariance_matrix[8];
    let curvature = if eigen_sum != 0.0 {
        (eigen_value / eigen_sum).abs()
    } else {
        0.0
    };

    (eigen_vector, curvature)
}

/// calculates the normal vectors and the curvature of the surface for the given point cloud
fn normal_estimation<'a, T: BorrowedBuffer<'a>>(point_cloud: &'a T) -> (Vector3<f64>, f64) {
    let covariance_matrix = compute_covariance_matrix(point_cloud).unwrap();

    let (eigen_vector, curvature) = solve_plane_parameter(&covariance_matrix);

    (eigen_vector, curvature)
}

#[cfg(test)]
mod tests {

    use pasture_core::{containers::VectorBuffer, nalgebra::Matrix3, nalgebra::Vector3};
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
    impl KdPoint for SimplePoint {
        type Scalar = f64;
        type Dim = typenum::U3;
        fn at(&self, k: usize) -> f64 {
            let position = self.position;
            position[k]
        }
    }

    #[test]
    fn test_compute_normal_sub() {
        let points: Vec<SimplePoint> = vec![
            SimplePoint {
                position: Vector3::new(1.0, 0.0, 0.0),
                intensity: 42,
            },
            SimplePoint {
                position: Vector3::new(0.0, 1.0, 0.0),
                intensity: 84,
            },
            SimplePoint {
                position: Vector3::new(1.0, 1.0, 0.0),
                intensity: 84,
            },
            SimplePoint {
                position: Vector3::new(-1.0, 0.0, 0.0),
                intensity: 84,
            },
        ];

        let interleaved = points.into_iter().collect::<VectorBuffer>();

        let centroid = compute_centroid(&interleaved);
        let result_centroid = Vector3::<f64>::new(0.25, 0.5, 0.0);
        assert_eq!(centroid, result_centroid);

        let covariance_matrix = compute_covariance_matrix(&interleaved).unwrap();
        let result = Matrix3::new(
            0.6875 * 4.0,
            0.125 * 4.0,
            0.0,
            0.125 * 4.0,
            0.25 * 4.0,
            0.0,
            0.0,
            0.0,
            0.0,
        );
        assert_eq!(covariance_matrix, result);

        let (normal_vector, curvature) = solve_plane_parameter(&covariance_matrix);

        assert_eq!(normal_vector[0], 0.0);
        assert_eq!(normal_vector[1], 0.0);
        assert_ne!(normal_vector[2], 0.0);
        assert_eq!(curvature, 0.0);
    }

    #[test]
    fn test_covariance_error() {
        let points: Vec<SimplePoint> = vec![
            SimplePoint {
                position: Vector3::new(f64::NAN, 0.0, 0.0),
                intensity: 42,
            },
            SimplePoint {
                position: Vector3::new(0.0, 1.0, f64::NAN),
                intensity: 84,
            },
            SimplePoint {
                position: Vector3::new(1.0, 1.0, f64::NAN),
                intensity: 84,
            },
            SimplePoint {
                position: Vector3::new(-1.0, f64::NAN, 0.0),
                intensity: 84,
            },
        ];

        let interleaved = points.into_iter().collect::<VectorBuffer>();

        let result = compute_covariance_matrix(&interleaved);
        let expected_result = Err("The number of valid (finite and non-NaN values) points in a k nearest neighborhood is not enough to span a plane!");
        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_compute_normal() {
        let points: Vec<SimplePoint> = vec![
            SimplePoint {
                position: Vector3::new(1.0, 0.0, 0.0),
                intensity: 42,
            },
            SimplePoint {
                position: Vector3::new(0.0, 1.0, 0.0),
                intensity: 84,
            },
            SimplePoint {
                position: Vector3::new(1.0, 1.0, 0.0),
                intensity: 84,
            },
            SimplePoint {
                position: Vector3::new(-1.0, 0.0, 0.0),
                intensity: 84,
            },
        ];

        let interleaved = points.into_iter().collect::<VectorBuffer>();

        let solution_vec = compute_normals::<VectorBuffer, SimplePoint>(&interleaved, 3);
        for solution in solution_vec {
            assert_eq!(solution.0[0], 0.0);
            assert_eq!(solution.0[1], 0.0);
            assert_ne!(solution.0[2], 0.0);
            assert_eq!(solution.1, 0.0);
        }
    }

    #[test]
    #[should_panic(
        expected = "The point cloud is too small. Please use a point cloud that has 3 or more points!"
    )]
    fn test_compute_normal_not_enough_points_1() {
        let points: Vec<SimplePoint> = vec![SimplePoint {
            position: Vector3::new(1.0, 0.0, 0.0),
            intensity: 42,
        }];

        let interleaved = points.into_iter().collect::<VectorBuffer>();

        let _solution_vec = compute_normals::<VectorBuffer, SimplePoint>(&interleaved, 3);
    }
    #[test]
    #[should_panic(
        expected = "The point cloud is too small. Please use a point cloud that has 3 or more points!"
    )]
    fn test_compute_normal_not_enough_points_2() {
        let points: Vec<SimplePoint> = vec![
            SimplePoint {
                position: Vector3::new(1.0, 0.0, 0.0),
                intensity: 42,
            },
            SimplePoint {
                position: Vector3::new(0.0, 1.0, 0.0),
                intensity: 84,
            },
        ];

        let interleaved = points.into_iter().collect::<VectorBuffer>();

        let _solution_vec = compute_normals::<VectorBuffer, SimplePoint>(&interleaved, 3);
    }

    #[test]
    #[should_panic(expected = "The k nearest neigbors attribute is too small!")]
    fn test_compute_normal_knn_too_small_1() {
        let points: Vec<SimplePoint> = vec![
            SimplePoint {
                position: Vector3::new(1.0, 0.0, 0.0),
                intensity: 42,
            },
            SimplePoint {
                position: Vector3::new(0.0, 1.0, 0.0),
                intensity: 84,
            },
            SimplePoint {
                position: Vector3::new(1.0, 1.0, 0.0),
                intensity: 84,
            },
            SimplePoint {
                position: Vector3::new(-1.0, 0.0, 0.0),
                intensity: 84,
            },
        ];

        let interleaved = points.into_iter().collect::<VectorBuffer>();

        let _solution_vec = compute_normals::<VectorBuffer, SimplePoint>(&interleaved, 1);
    }
    #[test]
    #[should_panic(expected = "The k nearest neigbors attribute is too small!")]
    fn test_compute_normal_knn_too_small_2() {
        let points: Vec<SimplePoint> = vec![
            SimplePoint {
                position: Vector3::new(1.0, 0.0, 0.0),
                intensity: 42,
            },
            SimplePoint {
                position: Vector3::new(0.0, 1.0, 0.0),
                intensity: 84,
            },
            SimplePoint {
                position: Vector3::new(1.0, 1.0, 0.0),
                intensity: 84,
            },
            SimplePoint {
                position: Vector3::new(-1.0, 0.0, 0.0),
                intensity: 84,
            },
        ];

        let interleaved = points.into_iter().collect::<VectorBuffer>();

        let _solution_vec = compute_normals::<VectorBuffer, SimplePoint>(&interleaved, 2);
    }
}
