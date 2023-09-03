use std::vec;

use pasture_core::{
    layout::attributes::POSITION_3D,
    nalgebra::Vector3, containers::BorrowedBuffer,
};
use rand::Rng;
use rayon::prelude::*;

/// Represents a line between two points
/// the ranking shows how many points of the pointcloud are inliers for this specific line
#[derive(Debug)]
pub struct Line {
    first: Vector3<f64>,
    second: Vector3<f64>,
    ranking: usize,
}

/// Represents a plane in coordinate-form: ax + by + cz + d = 0
/// the ranking shows how many points of the pointcloud are inliers for this specific plane
#[derive(Debug)]
pub struct Plane {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    ranking: usize,
}

/// calculates the distance between a point and a plane
fn distance_point_plane(point: &Vector3<f64>, plane: &Plane) -> f64 {
    let d = (plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d).abs();
    let e = (plane.a * plane.a + plane.b * plane.b + plane.c * plane.c).sqrt();
    d / e
}

/// calculates the distance between a point and a line
/// careful: seems to be slow in debug, but is really fast in release
fn distance_point_line(point: &Vector3<f64>, line: &Line) -> f64 {
    (line.second - line.first)
        .cross(&(line.first - point))
        .norm()
        / (line.second - line.first).norm()
}

/// generates a random plane from three points of the buffer
fn generate_rng_plane<'a, T: BorrowedBuffer<'a>>(buffer: &'a T) -> Plane {
    // choose three random points from the pointcloud
    let mut rng = rand::thread_rng();
    let rand1 = rng.gen_range(0..buffer.len());
    let mut rand2 = rng.gen_range(0..buffer.len());
    while rand1 == rand2 {
        rand2 = rng.gen_range(0..buffer.len());
    }
    let mut rand3 = rng.gen_range(0..buffer.len());
    // make sure we have 3 unique random numbers to generate the plane model
    while rand2 == rand3 || rand1 == rand3 {
        rand3 = rng.gen_range(0..buffer.len());
    }
    let p_a: Vector3<f64> = buffer.view_attribute(&POSITION_3D).at(rand1);
    let p_b: Vector3<f64> = buffer.view_attribute(&POSITION_3D).at(rand2);
    let p_c: Vector3<f64> = buffer.view_attribute(&POSITION_3D).at(rand3);

    // compute plane from the three positions
    let vec1 = p_b - p_a;
    let vec2 = p_c - p_a;
    let normal = vec1.cross(&vec2);
    let d = -normal.dot(&p_a);
    Plane {
        a: normal.x,
        b: normal.y,
        c: normal.z,
        d,
        ranking: 0,
    }
}

/// generates a random line from two points of the buffer
fn generate_rng_line<'a, T: BorrowedBuffer<'a>>(buffer: &'a T) -> Line {
    // choose two random points from the pointcloud
    let mut rng = rand::thread_rng();
    let rand1 = rng.gen_range(0..buffer.len());
    let mut rand2 = rng.gen_range(0..buffer.len());
    // make sure we have two unique points
    while rand1 == rand2 {
        rand2 = rng.gen_range(0..buffer.len());
    }
    // generate line from the two points
    Line {
        first: buffer.view_attribute(&POSITION_3D).at(rand1),
        second: buffer.view_attribute(&POSITION_3D).at(rand2),
        ranking: 0,
    }
}

fn generate_line_model<'a, T: BorrowedBuffer<'a>>(buffer: &'a T, distance_threshold: f64) -> (Line, Vec<usize>) {
    // generate random line from three points in the buffer
    let mut curr_hypo = generate_rng_line(buffer);
    let mut curr_positions = vec![];
    // find all points that belong to the line
    for (index, p) in buffer
        .view_attribute::<Vector3<f64>>(&POSITION_3D)
        .into_iter()
        .enumerate()
    {
        let distance = distance_point_line(&p, &curr_hypo);
        if distance < distance_threshold {
            // we found a point of the line
            curr_positions.push(index);
            curr_hypo.ranking += 1;
        }
    }
    // return current line and positions
    (curr_hypo, curr_positions)
}

fn generate_plane_model<'a, T: BorrowedBuffer<'a>>(buffer: &'a T,
    distance_threshold: f64,
) -> (Plane, Vec<usize>) {
    // generate random plane from three points in the buffer
    let mut curr_hypo = generate_rng_plane(buffer);
    // find all points that belong to the plane
    let mut curr_positions = vec![];

    for (index, p) in buffer
        .view_attribute::<Vector3<f64>>(&POSITION_3D)
        .into_iter()
        .enumerate()
    {
        let distance = distance_point_plane(&p, &curr_hypo);
        if distance < distance_threshold {
            // we found a point that belongs to the plane
            curr_hypo.ranking += 1;
            curr_positions.push(index);
        }
    }
    (curr_hypo, curr_positions)
}

/// Ransac Plane Segmentation in parallel.
/// Returns the plane with the highest rating/most inliers and the associated indices of the inliers.
/// Iterates over all points in the `buffer`.
/// The `distance_threshold` sets the maximum distance to the plane that a point is counted as an inlier from.
/// With `num_of_iterations` the number of iterations that the algorithm performs can be chosen.
///
/// # Examples
///
/// ```
/// # use pasture_core::nalgebra::Vector3;
/// # use pasture_core::containers::*;
/// # use pasture_core::layout::PointType;
/// # use pasture_derive::PointType;
/// # use pasture_algorithms::segmentation::ransac_plane_par;
/// #[repr(C)]
/// #[derive(PointType, Debug, Copy, Clone, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
/// struct SimplePoint {
///     #[pasture(BUILTIN_POSITION_3D)]
///    pub position: Vector3<f64>,
/// }
/// let mut points = vec![];
/// // generate some inliers
/// for i in 0..200{
///     points.push(SimplePoint{position: Vector3::new(0.0, f64::from(i), f64::from(i*i))});
/// }
/// // generate an outlier
/// points.push(SimplePoint{position: Vector3::new(9.0, 0.0, 0.0)});
/// let buffer = points.into_iter().collect::<HashMapBuffer>();
/// let plane_and_indices = ransac_plane_par(&buffer, 0.5, 10);
/// for i in 0..199{
///     // inliers are in the plane
///     assert!(plane_and_indices.1.contains(&(i as usize)));
/// }
/// // outlier is not in the plane
/// assert!(!plane_and_indices.1.contains(&200));
/// ```
///
/// # Panics
///
/// If the size of the buffer is < 3.
pub fn ransac_plane_par<'a, T: BorrowedBuffer<'a> + Sync>(buffer: &'a T,
    distance_threshold: f64,
    num_of_iterations: usize,
) -> (Plane, Vec<usize>) {
    if buffer.len() < 3 {
        panic!("buffer needs to include at least 3 points to generate a plane.");
    }
    // iterate in parallel over num_of_iterations
    (0..num_of_iterations)
        .into_par_iter()
        .map(|_x| {
            // generate one model for the current iteration
            generate_plane_model(buffer, distance_threshold)
        })
        // get the best plane-model from all iterations (highest ranking)
        .max_by(|(x, _y), (a, _b)| x.ranking.cmp(&a.ranking))
        .unwrap()
}

/// Ransac Plane Segmentation in serial (for maximum speed use ransac_plane_par).
/// Returns the plane with the highest rating/most inliers and the associated indices of the inliers.
/// Iterates over all points in the `buffer`.
/// The `distance_threshold` sets the maximum distance to the plane that a point is counted as an inlier from.
/// With `num_of_iterations` the number of iterations that the algorithm performs can be chosen.
///
/// # Examples
///
/// ```
/// # use pasture_core::nalgebra::Vector3;
/// # use pasture_core::containers::*;
/// # use pasture_core::layout::PointType;
/// # use pasture_derive::PointType;
/// # use pasture_algorithms::segmentation::ransac_plane_serial;
/// #[repr(C)]
/// #[derive(PointType, Debug, Copy, Clone, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
/// struct SimplePoint {
///     #[pasture(BUILTIN_POSITION_3D)]
///    pub position: Vector3<f64>,
/// }
/// let mut points = vec![];
/// // generate some inliers
/// for i in 0..200{
///     points.push(SimplePoint{position: Vector3::new(0.0, f64::from(i), f64::from(i*i))});
/// }
/// // generate an outlier
/// points.push(SimplePoint{position: Vector3::new(9.0, 0.0, 0.0)});
/// let buffer = points.into_iter().collect::<HashMapBuffer>();
/// let plane_and_indices = ransac_plane_serial(&buffer, 0.5, 10);
/// for i in 0..199{
///     // inliers are in the plane
///     assert!(plane_and_indices.1.contains(&(i as usize)));
/// }
/// // outlier is not in the plane
/// assert!(!plane_and_indices.1.contains(&200));
/// ```
///
/// # Panics
///
/// If the size of the buffer is < 3.
pub fn ransac_plane_serial<'a, T: BorrowedBuffer<'a> + Sync>(buffer: &'a T,
    distance_threshold: f64,
    num_of_iterations: usize,
) -> (Plane, Vec<usize>) {
    if buffer.len() < 3 {
        panic!("buffer needs to include at least 3 points to generate a plane.");
    }
    (0..num_of_iterations)
        .map(|_x| 
            // generate one model for the current iteration
            generate_plane_model(buffer, distance_threshold))
        // get the best plane-model from all iterations (highest ranking)
        .max_by(|(x, _y), (a, _b)| x.ranking.cmp(&a.ranking))
        .unwrap()
}

/// Ransac Line Segmentation in parallel.
/// Returns the line with the highest rating/most inliers and the associated indices of the inliers.
/// Iterates over all points in the `buffer`.
/// The `distance_threshold` sets the maximum distance to the plane that a point is counted as an inlier from.
/// With `num_of_iterations` the number of iterations that the algorithm performs can be chosen.
///
/// # Examples
///
/// ```
/// # use pasture_core::nalgebra::Vector3;
/// # use pasture_core::containers::*;
/// # use pasture_core::layout::PointType;
/// # use pasture_derive::PointType;
/// # use pasture_algorithms::segmentation::ransac_line_par;
/// #[repr(C)]
/// #[derive(PointType, Debug, Copy, Clone, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
/// struct SimplePoint {
///     #[pasture(BUILTIN_POSITION_3D)]
///    pub position: Vector3<f64>,
/// }
/// let mut points = vec![];
/// // generate some inliers
/// for i in 0..200{
///     points.push(SimplePoint{position: Vector3::new(0.0, 0.0, f64::from(i))});
/// }
/// // generate an outlier
/// points.push(SimplePoint{position: Vector3::new(9.0, 0.0, 0.0)});
/// let buffer = points.into_iter().collect::<HashMapBuffer>();
/// let line_and_indices = ransac_line_par(&buffer, 0.5, 10);
/// for i in 0..199{
///     // inliers are in the plane
///     assert!(line_and_indices.1.contains(&(i as usize)));
/// }
/// // outlier is not in the plane
/// assert!(!line_and_indices.1.contains(&200));
/// ```
///
/// # Panics
///
/// If the size of the buffer is < 2.
pub fn ransac_line_par<'a, T: BorrowedBuffer<'a> + Sync>(buffer: &'a T,
    distance_threshold: f64,
    num_of_iterations: usize,
) -> (Line, Vec<usize>) {
    if buffer.len() < 2 {
        panic!("buffer needs to include at least 2 points to generate a line.");
    }
    // iterate num_of_iterations in parallel
    (0..num_of_iterations)
        .into_par_iter()
        .map(|_x| 
            // generate one model for the current iteration
            generate_line_model(buffer, distance_threshold))
        // get the best line-model from all iterations (highest ranking)
        .max_by(|(x, _y), (a, _b)| x.ranking.cmp(&a.ranking))
        .unwrap()
}

/// Ransac Line Segmentation in serial (for maximum speed use ransac_line_par).
/// Returns the line with the highest rating/most inliers and the associated indices of the inliers.
/// Iterates over all points in the `buffer`.
/// The `distance_threshold` sets the maximum distance to the plane that a point is counted as an inlier from.
/// With `num_of_iterations` the number of iterations that the algorithm performs can be chosen.
///
/// # Examples
///
/// ```
/// # use pasture_core::nalgebra::Vector3;
/// # use pasture_core::containers::*;
/// # use pasture_core::layout::PointType;
/// # use pasture_derive::PointType;
/// # use pasture_algorithms::segmentation::ransac_line_serial;
/// #[repr(C)]
/// #[derive(PointType, Debug, Copy, Clone, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
/// struct SimplePoint {
///     #[pasture(BUILTIN_POSITION_3D)]
///    pub position: Vector3<f64>,
/// }
/// let mut points = vec![];
/// // generate some inliers
/// for i in 0..200{
///     points.push(SimplePoint{position: Vector3::new(0.0, 0.0, f64::from(i))});
/// }
/// // generate an outlier
/// points.push(SimplePoint{position: Vector3::new(9.0, 0.0, 0.0)});
/// let buffer = points.into_iter().collect::<HashMapBuffer>();
/// let line_and_indices = ransac_line_serial(&buffer, 0.5, 10);
/// for i in 0..199{
///     // inliers are in the plane
///     assert!(line_and_indices.1.contains(&(i as usize)));
/// }
/// // outlier is not in the plane
/// assert!(!line_and_indices.1.contains(&200));
/// ```
///
/// # Panics
///
/// If the size of the buffer is < 2.
pub fn ransac_line_serial<'a, T: BorrowedBuffer<'a>>(buffer: &'a T,
    distance_threshold: f64,
    num_of_iterations: usize,
) -> (Line, Vec<usize>) {
    if buffer.len() < 2 {
        panic!("buffer needs to include at least 2 points to generate a line.");
    }

    // iterate num_of_iterations in parallel
    (0..num_of_iterations)
        
        .map(|_x| 
            // generate one model for the current iteration
            generate_line_model(buffer, distance_threshold))
        // get the best line-model from all iterations (highest ranking)
        .max_by(|(x, _y), (a, _b)| x.ranking.cmp(&a.ranking))
        .unwrap()
}

#[cfg(test)]
mod tests {

    use pasture_core::{nalgebra::Vector3, containers::HashMapBuffer,
    };
    use pasture_derive::PointType;

    use super::*;

    #[repr(C)]
    #[derive(PointType, Debug, Copy, Clone, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
    pub struct SimplePoint {
        #[pasture(BUILTIN_POSITION_3D)]
        pub position: Vector3<f64>,
    }

    fn setup_point_cloud() -> HashMapBuffer {
        // generate random points for the pointcloud
        (2..2002)
            
            .map(|p| {
                // let mut rng = rand::thread_rng();
                // generate plane points (along x- and y-axis)
                let mut point = SimplePoint {
                    // position: Vector3::new(rng.gen_range(0.0..100.0), rng.gen_range(0.0..100.0), 1.0),
                    position: Vector3::new(p as f64, (p * p) as f64, 1.0),
                };
                // generate z-axis points for the line
                if p % 5 == 0 {
                    point.position = Vector3::new(0.0, 0.0, (p * p) as f64);
                }
                // generate outliers
                if p % 50 == 0 {
                    point.position.z = (p * p) as f64;
                }
                point
            })
            .collect()
    }

    #[test]
    fn test_ransac_plane_par() {
        let buffer = setup_point_cloud();
        let (_plane, indices) = ransac_plane_par(&buffer, 0.1, 300);
        assert!(indices.len() == 1600);
        for i in 0..2000 {
            if i % 5 != 3 {
                assert!(indices.contains(&i));
            }
        }
    }

    #[test]
    fn test_ransac_plane_serial() {
        let buffer = setup_point_cloud();
        let (_plane, indices) = ransac_plane_serial(&buffer, 0.1, 300);
        assert!(indices.len() == 1600);
        for i in 0..2000 {
            if i % 5 != 3 {
                assert!(indices.contains(&i));
            }
        }
    }

    #[test]
    fn test_ransac_line_par() {
        let buffer = setup_point_cloud();
        let (_plane, indices) = ransac_line_par(&buffer, 0.1, 300);
        assert!(indices.len() == 400);
        for i in 0..2000 {
            if i % 5 == 3 {
                assert!(indices.contains(&i));
            }
        }
    }

    #[test]
    fn test_ransac_line_serial() {
        let buffer = setup_point_cloud();
        let (_plane, indices) = ransac_line_serial(&buffer, 0.1, 300);
        assert!(indices.len() == 400);
        for i in 0..2000 {
            if i % 5 == 3 {
                assert!(indices.contains(&i));
            }
        }
    }
}
