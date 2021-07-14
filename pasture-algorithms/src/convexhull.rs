use std::hash::{Hash, Hasher};
use std::collections::HashSet;
use pasture_core::{
    containers::{PointBuffer, PointBufferExt},
    layout::attributes::{POSITION_3D},
    nalgebra::{Vector3},
};

#[derive(Clone, Copy)]
struct Triangle {
    a: usize,
    b: usize,
    c: usize,
    normal: Vector3<f64>,
}

#[derive(Eq, Clone, Copy)]
struct Edge {
    a: usize,
    b: usize,
}

impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        self.a == other.a && self.b == other.b ||
        self.a == other.b && self.b == other.a
    }
}

impl Hash for Edge {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.a * self.b).hash(state);
    }
}

/// Convex Hull generation as triangle mesh
/// Returns the convex hull as a vector of tuples of size 3 that contains the indices of the triangle vertices within the input buffer that form a convex hull around all input points
/// or an error if less than 3 linearily independent points were given in the input buffer.
pub fn convex_hull_as_triangle_mesh<T: PointBuffer>(buffer: &T) -> Result<Vec<(usize, usize, usize)>, &'static str> {
    let triangles = create_convex_hull(buffer);
    if triangles.len() < 2 {
        return Err("input buffer cointains too few linearly independent points");
    }
    let mut triangle_indices = Vec::new();
    for tri in triangles {
        triangle_indices.push((tri.a, tri.b, tri.c));
    }
    return Ok(triangle_indices);
}

/// Convex Hull generation as points
/// Returns the convex hull as a vector that contains the indices the points forming a convex hull around all input points.
pub fn convex_hull_as_points<T: PointBuffer>(buffer: &T) -> Vec<usize> {
    let triangles = create_convex_hull(buffer);
    let mut points = HashSet::new();
    for tri in triangles {
        points.insert(tri.a);
        points.insert(tri.b);
        points.insert(tri.c);
    }
    let point_indices: Vec<usize> = points.into_iter().collect();
    return point_indices;
}

fn create_convex_hull<T: PointBuffer>(buffer: &T) -> Vec<Triangle> {
    let mut triangles: Vec<Triangle> = Vec::new();
    let position_attribute = match buffer
        .point_layout()
        .get_attribute_by_name(POSITION_3D.name())
    {
        Some(a) => a,
        None => return triangles,
    };

    let mut pointid: usize = 0;
    if position_attribute.datatype() == POSITION_3D.datatype() {
        for point in buffer.iter_attribute::<Vector3<f64>>(&POSITION_3D) {
            iteration(buffer, pointid, point, &mut triangles);
            pointid += 1;
        }
    } else {
        for point in buffer.iter_attribute_as::<Vector3<f64>>(&POSITION_3D) {
            iteration(buffer, pointid, point, &mut triangles);
            pointid += 1;
        }
    };
    return triangles;
}

/// Performs a single iteration of the cunvex hull algorithm.
/// `pointid`: current index within the buffer
/// `point`: current point to be checked against the convex hull, possibly extends the convex hull
/// `triangles`: the set of triangles forming the convex hull
/// Each iteration receives a convex hull and checks it against the current point within the buffer.
/// If the point lies within the current convex hull no changes have to be made.
/// If the point lies outside of the current convex hull the hull has to be extended to include the current point.
/// If 'triangles' contain only one entry: no full triangle has been found yet. In case of linearily dependant points no second triangle is added.
/// If all 'triangles' are in a plane with 'point' a special triangulation procedure is needed to prevent a degenerated triangle mesh.
///
/// TODO: compare to epsilons instead of 0.0
fn iteration<T: PointBuffer>(buffer: &T, pointid: usize, point: Vector3<f64>, triangles: &mut Vec<Triangle>) {
    if pointid == 0 {
        triangles.push(Triangle{ a: 0, b: 0, c: 0, normal: point })
    }
    else if triangles.len() == 1 {
        let mut first = &mut triangles[0];
        if pointid == 1 {
            first.b = pointid;
        }
        else {
            let first_a = buffer.get_attribute(&POSITION_3D, first.a);
            let first_b = buffer.get_attribute(&POSITION_3D, first.b);
            let ab: Vector3<f64> = first_b - first_a;
            let ab_mag_sqr = ab.magnitude_squared();
            if ab_mag_sqr == 0.0 {
                first.b = pointid;
            }
            else {
                let ac: Vector3<f64> = point - first_a;
                let ab_norm = ab.normalize();
                let ac_ab_projected_length = ac.dot(&ab_norm);
                if f64::abs(ac_ab_projected_length) == ac.magnitude() {
                    if ac_ab_projected_length >= 0.0 {
                        if ac.magnitude_squared() > ab_mag_sqr {
                            first.b = pointid;
                        }
                    }
                    else {
                        first.a = pointid;
                    }
                }
                else {
                    first.c = pointid;
                    first.normal = calc_normal(first_a, first_b, point);

                    let first = triangles[0];
                    triangles.push(Triangle{ a: first.a, b: first.c, c: first.b, normal: -first.normal })
                }
            }
        }
    }
    else {
        let mut outer_edges = HashSet::new();
        let mut inner_edges = HashSet::new();
        let mut planar_triangles = Vec::new();

        triangles.retain(|tri| {
            let tri_a: Vector3<f64> = buffer.get_attribute(&POSITION_3D, tri.a);
            let pa: Vector3<f64> = tri_a - point;
            let dot = pa.dot(&tri.normal);
            if dot < 0.0 {
                process_edge(tri.a, tri.b, &mut outer_edges, &mut inner_edges);
                process_edge(tri.b, tri.c, &mut outer_edges, &mut inner_edges);
                process_edge(tri.c, tri.a, &mut outer_edges, &mut inner_edges);
                return false;
            }
            else if dot == 0.0 {
                planar_triangles.push(tri.clone());
            }
            return true;
        });
        
        if outer_edges.len() > 0 || inner_edges.len() > 0 {
            for edge in outer_edges.iter() {
                let edge_a: Vector3<f64> = buffer.get_attribute(&POSITION_3D, edge.a);
                let edge_b: Vector3<f64> = buffer.get_attribute(&POSITION_3D, edge.b);
                triangles.push(Triangle{ a: edge.a, b: edge.b, c: pointid, normal: calc_normal(edge_a, edge_b, point)});
            }
        }
        else {
            let mut edges_facing_point = Vec::new();
            let mut edge_distances = Vec::new();
            let mut edge_triangle_id = Vec::new();
            for (i, pt) in planar_triangles.iter().enumerate() {
                let planar_a = buffer.get_attribute(&POSITION_3D, pt.a);
                let planar_b = buffer.get_attribute(&POSITION_3D, pt.b);
                let planar_c = buffer.get_attribute(&POSITION_3D, pt.c);
                let dist_ab = dist_point_to_edge(point, planar_a, planar_b, pt.normal);
                if dist_ab > 0.0 {
                    edges_facing_point.push(Edge{a: pt.a, b: pt.b});
                    edge_distances.push(dist_ab);
                    edge_triangle_id.push(i);
                }
                let dist_bc = dist_point_to_edge(point, planar_b, planar_c, pt.normal);
                if dist_bc > 0.0 {
                    edges_facing_point.push(Edge{a: pt.b, b: pt.c});
                    edge_distances.push(dist_bc);
                    edge_triangle_id.push(i);
                }
                let dist_ca = dist_point_to_edge(point, planar_c, planar_a, pt.normal);
                if dist_ca > 0.0 {
                    edges_facing_point.push(Edge{a: pt.c, b: pt.a});
                    edge_distances.push(dist_ca);
                    edge_triangle_id.push(i);
                }
            }

            let mut edge_triangle_normals = Vec::new();
            let edgenum = edges_facing_point.len();
            for i in (0..edgenum).rev() {
                let edg = edges_facing_point[0];
                let dist = edge_distances[i];
                let edg_a: Vector3<f64> = buffer.get_attribute(&POSITION_3D, edg.a);
                let edg_b: Vector3<f64> = buffer.get_attribute(&POSITION_3D, edg.b);
                let edg_a_b: Vector3<f64> = edg_b - edg_a;
                let edg_length = edg_a_b.magnitude();
                let edg_triangle_normal: Vector3<f64> = planar_triangles[edge_triangle_id[i]].normal;
                let mut remove = false;
                for other_edge_id in 0..edgenum {
                    if other_edge_id != i && edge_distances[other_edge_id] < dist {
                        let other_edg_triangle_normal = planar_triangles[edge_triangle_id[other_edge_id]].normal;
                        if edg_triangle_normal.dot(&other_edg_triangle_normal) > 0.0 {
                            let other_edg = edges_facing_point[other_edge_id];
                            let other_edg_a: Vector3<f64> = buffer.get_attribute(&POSITION_3D, other_edg.a);
                            let other_edg_b: Vector3<f64> = buffer.get_attribute(&POSITION_3D, other_edg.b);
                            let edg_a_other_edg_a = other_edg_a - edg_a;
                            let edg_a_other_edg_b = other_edg_b - edg_a;
                            let other_a_region: u8;
                            let other_b_region: u8;
                            let dot_edg_to_other_edg_a = edg_a_b.dot(&edg_a_other_edg_a);
                            let dot_edg_to_other_edg_b = edg_a_b.dot(&edg_a_other_edg_b);
                            if dot_edg_to_other_edg_a < 0.0 {
                                other_a_region = 1;
                            }
                            else if dot_edg_to_other_edg_a < edg_length {
                                remove = true;
                                break;
                            }
                            else {
                                other_a_region = 2;
                            }
                            if dot_edg_to_other_edg_b < 0.0 {
                                other_b_region = 1;
                            }
                            else if dot_edg_to_other_edg_b < edg_length {
                                remove = true;
                                break;
                            }
                            else {
                                other_b_region = 2;
                            }
                            if other_a_region != other_b_region {
                                remove = true;
                                break;
                            }
                        }
                    }
                }
                if remove {
                    edges_facing_point.remove(i);
                    edge_distances.remove(i);
                    edge_triangle_id.remove(i);
                }
                else {
                    edge_triangle_normals.insert(0, -edg_triangle_normal);
                }
            }

            for i in 0..edges_facing_point.len() {
                let edg = edges_facing_point[i];
                triangles.push(Triangle{ a: edg.a, b: edg.b, c: pointid, normal: edge_triangle_normals[i]});
            }
        }
    }
}

/// Calculates the distance of a point to an edge of a triangle. Assumes the point to be in the same plane as the triangle.
/// `point`: the point that lies in the same plane as the edge
/// `edge_a`: first vertex of the edge
/// `edge_b`: second vertex of the edge
/// 'triangle_normal': the normal of the triangle the edge belongs to
fn dist_point_to_edge(point: Vector3<f64>, edge_a: Vector3<f64>, edge_b: Vector3<f64>, triangle_normal: Vector3<f64>) -> f64 {
    let pa = edge_a - point;
    let edge_ab: Vector3<f64> = edge_b - edge_a;
    let edge_ab_normal = triangle_normal.cross(&edge_ab);
    return edge_ab_normal.dot(&pa);
}

/// Adds the given edge to the set of outer edges. If the given edge is already contained in the set of outer edges it is removed and added to the set of inner edges.
/// `a`: first vertex of the edge
/// `b`: second vertex of the edge
/// `outer_edges`: the set of outer edges
/// `inner_edges`: the set of inner edges
fn process_edge(a: usize, b: usize, outer_edges: &mut HashSet<Edge>, inner_edges: &mut HashSet<Edge>) {
    let e = Edge{a, b};
    if !outer_edges.insert(e) {
        outer_edges.remove(&e);
        inner_edges.insert(e);
    }
}

/// Calculates the normal of a triangle formed b three points.
/// `a`: first vertex of the triangle
/// `b`: second vertex of the triangle
/// `c`: third vertex of the triangle
fn calc_normal(a: Vector3<f64>, b: Vector3<f64>, c: Vector3<f64>) -> Vector3<f64> {
    let ab: Vector3<f64> = b - a;
    let ac: Vector3<f64> = c - a;
    return ab.cross(&ac);
}

#[cfg(test)]
mod tests {
    use pasture_core::{containers::PerAttributeVecPointStorage, layout::PointType, nalgebra::Vector3};
    use crate::convexhull;
    use pasture_derive::PointType;
    use anyhow::Result;
    use rand::{distributions::Uniform, thread_rng, Rng};

    #[derive(PointType, Default)]
    #[repr(C)]
    struct TestPointTypeSmall {
        #[pasture(BUILTIN_POSITION_3D)]
        pub position: Vector3<f64>,
    }

    // Internal Tests
    fn test_normals_for_triangles(triangles: Vec<convexhull::Triangle>, normals: Vec<Vector3<f64>>) {
        for n in normals {
            let mut found = false;
            for t in triangles.iter() {
                if f64::abs(t.normal.normalize().dot(&n) - 1.0) < 0.0001 {
                    found = true;
                    break;
                }
            }
            assert!(found);
        }
    }

    #[test]
    fn test_convex_simple_triangle() -> Result<()> {
        let mut buffer = PerAttributeVecPointStorage::with_capacity(3, TestPointTypeSmall::layout());
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(0.0, 0.0, 0.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(1.0, 0.0, 0.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(0.0, 0.0, 1.0) });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 2);
        let normals = vec![Vector3::new(0.0, 1.0, 0.0), Vector3::new(0.0, -1.0, 0.0)];
        test_normals_for_triangles(result, normals);

        Ok(())
    }

    #[test]
    fn test_convex_simple_tet_4_points() -> Result<()> {
        let mut buffer = PerAttributeVecPointStorage::with_capacity(3, TestPointTypeSmall::layout());
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(0.0, 0.0, 0.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(1.0, 0.0, 0.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(0.0, 0.0, 1.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(0.0, 1.0, 0.0) });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 4);
        let normals = vec![Vector3::new(-1.0, 0.0, 0.0), Vector3::new(0.0, -1.0, 0.0),
                Vector3::new(0.0, 0.0, -1.0), Vector3::new(1.0, 1.0, 1.0).normalize()];
        test_normals_for_triangles(result, normals);

        Ok(())
    }

    #[test]
    fn test_convex_simple_tet_5_points() -> Result<()> {
        let mut buffer = PerAttributeVecPointStorage::with_capacity(3, TestPointTypeSmall::layout());
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(0.0, 0.0, 0.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(1.0, 0.0, 0.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(0.0, 0.0, 1.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(0.0, 1.0, 0.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(-1.0, -1.0, -1.0) });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 4);
        let normals = vec![Vector3::new(1.0, 1.0, 1.0).normalize(), Vector3::new(1.0, 1.0, -3.0).normalize(), 
                Vector3::new(1.0, -3.0, 1.0).normalize(), Vector3::new(-3.0, 1.0, 1.0).normalize()];
        test_normals_for_triangles(result, normals);

        Ok(())
    }

    #[test]
    fn test_convex_1_point() -> Result<()> {
        let mut buffer = PerAttributeVecPointStorage::with_capacity(1, TestPointTypeSmall::layout());
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(0.0, 0.0, 0.0) });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].a, 0);

        Ok(())
    }

    #[test]
    fn test_convex_line_2_points() -> Result<()> {
        let mut buffer = PerAttributeVecPointStorage::with_capacity(3, TestPointTypeSmall::layout());
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(0.0, 0.0, 0.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(1.0, 0.0, 0.0) });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].a, 0);
        assert_eq!(result[0].b, 1);

        Ok(())
    }

    #[test]
    fn test_convex_line_3_points() -> Result<()> {
        let mut buffer = PerAttributeVecPointStorage::with_capacity(3, TestPointTypeSmall::layout());
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(0.0, 0.0, 0.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(1.0, 0.0, 0.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(2.0, 0.0, 0.0) });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].a, 0);
        assert_eq!(result[0].b, 2);

        Ok(())
    }

    #[test]
    fn test_convex_line_4_points() -> Result<()> {
        let mut buffer = PerAttributeVecPointStorage::with_capacity(3, TestPointTypeSmall::layout());
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(0.0, 0.0, 0.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(1.0, 0.0, 0.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(2.0, 0.0, 0.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(-1.0, 0.0, 0.0) });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].a, 3);
        assert_eq!(result[0].b, 2);

        Ok(())
    }

    #[test]
    fn test_convex_plane_4_points() -> Result<()> {
        let mut buffer = PerAttributeVecPointStorage::with_capacity(3, TestPointTypeSmall::layout());
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(0.0, 0.0, 0.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(0.0, 0.0, 1.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(1.0, 0.0, 0.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(1.0, 0.0, 1.0) });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 4);
        let normals = vec![Vector3::new(0.0, 1.0, 0.0), Vector3::new(0.0, -1.0, 0.0)];
        test_normals_for_triangles(result, normals);

        Ok(())
    }

    #[test]
    fn test_convex_random_points_in_box_create_box_first() -> Result<()> {
        let mut buffer = PerAttributeVecPointStorage::with_capacity(28, TestPointTypeSmall::layout());
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(-1.0, -1.0, -1.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(-1.0, -1.0, 1.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(-1.0, 1.0, -1.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(-1.0, 1.0, 1.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(1.0, -1.0, -1.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(1.0, -1.0, 1.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(1.0, 1.0, -1.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(1.0, 1.0, 1.0) });
        let mut rng = thread_rng();
        for _ in 0..20 {
            buffer.push_point(TestPointTypeSmall { position: Vector3::new(rng.sample(Uniform::new(-0.9, 0.9)),
                rng.sample(Uniform::new(-0.9, 0.9)), rng.sample(Uniform::new(-0.9, 0.9))) });
        }
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 12);
        let normals = vec![Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0), Vector3::new(0.0, 0.0, 1.0),
                Vector3::new(-1.0, 0.0, 0.0), Vector3::new(0.0, -1.0, 0.0), Vector3::new(0.0, 0.0, -1.0)];
        test_normals_for_triangles(result, normals);

        Ok(())
    }

    #[test]
    fn test_convex_random_points_in_box_create_box_last() -> Result<()> {
        let mut buffer = PerAttributeVecPointStorage::with_capacity(28, TestPointTypeSmall::layout());
        let mut rng = thread_rng();
        for _ in 0..20 {
            buffer.push_point(TestPointTypeSmall { position: Vector3::new(rng.sample(Uniform::new(-0.9, 0.9)),
                rng.sample(Uniform::new(-0.9, 0.9)), rng.sample(Uniform::new(-0.9, 0.9))) });
        }
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(-1.0, -1.0, -1.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(-1.0, -1.0, 1.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(-1.0, 1.0, -1.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(-1.0, 1.0, 1.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(1.0, -1.0, -1.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(1.0, -1.0, 1.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(1.0, 1.0, -1.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(1.0, 1.0, 1.0) });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 12);
        let normals = vec![Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0), Vector3::new(0.0, 0.0, 1.0),
                Vector3::new(-1.0, 0.0, 0.0), Vector3::new(0.0, -1.0, 0.0), Vector3::new(0.0, 0.0, -1.0)];
        test_normals_for_triangles(result, normals);

        Ok(())
    }

    // Interface Tests
    #[test]
    fn test_convex_0_point_output_mesh_error() -> Result<()> {
        let buffer = PerAttributeVecPointStorage::with_capacity(0, TestPointTypeSmall::layout());
        let result = convexhull::convex_hull_as_triangle_mesh(&buffer);

        assert_eq!(result, Err("input buffer cointains too few linearly independent points"));

        Ok(())
    }

    #[test]
    fn test_convex_1_point_output_mesh_error() -> Result<()> {
        let mut buffer = PerAttributeVecPointStorage::with_capacity(1, TestPointTypeSmall::layout());
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(0.0, 0.0, 0.0) });
        let result = convexhull::convex_hull_as_triangle_mesh(&buffer);

        assert_eq!(result, Err("input buffer cointains too few linearly independent points"));

        Ok(())
    }

    #[test]
    fn test_convex_2_point_output_mesh_error() -> Result<()> {
        let mut buffer = PerAttributeVecPointStorage::with_capacity(1, TestPointTypeSmall::layout());
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(0.0, 0.0, 0.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(1.0, 0.0, 0.0) });
        let result = convexhull::convex_hull_as_triangle_mesh(&buffer);

        assert_eq!(result, Err("input buffer cointains too few linearly independent points"));

        Ok(())
    }

    #[test]
    fn test_convex_3_point_output_mesh_error_same_point() -> Result<()> {
        let mut buffer = PerAttributeVecPointStorage::with_capacity(1, TestPointTypeSmall::layout());
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(0.0, 0.0, 0.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(0.0, 0.0, 0.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(0.0, 0.0, 0.0) });
        let result = convexhull::convex_hull_as_triangle_mesh(&buffer);

        assert_eq!(result, Err("input buffer cointains too few linearly independent points"));

        Ok(())
    }

    #[test]
    fn test_convex_3_point_output_mesh_error_line() -> Result<()> {
        let mut buffer = PerAttributeVecPointStorage::with_capacity(1, TestPointTypeSmall::layout());
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(0.0, 0.0, 0.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(1.0, 0.0, 0.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(2.0, 0.0, 0.0) });
        let result = convexhull::convex_hull_as_triangle_mesh(&buffer);

        assert_eq!(result, Err("input buffer cointains too few linearly independent points"));

        Ok(())
    }

    #[test]
    fn test_convex_3_point_output_mesh_no_error() -> Result<()> {
        let mut buffer = PerAttributeVecPointStorage::with_capacity(1, TestPointTypeSmall::layout());
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(0.0, 0.0, 0.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(1.0, 0.0, 0.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(0.0, 1.0, 0.0) });
        let result = convexhull::convex_hull_as_triangle_mesh(&buffer);
        let result_unwrapped = result.unwrap();

        assert_eq!(result_unwrapped.len(), 2);
        assert!(result_unwrapped.contains(&(0, 1, 2)));
        assert!(result_unwrapped.contains(&(0, 2, 1)));

        Ok(())
    }

    #[test]
    fn test_convex_3_point_output_points_line() -> Result<()> {
        let mut buffer = PerAttributeVecPointStorage::with_capacity(1, TestPointTypeSmall::layout());
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(0.0, 0.0, 0.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(1.0, 0.0, 0.0) });
        buffer.push_point(TestPointTypeSmall { position: Vector3::new(2.0, 0.0, 0.0) });
        let result = convexhull::convex_hull_as_points(&buffer);

        assert_eq!(result.len(), 2);
        assert!(result.contains(&0));
        assert!(result.contains(&2));

        Ok(())
    }
}