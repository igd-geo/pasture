use anyhow::{anyhow, Result};
use pasture_core::containers::BorrowedBuffer;
use pasture_core::{layout::attributes::POSITION_3D, nalgebra::Vector3};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::usize;

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
        self.a == other.a && self.b == other.b || self.a == other.b && self.b == other.a
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
///
/// #Panics
///
/// If the PointBuffer doesn't cointain a POSITION_3D attribute.
pub fn convex_hull_as_triangle_mesh<'a, T: BorrowedBuffer<'a>>(
    buffer: &'a T,
) -> Result<Vec<Vector3<usize>>> {
    let triangles = create_convex_hull(buffer);
    if triangles.len() < 2 {
        return Err(anyhow!(
            "input buffer cointains too few linearly independent points"
        ));
    }
    let mut triangle_indices = Vec::new();
    for tri in triangles {
        triangle_indices.push(Vector3::new(tri.a, tri.b, tri.c));
    }
    return Ok(triangle_indices);
}

/// Convex Hull generation as points
/// Returns the convex hull as an unsorted vector that contains the indices the points forming a convex hull around all input points.
///
/// #Panics
///
/// If the PointBuffer doesn't cointain a POSITION_3D attribute.
pub fn convex_hull_as_points<'a, T: BorrowedBuffer<'a>>(buffer: &'a T) -> Vec<usize> {
    let triangles = create_convex_hull(buffer);
    let mut points = HashSet::new();
    if triangles.len() > 1 {
        for tri in triangles {
            points.insert(tri.a);
            points.insert(tri.b);
            points.insert(tri.c);
        }
    } else {
        let tri = triangles[0];
        points.insert(tri.a);
        points.insert(tri.b);
        if tri.c != 0 {
            points.insert(tri.c);
        }
    }
    let point_indices: Vec<usize> = points.into_iter().collect();
    return point_indices;
}

fn create_convex_hull<'a, T: BorrowedBuffer<'a>>(buffer: &'a T) -> Vec<Triangle> {
    let mut triangles: Vec<Triangle> = Vec::new();
    let position_attribute = match buffer
        .point_layout()
        .get_attribute_by_name(POSITION_3D.name())
    {
        Some(a) => a,
        None => {
            panic!("point buffer contains no position attribute")
        }
    };

    let mut pointid: usize = 0;
    if position_attribute.datatype() == POSITION_3D.datatype() {
        for point in buffer.view_attribute::<Vector3<f64>>(&POSITION_3D) {
            iteration(buffer, pointid, point, &mut triangles);
            pointid += 1;
        }
    } else {
        for point in buffer
            .view_attribute_with_conversion::<Vector3<f64>>(&POSITION_3D)
            .expect("Can't convert POSITION_3D attribute")
        {
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
fn iteration<'a, T: BorrowedBuffer<'a>>(
    buffer: &'a T,
    pointid: usize,
    point: Vector3<f64>,
    triangles: &mut Vec<Triangle>,
) {
    if pointid == 0 {
        triangles.push(Triangle {
            a: 0,
            b: 0,
            c: 0,
            normal: point,
        })
    } else if triangles.len() == 1 {
        let first = &mut triangles[0];
        if pointid == 1 {
            first.b = pointid;
        } else {
            let first_a = buffer.view_attribute(&POSITION_3D).at(first.a);
            let first_b = buffer.view_attribute(&POSITION_3D).at(first.b);
            let ab: Vector3<f64> = first_b - first_a;
            let ab_mag_sqr = ab.magnitude_squared();
            if ab_mag_sqr == 0.0 {
                first.b = pointid;
            } else {
                let ac: Vector3<f64> = point - first_a;
                let ab_norm = ab.normalize();
                let ac_ab_projected_length = ac.dot(&ab_norm);
                if f64::abs(ac_ab_projected_length) == ac.magnitude() {
                    if ac_ab_projected_length >= 0.0 {
                        if ac.magnitude_squared() > ab_mag_sqr {
                            first.b = pointid;
                        }
                    } else {
                        first.a = pointid;
                    }
                } else {
                    first.c = pointid;
                    first.normal = calc_normal(first_a, first_b, point);

                    let first = triangles[0];
                    triangles.push(Triangle {
                        a: first.a,
                        b: first.c,
                        c: first.b,
                        normal: -first.normal,
                    })
                }
            }
        }
    } else {
        let mut outer_edges = HashSet::new();
        let mut inner_edges = HashSet::new();
        let mut planar_triangles = Vec::new();

        triangles.retain(|tri| {
            let tri_a: Vector3<f64> = buffer.view_attribute(&POSITION_3D).at(tri.a);
            let pa: Vector3<f64> = tri_a - point;
            let dot = pa.dot(&tri.normal);
            if dot < 0.0 {
                add_edge_to_outer_or_inner_edges(tri.a, tri.b, &mut outer_edges, &mut inner_edges);
                add_edge_to_outer_or_inner_edges(tri.b, tri.c, &mut outer_edges, &mut inner_edges);
                add_edge_to_outer_or_inner_edges(tri.c, tri.a, &mut outer_edges, &mut inner_edges);
                return false;
            } else if dot == 0.0 {
                planar_triangles.push(tri.clone());
            }
            return true;
        });

        if outer_edges.len() > 0 || inner_edges.len() > 0 {
            for edge in outer_edges.iter() {
                let edge_a: Vector3<f64> = buffer.view_attribute(&POSITION_3D).at(edge.a);
                let edge_b: Vector3<f64> = buffer.view_attribute(&POSITION_3D).at(edge.b);
                triangles.push(Triangle {
                    a: edge.a,
                    b: edge.b,
                    c: pointid,
                    normal: calc_normal(edge_a, edge_b, point),
                });
            }
        } else {
            // Find all edges of the triangle of which the edge-normal is facing the point.
            let mut edges_facing_point = Vec::new();
            let mut edge_triangle_id = Vec::new();
            let position_view = buffer.view_attribute(&POSITION_3D);
            for (i, pt) in planar_triangles.iter().enumerate() {
                let planar_a = position_view.at(pt.a);
                let planar_b = position_view.at(pt.b);
                let planar_c = position_view.at(pt.c);
                let dist_ab = dist_point_to_edge(point, planar_a, planar_b, pt.normal);
                if dist_ab >= 0.0 {
                    edges_facing_point.push(Edge { a: pt.a, b: pt.b });
                    edge_triangle_id.push(i);
                }
                let dist_bc = dist_point_to_edge(point, planar_b, planar_c, pt.normal);
                if dist_bc >= 0.0 {
                    edges_facing_point.push(Edge { a: pt.b, b: pt.c });
                    edge_triangle_id.push(i);
                }
                let dist_ca = dist_point_to_edge(point, planar_c, planar_a, pt.normal);
                if dist_ca >= 0.0 {
                    edges_facing_point.push(Edge { a: pt.c, b: pt.a });
                    edge_triangle_id.push(i);
                }
                if dist_ab < 0.0 && dist_bc < 0.0 && dist_ca < 0.0 {
                    edges_facing_point.clear();
                    break;
                }
            }

            // Remove all edges occluded by other edges.
            let mut edge_triangle_normals = Vec::new();
            for i in (0..edges_facing_point.len()).rev() {
                let edg = edges_facing_point[i];
                let edg_a: Vector3<f64> = position_view.at(edg.a);
                let edg_b: Vector3<f64> = position_view.at(edg.b);
                let dist_edg_a_p = (edg_a - point).magnitude_squared();
                let dist_edg_b_p = (edg_b - point).magnitude_squared();
                let dist_edg_p = dist_point_to_line_segment(point, edg_a, edg_b);
                let edg_triangle_normal: Vector3<f64> =
                    planar_triangles[edge_triangle_id[i]].normal;
                let mut remove = false;
                for other_edge_id in 0..edges_facing_point.len() {
                    if other_edge_id != i {
                        let other_edg_triangle_normal =
                            planar_triangles[edge_triangle_id[other_edge_id]].normal;
                        if edg_triangle_normal.dot(&other_edg_triangle_normal) > 0.0 {
                            let other_edg = edges_facing_point[other_edge_id];
                            let other_edg_a: Vector3<f64> = position_view.at(other_edg.a);
                            let other_edg_b: Vector3<f64> = position_view.at(other_edg.b);
                            let point_other_edg_a = other_edg_a - point;
                            let point_other_edg_b = other_edg_b - point;
                            let point_other_edg_a_norm =
                                other_edg_triangle_normal.cross(&point_other_edg_a);
                            let point_other_edg_b_norm =
                                other_edg_triangle_normal.cross(&point_other_edg_b);
                            let edg_a_other_edg_a = edg_a - other_edg_a;
                            let edg_a_other_edg_b = edg_a - other_edg_b;
                            let edg_b_other_edg_a = edg_b - other_edg_a;
                            let edg_b_other_edg_b = edg_b - other_edg_b;
                            let ea_oea_dot_border_a =
                                point_other_edg_a_norm.dot(&edg_a_other_edg_a);
                            let eb_oea_dot_border_a =
                                point_other_edg_a_norm.dot(&edg_b_other_edg_a);
                            let ea_oeb_dot_border_b =
                                point_other_edg_b_norm.dot(&edg_a_other_edg_b);
                            let eb_oeb_dot_border_b =
                                point_other_edg_b_norm.dot(&edg_b_other_edg_b);
                            if ea_oea_dot_border_a < 0.0 && ea_oeb_dot_border_b > 0.0 {
                                let dist_other_edg_p = f64::min(
                                    point_other_edg_a.magnitude_squared(),
                                    point_other_edg_b.magnitude_squared(),
                                );
                                if dist_edg_a_p > dist_other_edg_p {
                                    remove = true;
                                    break;
                                }
                            }
                            if eb_oea_dot_border_a < 0.0 && eb_oeb_dot_border_b > 0.0 {
                                let dist_other_edg_p = f64::min(
                                    point_other_edg_a.magnitude_squared(),
                                    point_other_edg_b.magnitude_squared(),
                                );
                                if dist_edg_b_p > dist_other_edg_p {
                                    remove = true;
                                    break;
                                }
                            }
                            if (ea_oea_dot_border_a < 0.0 && eb_oeb_dot_border_b > 0.0)
                                || (eb_oea_dot_border_a < 0.0 && ea_oeb_dot_border_b > 0.0)
                            {
                                let dist_other_edg_p =
                                    dist_point_to_line_segment(point, other_edg_a, other_edg_b);
                                if dist_edg_p > dist_other_edg_p {
                                    remove = true;
                                    break;
                                }
                            }
                        }
                    }
                }
                if remove {
                    edges_facing_point.remove(i);
                    edge_triangle_id.remove(i);
                } else {
                    edge_triangle_normals.insert(0, -edg_triangle_normal);
                }
            }

            // Remove all triangles with vertices that are contained in two edges facing point
            let edgenum = edges_facing_point.len();
            if edgenum > 2 {
                let mut edges_to_remove = HashSet::new();
                let mut vertices_on_one_edge_start = HashMap::new();
                let mut vertices_on_one_edge_end = HashMap::new();
                let mut vertices_on_two_edges = HashMap::new();
                for facing_edge in edges_facing_point.iter() {
                    let res_a = vertices_on_one_edge_start
                        .insert(facing_edge.a, (facing_edge.b, facing_edge.clone()));
                    if res_a.is_some() && res_a.unwrap().0 != facing_edge.b {
                        vertices_on_one_edge_start.remove(&facing_edge.a);
                        vertices_on_two_edges
                            .insert(facing_edge.a, (res_a.unwrap().1, facing_edge.clone()));
                    }
                    let res_b = vertices_on_one_edge_end
                        .insert(facing_edge.b, (facing_edge.a, facing_edge.clone()));
                    if res_b.is_some() && res_b.unwrap().0 != facing_edge.a {
                        vertices_on_one_edge_end.remove(&facing_edge.b);
                        vertices_on_two_edges
                            .insert(facing_edge.b, (res_b.unwrap().1, facing_edge.clone()));
                    }
                }
                let mut triangles_to_remove = Vec::new();
                for t_id in 0..triangles.len() {
                    let tri = triangles.get(t_id).unwrap();
                    let res_a = vertices_on_two_edges.get(&tri.a);
                    let res_b = vertices_on_two_edges.get(&tri.b);
                    let res_c = vertices_on_two_edges.get(&tri.c);
                    let a_on_two_edges = res_a.is_some();
                    let b_on_two_edges = res_b.is_some();
                    let c_on_two_edges = res_c.is_some();
                    if a_on_two_edges || b_on_two_edges || c_on_two_edges {
                        triangles_to_remove.push(t_id);
                        if c_on_two_edges {
                            edges_facing_point.push(Edge { a: tri.a, b: tri.b });
                            edges_to_remove.insert(Edge { a: tri.b, b: tri.c });
                            edges_to_remove.insert(Edge { a: tri.c, b: tri.a });
                            edge_triangle_normals.push(tri.normal);
                        }
                        if b_on_two_edges {
                            edges_facing_point.push(Edge { a: tri.c, b: tri.a });
                            edges_to_remove.insert(Edge { a: tri.a, b: tri.b });
                            edges_to_remove.insert(Edge { a: tri.b, b: tri.c });
                            edge_triangle_normals.push(tri.normal);
                        }
                        if a_on_two_edges {
                            edges_facing_point.push(Edge { a: tri.b, b: tri.c });
                            edges_to_remove.insert(Edge { a: tri.c, b: tri.a });
                            edges_to_remove.insert(Edge { a: tri.a, b: tri.b });
                            edge_triangle_normals.push(tri.normal);
                        }
                    }
                }
                for t_id_remove in triangles_to_remove.iter().rev() {
                    triangles.remove(*t_id_remove);
                }
                for efp in (0..edges_facing_point.len()).rev() {
                    if edges_to_remove.contains(edges_facing_point.get(efp).unwrap()) {
                        edges_facing_point.remove(efp);
                        edge_triangle_normals.remove(efp);
                    }
                }
            }

            for i in 0..edges_facing_point.len() {
                let edg = edges_facing_point[i];
                triangles.push(Triangle {
                    a: edg.a,
                    b: edg.b,
                    c: pointid,
                    normal: edge_triangle_normals[i],
                });
            }
        }
    }
}

/// Calculates the distance of a point to an edge of a triangle. Assumes the point to be in the same plane as the triangle.
/// `point`: the point that lies in the same plane as the edge
/// `edge_a`: first vertex of the edge
/// `edge_b`: second vertex of the edge
/// 'triangle_normal': the normal of the triangle the edge belongs to
fn dist_point_to_edge(
    point: Vector3<f64>,
    edge_a: Vector3<f64>,
    edge_b: Vector3<f64>,
    triangle_normal: Vector3<f64>,
) -> f64 {
    let pa = edge_a - point;
    let edge_ab: Vector3<f64> = edge_b - edge_a;
    let edge_ab_normal = triangle_normal.cross(&edge_ab);
    return edge_ab_normal.dot(&pa);
}

/// Calculates the distance of a point to a line segment.
/// `point`: the point that lies in the same plane as the edge
/// `line_a`: first vertex of the line segment
/// `line_b`: second vertex of the line segment
fn dist_point_to_line_segment(
    point: Vector3<f64>,
    segment_a: Vector3<f64>,
    segment_b: Vector3<f64>,
) -> f64 {
    let ab: Vector3<f64> = segment_b - segment_a;
    let ap: Vector3<f64> = point - segment_a;

    if ap.dot(&ab) <= 0.0 {
        return ap.magnitude();
    }

    let bp = point - segment_b;
    if bp.dot(&ab) >= 0.0 {
        return bp.magnitude();
    }

    return ab.cross(&ap).magnitude() / ab.magnitude();
}

/// Adds the given edge to the set of outer edges. If the given edge is already contained in the set of outer edges it is removed and added to the set of inner edges.
/// `a`: first vertex of the edge
/// `b`: second vertex of the edge
/// `outer_edges`: the set of outer edges
/// `inner_edges`: the set of inner edges
fn add_edge_to_outer_or_inner_edges(
    a: usize,
    b: usize,
    outer_edges: &mut HashSet<Edge>,
    inner_edges: &mut HashSet<Edge>,
) {
    let e = Edge { a, b };
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
    use crate::convexhull;
    use anyhow::Result;
    use pasture_core::{
        containers::{BorrowedBuffer, BorrowedMutBuffer, HashMapBuffer},
        layout::attributes::POSITION_3D,
        layout::PointType,
        nalgebra::Vector3,
    };
    use pasture_derive::PointType;
    use rand::{distributions::Uniform, thread_rng, Rng};

    #[derive(
        PointType, Default, Copy, Clone, Debug, bytemuck::AnyBitPattern, bytemuck::NoUninit,
    )]
    #[repr(C)]
    struct TestPointTypeSmall {
        #[pasture(BUILTIN_POSITION_3D)]
        pub position: Vector3<f64>,
    }

    // Internal Tests
    fn test_normals_for_triangles(
        triangles: &Vec<convexhull::Triangle>,
        normals: &Vec<Vector3<f64>>,
    ) {
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

    fn test_all_points_inside_hull<'a, T: BorrowedBuffer<'a>>(
        buffer: &'a T,
        triangles: &Vec<convexhull::Triangle>,
    ) {
        let position_attribute = buffer
            .point_layout()
            .get_attribute_by_name(POSITION_3D.name())
            .unwrap();
        if position_attribute.datatype() == POSITION_3D.datatype() {
            for point in buffer.view_attribute::<Vector3<f64>>(&POSITION_3D) {
                for t in triangles.iter() {
                    let a: Vector3<f64> = buffer.view_attribute(&POSITION_3D).at(t.a);
                    let pa = a - point;
                    assert!(pa.dot(&t.normal) >= -0.0000001);
                }
            }
        } else {
            let position_view = buffer
                .view_attribute_with_conversion::<Vector3<f64>>(&POSITION_3D)
                .expect("Can't convert POSITION_3D to Vector3<f64>");
            for idx in 0..buffer.len() {
                let point = position_view.at(idx);
                for t in triangles.iter() {
                    let a: Vector3<f64> = position_view.at(t.a);
                    let pa = a - point;
                    assert!(pa.dot(&t.normal) >= -0.0000001);
                }
            }
        };
    }

    #[test]
    fn test_convex_simple_triangle() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(3, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 0.0, 1.0),
        });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 2);
        let normals = vec![Vector3::new(0.0, 1.0, 0.0), Vector3::new(0.0, -1.0, 0.0)];
        test_normals_for_triangles(&result, &normals);

        Ok(())
    }

    #[test]
    fn test_convex_simple_tet_4_points() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(4, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 0.0, 1.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 1.0, 0.0),
        });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 4);
        let normals = vec![
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(1.0, 1.0, 1.0).normalize(),
        ];
        test_normals_for_triangles(&result, &normals);
        test_all_points_inside_hull(&buffer, &result);

        Ok(())
    }

    #[test]
    fn test_convex_simple_tet_5_points() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(5, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 0.0, 1.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, -1.0, -1.0),
        });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 4);
        let normals = vec![
            Vector3::new(1.0, 1.0, 1.0).normalize(),
            Vector3::new(1.0, 1.0, -3.0).normalize(),
            Vector3::new(1.0, -3.0, 1.0).normalize(),
            Vector3::new(-3.0, 1.0, 1.0).normalize(),
        ];
        test_normals_for_triangles(&result, &normals);
        test_all_points_inside_hull(&buffer, &result);

        Ok(())
    }

    #[test]
    fn test_convex_1_point() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(1, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 0.0, 0.0),
        });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].a, 0);

        Ok(())
    }

    #[test]
    fn test_convex_line_2_points() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(2, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 0.0, 0.0),
        });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].a, 0);
        assert_eq!(result[0].b, 1);

        Ok(())
    }

    #[test]
    fn test_convex_line_3_points() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(3, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(2.0, 0.0, 0.0),
        });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].a, 0);
        assert_eq!(result[0].b, 2);

        Ok(())
    }

    #[test]
    fn test_convex_line_4_points() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(4, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(2.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, 0.0, 0.0),
        });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].a, 3);
        assert_eq!(result[0].b, 2);

        Ok(())
    }

    #[test]
    fn test_convex_plane_4_points() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(4, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 0.0, 1.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 0.0, 1.0),
        });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 4);
        let normals = vec![Vector3::new(0.0, 1.0, 0.0), Vector3::new(0.0, -1.0, 0.0)];
        test_normals_for_triangles(&result, &normals);
        test_all_points_inside_hull(&buffer, &result);

        Ok(())
    }

    #[test]
    fn test_convex_2d_point_in_square() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(5, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, -1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, -1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, 1.0, 0.0),
        });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 4);
        let normals = vec![Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.0, 0.0, -1.0)];
        test_normals_for_triangles(&result, &normals);
        test_all_points_inside_hull(&buffer, &result);

        Ok(())
    }

    #[test]
    fn test_convex_2d_point_next_to_square_1() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(5, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, -1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, -1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, 1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(2.0, 0.0, 0.0),
        });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 6);
        let normals = vec![Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.0, 0.0, -1.0)];
        test_normals_for_triangles(&result, &normals);
        test_all_points_inside_hull(&buffer, &result);

        Ok(())
    }

    #[test]
    fn test_convex_2d_point_next_to_square_2() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(5, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, -1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, -1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, 1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 2.0, 0.0),
        });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 6);
        let normals = vec![Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.0, 0.0, -1.0)];
        test_normals_for_triangles(&result, &normals);
        test_all_points_inside_hull(&buffer, &result);

        Ok(())
    }

    #[test]
    fn test_convex_2d_point_next_to_square_3() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(5, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, -1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, -1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, 1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(2.0, 2.0, 0.0),
        });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 4);
        let normals = vec![Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.0, 0.0, -1.0)];
        test_normals_for_triangles(&result, &normals);
        test_all_points_inside_hull(&buffer, &result);

        Ok(())
    }

    #[test]
    fn test_convex_2d_point_next_to_square_4() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(5, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, -1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, -1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, 1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-2.0, 2.0, 0.0),
        });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 4);
        let normals = vec![Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.0, 0.0, -1.0)];
        test_normals_for_triangles(&result, &normals);
        test_all_points_inside_hull(&buffer, &result);

        Ok(())
    }

    #[test]
    fn test_convex_random_1d_points_in_box_create_box_first() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(22, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 0.0, 0.0),
        });
        let mut rng = thread_rng();
        for _ in 0..20 {
            buffer.view_mut().push_point(TestPointTypeSmall {
                position: Vector3::new(rng.sample(Uniform::new(-0.9, 0.9)), 0.0, 0.0),
            });
        }
        let result = convexhull::convex_hull_as_points(&buffer);

        assert_eq!(result.len(), 2);
        assert!(result.contains(&0));
        assert!(result.contains(&1));

        Ok(())
    }

    #[test]
    fn test_convex_random_1d_points_in_box_create_box_last() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(22, TestPointTypeSmall::layout());
        let mut rng = thread_rng();
        for _ in 0..20 {
            buffer.view_mut().push_point(TestPointTypeSmall {
                position: Vector3::new(rng.sample(Uniform::new(-0.9, 0.9)), 0.0, 0.0),
            });
        }
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 0.0, 0.0),
        });
        let result = convexhull::convex_hull_as_points(&buffer);

        assert_eq!(result.len(), 2);
        assert!(result.contains(&20));
        assert!(result.contains(&21));

        Ok(())
    }

    #[test]
    fn test_convex_random_2d_points_in_box_create_box_first() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(24, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, -1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, -1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, 1.0, 0.0),
        });
        let mut rng = thread_rng();
        for _ in 0..20 {
            buffer.view_mut().push_point(TestPointTypeSmall {
                position: Vector3::new(
                    rng.sample(Uniform::new(-0.9, 0.9)),
                    rng.sample(Uniform::new(-0.9, 0.9)),
                    0.0,
                ),
            });
        }
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 4);
        let normals = vec![Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.0, 0.0, -1.0)];
        test_normals_for_triangles(&result, &normals);
        test_all_points_inside_hull(&buffer, &result);

        Ok(())
    }

    #[test]
    fn test_convex_2d_points_in_box_create_box_last_case_1() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(6, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.5, 0.2, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-0.5, -0.3, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, -1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, -1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, 1.0, 0.0),
        });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 4);
        let normals = vec![Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.0, 0.0, -1.0)];
        test_normals_for_triangles(&result, &normals);
        test_all_points_inside_hull(&buffer, &result);

        Ok(())
    }

    #[test]
    fn test_convex_2d_points_in_box_create_box_last_case_2() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(6, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.2, 0.1, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-0.9, 0.3, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, -1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, -1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, 1.0, 0.0),
        });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 4);
        let normals = vec![Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.0, 0.0, -1.0)];
        test_normals_for_triangles(&result, &normals);
        test_all_points_inside_hull(&buffer, &result);

        Ok(())
    }

    #[test]
    fn test_convex_2d_points_in_box_create_box_last_case_3() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(7, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-0.3, -0.3, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.9, -0.4, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.2, 0.1, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, -1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, -1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, 1.0, 0.0),
        });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 4);
        let normals = vec![Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.0, 0.0, -1.0)];
        test_normals_for_triangles(&result, &normals);
        test_all_points_inside_hull(&buffer, &result);

        Ok(())
    }

    #[test]
    fn test_convex_random_points_in_box_create_box_first() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(28, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, -1.0, -1.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, -1.0, 1.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, 1.0, -1.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, 1.0, 1.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, -1.0, -1.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, -1.0, 1.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 1.0, -1.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 1.0, 1.0),
        });
        let mut rng = thread_rng();
        for _ in 0..20 {
            buffer.view_mut().push_point(TestPointTypeSmall {
                position: Vector3::new(
                    rng.sample(Uniform::new(-0.9, 0.9)),
                    rng.sample(Uniform::new(-0.9, 0.9)),
                    rng.sample(Uniform::new(-0.9, 0.9)),
                ),
            });
        }
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 12);
        let normals = vec![
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, 0.0, -1.0),
        ];
        test_normals_for_triangles(&result, &normals);
        test_all_points_inside_hull(&buffer, &result);

        Ok(())
    }

    #[test]
    fn test_convex_random_points_in_box_create_box_last() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(28, TestPointTypeSmall::layout());
        let mut rng = thread_rng();
        for _ in 0..20 {
            buffer.view_mut().push_point(TestPointTypeSmall {
                position: Vector3::new(
                    rng.sample(Uniform::new(-0.9, 0.9)),
                    rng.sample(Uniform::new(-0.9, 0.9)),
                    rng.sample(Uniform::new(-0.9, 0.9)),
                ),
            });
        }
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, -1.0, -1.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, -1.0, 1.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, 1.0, -1.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, 1.0, 1.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, -1.0, -1.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, -1.0, 1.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 1.0, -1.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 1.0, 1.0),
        });
        let result = convexhull::create_convex_hull(&buffer);

        assert_eq!(result.len(), 12);
        let normals = vec![
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, 0.0, -1.0),
        ];
        test_normals_for_triangles(&result, &normals);
        test_all_points_inside_hull(&buffer, &result);

        Ok(())
    }

    #[test]
    fn test_convex_random_points() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(100, TestPointTypeSmall::layout());
        let mut rng = thread_rng();
        for _ in 0..100 {
            buffer.view_mut().push_point(TestPointTypeSmall {
                position: Vector3::new(
                    rng.sample(Uniform::new(-100.0, 100.0)),
                    rng.sample(Uniform::new(-100.0, 100.0)),
                    rng.sample(Uniform::new(-100.0, 100.0)),
                ),
            });
        }
        let result = convexhull::create_convex_hull(&buffer);

        test_all_points_inside_hull(&buffer, &result);

        Ok(())
    }

    // Interface Tests
    #[test]
    fn test_convex_0_point_output_mesh_error() -> Result<()> {
        let buffer = HashMapBuffer::with_capacity(0, TestPointTypeSmall::layout());
        let result = convexhull::convex_hull_as_triangle_mesh(&buffer);

        assert_eq!(
            result.unwrap_err().to_string(),
            "input buffer cointains too few linearly independent points"
        );

        Ok(())
    }

    #[test]
    fn test_convex_1_point_output_mesh_error() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(1, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 0.0, 0.0),
        });
        let result = convexhull::convex_hull_as_triangle_mesh(&buffer);

        assert_eq!(
            result.unwrap_err().to_string(),
            "input buffer cointains too few linearly independent points"
        );

        Ok(())
    }

    #[test]
    fn test_convex_2_point_output_mesh_error() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(2, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 0.0, 0.0),
        });
        let result = convexhull::convex_hull_as_triangle_mesh(&buffer);

        assert_eq!(
            result.unwrap_err().to_string(),
            "input buffer cointains too few linearly independent points"
        );

        Ok(())
    }

    #[test]
    fn test_convex_3_point_output_mesh_error_same_point() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(3, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 0.0, 0.0),
        });
        let result = convexhull::convex_hull_as_triangle_mesh(&buffer);

        assert_eq!(
            result.unwrap_err().to_string(),
            "input buffer cointains too few linearly independent points"
        );

        Ok(())
    }

    #[test]
    fn test_convex_3_point_output_mesh_error_line() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(3, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(2.0, 0.0, 0.0),
        });
        let result = convexhull::convex_hull_as_triangle_mesh(&buffer);

        assert_eq!(
            result.unwrap_err().to_string(),
            "input buffer cointains too few linearly independent points"
        );

        Ok(())
    }

    #[test]
    fn test_convex_3_point_output_mesh_no_error() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(3, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 1.0, 0.0),
        });
        let result = convexhull::convex_hull_as_triangle_mesh(&buffer);
        let result_unwrapped = result.unwrap();

        assert_eq!(result_unwrapped.len(), 2);
        assert!(result_unwrapped.contains(&Vector3::new(0, 1, 2)));
        assert!(result_unwrapped.contains(&Vector3::new(0, 2, 1)));

        Ok(())
    }

    #[test]
    fn test_convex_3_point_output_points_line() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(3, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(2.0, 0.0, 0.0),
        });
        let result = convexhull::convex_hull_as_points(&buffer);

        assert_eq!(result.len(), 2);
        assert!(result.contains(&0));
        assert!(result.contains(&2));

        Ok(())
    }

    #[test]
    fn test_convex_4_point_output_point_in_triangle() -> Result<()> {
        let mut buffer = HashMapBuffer::with_capacity(4, TestPointTypeSmall::layout());
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 0.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(-1.0, -1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(1.0, -1.0, 0.0),
        });
        buffer.view_mut().push_point(TestPointTypeSmall {
            position: Vector3::new(0.0, 1.0, 0.0),
        });
        let result = convexhull::convex_hull_as_points(&buffer);

        assert_eq!(result.len(), 3);
        assert!(result.contains(&1));
        assert!(result.contains(&2));
        assert!(result.contains(&3));

        Ok(())
    }

    #[derive(
        PointType, Default, Copy, Clone, Debug, bytemuck::AnyBitPattern, bytemuck::NoUninit,
    )]
    #[repr(C)]
    struct TestPointTypeNoPositions {
        #[pasture(BUILTIN_INTENSITY)]
        pub intensity: u16,
    }

    #[test]
    #[should_panic(expected = "point buffer contains no position attribute")]
    fn test_convex_no_positions_panic() {
        let mut buffer = HashMapBuffer::with_capacity(1, TestPointTypeNoPositions::layout());
        buffer
            .view_mut()
            .push_point(TestPointTypeNoPositions { intensity: 1 });
        let _result = convexhull::convex_hull_as_triangle_mesh(&buffer);
    }
}
