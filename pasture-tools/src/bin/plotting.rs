use pasture_core::{
    math::expand_bits_by_3,
    math::reverse_bits,
    math::{MortonIndex64, AABB},
    nalgebra::Point3,
    nalgebra::Vector3,
};
use plotters::coord::types::RangedCoordf32;
use plotters::prelude::*;
use rand::{rngs::SmallRng, Rng, SeedableRng};

fn reversed_morton_index(point: &Point3<f64>, bounds: &AABB<f64>) -> MortonIndex64 {
    let normalized_extent = (2.0_f64.powf(21 as f64)) / bounds.extent().x;
    let normalized_point = (point - bounds.min()).component_mul(&Vector3::new(
        normalized_extent,
        normalized_extent,
        normalized_extent,
    ));

    let max_index = (1_u64 << 21) - 1;
    let grid_index_x = u64::min(normalized_point.x as u64, max_index);
    let grid_index_y = u64::min(normalized_point.y as u64, max_index);
    let grid_index_z = u64::min(normalized_point.z as u64, max_index);

    let x_bits = expand_bits_by_3(grid_index_x);
    let y_bits = expand_bits_by_3(grid_index_y);
    let z_bits = expand_bits_by_3(grid_index_z);

    let index = (z_bits << 2) | (y_bits << 1) | x_bits;
    // reverse the bits of the index, so that the LSB becomes the MSB and vice versa
    MortonIndex64::from_raw(reverse_bits(index))
}

fn gen_random_points(count: usize) -> Vec<Point3<f64>> {
    //let mut rng = thread_rng();
    let mut rng = SmallRng::seed_from_u64(14823937);

    (0..count)
        .map(|_| Point3::<f64>::new(rng.gen(), rng.gen(), 0.0))
        .collect()
}

fn sample_points() -> Vec<(Point3<f64>, MortonIndex64)> {
    let points = gen_random_points(64);
    let bounds =
        AABB::from_min_max_unchecked(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
    let mut points_and_morton_indices = points
        .iter()
        .map(|p| (*p, reversed_morton_index(p, &bounds)))
        .collect::<Vec<_>>();

    points_and_morton_indices.sort_by(|a, b| a.1.cmp(&b.1));

    points_and_morton_indices
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let points = sample_points();

    let root = BitMapBackend::new("plot_morton_rev.png", (512, 512)).into_drawing_area();

    root.fill(&RGBColor(240, 200, 200))?;

    let root = root.apply_coord_spec(Cartesian2d::<RangedCoordf32, RangedCoordf32>::new(
        -0.1f32..1.1f32,
        -0.1f32..1.1f32,
        (0..512, 0..512),
    ));

    let dot_and_label = |x: f32, y: f32, idx: usize| {
        return EmptyElement::at((x, y))
            + Circle::new((0, 0), 3, ShapeStyle::from(&BLACK).filled())
            + Text::new(
                format!("({})", idx),
                (10, 0),
                ("sans-serif", 15.0).into_font(),
            );
    };

    for (idx, (p, _)) in points.iter().enumerate() {
        root.draw(&dot_and_label(p.x as f32, p.y as f32, idx))?;
    }

    // root.draw(&dot_and_label(0.5, 0.6))?;
    // root.draw(&dot_and_label(0.25, 0.33))?;
    // root.draw(&dot_and_label(0.8, 0.8))?;
    Ok(())
}
