use std::{fs::File, io::BufWriter};

use anyhow::Result;
use pasture_core::{
    containers::{PerAttributeVecPointStorage, OwningPointBuffer},
    layout::PointType,
    math::AABB,
    nalgebra::{Point3, Vector3},
};
use pasture_derive::PointType;
use pasture_io::{
    base::PointWriter,
    tiles3d::{BoundingVolume, PntsWriter, Refinement, RootTileset, Tileset, TilesetBuilder},
};

#[derive(Copy, Clone, Debug, PointType)]
#[repr(C, packed)]
struct Point {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f32>,
    #[pasture(BUILTIN_COLOR_RGB)]
    pub color: Vector3<u8>,
    #[pasture(BUILTIN_NORMAL)]
    pub normal: Vector3<f32>,
}

/// Generates a bunch of points with a nice pattern
fn gen_points() -> PerAttributeVecPointStorage {
    let points_per_axis = 64;
    let height = 4.0;
    let mut buffer = PerAttributeVecPointStorage::with_capacity(points_per_axis, Point::layout());

    let z_at = |x: f32, y: f32| -> f32 {
        let z1 = (x as f32 / points_per_axis as f32 * 8.0).sin();
        let z2 = (y as f32 / points_per_axis as f32 * 12.0).sin();
        z1 * z2 * height
    };

    for y in 0..points_per_axis {
        for x in 0..points_per_axis {
            let z = z_at(x as f32, y as f32);

            let w = z_at((x - 1) as f32, y as f32);
            let e = z_at((x + 1) as f32, y as f32);
            let n = z_at(x as f32, (y - 1) as f32);
            let s = z_at(x as f32, (y + 1) as f32);

            let dnx = e - w;
            let dny = s - n;

            let normal: Vector3<f32> = (Vector3::new(1.0, 0.0, 0.0) * dnx
                + Vector3::new(0.0, 1.0, 0.0) * dny
                + Vector3::new(0.0, 0.0, 1.0))
            .normalize();

            buffer.push_point(Point {
                position: Vector3::new(x as f32, y as f32, z),
                color: Vector3::new((x * 4) as u8, (y * 4) as u8, 0),
                normal,
            });
        }
    }

    buffer
}

fn create_tileset_for_points() -> RootTileset {
    // Using some approximate bounds here, but you could also use `calculate_bounds` from `pasture-algorithms` to
    // get a tight-fitting bounding box
    let bounds = AABB::from_min_max(Point3::new(0.0, 0.0, -4.0), Point3::new(63.0, 63.0, 4.0));

    // We have a single tileset in this example, which has the given bounds, references the .pnts file
    // and has some extra parameters that are required for the visualization in e.g. Cesium
    let tileset: Tileset = TilesetBuilder::new()
        .bounding_volume(BoundingVolume::Box(bounds.into()))
        .content("points.pnts".into(), None)
        .geometric_error(16.0)
        .refinement(Refinement::Add)
        .into();

    RootTileset {
        geometric_error: 16.0,
        root: tileset,
        ..Default::default()
    }
}

fn main() -> Result<()> {
    // 3D Tiles is made up of 'tilesets', which are stored inside a 'tileset.json' file. Each tileset represents
    // a piece of geometry (a point cloud in our case) associated with some meta-information, such as the bounding
    // volume of the geometry. Point clouds themselves are stored in the '.pnts' format with 3D Tiles, so we first
    // write some points into the points.pnts file, then create a corresponding tileset and write it into the tileset.json
    // file. The tileset internally references the points.pnts file

    let points = gen_points();
    {
        let mut writer = PntsWriter::from_write_and_layout(
            BufWriter::new(File::create("points.pnts")?),
            Point::layout(),
        );
        writer.write(&points)?;
    }

    let tileset = create_tileset_for_points();
    serde_json::to_writer_pretty(BufWriter::new(File::create("tileset.json")?), &tileset)?;

    Ok(())
}
