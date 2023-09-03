use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use pasture_core::{
    math::AABB,
    nalgebra::{Matrix4, Vector3},
};

/// 3D Tiles refinement strategy
#[derive(Copy, Clone, Serialize, Deserialize, PartialEq, Eq, Debug)]
pub enum Refinement {
    /// Refine by replacing the tileset
    #[serde(rename = "REPLACE")]
    Replace,
    /// Refine by adding the tileset to existing tilesets
    #[serde(rename = "ADD")]
    Add,
}

/// 3D Tiles bounding region
#[derive(Copy, Clone, Serialize, Deserialize, Default, PartialEq, Debug)]
pub struct BoundingRegion([f64; 6]);

impl BoundingRegion {
    pub fn new(
        west: f64,
        south: f64,
        east: f64,
        north: f64,
        min_height: f64,
        max_height: f64,
    ) -> Self {
        Self([west, south, east, north, min_height, max_height])
    }

    pub fn west(&self) -> f64 {
        self.0[0]
    }

    pub fn south(&self) -> f64 {
        self.0[1]
    }

    pub fn east(&self) -> f64 {
        self.0[2]
    }

    pub fn north(&self) -> f64 {
        self.0[3]
    }

    pub fn min_height(&self) -> f64 {
        self.0[4]
    }

    pub fn max_height(&self) -> f64 {
        self.0[5]
    }
}

/// 3D Tiles oriented bounding box
#[derive(Copy, Clone, Serialize, Deserialize, Default, PartialEq, Debug)]
pub struct BoundingBox([f64; 12]);

impl BoundingBox {
    pub fn new(center: Vector3<f64>, x: Vector3<f64>, y: Vector3<f64>, z: Vector3<f64>) -> Self {
        Self([
            center.x, center.y, center.z, x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z,
        ])
    }

    pub fn center(&self) -> Vector3<f64> {
        Vector3::new(self.0[0], self.0[1], self.0[2])
    }

    pub fn x(&self) -> Vector3<f64> {
        Vector3::new(self.0[3], self.0[4], self.0[5])
    }

    pub fn y(&self) -> Vector3<f64> {
        Vector3::new(self.0[6], self.0[7], self.0[8])
    }

    pub fn z(&self) -> Vector3<f64> {
        Vector3::new(self.0[9], self.0[10], self.0[11])
    }
}

impl From<AABB<f64>> for BoundingBox {
    fn from(aabb: AABB<f64>) -> Self {
        (&aabb).into()
    }
}

impl From<&AABB<f64>> for BoundingBox {
    fn from(aabb: &AABB<f64>) -> Self {
        let half_extent = aabb.extent() * 0.5;
        Self::new(
            aabb.center().coords,
            Vector3::new(half_extent.x, 0.0, 0.0),
            Vector3::new(0.0, half_extent.y, 0.0),
            Vector3::new(0.0, 0.0, half_extent.z),
        )
    }
}

/// 3D Tiles bounding sphere
#[derive(Copy, Clone, Serialize, Deserialize, Default, PartialEq, Debug)]
pub struct BoundingSphere([f64; 4]);

impl BoundingSphere {
    pub fn new(center: Vector3<f64>, radius: f64) -> Self {
        Self([center.x, center.y, center.z, radius])
    }

    pub fn center(&self) -> Vector3<f64> {
        Vector3::new(self.0[0], self.0[1], self.0[2])
    }

    pub fn radius(&self) -> f64 {
        self.0[3]
    }
}

/// 3D Tiles bounding volume
#[derive(Copy, Clone, Serialize, Deserialize, PartialEq, Debug)]
pub enum BoundingVolume {
    #[serde(rename = "region")]
    Region(BoundingRegion),
    #[serde(rename = "box")]
    Box(BoundingBox),
    #[serde(rename = "sphere")]
    Sphere(BoundingSphere),
}

impl Default for BoundingVolume {
    fn default() -> Self {
        Self::Region(Default::default())
    }
}

/// Content of a `Tileset`. This refers to the file that contains the actual geometry, in the case of pasture
/// this will mostly be `.pnts` files.
#[derive(Clone, Serialize, Deserialize, Default, PartialEq, Debug)]
pub struct TilesetContent {
    #[serde(rename = "boundingVolume", skip_serializing_if = "Option::is_none")]
    pub bounding_volume: Option<BoundingVolume>,
    pub uri: String,
}

/// 3D Tiles tileset primitive
#[derive(Clone, Serialize, Deserialize, Default, PartialEq, Debug)]
pub struct Tileset {
    #[serde(rename = "geometricError")]
    pub geometric_error: f64,
    #[serde(rename = "refine", skip_serializing_if = "Option::is_none")]
    pub refinement: Option<Refinement>,
    #[serde(rename = "boundingVolume")]
    pub bounding_volume: BoundingVolume,
    #[serde(
        rename = "viewerRequestVolume",
        skip_serializing_if = "Option::is_none"
    )]
    pub viewer_request_volume: Option<BoundingVolume>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<TilesetContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transform: Option<Matrix4<f64>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<Tileset>,
}

/// Builder for `Tileset` structures
#[derive(Default)]
pub struct TilesetBuilder {
    tileset: Tileset,
}

impl TilesetBuilder {
    /// Sets the geometric error of the `Tileset` to the given value
    /// # Panics
    /// If the `geometric_error` is less than 0
    pub fn geometric_error(mut self, geometric_error: f64) -> Self {
        if geometric_error < 0.0 {
            panic!("Geometric error must be >= 0");
        }
        self.tileset.geometric_error = geometric_error;
        self
    }

    /// Sets the refinement strategy to use for the `Tileset`
    pub fn refinement(mut self, refinement: Refinement) -> Self {
        self.tileset.refinement = Some(refinement);
        self
    }

    /// Sets the given `bounding_volume` for the `Tileset`
    pub fn bounding_volume(mut self, bounding_volume: BoundingVolume) -> Self {
        self.tileset.bounding_volume = bounding_volume;
        self
    }

    /// Sets the given `viewer_request_volume` for the `Tileset`
    pub fn viewer_request_volume(mut self, viewer_request_volume: BoundingVolume) -> Self {
        self.tileset.viewer_request_volume = Some(viewer_request_volume);
        self
    }

    /// Sets the given content for the `Tileset`. `uri` is mandatory but the `bounding_volume` is optional. If it is not
    /// set, the bounding volume of the `Tileset` will be used
    pub fn content(mut self, uri: String, bounding_volume: Option<BoundingVolume>) -> Self {
        self.tileset.content = Some(TilesetContent {
            bounding_volume,
            uri,
        });
        self
    }

    /// Sets a transformation matrix for the `Tileset`
    pub fn transform(mut self, transform: Matrix4<f64>) -> Self {
        self.tileset.transform = Some(transform);
        self
    }

    /// Adds the given `Tileset` as a child of this `Tileset`
    pub fn add_child(mut self, child: Tileset) -> Self {
        self.tileset.children.push(child);
        self
    }

    /// Adds the given `Tilesets` as children of this `Tileset`
    pub fn add_children<I: IntoIterator<Item = Tileset>>(mut self, children: I) -> Self {
        self.tileset.children.extend(children);
        self
    }
}

impl From<TilesetBuilder> for Tileset {
    fn from(val: TilesetBuilder) -> Self {
        val.tileset
    }
}

/// Version information of a `Tileset`. Identifies both the 3D Tiles version that is used, as well as
/// an optional version of the `Tileset` itself
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq, Debug)]
pub struct TilesetAssetInfo {
    /// 3D Tiles version that the `Tileset` uses. This defaults to version `"1.0"`
    pub version: String,
    #[serde(rename = "tilesetVersion", skip_serializing_if = "Option::is_none")]
    pub tileset_version: Option<String>,
}

impl Default for TilesetAssetInfo {
    fn default() -> Self {
        Self {
            version: "1.0".into(),
            tileset_version: None,
        }
    }
}

/// Property inside the root tileset.json
/// TODO What are valid types for minimum/maximum?
#[derive(Clone, Serialize, Deserialize, Default, PartialEq, Debug)]
pub struct TilesetProperty {
    pub minimum: f64,
    pub maximum: f64,
}

/// Root tileset within a `tileset.json` file
#[derive(Clone, Serialize, Deserialize, Default, PartialEq, Debug)]
pub struct RootTileset {
    pub asset: TilesetAssetInfo,
    pub properties: Option<HashMap<String, TilesetProperty>>,
    #[serde(rename = "geometricError")]
    pub geometric_error: f64,
    pub root: Tileset,
}

#[cfg(test)]
mod tests {
    use std::{fs::File, path::PathBuf};

    use super::*;

    fn get_test_tileset_path() -> PathBuf {
        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("resources/test/tileset.json");
        test_file_path
    }

    /// Get example tileset which corresponds to the tileset.json file in resources/test
    fn get_example_tileset() -> RootTileset {
        let mut tileset = RootTileset {
            asset: TilesetAssetInfo {
                version: "1.0".into(),
                tileset_version: Some("e575c6f1-a45b-420a-b172-6449fa6e0a59".into()),
            },
            ..Default::default()
        };
        tileset.properties = Some(HashMap::new());
        tileset.properties.as_mut().unwrap().insert(
            "Height".into(),
            TilesetProperty {
                minimum: 1.0,
                maximum: 241.6,
            },
        );
        tileset.geometric_error = 494.509;

        let inner_tileset: Tileset = TilesetBuilder::default()
            .bounding_volume(BoundingVolume::Region(BoundingRegion::new(
                -0.000568296657741,
                0.8987233516605286,
                0.0001164658209855,
                0.8990603398325034,
                0.0,
                241.6,
            )))
            .geometric_error(268.378)
            .refinement(Refinement::Add)
            .content(
                "0/0/0.b3dm".into(),
                Some(BoundingVolume::Region(BoundingRegion::new(
                    -0.000400169090897,
                    0.8988700116775743,
                    0.0001009672972278,
                    0.8989625664878067,
                    0.0,
                    241.6,
                ))),
            )
            .into();

        let mut root_tileset = inner_tileset.clone();
        root_tileset.children = vec![inner_tileset];

        tileset.root = root_tileset;

        tileset
    }

    #[test]
    fn test_deser_tileset() {
        let tileset_json_path = get_test_tileset_path();
        let tileset: RootTileset = serde_json::from_reader(
            File::open(tileset_json_path).expect("Could not open test tileset.json"),
        )
        .expect("Error while deserializing tileset JSON");

        let example_tileset = get_example_tileset();
        assert_eq!(example_tileset, tileset);
    }

    #[test]
    fn test_ser_deser_tileset() {
        let example_tileset = get_example_tileset();
        let as_json =
            serde_json::to_string(&example_tileset).expect("Error while serializing tileset JSON");
        let tileset_again =
            serde_json::from_str(as_json.as_str()).expect("Error while deserializing tileset JSON");
        assert_eq!(example_tileset, tileset_again);
    }
}
