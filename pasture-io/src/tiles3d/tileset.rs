use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use pasture_core::nalgebra::{Matrix4, Vector3};

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

#[derive(Clone, Serialize, Deserialize, Default, PartialEq, Debug)]
pub struct TilesetContent {
    #[serde(rename = "boundingVolume")]
    pub bounding_volume: BoundingVolume,
    pub uri: String,
}

/// 3D Tiles tileset primitive
#[derive(Clone, Serialize, Deserialize, Default, PartialEq, Debug)]
pub struct Tileset {
    #[serde(rename = "geometricError")]
    pub geometric_error: f64,
    #[serde(rename = "refine")]
    pub refinement: Option<Refinement>,
    #[serde(rename = "boundingVolume")]
    pub bounding_volume: BoundingVolume,
    #[serde(rename = "viewerRequestVolume")]
    pub viewer_request_volume: Option<BoundingVolume>,
    pub content: TilesetContent,
    pub transform: Option<Matrix4<f64>>,
    #[serde(default)]
    pub children: Vec<Tileset>,
}

#[derive(Clone, Serialize, Deserialize, Default, PartialEq, Eq, Debug)]
pub struct TilesetAssetInfo {
    pub version: String,
    #[serde(rename = "tilesetVersion")]
    pub tileset_version: String,
}

/// Property inside the root tileset.json
/// TODO What are valid types for minimum/maximum?
#[derive(Clone, Serialize, Deserialize, Default, PartialEq, Debug)]
pub struct TilesetProperty {
    pub minimum: f64,
    pub maximum: f64,
}

#[derive(Clone, Serialize, Deserialize, Default, PartialEq, Debug)]
pub struct RootTileset {
    pub asset: TilesetAssetInfo,
    pub properties: HashMap<String, TilesetProperty>,
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
        let mut tileset: RootTileset = Default::default();
        tileset.asset = TilesetAssetInfo {
            version: "1.0".into(),
            tileset_version: "e575c6f1-a45b-420a-b172-6449fa6e0a59".into(),
        };
        tileset.properties.insert(
            "Height".into(),
            TilesetProperty {
                minimum: 1.0,
                maximum: 241.6,
            },
        );
        tileset.geometric_error = 494.509;

        let inner_tileset = Tileset {
            bounding_volume: BoundingVolume::Region(BoundingRegion::new(
                -0.000568296657741,
                0.8987233516605286,
                0.0001164658209855,
                0.8990603398325034,
                0.0,
                241.6,
            )),
            geometric_error: 268.378,
            refinement: Some(Refinement::Add),
            content: TilesetContent {
                bounding_volume: BoundingVolume::Region(BoundingRegion::new(
                    -0.000400169090897,
                    0.8988700116775743,
                    0.0001009672972278,
                    0.8989625664878067,
                    0.0,
                    241.6,
                )),
                uri: "0/0/0.b3dm".into(),
            },
            ..Default::default()
        };
        let mut root_tileset = inner_tileset.clone();
        root_tileset.children = vec![inner_tileset];

        tileset.root = root_tileset;

        tileset
    }

    #[test]
    fn test_deser_tileset() {
        let tileset_json_path = get_test_tileset_path();
        let tileset: RootTileset = serde_json::from_reader(
            File::open(&tileset_json_path).expect("Could not open test tileset.json"),
        )
        .expect("Error while deserializing tileset JSON");

        let example_tileset = get_example_tileset();
        assert_eq!(example_tileset, tileset);
    }
}
