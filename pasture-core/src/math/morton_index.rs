use nalgebra::{Point3, Vector3};
use std::fmt::Display;

use super::{expand_bits_by_3, AABB};

/// 64-bit 3D Morton index
#[derive(Debug, PartialEq, Eq, Copy, Clone, PartialOrd)]
pub struct MortonIndex64 {
    index: u64,
}

impl MortonIndex64 {
    pub const LEVELS: usize = 21;

    /// Creates a `MortonIndex64` from the given raw index
    ///
    /// Example:
    /// ```
    /// # use pasture_core::math::*;
    /// let morton_index = MortonIndex64::from_raw(1234);
    /// ```
    pub fn from_raw(index: u64) -> Self {
        Self { index }
    }

    /// Creates a `MortonIndex64` that encodes the given octree octants. The order of the octant numbering is unspecified, but
    /// each value in `octants` must be in [0;7]. Octants are encoded little-endian, i.e. the first octant is encoded in the
    /// most significant bits of the Morton index. `octants` may contain at most `MortonIndex64::LEVELS` entries.
    ///
    /// Example:
    /// ```
    /// # use pasture_core::math::*;
    /// // Create a Morton index for the octants 1->2->4 (octant 4 of octant 2 of octant 1 of the root node)
    /// let morton_index = MortonIndex64::from_octants(&[1,2,4]);
    /// assert_eq!(1, morton_index.get_octant_at_level(0));
    /// assert_eq!(2, morton_index.get_octant_at_level(1));
    /// assert_eq!(4, morton_index.get_octant_at_level(2));
    /// ```
    pub fn from_octants(octants: &[u8]) -> Self {
        if octants.len() > Self::LEVELS {
            panic!(
                "MortonIndex64::from_octants requires at most {} octant indices",
                Self::LEVELS
            );
        }

        let mut index = Self::from_raw(0);
        for (level, &octant) in octants.iter().enumerate() {
            index.set_octant_at_level(octant, level as u8);
        }
        index
    }

    /// Computes a `MortonIndex64` for the given `point` within `bounds`. Octant order is ZYX little-endian (MSB encodes Z, LSB encodes X)
    pub fn from_point_in_bounds(point: &Point3<f64>, bounds: &AABB<f64>) -> Self {
        let normalized_extent = (2.0_f64.powf(Self::LEVELS as f64)) / bounds.extent().x;
        let normalized_point = (point - bounds.min()).component_mul(&Vector3::new(
            normalized_extent,
            normalized_extent,
            normalized_extent,
        ));

        let max_index = (1_u64 << Self::LEVELS) - 1;
        let grid_index_x = u64::min(normalized_point.x as u64, max_index);
        let grid_index_y = u64::min(normalized_point.y as u64, max_index);
        let grid_index_z = u64::min(normalized_point.z as u64, max_index);

        let x_bits = expand_bits_by_3(grid_index_x);
        let y_bits = expand_bits_by_3(grid_index_y);
        let z_bits = expand_bits_by_3(grid_index_z);

        let index = (z_bits << 2) | (y_bits << 1) | x_bits;
        Self { index }
    }

    /// Returns the raw value of the associated `MortonIndex64`
    /// ```
    /// # use pasture_core::math::*;
    /// let morton_index = MortonIndex64::from_raw(1234);
    /// assert_eq!(1234, morton_index.index());
    /// ```
    pub fn index(&self) -> u64 {
        self.index
    }

    /// Sets the octant index at the specified level to `octant`. `level` must be less than `MortonIndex64::LEVELS` and
    /// `octant` must be in [0;7]. Octant numbering order is unspecified.
    ///
    /// Example:
    /// ```
    /// # use pasture_core::math::*;
    /// let mut morton_index = MortonIndex64::from_raw(0);
    /// morton_index.set_octant_at_level(3, 1);
    /// assert_eq!(3, morton_index.get_octant_at_level(1));
    /// ```
    pub fn set_octant_at_level(&mut self, octant: u8, level: u8) {
        if octant > 7 {
            panic!(
                "MortonIndex64::set_octant_at_level: Octant index {} is out of bounds!",
                octant
            );
        }
        if level as usize >= Self::LEVELS {
            panic!(
                "MortonIndex64::set_octant_at_level: Level {} is out of bounds!",
                level
            );
        }
        self.set_octant_at_level_unchecked(octant, level);
    }

    /// Unchecked version of `set_octant_at_level`
    pub fn set_octant_at_level_unchecked(&mut self, octant: u8, level: u8) {
        let bit_shift = (Self::LEVELS - level as usize - 1) * 3;
        let clear_bits_mask = !(0b111 << bit_shift);
        self.index = (self.index & clear_bits_mask) | ((octant as u64) << bit_shift);
    }

    /// Returns the octant index at the specified level in the associated `MortonIndex64`. `level` must be less than `MortonIndex64::LEVELS`
    pub fn get_octant_at_level(&self, level: u8) -> u8 {
        if level as usize >= Self::LEVELS {
            panic!(
                "MortonIndex64::get_octant_at_level: Level {} is out of bounds!",
                level
            );
        }
        self.get_octant_at_level_unchecked(level)
    }

    /// Unchecked version of 'get_octant_at_level'
    pub fn get_octant_at_level_unchecked(&self, level: u8) -> u8 {
        let bit_shift = (Self::LEVELS - level as usize - 1) * 3;
        ((self.index >> bit_shift) & 0b111) as u8
    }
}

impl Default for MortonIndex64 {
    fn default() -> Self {
        Self { index: 0 }
    }
}

impl Ord for MortonIndex64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.index.cmp(&other.index)
    }
}

/// 3D Morton index with a dynamic depth
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct DynamicMortonIndex {
    octants: Vec<u8>,
}

impl DynamicMortonIndex {
    /// Creates a new `DynamicMortonIndex` from the given octants. The order of the octant numbering is unspecified, but
    /// each value in `octants` must be in [0;7].The order of the indices goes from shallowest
    /// node to deepest node.
    ///
    /// Example:
    /// ```
    /// # use pasture_core::math::*;
    /// let mut dynamic_morton_index = DynamicMortonIndex::from_octants(&[1,2,4]);
    /// assert_eq!(3, dynamic_morton_index.depth());
    /// ```
    pub fn from_octants(octants: &[u8]) -> Self {
        if octants.iter().any(|&octant| octant > 7) {
            panic!("DynamicMortonIndex::from_octants: Octant indices must be in [0;7]!");
        }
        Self::from_octants_unchecked(octants)
    }

    /// Unchecked version of `from_octants`
    pub fn from_octants_unchecked(octants: &[u8]) -> Self {
        Self {
            octants: octants.to_vec(),
        }
    }

    /// Returns the depth of the associated `DynamicMortonIndex`. A depth of zero corresponds to the root node of an octree
    pub fn depth(&self) -> usize {
        self.octants.len()
    }

    /// Returns a reference to the octant indices of associated `DynamicMortonIndex`. The order of the indices goes from shallowest
    /// node to deepest node.
    pub fn octants(&self) -> &[u8] {
        self.octants.as_slice()
    }

    /// Adds a new octant to the associated `DynamicMortonIndex`. `octant` must be in [0;7]
    ///
    /// Example:
    /// ```
    /// # use pasture_core::math::*;
    /// let mut dynamic_morton_index = DynamicMortonIndex::from_octants(&[1,2,4]);
    /// dynamic_morton_index.add_octant(6);
    /// assert_eq!(4, dynamic_morton_index.depth());
    /// assert_eq!(&[1,2,4,6], dynamic_morton_index.octants());
    /// ```
    pub fn add_octant(&mut self, octant: u8) {
        if octant > 7 {
            panic!("DynamicMortonIndex::add_octant: Octant index must be in [0;7]!");
        }
        self.octants.push(octant);
    }

    /// Tries to truncate the associated `DynamicMortonIndex` to the given depth. If `depth` is greater than the
    /// current depth, this operation fails and returns `None`.
    ///
    /// Example:
    /// ```
    /// # use pasture_core::math::*;
    /// let mut dynamic_morton_index = DynamicMortonIndex::from_octants(&[1,2,4]);
    /// let truncated_index = dynamic_morton_index.truncate_to_depth(1);
    /// assert_eq!(Some(DynamicMortonIndex::from_octants(&[1])), truncated_index);
    ///
    /// let invalid_index = dynamic_morton_index.truncate_to_depth(10);
    /// assert_eq!(None, invalid_index);
    /// ```
    pub fn truncate_to_depth(&self, new_depth: usize) -> Option<Self> {
        if new_depth > self.depth() {
            return None;
        }

        Some(Self::from_octants_unchecked(&self.octants[0..new_depth]))
    }

    /// Returns an iterator over the octants in the associated `DynamicMortonIndex`
    pub fn iter(&self) -> impl Iterator<Item = &u8> {
        self.octants.iter()
    }

    /// Returns a mutable iterator over the octants in the associated `DynamicMortonIndex`
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut u8> {
        self.octants.iter_mut()
    }
}

impl Default for DynamicMortonIndex {
    fn default() -> Self {
        Self { octants: vec![] }
    }
}

impl Display for DynamicMortonIndex {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(fmt, "{:?}", self.octants)
    }
}

impl From<MortonIndex64> for DynamicMortonIndex {
    fn from(morton_index: MortonIndex64) -> Self {
        Self {
            octants: (0..21)
                .map(|level| morton_index.get_octant_at_level_unchecked(level))
                .collect(),
        }
    }
}
