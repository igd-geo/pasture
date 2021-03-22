use nalgebra::{Point3, Vector3};
use std::{char::from_digit, convert::TryFrom};
use std::{convert::TryInto, fmt::Display};

use super::{expand_bits_by_3, AABB};

/// Defines different ways of turning a Morton index into a string
pub enum MortonIndexNaming {
    /// Concatenate the octant numbers of all octants in the Morton index to form a string. This yields strings of
    /// like `0725463`, encoding octant `0` below root, `7` below `0` and so on. With this naming, a Morton index
    /// representing the root node yields an empty string
    AsOctantConcatenation,
    /// Like `AsOctantConcatenation`, but appends a `r` to the front of the string, so that a Morton index representing
    /// the root node yields the string `r`
    AsOctantConcatenationWithRoot,
    /// Encodes the X, Y, and Z coordinates within the grid that the Morton index describes, together with the depth
    /// of the Morton index. This yields strings of the form `4-15-8-3` for a Morton index at level 4 that represents
    /// the grid cell `(15,8,3)` at this level. A Morton index representing the root node always yields the string
    /// `0-0-0-0`, representing grid cell `(0,0,0)` at level 0. *Beware:* The `MortonIndex64` type always stores a node
    /// of depth 21, so a default-constructed `MortonIndex64` will yield `21-0-0-0` instead of `0-0-0-0`!
    AsGridCoordinates,
}

/// Error type for an octant index that is out of bounds
#[derive(Debug)]
pub struct OctantIndexOutOfBoundsError {
    wrong_index: u8,
}

impl OctantIndexOutOfBoundsError {
    pub fn new(wrong_index: u8) -> Self {
        Self { wrong_index }
    }
}

impl Display for OctantIndexOutOfBoundsError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(fmt, "Octant index {} is out of bounds!", self.wrong_index)
    }
}

impl std::error::Error for OctantIndexOutOfBoundsError {}

/// Error for a `DynamicMortonIndex` that is too large (i.e. it has too many levels) to be converted
/// into one of the fixed-size Morton index types (e.g. `MortonIndex64`, `MortonIndex64WithDepth`)
#[derive(Debug)]
pub struct DynamicMortonIndexTooLargeError {
    dynamic_morton_index: DynamicMortonIndex,
}

impl DynamicMortonIndexTooLargeError {
    pub fn new(dynamic_morton_index: DynamicMortonIndex) -> Self {
        Self {
            dynamic_morton_index,
        }
    }
}

impl Display for DynamicMortonIndexTooLargeError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            fmt,
            "DynamicMortonIndex {} has too many levels to be converted to a 64-bit Morton index!",
            self.dynamic_morton_index
                .to_string(MortonIndexNaming::AsOctantConcatenationWithRoot)
        )
    }
}

impl std::error::Error for DynamicMortonIndexTooLargeError {}

/// Wrapper type around an u8 that encodes the index of an octant within an octree. Since there are only 8 possible octants,
/// this type is only valid for values in `[0;7]` and acts as a safeguard to prevent out-of-bounds errors when passing around
/// and setting octant indices
#[derive(Debug, Copy, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub struct Octant(u8);

impl Octant {
    /// Constant for the octant with index 0
    pub const ZERO: Octant = Octant(0);
    /// Constant for the octant with index 1
    pub const ONE: Octant = Octant(1);
    /// Constant for the octant with index 2
    pub const TWO: Octant = Octant(2);
    /// Constant for the octant with index 3
    pub const THREE: Octant = Octant(3);
    /// Constant for the octant with index 4
    pub const FOUR: Octant = Octant(4);
    /// Constant for the octant with index 5
    pub const FIVE: Octant = Octant(5);
    /// Constant for the octant with index 6
    pub const SIX: Octant = Octant(6);
    /// Constant for the octant with index 7
    pub const SEVEN: Octant = Octant(7);

    /// The index of the associated `Octant` as an `u8` value
    pub fn index(&self) -> u8 {
        self.0
    }
}

impl From<Octant> for u8 {
    fn from(octant: Octant) -> Self {
        octant.0
    }
}

impl From<&Octant> for u8 {
    fn from(octant: &Octant) -> Self {
        octant.0
    }
}

impl From<&mut Octant> for u8 {
    fn from(octant: &mut Octant) -> Self {
        octant.0
    }
}

impl TryFrom<u8> for Octant {
    type Error = OctantIndexOutOfBoundsError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        if value > 7 {
            return Err(OctantIndexOutOfBoundsError::new(value));
        }
        Ok(Self(value))
    }
}

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

    /// Creates a `MortonIndex64` that encodes the given octree octants. The order of the octant numbering is unspecified, i.e.
    /// there is no assumption made which octant in 3D space has index 0, 1, etc.
    /// Octants are encoded little-endian, i.e. the first octant is encoded in the
    /// most significant bits of the Morton index. `octants` may contain at most `MortonIndex64::LEVELS` entries. If it contains less
    /// than `MortonIndex64::LEVELS` entries, the remaining octants will be zero.
    ///
    /// # Example:
    /// ```
    /// # use pasture_core::math::*;
    /// // Create a Morton index for the octants 1->2->4 (octant 4 of octant 2 of octant 1 of the root node)
    /// let morton_index = MortonIndex64::from_octants(&[Octant::ONE, Octant::TWO, Octant::FOUR]);
    /// assert_eq!(Some(Octant::ONE), morton_index.get_octant_at_level(1));
    /// assert_eq!(Some(Octant::TWO), morton_index.get_octant_at_level(2));
    /// assert_eq!(Some(Octant::FOUR), morton_index.get_octant_at_level(3));
    /// ```
    pub fn from_octants(octants: &[Octant]) -> Self {
        if octants.len() > Self::LEVELS {
            panic!(
                "MortonIndex64::from_octants requires at most {} octant indices",
                Self::LEVELS
            );
        }

        let mut index = Self::from_raw(0);
        for (level, octant) in octants.iter().enumerate() {
            index.set_octant_at_level((level + 1) as u8, *octant);
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

    /// Sets the octant index at the specified `level` to `octant`. `level` must be in `[1;MortonIndex64::LEVELS]`. This may seem surprising, however
    /// in an octree level 0 represents the root node and there is only one node at this level. The first level that has octants with respect to its
    /// parent node thus is level 1!
    /// `octant` must be in [0;7]. Octant numbering order is unspecified.
    ///
    /// Example:
    /// ```
    /// # use pasture_core::math::*;
    /// let mut morton_index = MortonIndex64::from_raw(0);
    /// morton_index.set_octant_at_level(3, Octant::ONE);
    /// assert_eq!(Some(Octant::ONE), morton_index.get_octant_at_level(3));
    /// ```
    ///
    /// # Panics
    ///
    /// If `level` is either zero (see comment above) or `level` is greater than `MortonIndex64::LEVELS`.
    pub fn set_octant_at_level(&mut self, level: u8, octant: Octant) {
        if level as usize > Self::LEVELS {
            panic!(
                "MortonIndex64::set_octant_at_level: Level {} is out of bounds!",
                level
            );
        }
        self.set_octant_at_level_unchecked(level, octant.into());
    }

    /// Unchecked version of `set_octant_at_level`
    pub fn set_octant_at_level_unchecked(&mut self, level: u8, octant: u8) {
        let bit_shift = (Self::LEVELS - level as usize) * 3;
        let clear_bits_mask = !(0b111 << bit_shift);
        self.index = (self.index & clear_bits_mask) | ((octant as u64) << bit_shift);
    }

    /// Returns the octant index at the specified level in the associated `MortonIndex64`. `level` describes the level relative
    /// to the root node. Since the root node is only a single node and has no parent, there is no octant at level 0, so `None`
    /// is returned in this case.
    ///
    /// # Example
    /// ```
    /// # use pasture_core::math::*;
    /// let mut morton_index = MortonIndex64::from_raw(0);
    /// morton_index.set_octant_at_level(3, Octant::ONE);
    /// assert_eq!(Some(Octant::ONE), morton_index.get_octant_at_level(3));
    /// ```
    ///
    /// # Panics
    ///
    /// If level is greater than `MortonIndex64::LEVELS`
    pub fn get_octant_at_level(&self, level: u8) -> Option<Octant> {
        if level == 0 {
            return None;
        }
        if level as usize > Self::LEVELS {
            panic!(
                "MortonIndex64::get_octant_at_level: Level {} is out of bounds!",
                level
            );
        }

        Some(
            self.get_octant_at_level_unchecked(level)
                .try_into()
                .unwrap(),
        )
    }

    /// Unchecked version of 'get_octant_at_level'. In contrast to `get_octant_at_level`, this always returns zero for
    /// level 0 (instead of `None`). Calling this method with a value for `level` that is greater than `MortonIndex64::LEVELS`
    /// is undefined behaviour!
    pub fn get_octant_at_level_unchecked(&self, level: u8) -> u8 {
        let bit_shift = (Self::LEVELS - level as usize) * 3;
        ((self.index >> bit_shift) & 0b111) as u8
    }

    /// Returns a `MortonIndex64WithDepth` from the associated `MortonIndex64` with the given depth
    ///
    /// # Panics
    ///
    /// If depth is greater than `MortonIndex64::LEVELS`
    pub fn with_depth(&self, depth: u8) -> MortonIndex64WithDepth {
        if depth as usize > Self::LEVELS {
            panic!(
                "MortonIndex64::with_depth: depth must not be greater than MortonIndex64::LEVELS!"
            );
        }
        // depth 0 == lowest 63 bits not set
        // depth 1 == lowest 60 bits not set
        // ...
        // depth 20 == lowest 3 bits not set
        // depth 21 == all bits set
        let shift = (Self::LEVELS - depth as usize) * 3;
        let mask = !((1_u64 << shift as u64) - 1);
        MortonIndex64WithDepth {
            index: self.index & mask,
            depth: depth,
        }
    }

    /// Converts the associated `MortonIndex64` into an XYZ index within a 3D grid
    pub fn as_grid_index(&self) -> Vector3<u32> {
        let mut x_idx: u32 = 0;
        let mut y_idx: u32 = 0;
        let mut z_idx: u32 = 0;

        for level in 0..Self::LEVELS {
            let octant = self.get_octant_at_level_unchecked((level + 1) as u8);

            x_idx <<= 1;
            y_idx <<= 1;
            z_idx <<= 1;

            x_idx = x_idx | (octant & 1) as u32;
            y_idx = y_idx | ((octant >> 1) & 1) as u32;
            z_idx = z_idx | ((octant >> 2) & 1) as u32;
        }

        Vector3::new(x_idx, y_idx, z_idx)
    }

    /// Returns a string representation of the associated `MortonIndex64` with the given `MortonIndexNaming`
    pub fn to_string(&self, naming: MortonIndexNaming) -> String {
        match naming {
            MortonIndexNaming::AsOctantConcatenation => self.to_string_octants(),
            MortonIndexNaming::AsOctantConcatenationWithRoot => {
                format!("r{}", self.to_string_octants())
            }
            MortonIndexNaming::AsGridCoordinates => self.to_string_grid_cells(),
        }
    }

    fn to_string_octants(&self) -> String {
        let mut str = String::with_capacity(Self::LEVELS);
        for level in 0..Self::LEVELS {
            let octant_at_level = self.get_octant_at_level_unchecked((level + 1) as u8);
            str.push(
                from_digit(octant_at_level as u32, 10).expect("Could not convert octant to digit!"),
            );
        }
        str
    }

    fn to_string_grid_cells(&self) -> String {
        let grid_index = self.as_grid_index();

        format!(
            "{}-{}-{}-{}",
            Self::LEVELS,
            grid_index.x,
            grid_index.y,
            grid_index.z
        )
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

impl From<u64> for MortonIndex64 {
    fn from(val: u64) -> Self {
        Self::from_raw(val)
    }
}

impl From<&u64> for MortonIndex64 {
    fn from(val: &u64) -> Self {
        Self::from_raw(*val)
    }
}

/// 64-bit 3D Morton index with depth information. This is the more efficient, but more constraint
/// version of `DynamicMortonIndex` because it can store at maximum 21 levels. *Important note:* Since
/// a Morton index defines a node in an octree, one has to define what index the root node has. As
/// each 3 bits within a 3D Morton index identify an octant at a certain level, there is no level that
/// defines the root node since it is just a single node. We now define that the root node is always at
/// level 0, so the `level` parameter of this structure is always relative to the root node. A
/// `MortonIndex64WithLevel` with a `level` of 0 is therefore always zero. The first octant lies at level 1.
/// This can be a bit confusing because it makes the code prone to off-by-one errors, however it is better
/// than the alternative (IMO) where root node has level -1, or None.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct MortonIndex64WithDepth {
    index: u64,
    depth: u8,
}

impl MortonIndex64WithDepth {
    /// Creates a new `MortonIndex64WithDepth` from the given octants. `octants.len()` must be `<= MortonIndex64::LEVELS`
    ///
    /// # Example
    /// ```
    /// # use pasture_core::math::*;
    /// // Create a Morton index for the octants 1->2->4 (octant 4 of octant 2 of octant 1 of the root node)
    /// let morton_index_with_depth = MortonIndex64WithDepth::from_octants(&[Octant::ONE, Octant::TWO, Octant::FOUR]);
    /// assert_eq!(3, morton_index_with_depth.depth());
    /// assert_eq!(Some(Octant::ONE), morton_index_with_depth.get_octant_at_level(1));
    /// assert_eq!(Some(Octant::TWO), morton_index_with_depth.get_octant_at_level(2));
    /// assert_eq!(Some(Octant::FOUR), morton_index_with_depth.get_octant_at_level(3));
    /// ```
    pub fn from_octants(octants: &[Octant]) -> Self {
        let full_morton_index = MortonIndex64::from_octants(octants);
        full_morton_index.with_depth(octants.len() as u8)
    }

    /// Returns the raw integer value of the associated `MortonIndex64WithDepth`. This index is similar
    /// to what `MortonIndex64::index()` returns, however only the highest `3 * depth()` bits are used
    /// (ignoring the MSB which is always unused for 64 bit Morton indices). All lower bits (and the MSB)
    /// are zero!
    pub fn raw_index(&self) -> u64 {
        self.index
    }

    /// Returns the depth of the octree node represented by the associated `MortonIndex64WithDepth`. A
    /// depth of zero represents the root node and corresponds to a `raw_index()` where no bits are used
    pub fn depth(&self) -> u8 {
        self.depth
    }

    /// Returns the octant index relative to the parent node for the node at the given `level`. Since
    /// level zero represents the root node, and the root node has no parent, `None` is returned for
    /// level zero.
    ///
    /// # Panics
    ///
    /// If `level` is greater than `depth()`
    pub fn get_octant_at_level(&self, level: u8) -> Option<Octant> {
        if level == 0 {
            return None;
        }
        if level > self.depth {
            panic!("MortonIndex64WithDepth::get_octant_at_level: level must not be larger than self.depth()!");
        }

        let bit_shift = (MortonIndex64::LEVELS - level as usize) * 3;
        let relevant_bits = (self.index >> bit_shift) & 0b111;
        Some(
            (relevant_bits as u8)
                .try_into()
                .expect("Octant index was greater than 7 but shouldn't be!"),
        )
    }

    /// Returns the child node at the given `octant` from the associated `MortonIndex64WithDepth`. If the associated
    /// index is already of maximum depth (`MortonIndex64::LEVELS`), returns `None`
    pub fn child(&self, octant: Octant) -> Option<Self> {
        if self.depth() as usize == MortonIndex64::LEVELS {
            return None;
        }

        // Set only the bits of the new octant within index
        let shift = (MortonIndex64::LEVELS - self.depth() as usize - 1) * 3;
        let bits = (octant.0 as u64) << shift;
        Some(Self {
            index: self.index | bits,
            depth: self.depth() + 1,
        })
    }

    /// Unchecked version of `get_octant_at_level`
    fn get_octant_at_level_unchecked(&self, level: u8) -> u8 {
        let bit_shift = (MortonIndex64::LEVELS - level as usize) * 3;
        let relevant_bits = (self.index >> bit_shift) & 0b111;
        relevant_bits as u8
    }

    /// Returns a version of the associated `MortonIndex64WithDepth` with the given lower depth
    pub fn with_lower_depth(&self, lower_depth: u8) -> Self {
        assert!(lower_depth < self.depth);
        // Trim of the lower bits of the index
        let shift = (MortonIndex64::LEVELS - lower_depth as usize) * 3;
        let mask = !((1_u64 << shift as u64) - 1);
        Self {
            index: self.raw_index() & mask,
            depth: lower_depth,
        }
    }

    /// Converts the associated `MortonIndex64WithDepth` into an XYZ index within a 3D grid of size `(2^self.depth)x(2^self.depth)x(2^self.depth)`
    pub fn as_grid_index(&self) -> Vector3<u32> {
        let mut x_idx: u32 = 0;
        let mut y_idx: u32 = 0;
        let mut z_idx: u32 = 0;

        for level in 0..self.depth {
            let octant = self.get_octant_at_level_unchecked((level + 1) as u8);

            x_idx <<= 1;
            y_idx <<= 1;
            z_idx <<= 1;

            x_idx = x_idx | (octant & 1) as u32;
            y_idx = y_idx | ((octant >> 1) & 1) as u32;
            z_idx = z_idx | ((octant >> 2) & 1) as u32;
        }

        Vector3::new(x_idx, y_idx, z_idx)
    }

    /// Returns a string representation of the associated `MortonIndex64WithDepth` with the given `MortonIndexNaming`
    pub fn to_string(&self, naming: MortonIndexNaming) -> String {
        match naming {
            MortonIndexNaming::AsOctantConcatenation => self.to_string_octants(),
            MortonIndexNaming::AsOctantConcatenationWithRoot => {
                format!("r{}", self.to_string_octants())
            }
            MortonIndexNaming::AsGridCoordinates => self.to_string_grid_cells(),
        }
    }

    fn to_string_octants(&self) -> String {
        let mut str = String::with_capacity(self.depth as usize);
        for level in 0..self.depth {
            let octant_at_level = self.get_octant_at_level(level + 1).unwrap();
            str.push(
                from_digit(octant_at_level.0 as u32, 10)
                    .expect("Could not convert octant to digit!"),
            );
        }
        str
    }

    fn to_string_grid_cells(&self) -> String {
        let mut x_idx: u64 = 0;
        let mut y_idx: u64 = 0;
        let mut z_idx: u64 = 0;

        for level in 0..self.depth {
            let octant = self.get_octant_at_level(level + 1).unwrap().0;

            x_idx <<= 1;
            y_idx <<= 1;
            z_idx <<= 1;

            x_idx = x_idx | (octant & 1) as u64;
            y_idx = y_idx | ((octant >> 1) & 1) as u64;
            z_idx = z_idx | ((octant >> 2) & 1) as u64;
        }

        format!("{}-{}-{}-{}", self.depth, x_idx, y_idx, z_idx)
    }
}

impl Default for MortonIndex64WithDepth {
    fn default() -> Self {
        Self { index: 0, depth: 0 }
    }
}

/// 3D Morton index with a dynamic depth
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct DynamicMortonIndex {
    octants: Vec<Octant>,
}

impl DynamicMortonIndex {
    /// Creates a new `DynamicMortonIndex` from the given octants. The order of the octant numbering is unspecified.
    /// The order of the indices goes from shallowest node to deepest node.
    ///
    /// Example:
    /// ```
    /// # use pasture_core::math::*;
    /// let mut dynamic_morton_index = DynamicMortonIndex::from_octants(&[Octant::ONE, Octant::TWO, Octant::FOUR]);
    /// assert_eq!(3, dynamic_morton_index.depth());
    /// ```
    pub fn from_octants(octants: &[Octant]) -> Self {
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
    pub fn octants(&self) -> &[Octant] {
        self.octants.as_slice()
    }

    /// Adds a new octant to the associated `DynamicMortonIndex`
    ///
    /// Example:
    /// ```
    /// # use pasture_core::math::*;
    /// let mut dynamic_morton_index = DynamicMortonIndex::from_octants(&[Octant::ONE, Octant::TWO, Octant::FOUR]);
    /// dynamic_morton_index.add_octant(Octant::SIX);
    /// assert_eq!(4, dynamic_morton_index.depth());
    /// assert_eq!(&[Octant::ONE, Octant::TWO, Octant::FOUR, Octant::SIX], dynamic_morton_index.octants());
    /// ```
    pub fn add_octant(&mut self, octant: Octant) {
        self.octants.push(octant);
    }

    /// Tries to truncate the associated `DynamicMortonIndex` to the given depth. If `depth` is greater than the
    /// current depth, this operation fails and returns `None`.
    ///
    /// Example:
    /// ```
    /// # use pasture_core::math::*;
    /// let mut dynamic_morton_index = DynamicMortonIndex::from_octants(&[Octant::ONE, Octant::TWO, Octant::FOUR]);
    /// let truncated_index = dynamic_morton_index.truncate_to_depth(1);
    /// assert_eq!(Some(DynamicMortonIndex::from_octants(&[Octant::ONE])), truncated_index);
    ///
    /// let invalid_index = dynamic_morton_index.truncate_to_depth(10);
    /// assert_eq!(None, invalid_index);
    /// ```
    pub fn truncate_to_depth(&self, new_depth: usize) -> Option<Self> {
        if new_depth > self.depth() {
            return None;
        }

        Some(Self::from_octants(&self.octants[0..new_depth]))
    }

    /// Returns an iterator over the octants in the associated `DynamicMortonIndex`
    pub fn iter(&self) -> impl Iterator<Item = &Octant> {
        self.octants.iter()
    }

    /// Returns a mutable iterator over the octants in the associated `DynamicMortonIndex`
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Octant> {
        self.octants.iter_mut()
    }

    /// Returns a string representation of the associated `MortonIndex64` with the given `MortonIndexNaming`
    pub fn to_string(&self, naming: MortonIndexNaming) -> String {
        match naming {
            MortonIndexNaming::AsOctantConcatenation => self.to_string_octants(),
            MortonIndexNaming::AsOctantConcatenationWithRoot => {
                format!("r{}", self.to_string_octants())
            }
            MortonIndexNaming::AsGridCoordinates => self.to_string_grid_cells(),
        }
    }

    fn to_string_octants(&self) -> String {
        self.octants
            .iter()
            .map(|&octant| {
                from_digit(octant.0 as u32, 10).expect("Could not convert digit to char!")
            })
            .collect::<String>()
    }

    fn to_string_grid_cells(&self) -> String {
        let mut x_idx: u64 = 0;
        let mut y_idx: u64 = 0;
        let mut z_idx: u64 = 0;

        for octant in self.octants.iter() {
            x_idx <<= 1;
            y_idx <<= 1;
            z_idx <<= 1;

            x_idx = x_idx | (octant.0 & 1) as u64;
            y_idx = y_idx | ((octant.0 >> 1) & 1) as u64;
            z_idx = z_idx | ((octant.0 >> 2) & 1) as u64;
        }

        format!("{}-{}-{}-{}", self.depth(), x_idx, y_idx, z_idx)
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
                .map(|level| {
                    morton_index
                        .get_octant_at_level_unchecked(level)
                        .try_into()
                        .unwrap()
                })
                .collect(),
        }
    }
}

impl From<MortonIndex64WithDepth> for DynamicMortonIndex {
    fn from(morton_index: MortonIndex64WithDepth) -> Self {
        Self {
            octants: (1..=morton_index.depth())
                .map(|level| {
                    morton_index
                        .get_octant_at_level_unchecked(level)
                        .try_into()
                        .unwrap()
                })
                .collect(),
        }
    }
}

impl From<&MortonIndex64WithDepth> for DynamicMortonIndex {
    fn from(morton_index: &MortonIndex64WithDepth) -> Self {
        Self {
            octants: (1..=morton_index.depth())
                .map(|level| {
                    morton_index
                        .get_octant_at_level_unchecked(level)
                        .try_into()
                        .unwrap()
                })
                .collect(),
        }
    }
}

impl TryFrom<DynamicMortonIndex> for MortonIndex64WithDepth {
    type Error = DynamicMortonIndexTooLargeError;

    fn try_from(value: DynamicMortonIndex) -> Result<Self, Self::Error> {
        if value.depth() > MortonIndex64::LEVELS {
            return Err(DynamicMortonIndexTooLargeError::new(value));
        }
        let static_morton_index = MortonIndex64::from_octants(value.octants());
        Ok(static_morton_index.with_depth(value.depth() as u8))
    }
}

impl TryFrom<&DynamicMortonIndex> for MortonIndex64WithDepth {
    type Error = DynamicMortonIndexTooLargeError;

    fn try_from(value: &DynamicMortonIndex) -> Result<Self, Self::Error> {
        if value.depth() > MortonIndex64::LEVELS {
            return Err(DynamicMortonIndexTooLargeError::new(value.clone()));
        }
        let static_morton_index = MortonIndex64::from_octants(value.octants());
        Ok(static_morton_index.with_depth(value.depth() as u8))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_21_example_octants() -> Vec<Octant> {
        vec![
            7, 3, 4, 5, 1, 6, 0, 2, 3, 6, 3, 4, 2, 5, 7, 5, 1, 3, 0, 6, 1,
        ]
        .iter()
        .map(|&raw_octant| (raw_octant as u8).try_into().unwrap())
        .collect()
    }

    #[test]
    fn test_morton_index_with_depth_from_morton_index() {
        let octants = get_21_example_octants();

        let static_morton_index = MortonIndex64::from_octants(octants.as_slice());

        {
            let l0_index = static_morton_index.with_depth(0);
            assert_eq!(0, l0_index.depth());
            assert_eq!(None, l0_index.get_octant_at_level(0));
        }

        {
            let l1_index = static_morton_index.with_depth(1);
            assert_eq!(1, l1_index.depth());
            assert_eq!(Some(octants[0]), l1_index.get_octant_at_level(1));
        }

        {
            let l2_index = static_morton_index.with_depth(2);
            assert_eq!(2, l2_index.depth());
            assert_eq!(Some(octants[0]), l2_index.get_octant_at_level(1));
            assert_eq!(Some(octants[1]), l2_index.get_octant_at_level(2));
        }

        {
            let all_levels_index = static_morton_index.with_depth(21);
            assert_eq!(21, all_levels_index.depth());
            for lvl in 0..21 {
                assert_eq!(
                    Some(octants[lvl]),
                    all_levels_index.get_octant_at_level((lvl + 1) as u8)
                );
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_morton_index_with_depth_get_out_of_bounds_octant() {
        let static_morton_index: MortonIndex64 = Default::default();
        let morton_index_with_depth = static_morton_index.with_depth(0);
        morton_index_with_depth.get_octant_at_level(1);
    }

    #[test]
    fn test_empty_morton_index_64_to_string_octant_naming() {
        let idx: MortonIndex64 = Default::default();
        assert_eq!(
            "0".repeat(21),
            idx.to_string(MortonIndexNaming::AsOctantConcatenation)
        );
    }

    #[test]
    fn test_morton_index_64_to_string_octant_naming() {
        let octants = get_21_example_octants();
        let expected_str = "734516023634257513061".to_owned();

        let idx = MortonIndex64::from_octants(octants.as_slice());
        assert_eq!(
            expected_str,
            idx.to_string(MortonIndexNaming::AsOctantConcatenation)
        );
    }

    #[test]
    fn test_empty_morton_index_64_to_string_octant_and_root_naming() {
        let idx: MortonIndex64 = Default::default();
        assert_eq!(
            format!("r{}", "0".repeat(21)),
            idx.to_string(MortonIndexNaming::AsOctantConcatenationWithRoot)
        );
    }

    #[test]
    fn test_morton_index_64_to_string_octant_and_root_naming() {
        let octants = get_21_example_octants();
        let expected_str = "r734516023634257513061".to_owned();

        let idx = MortonIndex64::from_octants(octants.as_slice());
        assert_eq!(
            expected_str,
            idx.to_string(MortonIndexNaming::AsOctantConcatenationWithRoot)
        );
    }

    #[test]
    fn test_empty_morton_index_64_to_string_grid_cell_naming() {
        let idx: MortonIndex64 = Default::default();
        assert_eq!(
            "21-0-0-0".to_owned(),
            idx.to_string(MortonIndexNaming::AsGridCoordinates)
        );
    }

    #[test]
    fn test_morton_index_64_to_string_grid_cell_naming() {
        let octants = get_21_example_octants();
        let x_index: u64 = octants
            .iter()
            .enumerate()
            .map(|(idx, octant)| {
                let x_bit = (octant.0 & 1) as u64;
                let shift = (MortonIndex64::LEVELS - idx - 1) as u64;
                x_bit << shift
            })
            .sum();
        let y_index: u64 = octants
            .iter()
            .enumerate()
            .map(|(idx, octant)| {
                let y_bit = ((octant.0 & 0b10) >> 1) as u64;
                let shift = (MortonIndex64::LEVELS - idx - 1) as u64;
                y_bit << shift
            })
            .sum();
        let z_index: u64 = octants
            .iter()
            .enumerate()
            .map(|(idx, octant)| {
                let z_bit = ((octant.0 & 0b100) >> 2) as u64;
                let shift = (MortonIndex64::LEVELS - idx - 1) as u64;
                z_bit << shift
            })
            .sum();

        let morton_idx = MortonIndex64::from_octants(octants.as_slice());

        let expected_str = format!("21-{}-{}-{}", x_index, y_index, z_index);

        assert_eq!(
            expected_str,
            morton_idx.to_string(MortonIndexNaming::AsGridCoordinates)
        );
    }

    #[test]
    fn test_empty_dynamic_morton_index_to_string_octant_naming() {
        let idx: DynamicMortonIndex = Default::default();
        assert_eq!("", idx.to_string(MortonIndexNaming::AsOctantConcatenation));
    }

    #[test]
    fn test_dynamic_morton_index_to_string_octant_naming() {
        let octants = get_21_example_octants();
        let expected_str = "734516023634257513061".to_owned();

        let idx = DynamicMortonIndex::from_octants(octants.as_slice());
        assert_eq!(
            expected_str,
            idx.to_string(MortonIndexNaming::AsOctantConcatenation)
        );
    }

    #[test]
    fn test_empty_dynamic_morton_index_to_string_octant_and_root_naming() {
        let idx: DynamicMortonIndex = Default::default();
        assert_eq!(
            "r".to_owned(),
            idx.to_string(MortonIndexNaming::AsOctantConcatenationWithRoot)
        );
    }

    #[test]
    fn test_dynamic_morton_index_to_string_octant_and_root_naming() {
        let octants = get_21_example_octants();
        let expected_str = "r734516023634257513061".to_owned();

        let idx = DynamicMortonIndex::from_octants(octants.as_slice());
        assert_eq!(
            expected_str,
            idx.to_string(MortonIndexNaming::AsOctantConcatenationWithRoot)
        );
    }

    #[test]
    fn test_empty_dynamic_morton_index_to_string_grid_cell_naming() {
        let idx: DynamicMortonIndex = Default::default();
        assert_eq!(
            "0-0-0-0".to_owned(),
            idx.to_string(MortonIndexNaming::AsGridCoordinates)
        );
    }

    #[test]
    fn test_dynamic_morton_index_to_string_grid_cell_naming() {
        let octants = get_21_example_octants();
        let x_index: u64 = octants
            .iter()
            .enumerate()
            .map(|(idx, octant)| {
                let x_bit = (octant.0 & 1) as u64;
                let shift = (MortonIndex64::LEVELS - idx - 1) as u64;
                x_bit << shift
            })
            .sum();
        let y_index: u64 = octants
            .iter()
            .enumerate()
            .map(|(idx, octant)| {
                let y_bit = ((octant.0 & 0b10) >> 1) as u64;
                let shift = (MortonIndex64::LEVELS - idx - 1) as u64;
                y_bit << shift
            })
            .sum();
        let z_index: u64 = octants
            .iter()
            .enumerate()
            .map(|(idx, octant)| {
                let z_bit = ((octant.0 & 0b100) >> 2) as u64;
                let shift = (MortonIndex64::LEVELS - idx - 1) as u64;
                z_bit << shift
            })
            .sum();

        let morton_idx = DynamicMortonIndex::from_octants(octants.as_slice());

        let expected_str = format!("21-{}-{}-{}", x_index, y_index, z_index);

        assert_eq!(
            expected_str,
            morton_idx.to_string(MortonIndexNaming::AsGridCoordinates)
        );
    }

    #[test]
    fn test_morton_index_with_depth_roundtrip() {
        let octants = get_21_example_octants();

        let static_morton_index = MortonIndex64::from_octants(octants.as_slice());
        let with_depth = static_morton_index.with_depth(5);

        let dynamic: DynamicMortonIndex = with_depth.clone().into();
        let again_with_depth: MortonIndex64WithDepth = dynamic.try_into().unwrap();
        assert_eq!(with_depth, again_with_depth);
    }

    #[test]
    fn test_morton_index_with_depth_child() {
        let octants = get_21_example_octants();

        let static_morton_index = MortonIndex64::from_octants(octants.as_slice());
        let with_depth = static_morton_index.with_depth(5);

        assert_eq!(
            Some(static_morton_index.with_depth(6)),
            with_depth.child(octants[5])
        );
        assert_eq!(None, static_morton_index.with_depth(21).child(Octant::ZERO));
    }

    #[test]
    fn test_morton_index_with_depth_lower_depth() {
        let octants = get_21_example_octants();

        let static_morton_index = MortonIndex64::from_octants(octants.as_slice());
        let with_depth = static_morton_index.with_depth(5);

        assert_eq!(
            with_depth.with_lower_depth(4),
            static_morton_index.with_depth(4)
        );
        assert_eq!(
            with_depth.with_lower_depth(2),
            static_morton_index.with_depth(2)
        );
        assert_eq!(
            with_depth.with_lower_depth(0),
            static_morton_index.with_depth(0)
        );
    }
}
