use anyhow::{bail, Result};
use kd_tree::KdSlice3;
use pasture_core::{
    containers::{PointBufferExt, PointBufferWriteable, PointBufferWriteableExt},
    layout::{attributes::POSITION_3D, PointAttributeDefinition},
    nalgebra::{Matrix3, Vector3},
};

use self::attributes::{
    ANISOTROPY, EIGENENTROPY, LINEARITY, OMNIVARIANCE, PLANARITY, SPHERICITY, SUM_OF_EIGENVALUES,
};

pub mod attributes {
    use pasture_core::layout::{PointAttributeDataType, PointAttributeDefinition};

    /// Point attribute for the 'linearity' structure measure, given by `(l1 - l2) / l1` with `l1` and `l2` being
    /// the two largest eigenvalues of the structure tensor
    pub const LINEARITY: PointAttributeDefinition =
        PointAttributeDefinition::custom("StructureMeasureLinearity", PointAttributeDataType::F32);

    /// Point attribute for the 'planarity' structure measure, given by `(l2 - l3) / l1` with `l1`, `l2` and `l3` being
    /// the three largest eigenvalues of the structure tensor
    pub const PLANARITY: PointAttributeDefinition =
        PointAttributeDefinition::custom("StructureMeasurePlanarity", PointAttributeDataType::F32);

    /// Point attribute for the 'sphericity' structure measure, given by `l3 / l1` with `l1` and `l3` being
    /// the first and third largest eigenvalues respectively of the structure tensor
    pub const SPHERICITY: PointAttributeDefinition =
        PointAttributeDefinition::custom("StructureMeasureSphericity", PointAttributeDataType::F32);

    /// Point attribute for the 'omnivariance' structure measure, given by `(l1 * l2 * l3)^1/3` with `l1`, `l2` and `l3` being
    /// the three largest eigenvalues of the structure tensor
    pub const OMNIVARIANCE: PointAttributeDefinition = PointAttributeDefinition::custom(
        "StructureMeasureOmnivariance",
        PointAttributeDataType::F32,
    );

    /// Point attribute for the 'anisotropy' structure measure, given by `l3 / l1` with `l1` and `l3` being
    /// the first and third largest eigenvalues respectively of the structure tensor
    pub const ANISOTROPY: PointAttributeDefinition =
        PointAttributeDefinition::custom("StructureMeasureAnisotropy", PointAttributeDataType::F32);

    /// Point attribute for the 'eigenentropy' structure measure, given by `-SUM(l_i * ln(l_i))` with `l_i` being
    /// the ith-largest eigenvalue of the structure tensor
    pub const EIGENENTROPY: PointAttributeDefinition = PointAttributeDefinition::custom(
        "StructureMeasureEigenentropy",
        PointAttributeDataType::F32,
    );

    /// Point attribute for the 'sum of eigenvalues' structure measure, given by the sum of the three eigenvalues
    /// of the structure tensor
    pub const SUM_OF_EIGENVALUES: PointAttributeDefinition = PointAttributeDefinition::custom(
        "StructureMeasureSumOfEigenvalues",
        PointAttributeDataType::F32,
    );
}

/// Calculates the linearity for each point of the given `PointBuffer` and stores it in the buffer in the `LINEARITY` point
/// attribute. Linearity calculation relies on the structure tensor, which has to be computed per-point from the local
/// neighbourhood of the point. To accelerate this calculation, a kd-tree accelerator is required that must contain all positions
/// of the associated `buffer`. The parameter `num_nearest_neighbours`defines the number of nearest neighbours to use at each point
/// in the calculation. It must be `>=3`. Linearity calculation can fail if the structure tensor can't be computed. In this
/// case this method will return an `Err`.
///
/// # Panics
/// If the `buffer` does not contain the a `POSITION_3D` attribute or the `LINEARITY` attribute.
/// If `num_nearest_neighbours` is less than 3.
/// If the `kd_tree` contains a different number of point than the `buffer`
pub fn calculate_linearity<B: PointBufferWriteable + ?Sized>(
    buffer: &mut B,
    kd_tree: &KdSlice3<[f64; 3]>,
    num_nearest_neighbours: usize,
) -> Result<()> {
    if !buffer.point_layout().has_attribute(&POSITION_3D) {
        panic!("Buffer must contain the POSITION_3D attribute");
    }
    if !buffer.point_layout().has_attribute(&LINEARITY) {
        panic!("Buffer must contain the LINEARITY attribute");
    }
    if num_nearest_neighbours < 3 {
        panic!("num_nearest_neighbours must be at least 3");
    }
    if kd_tree.items().len() != buffer.len() {
        panic!("kd_tree must contain the same number of points as the buffer");
    }

    compute_structure_measure(
        buffer,
        kd_tree,
        num_nearest_neighbours,
        &LINEARITY,
        linearity,
    )
}

/// Calculates the planarity for each point of the given `PointBuffer` and stores it in the buffer in the `PLANARITY` point
/// attribute. Planarity calculation relies on the structure tensor, which has to be computed per-point from the local
/// neighbourhood of the point. To accelerate this calculation, a kd-tree accelerator is required that must contain all positions
/// of the associated `buffer`. The parameter `num_nearest_neighbours`defines the number of nearest neighbours to use at each point
/// in the calculation. It must be `>=3`. Planarity calculation can fail if the structure tensor can't be computed. In this
/// case this method will return an `Err`.
///
/// # Panics
/// If the `buffer` does not contain the a `POSITION_3D` attribute or the `PLANARITY` attribute.
/// If `num_nearest_neighbours` is less than 3.
/// If the `kd_tree` contains a different number of point than the `buffer`
pub fn calculate_planarity<B: PointBufferWriteable + ?Sized>(
    buffer: &mut B,
    kd_tree: &KdSlice3<[f64; 3]>,
    num_nearest_neighbours: usize,
) -> Result<()> {
    if !buffer.point_layout().has_attribute(&POSITION_3D) {
        panic!("Buffer must contain the POSITION_3D attribute");
    }
    if !buffer.point_layout().has_attribute(&PLANARITY) {
        panic!("Buffer must contain the PLANARITY attribute");
    }
    if num_nearest_neighbours < 3 {
        panic!("num_nearest_neighbours must be at least 3");
    }
    if kd_tree.items().len() != buffer.len() {
        panic!("kd_tree must contain the same number of points as the buffer");
    }

    compute_structure_measure(
        buffer,
        kd_tree,
        num_nearest_neighbours,
        &PLANARITY,
        planarity,
    )
}

/// Calculates the sphericity for each point of the given `PointBuffer` and stores it in the buffer in the `SPHERICITY` point
/// attribute. Sphericity calculation relies on the structure tensor, which has to be computed per-point from the local
/// neighbourhood of the point. To accelerate this calculation, a kd-tree accelerator is required that must contain all positions
/// of the associated `buffer`. The parameter `num_nearest_neighbours`defines the number of nearest neighbours to use at each point
/// in the calculation. It must be `>=3`. Sphericity calculation can fail if the structure tensor can't be computed. In this
/// case this method will return an `Err`.
///
/// # Panics
/// If the `buffer` does not contain the a `POSITION_3D` attribute or the `SPHERICITY` attribute.
/// If `num_nearest_neighbours` is less than 3.
/// If the `kd_tree` contains a different number of point than the `buffer`
pub fn calculate_sphericity<B: PointBufferWriteable + ?Sized>(
    buffer: &mut B,
    kd_tree: &KdSlice3<[f64; 3]>,
    num_nearest_neighbours: usize,
) -> Result<()> {
    if !buffer.point_layout().has_attribute(&POSITION_3D) {
        panic!("Buffer must contain the POSITION_3D attribute");
    }
    if !buffer.point_layout().has_attribute(&SPHERICITY) {
        panic!("Buffer must contain the SPHERICITY attribute");
    }
    if num_nearest_neighbours < 3 {
        panic!("num_nearest_neighbours must be at least 3");
    }
    if kd_tree.items().len() != buffer.len() {
        panic!("kd_tree must contain the same number of points as the buffer");
    }

    compute_structure_measure(
        buffer,
        kd_tree,
        num_nearest_neighbours,
        &SPHERICITY,
        sphericity,
    )
}

/// Calculates the omnivariance for each point of the given `PointBuffer` and stores it in the buffer in the `OMNIVARIANCE` point
/// attribute. Omnivariance calculation relies on the structure tensor, which has to be computed per-point from the local
/// neighbourhood of the point. To accelerate this calculation, a kd-tree accelerator is required that must contain all positions
/// of the associated `buffer`. The parameter `num_nearest_neighbours`defines the number of nearest neighbours to use at each point
/// in the calculation. It must be `>=3`. Omnivariance calculation can fail if the structure tensor can't be computed. In this
/// case this method will return an `Err`.
///
/// # Panics
/// If the `buffer` does not contain the a `POSITION_3D` attribute or the `OMNIVARIANCE` attribute.
/// If `num_nearest_neighbours` is less than 3.
/// If the `kd_tree` contains a different number of point than the `buffer`
pub fn calculate_omnivariance<B: PointBufferWriteable + ?Sized>(
    buffer: &mut B,
    kd_tree: &KdSlice3<[f64; 3]>,
    num_nearest_neighbours: usize,
) -> Result<()> {
    if !buffer.point_layout().has_attribute(&POSITION_3D) {
        panic!("Buffer must contain the POSITION_3D attribute");
    }
    if !buffer.point_layout().has_attribute(&OMNIVARIANCE) {
        panic!("Buffer must contain the OMNIVARIANCE attribute");
    }
    if num_nearest_neighbours < 3 {
        panic!("num_nearest_neighbours must be at least 3");
    }
    if kd_tree.items().len() != buffer.len() {
        panic!("kd_tree must contain the same number of points as the buffer");
    }

    compute_structure_measure(
        buffer,
        kd_tree,
        num_nearest_neighbours,
        &OMNIVARIANCE,
        omnivariance,
    )
}

/// Calculates the anisotropy for each point of the given `PointBuffer` and stores it in the buffer in the `ANISOTROPY` point
/// attribute. Anisotropy calculation relies on the structure tensor, which has to be computed per-point from the local
/// neighbourhood of the point. To accelerate this calculation, a kd-tree accelerator is required that must contain all positions
/// of the associated `buffer`. The parameter `num_nearest_neighbours`defines the number of nearest neighbours to use at each point
/// in the calculation. It must be `>=3`. Anisotropy calculation can fail if the structure tensor can't be computed. In this
/// case this method will return an `Err`.
///
/// # Panics
/// If the `buffer` does not contain the a `POSITION_3D` attribute or the `ANISOTROPY` attribute.
/// If `num_nearest_neighbours` is less than 3.
/// If the `kd_tree` contains a different number of point than the `buffer`
pub fn calculate_anisotropy<B: PointBufferWriteable + ?Sized>(
    buffer: &mut B,
    kd_tree: &KdSlice3<[f64; 3]>,
    num_nearest_neighbours: usize,
) -> Result<()> {
    if !buffer.point_layout().has_attribute(&POSITION_3D) {
        panic!("Buffer must contain the POSITION_3D attribute");
    }
    if !buffer.point_layout().has_attribute(&ANISOTROPY) {
        panic!("Buffer must contain the ANISOTROPY attribute");
    }
    if num_nearest_neighbours < 3 {
        panic!("num_nearest_neighbours must be at least 3");
    }
    if kd_tree.items().len() != buffer.len() {
        panic!("kd_tree must contain the same number of points as the buffer");
    }

    compute_structure_measure(
        buffer,
        kd_tree,
        num_nearest_neighbours,
        &ANISOTROPY,
        anisotropy,
    )
}

/// Calculates the eigenentropy for each point of the given `PointBuffer` and stores it in the buffer in the `EIGENTROPY` point
/// attribute. Eigenentropy calculation relies on the structure tensor, which has to be computed per-point from the local
/// neighbourhood of the point. To accelerate this calculation, a kd-tree accelerator is required that must contain all positions
/// of the associated `buffer`. The parameter `num_nearest_neighbours`defines the number of nearest neighbours to use at each point
/// in the calculation. It must be `>=3`. Eigenentropy calculation can fail if the structure tensor can't be computed. In this
/// case this method will return an `Err`.
///
/// # Panics
/// If the `buffer` does not contain the a `POSITION_3D` attribute or the `EIGENENTROPY` attribute.
/// If `num_nearest_neighbours` is less than 3.
/// If the `kd_tree` contains a different number of point than the `buffer`
pub fn calculate_eigenentropy<B: PointBufferWriteable + ?Sized>(
    buffer: &mut B,
    kd_tree: &KdSlice3<[f64; 3]>,
    num_nearest_neighbours: usize,
) -> Result<()> {
    if !buffer.point_layout().has_attribute(&POSITION_3D) {
        panic!("Buffer must contain the POSITION_3D attribute");
    }
    if !buffer.point_layout().has_attribute(&EIGENENTROPY) {
        panic!("Buffer must contain the EIGENENTROPY attribute");
    }
    if num_nearest_neighbours < 3 {
        panic!("num_nearest_neighbours must be at least 3");
    }
    if kd_tree.items().len() != buffer.len() {
        panic!("kd_tree must contain the same number of points as the buffer");
    }

    compute_structure_measure(
        buffer,
        kd_tree,
        num_nearest_neighbours,
        &EIGENENTROPY,
        eigenentropy,
    )
}

/// Calculates the sum of eigenvalues for each point of the given `PointBuffer` and stores it in the buffer in the `SUM_OF_EIGENVALUES` point
/// attribute. Sum of eigenvalues calculation relies on the structure tensor, which has to be computed per-point from the local
/// neighbourhood of the point. To accelerate this calculation, a kd-tree accelerator is required that must contain all positions
/// of the associated `buffer`. The parameter `num_nearest_neighbours`defines the number of nearest neighbours to use at each point
/// in the calculation. It must be `>=3`. Sum of eigenvalues calculation can fail if the structure tensor can't be computed. In this
/// case this method will return an `Err`.
///
/// # Panics
/// If the `buffer` does not contain the a `POSITION_3D` attribute or the `SUM_OF_EIGENVALUES` attribute.
/// If `num_nearest_neighbours` is less than 3.
/// If the `kd_tree` contains a different number of point than the `buffer`
pub fn calculate_sum_of_eigenvalues<B: PointBufferWriteable + ?Sized>(
    buffer: &mut B,
    kd_tree: &KdSlice3<[f64; 3]>,
    num_nearest_neighbours: usize,
) -> Result<()> {
    if !buffer.point_layout().has_attribute(&POSITION_3D) {
        panic!("Buffer must contain the POSITION_3D attribute");
    }
    if !buffer.point_layout().has_attribute(&SUM_OF_EIGENVALUES) {
        panic!("Buffer must contain the SUM_OF_EIGENVALUES attribute");
    }
    if num_nearest_neighbours < 3 {
        panic!("num_nearest_neighbours must be at least 3");
    }
    if kd_tree.items().len() != buffer.len() {
        panic!("kd_tree must contain the same number of points as the buffer");
    }

    compute_structure_measure(
        buffer,
        kd_tree,
        num_nearest_neighbours,
        &SUM_OF_EIGENVALUES,
        sum_of_eigenvalues,
    )
}

fn compute_structure_measure<B: PointBufferWriteable + ?Sized, F: Fn(f64, f64, f64) -> f32>(
    buffer: &mut B,
    kd_tree: &KdSlice3<[f64; 3]>,
    num_nearest_neighbours: usize,
    structure_attribute: &PointAttributeDefinition,
    calc_attribute_fn: F,
) -> Result<()> {
    let mut structure_measures = Vec::with_capacity(buffer.len());

    for position in buffer.iter_attribute::<Vector3<f64>>(&POSITION_3D) {
        let knn = kd_tree.nearests(
            &[position.x, position.y, position.z],
            num_nearest_neighbours + 1,
        );
        // First element will be the point itself, we don't care for that so we remove it
        let knn = knn
            .iter()
            .skip(1)
            .map(|v| Vector3::new(v.item[0], v.item[1], v.item[2]))
            .collect::<Vec<_>>();

        // Structure measure calculation goes in three steps:
        // 1) calculate the 3D covariance matrix
        // 2) get the three largest eigenvalues from the covariance matrix
        // 3) calculate structure measure from there

        let centroid = calc_centroid(&knn);
        let covariance_matrix = calc_covariance_matrix(&centroid, &knn);
        let eigenvalues = covariance_matrix.eigenvalues();
        if eigenvalues.is_none() {
            bail!("Could not calculate eigenvalues at point {}", position);
        }
        let eigenvalues = eigenvalues.unwrap();
        if eigenvalues.len() != 3 {
            bail!("Number of eigenvalues is not 3");
        }

        let mut e1 = eigenvalues[0];
        let mut e2 = eigenvalues[1];
        let mut e3 = eigenvalues[2];

        //Tiny in-place bubblesort so that e1 ends up being the largest eigenvalue and e3 the smallest
        if e1 < e2 {
            std::mem::swap(&mut e1, &mut e2);
        }
        if e2 < e3 {
            std::mem::swap(&mut e2, &mut e3);
        }
        if e1 < e2 {
            std::mem::swap(&mut e1, &mut e2);
        }

        let sum_of_eigenvalues = e1 + e2 + e3;
        let e1_norm = e1 / sum_of_eigenvalues;
        let e2_norm = e2 / sum_of_eigenvalues;
        let e3_norm = e3 / sum_of_eigenvalues;

        let measure = calc_attribute_fn(e1_norm, e2_norm, e3_norm);
        structure_measures.push(measure);
    }

    for (idx, measure) in structure_measures.iter().enumerate() {
        buffer.set_attribute(structure_attribute, idx, *measure);
    }

    Ok(())
}

fn linearity(e1_norm: f64, e2_norm: f64, _e3_norm: f64) -> f32 {
    ((e1_norm - e2_norm) / e1_norm) as f32
}

fn planarity(e1_norm: f64, e2_norm: f64, e3_norm: f64) -> f32 {
    ((e2_norm - e3_norm) / e1_norm) as f32
}

fn sphericity(e1_norm: f64, _e2_norm: f64, e3_norm: f64) -> f32 {
    (e3_norm / e1_norm) as f32
}

fn omnivariance(e1_norm: f64, e2_norm: f64, e3_norm: f64) -> f32 {
    (e1_norm * e2_norm * e3_norm).powf(1.0 / 3.0) as f32
}

fn anisotropy(e1_norm: f64, _e2_norm: f64, e3_norm: f64) -> f32 {
    ((e1_norm - e3_norm) / e1_norm) as f32
}

fn eigenentropy(e1_norm: f64, e2_norm: f64, e3_norm: f64) -> f32 {
    (-(e1_norm * e1_norm.ln()) - (e2_norm * e2_norm.ln()) - (e3_norm * e3_norm.ln())) as f32
}

fn sum_of_eigenvalues(e1_norm: f64, e2_norm: f64, e3_norm: f64) -> f32 {
    (e1_norm + e2_norm + e3_norm) as f32
}

fn calc_centroid(knn: &[Vector3<f64>]) -> Vector3<f64> {
    let sum: Vector3<f64> = knn.iter().sum();

    sum / (knn.len() as f64)
}

fn calc_covariance_matrix(centroid: &Vector3<f64>, knn: &[Vector3<f64>]) -> Matrix3<f64> {
    let unweighted_covariance_matrix: Matrix3<f64> = knn
        .iter()
        .map(|v| {
            let diff: Vector3<f64> = v - centroid;
            let diff_transposed = diff.transpose();
            diff * diff_transposed
        })
        .sum();
    let weight_factor = 1.0 / (knn.len() as f64 + 1.0);
    unweighted_covariance_matrix * weight_factor
}
