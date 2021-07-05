use std::path::PathBuf;

use pasture_core::nalgebra::Vector3;

/// Returns the resource/test/folder
pub(crate) fn get_test_file_path(filename: &str) -> PathBuf {
    let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    test_file_path.push(format!("resources/test/{}", filename));
    test_file_path
}



pub(crate) fn test_data_positions() -> Vec<Vector3<f64>> {
    vec![
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(1.0, 1.0, 1.0),
        Vector3::new(2.0, 2.0, 2.0),
        Vector3::new(3.0, 3.0, 3.0),
        Vector3::new(4.0, 4.0, 4.0),
        Vector3::new(5.0, 5.0, 5.0),
        Vector3::new(6.0, 6.0, 6.0),
        Vector3::new(7.0, 7.0, 7.0),
        Vector3::new(8.0, 8.0, 8.0),
        Vector3::new(9.0, 9.0, 9.0),
    ]
}