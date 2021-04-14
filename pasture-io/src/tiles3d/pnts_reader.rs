use std::io::{Read, Seek};

use pasture_core::layout::PointLayout;

use crate::las::LASMetadata;

pub struct PnstReader<R: Read + Seek> {
    reader: R,
    metadata: LASMetadata,
    layout: PointLayout,
    current_point_index: usize,
}
