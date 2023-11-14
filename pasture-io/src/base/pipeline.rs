use std::path::Path;

use anyhow::Result;

pub trait PipelineStage {}

trait PointInput {}

pub trait PointOutput {}

pub fn run_pipeline<P: AsRef<Path>>(
    input_files: &[P],
    stages: Vec<Box<dyn PipelineStage>>,
    output: Box<dyn PointOutput>,
) -> Result<()> {
    todo!()
}
