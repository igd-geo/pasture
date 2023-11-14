use std::{
    io::SeekFrom,
    ops::Range,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, bail, Result};
use gat_lending_iterator::LendingIterator;
use itertools::Itertools;
use pasture_core::{
    containers::{BorrowedBuffer, MakeBufferFromLayout, OwningBuffer, VectorBuffer},
    layout::PointLayout,
    math::AABB,
};
use walkdir::WalkDir;

use crate::base::{GenericPointReader, PointReader};

use super::SeekToPoint;

#[derive(Debug, Clone)]
pub struct ChunkInfo {
    pub file_id: usize,
    pub point_range_in_file: Range<usize>,
}

pub struct Pointcloud {
    files: Vec<PathBuf>,
    bounds: Option<AABB<f64>>,
    num_points: usize,
    points_per_file: Vec<usize>,
    common_point_layout: PointLayout,
}

impl Pointcloud {
    pub fn from_dir<P: AsRef<Path>>(
        dir: P,
        requested_point_layout: Option<PointLayout>,
    ) -> Result<Self> {
        let mut matching_files = WalkDir::new(dir)
            .into_iter()
            .filter_map(|entry| -> Option<Result<PathBuf>> {
                match entry {
                    Ok(entry) => {
                        if !entry.path().is_file() {
                            None
                        } else {
                            match GenericPointReader::is_supported_file(entry.path()) {
                                Ok(true) => Some(Ok(entry.path().to_path_buf())),
                                Ok(false) => None,
                                Err(why) => Some(Err(why)),
                            }
                        }
                    }
                    Err(why) => Some(Err(why.into())),
                }
            })
            .collect::<Result<Vec<_>>>()?;
        // Sort file paths lexicographically
        matching_files.sort();

        if matching_files.is_empty() {
            return Ok(Self {
                bounds: None,
                common_point_layout: Default::default(),
                files: matching_files,
                points_per_file: Default::default(),
                num_points: 0,
            });
        }

        let metas_and_layouts = matching_files
            .iter()
            .map(|file| {
                let reader = GenericPointReader::open_file(file)?;
                Ok((
                    reader.get_metadata().clone_into_box(),
                    reader.get_default_point_layout().clone(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        let bounds = metas_and_layouts
            .iter()
            .map(|(meta, _)| meta.bounds())
            .reduce(|a, b| match (a, b) {
                (Some(a), Some(b)) => Some(AABB::union(&a, &b)),
                _ => None,
            })
            .flatten();

        let points_per_file = metas_and_layouts
            .iter()
            .map(|(meta, _)| {
                meta.number_of_points()
                    .ok_or_else(|| anyhow!("Could not determine number of points"))
            })
            .collect::<Result<Vec<_>>>()?;
        let num_points = points_per_file.iter().copied().sum();

        let common_point_layout = match requested_point_layout {
            Some(requested_layout) => requested_layout,
            None => {
                let common_layout = metas_and_layouts[0].1.clone();
                if !metas_and_layouts
                    .iter()
                    .all(|(_, layout)| *layout == common_layout)
                {
                    bail!("Not all point cloud files have the same PointLayout. Pass a requested common PointLayout to this function if you want to read this point cloud with a unified layout");
                }
                common_layout
            }
        };

        Ok(Self {
            bounds,
            common_point_layout,
            files: matching_files,
            num_points,
            points_per_file,
        })
    }

    pub fn files(&self) -> &[PathBuf] {
        &self.files
    }

    pub fn num_points(&self) -> usize {
        self.num_points
    }

    pub fn stream<'a>(&'a self, chunk_size: usize) -> PointcloudStream<'a> {
        PointcloudStream::new(self, chunk_size)
    }
}

fn chunk_range(range: Range<usize>, chunk_size: usize) -> impl Iterator<Item = Range<usize>> {
    let len = range.len();
    let num_chunks = (len + chunk_size - 1) / chunk_size;
    (0..num_chunks)
        .map(move |chunk_id| (chunk_id * chunk_size)..((chunk_id + 1) * chunk_size).min(len))
}

pub struct PointcloudStream<'a> {
    pointcloud: &'a Pointcloud,
    buffer: VectorBuffer,
    chunks: Vec<ChunkInfo>,
    current_chunk: usize,
    current_reader: Option<(usize, GenericPointReader)>,
}

impl<'a> PointcloudStream<'a> {
    fn new(pointcloud: &'a Pointcloud, chunk_size: usize) -> Self {
        let chunks = pointcloud
            .points_per_file
            .iter()
            .enumerate()
            .flat_map(|(file_id, count_in_file)| {
                chunk_range(0..*count_in_file, chunk_size).map(move |chunk| ChunkInfo {
                    file_id,
                    point_range_in_file: chunk,
                })
            })
            .collect_vec();

        Self {
            buffer: VectorBuffer::new_from_layout(pointcloud.common_point_layout.clone()),
            chunks,
            current_chunk: 0,
            pointcloud,
            current_reader: None,
        }
    }

    fn read_next_chunk(&mut self) -> Result<(ChunkInfo, &VectorBuffer)> {
        let current_chunk = self.chunks[self.current_chunk].clone();
        self.current_chunk += 1;
        let reader = match self.current_reader.as_mut() {
            Some((file_id, reader)) => {
                if *file_id != current_chunk.file_id {
                    let new_reader = GenericPointReader::open_file(
                        &self.pointcloud.files[current_chunk.file_id],
                    )?;
                    *file_id = current_chunk.file_id;
                    *reader = new_reader;
                }
                reader
            }
            None => {
                let reader =
                    GenericPointReader::open_file(&self.pointcloud.files[current_chunk.file_id])?;
                self.current_reader = Some((current_chunk.file_id, reader));
                &mut self.current_reader.as_mut().unwrap().1
            }
        };
        let num_points_in_current_chunk = current_chunk.point_range_in_file.len();
        if self.buffer.len() < num_points_in_current_chunk {
            self.buffer.resize(num_points_in_current_chunk);
        }
        reader.read_into(&mut self.buffer, num_points_in_current_chunk)?;

        Ok((current_chunk, &self.buffer))
    }

    pub fn for_each<F: FnMut(Result<(ChunkInfo, &VectorBuffer)>)>(&mut self, mut f: F) {
        while self.current_chunk < self.chunks.len() {
            f(self.read_next_chunk())
        }
    }
}

// impl<'a> LendingIterator for PointcloudStream<'a> {
//     type Item<'b> = Result<(ChunkInfo, &'b VectorBuffer)> where Self: 'b, 'a: 'b;

//     fn next(&mut self) -> Option<Self::Item<'_>> {
//         if self.current_chunk == self.chunks.len() {
//             return None;
//         }

//         Some(self.read_next_chunk())
//     }
// }
