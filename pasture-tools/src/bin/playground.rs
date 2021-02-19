#![warn(clippy::all)]

use std::{
    fs::read_dir,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Result};
use clap::{App, Arg};
use log::{info, warn};
use pasture_core::{
    containers::{attribute, InterleavedVecPointStorage, PointBufferWriteable},
    layout::attributes::POSITION_3D,
    nalgebra::Vector3,
};
use pasture_io::base::PointReader;
use pasture_io::las::LASReader;

struct Args {
    pub input_files: Vec<PathBuf>,
    pub output_dir: PathBuf,
}

fn get_all_input_files<P: AsRef<Path>>(input_path: P) -> Result<Vec<PathBuf>> {
    let path = input_path.as_ref();
    if !path.exists() {
        return Err(anyhow!("Input path {} does not exist!", path.display()));
    }

    if path.is_file() {
        return Ok(vec![path.into()]);
    }

    if path.is_dir() {
        match read_dir(path) {
            Ok(dir_iter) => {
                let files: Result<Vec<_>, _> = dir_iter
                    .map(|dir_entry| dir_entry.map(|entry| entry.path()))
                    .collect();
                return files.map_err(|e| e.into());
            }
            Err(why) => return Err(anyhow!("{}", why)),
        };
    }

    Err(anyhow!(
        "Input path {} is neither file nor directory!",
        path.display()
    ))
}

fn is_valid_file(file: &Path) -> bool {
    if !file.exists() {
        warn!("File {} does not exist!", file.display());
        return false;
    }
    file.extension()
        .map(|ex| {
            let is_valid = ex == "laz";
            if !is_valid {
                warn!("File {} is no LAZ file!", file.display());
            }
            is_valid
        })
        .unwrap_or(false)
}

fn sanitize_input_files(files: &mut Vec<PathBuf>) {
    let valid_files = files
        .drain(..)
        .filter(|file| is_valid_file(file))
        .collect::<Vec<_>>();
    *files = valid_files;
}

fn get_args() -> Result<Args> {
    let matches = App::new("pasture playground")
    .version("0.1")
    .author("Pascal Bormann <pascal.bormann@igd.fraunhofer.de>")
    .arg(Arg::with_name("input").short("i").value_name("INPUT").help("Input file or directory. Directories are scanned (non-recursively) for point cloud files with known file extensions"))
    .arg(Arg::with_name("output").short("o").value_name("OUTPUT").help("Output directory"))
    .get_matches();

    let input_dir = matches.value_of("INPUT").unwrap();
    let mut input_files = get_all_input_files(input_dir)?;
    sanitize_input_files(&mut input_files);

    let output_dir: PathBuf = matches.value_of("OUTPUT").unwrap().into();
    if !output_dir.exists() {
        std::fs::create_dir_all(&output_dir)?;
    }

    Ok(Args {
        input_files,
        output_dir,
    })
}

fn hilbertize_chunk(buffer: &mut InterleavedVecPointStorage) {
    // TODO Hilbert curve for chunk bounds, or for file bounds?
    // TODO Collect positions as vector, or multiple calls to attributes()?

    //let chunk_bounds =
    //let positions = attributes::<Vector3<f64>>(buffer, &POSITION_3D);
}

fn hilbertize_file(laz_file: &Path, output_dir: &Path) -> Result<()> {
    let chunk_size = 50_000;
    let mut reader = LASReader::from_path(laz_file)?;

    let mut buffer = InterleavedVecPointStorage::with_capacity(
        chunk_size,
        reader.get_default_point_layout().clone(),
    );
    let remaining_points = reader.remaining_points();
    let chunks = (remaining_points + (chunk_size - 1)) / chunk_size;

    let file_name = laz_file
        .file_name()
        .ok_or_else(|| anyhow!("Could not get file name of file {}", laz_file.display()))?;
    let output_file_path = output_dir.join(file_name);

    for _ in 0..chunks {
        let points_in_chunk = std::cmp::min(chunk_size, remaining_points);
        reader.read_into(&mut buffer, points_in_chunk)?;

        buffer.clear();
    }

    Ok(())
}

fn main() -> Result<()> {
    pretty_env_logger::init();

    let args = get_args()?;

    let mut reader = LASReader::from_path("/home/pbormann/data/geodata/pointclouds/datasets/navvis_m6_3rdFloor/navvis_m6_HQ3rdFloor.laz")?;

    let total_points = reader.remaining_points();
    let mut remaining_points = total_points;
    let chunk_size = 50_000;
    let mut buffer = InterleavedVecPointStorage::with_capacity(
        chunk_size,
        reader.get_default_point_layout().clone(),
    );
    let chunks = (remaining_points + (chunk_size - 1)) / chunk_size;

    for _ in 0..chunks {
        let points_in_chunk = std::cmp::min(chunk_size, remaining_points);
        reader.read_into(&mut buffer, points_in_chunk)?;
        eprintln!("Read {} points", total_points - remaining_points);
        remaining_points -= points_in_chunk;
        buffer.clear();
    }

    Ok(())
}
