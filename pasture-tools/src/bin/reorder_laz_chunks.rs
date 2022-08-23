#![warn(clippy::all)]

use std::{
    convert::TryFrom,
    fs::read_dir,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Result};
use clap::{value_t, App, Arg};
use log::{info, warn};
use morton_index::{dimensions::OctantOrdering, FixedDepthMortonIndex3D64};
use pasture_algorithms::bounds::calculate_bounds;
use pasture_core::{
    containers::PointBuffer,
    containers::{
        InterleavedVecPointStorage, PerAttributeVecPointStorage, PointBufferExt,
        PointBufferWriteable, OwningPointBuffer,
    },
    layout::attributes::POSITION_3D,
    math::AABB,
    nalgebra::{Point3, Vector3},
};
use pasture_io::las::LASReader;
use pasture_io::{base::PointReader, base::PointWriter, las::LASWriter};

struct Args {
    pub input_files: Vec<PathBuf>,
    pub output_dir: PathBuf,
    pub take_first_n: usize,
    pub use_chunk_bounds: bool,
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
    .arg(Arg::with_name("INPUT").short("i").takes_value(true).value_name("INPUT").help("Input file or directory. Directories are scanned (non-recursively) for point cloud files with known file extensions").required(true))
    .arg(Arg::with_name("OUTPUT").short("o").takes_value(true).value_name("OUTPUT").help("Output directory").required(true))
    .arg(Arg::with_name("N").short("n").takes_value(true).value_name("N").help("Take the first N points of each LAZ chunk").default_value("50000"))
    .arg(Arg::with_name("LOCAL_BOUNDS").long("chunk-bounds").help("Use the local bounds of each chunk for point reordering"))
    .get_matches();

    let input_dir = matches.value_of("INPUT").unwrap();
    let mut input_files = get_all_input_files(input_dir)?;
    sanitize_input_files(&mut input_files);

    let output_dir: PathBuf = matches.value_of("OUTPUT").unwrap().into();
    if !output_dir.exists() {
        std::fs::create_dir_all(&output_dir)?;
    }

    let take_first_n = value_t!(matches, "N", usize).unwrap();
    let use_chunk_bounds = matches.is_present("LOCAL_BOUNDS");

    Ok(Args {
        input_files,
        output_dir,
        take_first_n,
        use_chunk_bounds,
    })
}

fn reversed_morton_index(point: &Point3<f64>, bounds: &AABB<f64>) -> FixedDepthMortonIndex3D64 {
    let normalized_extent = (2.0_f64.powf(21 as f64)) / bounds.extent().x;
    let normalized_point = (point - bounds.min()).component_mul(&Vector3::new(
        normalized_extent,
        normalized_extent,
        normalized_extent,
    ));

    let max_index = (1_u64 << 21) - 1;
    let grid_index_x = u64::min(normalized_point.x as u64, max_index);
    let grid_index_y = u64::min(normalized_point.y as u64, max_index);
    let grid_index_z = u64::min(normalized_point.z as u64, max_index);

    // Inefficient implementation...
    let mut rev_cells = FixedDepthMortonIndex3D64::from_grid_index(
        Vector3::new(
            grid_index_x as usize,
            grid_index_y as usize,
            grid_index_z as usize,
        ),
        OctantOrdering::XYZ,
    )
    .cells()
    .collect::<Vec<_>>();
    rev_cells.reverse();
    FixedDepthMortonIndex3D64::try_from(rev_cells.as_slice()).unwrap()
}

fn hilbertize_chunk(
    input_buffer: &InterleavedVecPointStorage,
    output_buffer: &mut PerAttributeVecPointStorage,
    point_count: usize,
    file_bounds_cubed: &AABB<f64>,
    take_first_n: usize,
    use_chunk_bounds: bool,
) {
    // Sort the chunk in reverse Morton index order and take the first N points. Use either the chunks local bounds
    // or the file bounds for calculating the Morton indices

    let bounds = if use_chunk_bounds {
        let chunk_bounds =
            calculate_bounds(input_buffer).expect("Could not calculate chunk bounds!");
        chunk_bounds.as_cubic()
    } else {
        file_bounds_cubed.clone()
    };

    let mut indexed_morton_indices = {
        let positions = input_buffer.iter_attribute::<Vector3<f64>>(&POSITION_3D);

        positions
            .map(|pos| reversed_morton_index(&Point3::new(pos.x, pos.y, pos.z), &bounds))
            .enumerate()
            .take(point_count)
            .collect::<Vec<_>>()
    };
    indexed_morton_indices.sort_by(|a, b| a.1.cmp(&b.1));

    // TODO I have a real problem here: In an InterleavedPointBuffer I can't access/mutate a single attribute (e.g. COLOR_RGB)
    // but I have to. I can only access whole points as &mut, however I don't know the strongly typed T that corresponds to the
    // points in the buffer :/
    // Accessing mutable borrows of specific attributes in an interleaved buffer might be UB?! Maybe only if they are unaligned, but
    // then again we can't guarantee that the underlying PointLayout has all attributes properly aligned :(

    //Buffer should have reorder operation ?! General question: Which operations do we have to implement per-buffer and what
    // could we do with the standard algorithms

    for (source_index, _) in indexed_morton_indices.iter().take(take_first_n) {
        output_buffer.push(&input_buffer.slice(*source_index..*source_index + 1));
    }
}

fn reorder_file(laz_file: &Path, args: &Args) -> Result<()> {
    info!("Processing {}", laz_file.display());

    let chunk_size = 50_000;
    let mut reader = LASReader::from_path(laz_file)?;

    let mut input_buffer = InterleavedVecPointStorage::with_capacity(
        chunk_size,
        reader.get_default_point_layout().clone(),
    );
    let mut output_buffer =
        PerAttributeVecPointStorage::with_capacity(chunk_size, input_buffer.point_layout().clone());

    let mut remaining_points = reader.remaining_points();
    let chunks = (remaining_points + (chunk_size - 1)) / chunk_size;
    let bounds = reader.get_metadata().bounds().unwrap().as_cubic();

    let file_name = laz_file
        .file_name()
        .ok_or_else(|| anyhow!("Could not get file name of file {}", laz_file.display()))?;
    let output_file_path = args.output_dir.join(file_name);
    let output_header = reader.header().clone();
    let mut writer = LASWriter::from_path_and_header(&output_file_path, output_header)?;

    for chunk_index in 0..chunks {
        let points_in_chunk = std::cmp::min(chunk_size, remaining_points);
        remaining_points -= points_in_chunk;
        reader.read_into(&mut input_buffer, points_in_chunk)?;

        hilbertize_chunk(
            &input_buffer,
            &mut output_buffer,
            points_in_chunk,
            &bounds,
            args.take_first_n,
            args.use_chunk_bounds,
        );
        writer.write(&output_buffer)?;

        input_buffer.clear();
        output_buffer.clear();

        info!("Chunk {}/{}", chunk_index + 1, chunks);
    }

    info!("Wrote file {}", output_file_path.display());

    Ok(())
}

fn main() -> Result<()> {
    pretty_env_logger::init();

    let args = get_args()?;

    info!("Processing {} files", args.input_files.len());

    for file in args.input_files.iter() {
        reorder_file(file, &args)?;
    }

    Ok(())
}
