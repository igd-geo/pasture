/// Try to read LAS files as fast as possible, while doing some 'useful' work on the read points
/// The idea is to measure three different strategies for reading and parsing point cloud files:
/// - Sequential (perform IO, then parse, then search, all on the same thread)
/// - Parallel (perform IO, then parse, then search, in chunks on multiple parallel threads)
/// - Pipelined (perform IO on some threads, then parsing and searching on different threads)
///
/// TODO: There could also be a fully pipelined implementation where IO, parsing, and searching all happen on
/// different threads. A potential implementation would be a task-based processing system, where there are a
/// fixed number of threads and each thread processes a work-item and potentially produces new work-items. The
/// initial work-items are IO tasks (i.e. reading a chunk of data from the input file)
use std::{
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    num::NonZeroUsize,
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
    process::Command,
    sync::{atomic::AtomicUsize, Arc},
    thread::available_parallelism,
    time::{Duration, Instant},
};

use anyhow::{bail, Context, Result};
use clap::Parser;
use human_repr::HumanThroughput;
use las::{Header, Transform};
use memmap2::Advice;
use nalgebra::{clamp, Point3, Vector3};
use pasture_core::{
    containers::{
        BorrowedBuffer, ColumnarBuffer, ExternalMemoryBuffer, HashMapBuffer, InterleavedBuffer,
        MakeBufferFromLayout, OwningBuffer, VectorBuffer,
    },
    layout::{attributes::POSITION_3D, PointLayout},
    math::AABB,
};
use pasture_io::{
    base::{
        pipeline::{ConcurrentFileReader, ConcurrentPipeline, PipelineWorkItem},
        PointReader, SeekToPoint,
    },
    las::{
        get_default_las_converter, point_layout_from_las_metadata, LASMetadata, LASReader,
        ATTRIBUTE_LOCAL_LAS_POSITION,
    },
};
use rayon::prelude::*;

fn flush_disk_cache() -> Result<()> {
    eprintln!("Flushing disk cache");
    let sync_output = Command::new("sync")
        .output()
        .context("Could not execute sync command")?;
    if !sync_output.status.success() {
        bail!("Sync command failed with exit code {}", sync_output.status);
    }

    if std::env::consts::OS == "macos" {
        let purge_output = Command::new("purge")
            .output()
            .context("Could not execute purge command")?;
        if !purge_output.status.success() {
            bail!(
                "Purge command failed with exit code {}",
                purge_output.status
            );
        }
    } else if std::env::consts::OS == "linux" {
        let mut drop_caches = OpenOptions::new()
            .write(true)
            .open("/proc/sys/vm/drop_caches")?;
        drop_caches.write_all("3".as_bytes())?;
    }

    Ok(())
}

fn to_local_integer_position(
    position_world: &Vector3<f64>,
    las_transforms: &pasture_io::las_rs::Vector<Transform>,
) -> Vector3<i32> {
    let local_x = (position_world.x / las_transforms.x.scale) - las_transforms.x.offset;
    let local_y = (position_world.y / las_transforms.y.scale) - las_transforms.y.offset;
    let local_z = (position_world.z / las_transforms.z.scale) - las_transforms.z.offset;
    Vector3::new(
        clamp(local_x, i32::MIN as f64, i32::MAX as f64) as i32,
        clamp(local_y, i32::MIN as f64, i32::MAX as f64) as i32,
        clamp(local_z, i32::MIN as f64, i32::MAX as f64) as i32,
    )
}

#[derive(Parser)]
struct Args {
    file: PathBuf,
}

trait Query<B: for<'a> BorrowedBuffer<'a>>: Sync {
    fn process(&self, data: &B) -> Result<()>;
    fn finish(&self) -> Result<()>;
    fn required_point_layout(&self) -> PointLayout;
}

struct BoundsQuery {
    bounds: AABB<f64>,
    num_matches: AtomicUsize,
}

impl BoundsQuery {
    pub fn new(bounds: AABB<f64>) -> Self {
        Self {
            bounds,
            num_matches: AtomicUsize::new(0),
        }
    }
}

impl<B: for<'a> BorrowedBuffer<'a>> Query<B> for BoundsQuery {
    fn process(&self, data: &B) -> Result<()> {
        let mut num_matches: usize = 0;
        let positions = data.view_attribute::<Vector3<f64>>(&POSITION_3D);
        for pos in positions.into_iter() {
            if !self.bounds.contains(&pos.into()) {
                continue;
            }
            num_matches += 1;
        }

        self.num_matches
            .fetch_add(num_matches, std::sync::atomic::Ordering::SeqCst);

        Ok(())
    }

    fn finish(&self) -> Result<()> {
        println!(
            "Matches: {}",
            self.num_matches.load(std::sync::atomic::Ordering::SeqCst)
        );
        Ok(())
    }

    fn required_point_layout(&self) -> PointLayout {
        IntoIterator::into_iter([POSITION_3D]).collect()
    }
}

struct LocalBoundsQuery {
    local_bounds: AABB<i32>,
    num_matches: AtomicUsize,
}

impl LocalBoundsQuery {
    pub fn new(global_bounds: AABB<f64>, las_header: &Header) -> Self {
        let local_min =
            to_local_integer_position(&global_bounds.min().coords, las_header.transforms());
        let local_max =
            to_local_integer_position(&global_bounds.max().coords, las_header.transforms());
        Self {
            local_bounds: AABB::from_min_max(local_min.into(), local_max.into()),
            num_matches: AtomicUsize::new(0),
        }
    }
}

impl<B: for<'a> BorrowedBuffer<'a>> Query<B> for LocalBoundsQuery {
    fn process(&self, data: &B) -> Result<()> {
        let local_positions = data.view_attribute::<Vector3<i32>>(&ATTRIBUTE_LOCAL_LAS_POSITION);
        for pos in local_positions.into_iter() {
            if !self.local_bounds.contains(&pos.into()) {
                continue;
            }
            self.num_matches
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }

        Ok(())
    }

    fn finish(&self) -> Result<()> {
        println!(
            "Matches: {}",
            self.num_matches.load(std::sync::atomic::Ordering::SeqCst)
        );
        Ok(())
    }

    fn required_point_layout(&self) -> PointLayout {
        IntoIterator::into_iter([ATTRIBUTE_LOCAL_LAS_POSITION]).collect()
    }
}

fn read_sequential(
    file: &Path,
    exact_memory_layout: bool,
    query: impl Query<VectorBuffer>,
) -> Result<()> {
    let mut reader = LASReader::from_path(file, exact_memory_layout)?;
    let required_layout = query.required_point_layout();
    let mut data = VectorBuffer::new_from_layout(required_layout);
    data.resize(reader.remaining_points());
    reader.read_into(&mut data, reader.remaining_points())?;
    query.process(&data)?;
    query.finish()?;
    Ok(())
}

fn read_parallel(
    file: &Path,
    exact_memory_layout: bool,
    query: impl Query<VectorBuffer>,
) -> Result<()> {
    let num_threads = available_parallelism()?.get();
    (0..num_threads)
        .into_par_iter()
        .map(|id| -> Result<()> {
            let t1 = Instant::now();
            let mut reader = LASReader::from_path(file, exact_memory_layout)?;
            let chunk_size = (reader.remaining_points() + num_threads - 1) / num_threads;
            reader.seek_point(SeekFrom::Start((id * chunk_size) as u64))?;
            let clamped_chunk_size = chunk_size.min(reader.remaining_points());

            let required_layout = query.required_point_layout();
            let mut data = VectorBuffer::new_from_layout(required_layout);
            data.resize(clamped_chunk_size);
            reader.read_into(&mut data, clamped_chunk_size)?;
            let read_parse_duration = t1.elapsed().as_secs_f64();
            let read_parse_throughput = clamped_chunk_size as f64 / read_parse_duration;
            let read_parse_throughput_bytes = read_parse_throughput
                * reader.get_default_point_layout().size_of_point_entry() as f64;
            eprintln!(
                "Read&parse throughput: {:.2} ({:.2})",
                read_parse_throughput.human_throughput("points"),
                read_parse_throughput_bytes.human_throughput_bytes()
            );

            let t2 = Instant::now();
            query.process(&data)?;
            let filter_throughput = data.len() as f64 / t2.elapsed().as_secs_f64();
            eprintln!(
                "Filter throughput: {:.2}",
                filter_throughput.human_throughput("points")
            );
            Ok(())
        })
        .collect::<Result<_>>()?;
    query.finish()?;
    Ok(())
}

fn read_pipelined(
    file: &Path,
    exact_memory_layout: bool,
    las_metadata_in_exact_layout: &LASMetadata,
    query: impl Query<VectorBuffer>,
) -> Result<()> {
    if exact_memory_layout {
        todo!()
    } else {
        let las_metadata_parsed_layout = LASReader::from_path(file, false)?.las_metadata().clone();
        let exact_point_layout =
            point_layout_from_las_metadata(las_metadata_in_exact_layout, true)?;

        const DEFAULT_CHUNK_SIZE: usize = 2 * 1024 * 1024;
        let las_header = las_metadata_parsed_layout.raw_las_header().unwrap();
        let raw_las_header = las_header.clone().into_raw()?;
        let first_point_record_byte = raw_las_header.offset_to_point_data as usize;
        // Clamp default chunk size to multiple of point record size
        let chunk_size = (DEFAULT_CHUNK_SIZE / exact_point_layout.size_of_point_entry() as usize)
            * exact_point_layout.size_of_point_entry() as usize;

        let query = &query;

        // Launch concurrent reader and a bunch of parse threads
        let (reader, memory_receiver) = ConcurrentFileReader::read(
            file,
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(chunk_size).unwrap(),
            first_point_record_byte,
        );

        std::thread::scope(|scope| -> Result<()> {
            let concurrency = available_parallelism()?.get();
            for _ in 0..concurrency {
                scope.spawn(|| -> Result<()> {
                    let required_layout = query.required_point_layout();
                    let converter = get_default_las_converter(
                        &exact_point_layout,
                        &required_layout,
                        las_header,
                    )?;
                    while let Ok(data) = memory_receiver.recv() {
                        // Parse data chunk
                        let input_buffer =
                            ExternalMemoryBuffer::new(&data.data, exact_point_layout.clone());
                        let converted_buffer = converter.convert::<VectorBuffer, _>(&input_buffer);
                        query.process(&converted_buffer)?;
                    }
                    Ok(())
                });
            }

            Ok(())
        })?;

        reader.join()?;

        query.finish()?;

        Ok(())
    }
}

struct State<Q: Query<VectorBuffer>> {
    file: PathBuf,
    exact_point_layout: PointLayout,
    query_point_layout: PointLayout,
    las_header: Header,
    query: Q,
}

const TASK_TYPE_IO: usize = 1;
const TASK_TYPE_PARSE: usize = 2;
const TASK_TYPE_FILTER: usize = 3;

fn make_filter_task<Q: Query<VectorBuffer> + Sized + Send + 'static>(
    points: VectorBuffer,
    state: Arc<State<Q>>,
) -> PipelineWorkItem {
    PipelineWorkItem::new(
        TASK_TYPE_FILTER,
        points.len(),
        move || -> Result<Vec<PipelineWorkItem>> {
            let _span = tracy_client::span!("filter");
            state.query.process(&points)?;
            // eprintln!("Filtered {} points", points.len());
            Ok(vec![])
        },
    )
}

fn make_parse_task<Q: Query<VectorBuffer> + Sized + Send + 'static>(
    data: Vec<u8>,
    state: Arc<State<Q>>,
) -> PipelineWorkItem {
    let num_points = data.len() / state.exact_point_layout.size_of_point_entry() as usize;
    PipelineWorkItem::new(
        TASK_TYPE_PARSE,
        num_points,
        move || -> Result<Vec<PipelineWorkItem>> {
            let _span = tracy_client::span!("convert");
            let converter = {
                let _span = tracy_client::span!("make_converter");
                get_default_las_converter(
                    &state.exact_point_layout,
                    &state.query_point_layout,
                    &state.las_header,
                )?
            };
            let source_buffer = ExternalMemoryBuffer::new(&data, state.exact_point_layout.clone());
            let parsed_points = converter.convert::<VectorBuffer, _>(&source_buffer);
            // eprintln!("Parsed {} points", parsed_points.len());
            Ok(vec![make_filter_task(parsed_points, state.clone())])
        },
    )
}

fn make_read_task<Q: Query<VectorBuffer> + Sized + Send + 'static>(
    chunk_start_byte: usize,
    chunk_size: usize,
    use_mmap: bool,
    state: Arc<State<Q>>,
) -> PipelineWorkItem {
    PipelineWorkItem::new(
        TASK_TYPE_IO,
        chunk_size,
        move || -> Result<Vec<PipelineWorkItem>> {
            let _span = tracy_client::span!("read");
            // Load the binary chunk and create a new parse task
            // TODO Maybe share the file pointer? Don't know what the overhead of File::open is
            let timer = Instant::now();
            let buf = if use_mmap {
                let mmap = unsafe { memmap2::Mmap::map(&File::open(&state.file)?)? };
                // TODO Advice::Sequential and ::Random make things very slow on MacOS
                // mmap.advise(Advice::Sequential)?;
                // mmap.advise(Advice::Random)?;
                mmap.advise(Advice::Normal)?;
                mmap[chunk_start_byte..(chunk_start_byte + chunk_size)].to_vec()
            } else {
                let mut file = {
                    let _span = tracy_client::span!("open file");
                    File::open(&state.file)?
                };
                file.seek(SeekFrom::Start(chunk_start_byte as u64))?;
                let mut buf = vec![0; chunk_size];
                {
                    let _span = tracy_client::span!("read file");
                    file.read_exact(&mut buf)?;
                }
                buf
            };
            let throughput_mibs =
                (chunk_size as f64 / (1024.0 * 1024.0)) / timer.elapsed().as_secs_f64();
            tracy_client::plot!("read_throughput", throughput_mibs);
            // eprintln!("Read chunk @ {chunk_start_byte}");
            Ok(vec![make_parse_task(buf, state.clone())])
        },
    )
}

fn read_fully_pipelined<Q: Query<VectorBuffer> + Sized + Send + 'static>(
    file: &Path,
    query: Q,
) -> Result<()> {
    // Require task-based processing system and some clever work-stealing: There should be groups of threads
    // that prioritize one type of task (e.g. IO threads, parse threads, search threads), but if any of these
    // threads starves for work, it can steal from a queue of another group of threads

    let las_metadata = LASReader::from_path(file, true)?.las_metadata().clone();
    let las_header = las_metadata.raw_las_header().unwrap();
    let exact_point_layout = point_layout_from_las_metadata(&las_metadata, true)?;
    let query_point_layout = query.required_point_layout();

    let num_points = las_metadata.point_count();
    let size_of_point = exact_point_layout.size_of_point_entry() as usize;
    let num_bytes = num_points * size_of_point;
    let start_of_point_records_in_file =
        las_header.clone().into_raw()?.offset_to_point_data as usize;

    let state = Arc::new(State {
        file: file.to_path_buf(),
        exact_point_layout,
        las_header: las_header.clone(),
        query,
        query_point_layout,
    });

    let pipeline = ConcurrentPipeline::new(
        [
            (TASK_TYPE_IO, num_bytes),
            (TASK_TYPE_PARSE, num_points),
            (TASK_TYPE_FILTER, num_points),
        ],
        available_parallelism()?.get(),
    );

    const IO_CHUNK_SIZE: usize = 2 * 1024 * 1024;
    let actual_io_chunk_size = (IO_CHUNK_SIZE / size_of_point) * size_of_point;
    let num_chunks = (num_bytes + actual_io_chunk_size - 1) / actual_io_chunk_size;
    pipeline.run((0..num_chunks).map(|chunk_id| {
        let chunk_size = if chunk_id == num_chunks - 1 {
            num_bytes - (chunk_id * actual_io_chunk_size)
        } else {
            actual_io_chunk_size
        };
        assert!(chunk_size % size_of_point == 0);
        let chunk_start_byte = start_of_point_records_in_file + chunk_id * actual_io_chunk_size;

        make_read_task(chunk_start_byte, chunk_size, false, state.clone())
    }))?;

    state.query.finish()?;

    Ok(())
}

fn get_bounds_query() -> BoundsQuery {
    BoundsQuery::new(AABB::from_min_max(
        Point3::new(-1.0, -1.0, -1.0),
        Point3::new(1.0, 1.0, 1.0),
    ))
}

fn get_local_bounds_query(las_header: &Header) -> LocalBoundsQuery {
    LocalBoundsQuery::new(
        AABB::from_min_max(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0)),
        las_header,
    )
}

fn main() -> Result<()> {
    let args = Args::parse();
    let _tracy_client = tracy_client::Client::start();

    let metadata = {
        let timer = Instant::now();
        let meta = LASReader::from_path(&args.file, true)?
            .las_metadata()
            .clone();
        eprintln!("Getting LAS metadata: {} ms", timer.elapsed().as_millis());
        meta
    };
    let las_header = metadata.raw_las_header().unwrap().clone();
    let total_file_size = args.file.metadata()?.len() as usize;
    let total_file_size_mib = total_file_size as f64 / (1024.0 * 1024.0);
    let get_throughput =
        |elapsed_time: &Duration| -> f64 { total_file_size_mib / elapsed_time.as_secs_f64() };

    let purge_cache = [true, false];
    for purge in &purge_cache {
        if *purge {
            flush_disk_cache()?;
        }
        {
            let timer = Instant::now();
            read_sequential(&args.file, false, get_bounds_query())?;
            let elapsed = timer.elapsed();
            eprintln!(
                "Sequential - {} - Parsed memory layout - {} ms - {} MiB/s",
                if *purge { "cold cache" } else { "hot cache" },
                elapsed.as_millis(),
                get_throughput(&elapsed),
            );
        }
    }

    for purge in &purge_cache {
        if *purge {
            flush_disk_cache()?;
        }
        {
            let timer = Instant::now();
            read_sequential(&args.file, true, get_local_bounds_query(&las_header))?;
            let elapsed = timer.elapsed();
            eprintln!(
                "Sequential - {} - Exact memory layout - {} ms - {} MiB/s",
                if *purge { "cold cache" } else { "hot cache" },
                elapsed.as_millis(),
                get_throughput(&elapsed),
            );
        }
    }

    for purge in &purge_cache {
        if *purge {
            flush_disk_cache()?;
        }
        {
            let timer = Instant::now();
            read_parallel(&args.file, false, get_bounds_query())?;
            let elapsed = timer.elapsed();
            eprintln!(
                "Parallel - {} - Parsed memory layout - {} ms - {} MiB/s",
                if *purge { "cold cache" } else { "hot cache" },
                elapsed.as_millis(),
                get_throughput(&elapsed),
            );
        }
    }

    for purge in &purge_cache {
        if *purge {
            flush_disk_cache()?;
        }
        {
            let timer = Instant::now();
            read_parallel(&args.file, true, get_local_bounds_query(&las_header))?;
            let elapsed = timer.elapsed();
            eprintln!(
                "Parallel - {} - Exact memory layout - {} ms - {} MiB/s",
                if *purge { "cold cache" } else { "hot cache" },
                elapsed.as_millis(),
                get_throughput(&elapsed),
            );
        }
    }

    for purge in &purge_cache {
        if *purge {
            flush_disk_cache()?;
        }
        {
            let timer = Instant::now();
            read_pipelined(&args.file, false, &metadata, get_bounds_query())?;
            let elapsed = timer.elapsed();
            eprintln!(
                "Pipelined - {} - Parsed memory layout - {} ms - {} MiB/s",
                if *purge { "cold cache" } else { "hot cache" },
                elapsed.as_millis(),
                get_throughput(&elapsed),
            );
        }
    }

    for purge in &purge_cache {
        if *purge {
            flush_disk_cache()?;
        }
        {
            let timer = Instant::now();
            read_fully_pipelined(&args.file, get_bounds_query())?;
            let elapsed = timer.elapsed();
            eprintln!(
                "Pipelined - {} - Parsed memory layout - {} ms - {} MiB/s",
                if *purge { "cold cache" } else { "hot cache" },
                elapsed.as_millis(),
                get_throughput(&elapsed),
            );
        }
    }

    Ok(())
}
