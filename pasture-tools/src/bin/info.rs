use std::{path::PathBuf, time::Instant};

use anyhow::Result;
use clap::{App, Arg};
use pasture_algorithms::minmax::minmax_attribute;
use pasture_core::{
    containers::{BorrowedBuffer, OwningBuffer, VectorBuffer},
    layout::attributes::NIR,
    layout::attributes::NUMBER_OF_RETURNS,
    layout::attributes::POINT_SOURCE_ID,
    layout::attributes::RETURN_NUMBER,
    layout::attributes::SCAN_ANGLE_RANK,
    layout::attributes::SCAN_DIRECTION_FLAG,
    layout::attributes::USER_DATA,
    layout::attributes::{
        CLASSIFICATION, COLOR_RGB, EDGE_OF_FLIGHT_LINE, GPS_TIME, INTENSITY, POSITION_3D,
    },
    layout::PointLayout,
    math::MinMax,
    nalgebra::Vector3,
};
use pasture_io::base::{GenericPointReader, PointReader, SeekToPoint};

struct Args {
    pub input_file: PathBuf,
    pub detailed: bool,
}

fn get_args() -> Result<Args> {
    let matches = App::new("pasture playground")
        .version("0.1")
        .author("Pascal Bormann <pascal.bormann@igd.fraunhofer.de>")
        .about("Prints information about the given point cloud file")
        .arg(
            Arg::with_name("INPUT")
                .short("i")
                .takes_value(true)
                .value_name("INPUT")
                .help("Input point cloud file")
                .required(true),
        )
        .arg(
            Arg::with_name("DETAILED")
                .short("d")
                .long("detailed")
                .help("Output a detailed analysis of the point cloud file, showing min and max values for all point attributes")
        )
        .get_matches();

    let input_file = PathBuf::from(matches.value_of("INPUT").unwrap());
    let detailed = matches.is_present("DETAILED");

    Ok(Args {
        input_file,
        detailed,
    })
}

fn print_attributes(point_layout: &PointLayout) {
    println!("Attributes");
    for attribute in point_layout.attributes() {
        println!("\t{}", attribute.name());
    }
}

macro_rules! minmax_chunk {
    ($minmax_tuple:ident, $buffer:ident, $attribute:expr, $type:ty) => {
        if $buffer
            .point_layout()
            .has_attribute_with_name($attribute.name())
        {
            let chunk_minmax: ($type, $type) = minmax_attribute(&$buffer, &$attribute).unwrap();
            match $minmax_tuple {
                None => $minmax_tuple = Some(chunk_minmax),
                Some((old_min, old_max)) => {
                    $minmax_tuple = Some((
                        old_min.infimum(&chunk_minmax.0),
                        old_max.supremum(&chunk_minmax.1),
                    ));
                }
            }
        }
    };
}

fn analyze_file<R: PointReader + SeekToPoint>(reader: &mut R) -> Result<()> {
    print_attributes(reader.get_default_point_layout());

    let total_points = reader.point_count()?;
    if total_points == 0 {
        return Ok(());
    }

    let t_start = Instant::now();

    println!("Analyzing minimum and maximum values for all point attributes...");

    let chunk_size = 1_000_000;
    let mut buffer =
        VectorBuffer::with_capacity(chunk_size, reader.get_default_point_layout().clone());
    let num_chunks = (total_points + chunk_size - 1) / chunk_size;
    //let num_chunks = 4;

    // We investigate all builtin attributes, even though not all might be present in the file
    let mut minmax_position = None;
    let mut minmax_intensity = None;
    let mut minmax_return_number = None;
    let mut minmax_number_of_returns = None;
    //TODO Extended bit attributes (classification flags, scanner channels)
    let mut minmax_scan_direction_flag = None;
    let mut minmax_edge_of_flight_line = None;
    let mut minmax_classification = None;
    let mut minmax_scan_angle_rank = None;
    let mut minmax_user_data = None;
    let mut minmax_point_source_id = None;
    let mut minmax_color_rgb = None;
    let mut minmax_gps_time = None;
    let mut minmax_nir = None;
    // TODO Waveform data

    for idx in 0..num_chunks {
        buffer.clear();

        let num_points_in_chunk = std::cmp::min(chunk_size, total_points - (idx * chunk_size));
        reader.read_into(&mut buffer, num_points_in_chunk)?;

        minmax_chunk!(minmax_position, buffer, POSITION_3D, Vector3<f64>);
        minmax_chunk!(minmax_intensity, buffer, INTENSITY, u16);
        minmax_chunk!(minmax_return_number, buffer, RETURN_NUMBER, u8);
        minmax_chunk!(minmax_number_of_returns, buffer, NUMBER_OF_RETURNS, u8);
        minmax_chunk!(minmax_scan_direction_flag, buffer, SCAN_DIRECTION_FLAG, u8);
        minmax_chunk!(minmax_edge_of_flight_line, buffer, EDGE_OF_FLIGHT_LINE, u8);
        minmax_chunk!(minmax_classification, buffer, CLASSIFICATION, u8);
        minmax_chunk!(minmax_scan_angle_rank, buffer, SCAN_ANGLE_RANK, i8);
        minmax_chunk!(minmax_user_data, buffer, USER_DATA, u8);
        minmax_chunk!(minmax_point_source_id, buffer, POINT_SOURCE_ID, u16);
        minmax_chunk!(minmax_color_rgb, buffer, COLOR_RGB, Vector3<u16>);
        minmax_chunk!(minmax_gps_time, buffer, GPS_TIME, f64);
        minmax_chunk!(minmax_nir, buffer, NIR, u16);

        // eprintln!(
        //     "Finding minmax of chunk: {:.2}s",
        //     inner_t_start.elapsed().as_secs_f64()
        // );
    }

    if let Some((min, max)) = minmax_position {
        println!("\tX:                      {}  {}", min.x, max.x);
        println!("\tY:                      {}  {}", min.y, max.y);
        println!("\tZ:                      {}  {}", min.z, max.z);
    }
    if let Some((min, max)) = minmax_intensity {
        println!("\tIntensity:              {}  {}", min, max);
    }
    if let Some((min, max)) = minmax_return_number {
        println!("\tReturn number:          {}  {}", min, max);
    }
    if let Some((min, max)) = minmax_number_of_returns {
        println!("\tNumber of returns:      {}  {}", min, max);
    }
    if let Some((min, max)) = minmax_scan_direction_flag {
        println!("\tScan direction flag:    {}  {}", min, max);
    }
    if let Some((min, max)) = minmax_edge_of_flight_line {
        println!("\tEdge of flight line:    {}  {}", min, max);
    }
    if let Some((min, max)) = minmax_classification {
        println!("\tClassification:         {}  {}", min, max);
    }
    if let Some((min, max)) = minmax_scan_angle_rank {
        println!("\tScan angle rank:        {}  {}", min, max);
    }
    if let Some((min, max)) = minmax_user_data {
        println!("\tUser data:              {}  {}", min, max);
    }
    if let Some((min, max)) = minmax_point_source_id {
        println!("\tPoint source ID:        {}  {}", min, max);
    }
    if let Some((min, max)) = minmax_color_rgb {
        println!("\tColor R:                {}  {}", min.x, max.x);
        println!("\tColor G:                {}  {}", min.y, max.y);
        println!("\tColor B:                {}  {}", min.z, max.z);
    }
    if let Some((min, max)) = minmax_gps_time {
        println!("\tGPS time:               {}  {}", min, max);
    }
    if let Some((min, max)) = minmax_nir {
        println!("\tNIR:                    {}  {}", min, max);
    }

    println!("Took {:.2}s", t_start.elapsed().as_secs_f64());

    Ok(())
}

fn main() -> Result<()> {
    let args = get_args()?;
    let mut reader = GenericPointReader::open_file(&args.input_file)?;
    let meta = reader.get_metadata();
    println!("pasture info report for {}", args.input_file.display());
    println!("{}", meta);

    if args.detailed {
        analyze_file(&mut reader)?;
    }

    Ok(())
}
