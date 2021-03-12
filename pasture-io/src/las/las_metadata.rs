use std::{any::Any, convert::TryInto, fmt::Display, path::Path};

use anyhow::{anyhow, Result};
use chrono::Datelike;
use las::{Bounds, Header};
use las_rs::{Vector, Vlr};
use pasture_core::{math::AABB, meta::Metadata, nalgebra::Point3};

/// Contains constants for possible named fields in a `LASMetadata` structure
pub mod named_fields {
    /// File source ID as per the LAS 1.4 specification
    pub const FILE_SOURCE_I_D: &'static str = "LASFIELD_FileSourceID";
    /// LAS file version
    pub const VERSION: &'static str = "LASFIELD_Version";
    /// System identifier as per the LAS 1.4 specification
    pub const SYSTEM_IDENTIFIER: &'static str = "LASFIELD_SystemIdentifier";
    /// Information about the generating software
    pub const GENERATING_SOFTWARE: &'static str = "LASFIELD_GeneratingSoftware";
    /// Day of year on which the file was created as per the LAS 1.4 specification
    pub const FILE_CREATION_DAY_OF_YEAR: &'static str = "LASFIELD_FileCreationDayOfYear";
    /// Year in which the file was created
    pub const FILE_CREATION_YEAR: &'static str = "LASFIELD_FileCreationYear";

    //TODO More fields
}

/// Converts a las-rs `Bounds` type into a pasture-core bounding box (`AABB<f64>`)
pub fn las_bounds_to_pasture_bounds(las_bounds: Bounds) -> AABB<f64> {
    let min_point = Point3::new(las_bounds.min.x, las_bounds.min.y, las_bounds.min.z);
    let max_point = Point3::new(las_bounds.max.x, las_bounds.max.y, las_bounds.max.z);
    AABB::from_min_max_unchecked(min_point, max_point)
}

/// Converts a pasture-core bounding box (`AABB<f64>`) into a las-rs `Bounds` type
pub fn pasture_bounds_to_las_bounds(bounds: &AABB<f64>) -> Bounds {
    Bounds {
        min: Vector {
            x: bounds.min().x,
            y: bounds.min().y,
            z: bounds.min().z,
        },
        max: Vector {
            x: bounds.max().x,
            y: bounds.max().y,
            z: bounds.max().z,
        },
    }
}

/// Tries to determine whether the given `path` represents a compressed LAZ file or an uncompressed LAS file
pub fn path_is_compressed_las_file<P: AsRef<Path>>(path: P) -> Result<bool> {
    path.as_ref()
        .extension()
        .map(|extension| extension == "laz")
        .ok_or(anyhow!(
            "Could not determine file extension of file {}",
            path.as_ref().display()
        ))
}

fn display_vlr(vlr: &Vlr, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    writeln!(f, "\t{}", vlr.description)?;
    writeln!(f, "\t\tUser:      {}", vlr.user_id)?;
    writeln!(f, "\t\tRecord:    {}", vlr.record_id)?;
    if vlr.data.len() > 32 {
        writeln!(f, "\t\tData:      {:?}...", &vlr.data[0..32])
    } else {
        writeln!(f, "\t\tData:      {:?}", vlr.data)
    }
}

/// `Metadata` implementation for LAS/LAZ files
#[derive(Debug, Clone)]
pub struct LASMetadata {
    bounds: AABB<f64>,
    point_count: usize,
    point_format: u8,
    raw_las_header: Option<Header>,
}

impl LASMetadata {
    /// Creates a new `LASMetadata` from the given parameters
    /// ```
    /// # use pasture_io::las::LASMetadata;
    /// # use pasture_core::math::AABB;
    /// # use pasture_core::nalgebra::Point3;
    ///
    /// let min = Point3::new(0.0, 0.0, 0.0);
    /// let max = Point3::new(1.0, 1.0, 1.0);
    /// let metadata = LASMetadata::new(AABB::from_min_max(min, max), 1024, 0);
    /// ```
    pub fn new(bounds: AABB<f64>, point_count: usize, point_format: u8) -> Self {
        Self {
            bounds,
            point_count,
            point_format,
            raw_las_header: None,
        }
    }

    /// Returns the number of points for the associated `LASMetadata`
    pub fn point_count(&self) -> usize {
        self.point_count
    }

    /// Returns the LAS point format for the associated `LASMetadata`
    pub fn point_format(&self) -> u8 {
        self.point_format
    }

    /// Returns the raw LAS header for the associated `LASMetadata`. This value is only present if the
    /// associated `LASMetadata` was created from a raw LAS header
    pub fn raw_las_header(&self) -> Option<&Header> {
        self.raw_las_header.as_ref()
    }
}

impl Display for LASMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "LAS Metadata")?;
        writeln!(f, "\tBounds (min):                {}", self.bounds.min())?;
        writeln!(f, "\tBounds (max):                {}", self.bounds.max())?;
        writeln!(f, "\tNumber of point records:     {}", self.point_count)?;
        writeln!(f, "\tPoint record format:         {}", self.point_format)?;

        if let Some(las_header) = &self.raw_las_header {
            writeln!(f, "Raw LAS header entries")?;
            //writeln!(f, "\tFile signature:          {}", las_header.si)
            writeln!(
                f,
                "\tFile source ID:              {}",
                las_header.file_source_id()
            )?;
            //writeln!(f, "\tGlobal encoding:         {}", las_header.);
            writeln!(f, "\tGUID:                        {}", las_header.guid())?;
            writeln!(f, "\tVersion:                     {}", las_header.version())?;
            writeln!(
                f,
                "\tSystem identifier:           {}",
                las_header.system_identifier()
            )?;
            writeln!(
                f,
                "\tGenerating software:         {}",
                las_header.generating_software()
            )?;
            writeln!(
                f,
                "\tFile creation date:          {}",
                las_header
                    .date()
                    .map(|date| date.to_string())
                    .unwrap_or("N/A".into())
            )?;
            writeln!(
                f,
                "\tPoint data format:           {}",
                las_header.point_format()
            )?;
            writeln!(
                f,
                "\tNumber of point records:     {}",
                las_header.number_of_points()
            )?;
            writeln!(
                f,
                "\tNumber of points by return:  {} {} {} {} {}",
                las_header.number_of_points_by_return(1).unwrap_or_default(),
                las_header.number_of_points_by_return(2).unwrap_or_default(),
                las_header.number_of_points_by_return(3).unwrap_or_default(),
                las_header.number_of_points_by_return(4).unwrap_or_default(),
                las_header.number_of_points_by_return(5).unwrap_or_default()
            )?;
            writeln!(
                f,
                "\tScale (x y z):               {} {} {}",
                las_header.transforms().x.scale,
                las_header.transforms().y.scale,
                las_header.transforms().z.scale
            )?;
            writeln!(
                f,
                "\tOffset (x y z):              {} {} {}",
                las_header.transforms().x.offset,
                las_header.transforms().y.offset,
                las_header.transforms().z.offset
            )?;
            writeln!(
                f,
                "\tMin (x y z):                 {} {} {}",
                las_header.bounds().min.x,
                las_header.bounds().min.y,
                las_header.bounds().min.z,
            )?;
            writeln!(
                f,
                "\tMax (x y z):                 {} {} {}",
                las_header.bounds().max.x,
                las_header.bounds().max.y,
                las_header.bounds().max.z,
            )?;

            if las_header.vlrs().is_empty() {
                writeln!(f, "No VLRs")?;
            } else {
                writeln!(f, "VLRs")?;
                for vlr in las_header.vlrs() {
                    display_vlr(vlr, f)?;
                }
            }

            if las_header.evlrs().is_empty() {
                writeln!(f, "No extended VLRs")?;
            } else {
                writeln!(f, "Extended VLRs")?;
                for evlr in las_header.evlrs() {
                    display_vlr(evlr, f)?;
                }
            }
        }

        Ok(())
    }
}

impl Metadata for LASMetadata {
    fn bounds(&self) -> Option<AABB<f64>> {
        Some(self.bounds)
    }

    fn number_of_points(&self) -> Option<usize> {
        Some(self.point_count)
    }

    fn get_named_field(&self, field_name: &str) -> Option<Box<dyn Any>> {
        match field_name {
            named_fields::FILE_CREATION_DAY_OF_YEAR => self
                .raw_las_header
                .as_ref()
                .and_then(|header| header.date())
                .map(|date| -> Box<dyn Any> {
                    let day_of_year: u16 = date.ordinal().try_into().unwrap();
                    Box::new(day_of_year)
                }),
            named_fields::FILE_CREATION_YEAR => self
                .raw_las_header
                .as_ref()
                .and_then(|header| header.date())
                .map(|date| -> Box<dyn Any> {
                    let year: u16 = date.year().try_into().unwrap();
                    Box::new(year)
                }),
            named_fields::FILE_SOURCE_I_D => self
                .raw_las_header
                .as_ref()
                .map(|header| -> Box<dyn Any> { Box::new(header.file_source_id()) }),
            named_fields::GENERATING_SOFTWARE => {
                self.raw_las_header.as_ref().map(|header| -> Box<dyn Any> {
                    Box::new(header.generating_software().to_owned())
                })
            }
            named_fields::SYSTEM_IDENTIFIER => self
                .raw_las_header
                .as_ref()
                .map(|header| -> Box<dyn Any> { Box::new(header.system_identifier().to_owned()) }),
            named_fields::VERSION => self
                .raw_las_header
                .as_ref()
                .map(|header| -> Box<dyn Any> { Box::new(header.version().to_string()) }),
            _ => None,
        }
    }

    fn clone_into_box(&self) -> Box<dyn Metadata> {
        Box::new(self.clone())
    }
}

impl From<&las::Header> for LASMetadata {
    fn from(header: &las::Header) -> Self {
        Self {
            bounds: las_bounds_to_pasture_bounds(header.bounds()),
            point_count: header.number_of_points() as usize,
            point_format: header
                .point_format()
                .to_u8()
                .expect("Invalid LAS point format"),
            raw_las_header: Some(header.clone()),
        }
    }
}

impl From<las::Header> for LASMetadata {
    fn from(header: las::Header) -> Self {
        (&header).into()
    }
}
