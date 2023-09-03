use anyhow::{bail, Result};
// This enum maps the different entrys on an ascii file to later map these entries to the corresponding attribute.
#[derive(Debug)]
pub(crate) enum PointDataType {
    Skip,
    CoordinateX,
    CoordinateY,
    CoordinateZ,       //Vec3f64
    Intensity,         //U16
    ReturnNumber,      //U8
    NumberOfReturns,   //U8
    Classification,    //U8
    UserData,          //U8
    ColorR,            //U16
    ColorG,            //U16
    ColorB,            //U16
    GpsTime,           //F64
    PointSourceID,     // U16
    EdgeOfFlightLine,  //bool
    ScanDirectionFlag, //bool
    ScanAngleRank,     //I8
    Nir,               //U16
}

impl std::fmt::Display for PointDataType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl PointDataType {
    //from LAStools
    //s - skip this number
    //x - x coordinate
    //y - y coordinate
    //z - z coordinate
    //i - intensity
    //r - ReturnNumber,
    //n - number of returns of given pulse,
    //c - classification
    //u - user data
    //R - red channel of RGB color
    //G - green channel of RGB color
    //B - blue channel of RGB color
    //t - gps time
    //p - point source ID
    //e - edge of flight line flag
    //d - direction of scan flag
    //a - scan angle rank
    //I - NIR channel
    pub(crate) fn get_parse_layout(format: &str) -> Result<Vec<PointDataType>> {
        let mut parse_layout = Vec::<PointDataType>::new();
        for character in format.chars() {
            match character {
                's' => parse_layout.push(PointDataType::Skip),
                'x' => parse_layout.push(PointDataType::CoordinateX),
                'y' => parse_layout.push(PointDataType::CoordinateY),
                'z' => parse_layout.push(PointDataType::CoordinateZ),
                'i' => parse_layout.push(PointDataType::Intensity),
                'n' => parse_layout.push(PointDataType::NumberOfReturns),
                'r' => parse_layout.push(PointDataType::ReturnNumber),
                'c' => parse_layout.push(PointDataType::Classification),
                't' => parse_layout.push(PointDataType::GpsTime),
                'u' => parse_layout.push(PointDataType::UserData),
                'p' => parse_layout.push(PointDataType::PointSourceID),
                'R' => parse_layout.push(PointDataType::ColorR),
                'G' => parse_layout.push(PointDataType::ColorG),
                'B' => parse_layout.push(PointDataType::ColorB),
                'I' => parse_layout.push(PointDataType::Nir),
                'a' => parse_layout.push(PointDataType::ScanAngleRank),
                'e' => parse_layout.push(PointDataType::EdgeOfFlightLine),
                'd' => parse_layout.push(PointDataType::ScanDirectionFlag),
                _ => {
                    bail!(
                        "FormatError can't interpret format literal '{}' in format string '{}'.",
                        character,
                        format
                    );
                }
            }
        }
        Ok(parse_layout)
    }
}
