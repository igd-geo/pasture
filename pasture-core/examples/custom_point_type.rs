use pasture_core::layout::PointType;
use pasture_core::nalgebra::Vector3;
use pasture_derive::PointType;

#[derive(PointType)]
#[repr(packed)]
struct CustomPointType {
    // #[point_attribute_intensity]
    //#[pasture(attribute = "INTENSITY")]
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: u16,
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f32>,
    #[pasture(attribute = "CUSTOM_ATTRIBUTE")]
    pub custom_attribute: f32,
}

//TODO I can distinguish between repr(Rust) repr(C) and repr(packed) now, but I also have to figure out the alignment of the whole struct, as well as the member alignments

fn main() {
    let layout = CustomPointType::layout();
    println!("Derived layout: {}", layout);
    //let p = CustomPointType {};
}
