use pasture_core::layout::PointType;
use pasture_derive::PointType;

#[derive(PointType)]
struct CustomPointType {
    // #[point_attribute_intensity]
    //#[pasture(attribute = "INTENSITY")]
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: u16,
    // #[point_attribute_position]
    //pub position: Vector3<f64>,
    #[pasture(attribute = "CUSTOM_ATTRIBUTE")]
    pub custom_attribute: f32,
}

fn main() {
    let layout = CustomPointType::layout();
    println!("Derived layout: {}", layout);
    //let p = CustomPointType {};
}
