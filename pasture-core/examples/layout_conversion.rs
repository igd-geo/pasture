use pasture_core::containers::{BorrowedBuffer, BorrowedBufferExt, OwningBuffer, VectorBuffer};
use pasture_core::layout::attributes::{COLOR_RGB, POSITION_3D};
use pasture_core::layout::conversion::BufferLayoutConverter;
use pasture_core::layout::{PointAttributeDataType, PointType};
use pasture_core::nalgebra::Vector3;
use pasture_derive::PointType;
use rand::prelude::Distribution;
use rand::{thread_rng, Rng};

#[derive(Copy, Clone, PointType, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
#[repr(C, packed)]
struct SourcePointType {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<i32>,
    #[pasture(BUILTIN_CLASSIFICATION)]
    pub classification: u8,
    #[pasture(BUILTIN_COLOR_RGB)]
    pub color: Vector3<u16>,
}

struct PointDistribution;

impl Distribution<SourcePointType> for PointDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> SourcePointType {
        SourcePointType {
            position: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
            classification: rng.gen(),
            color: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
        }
    }
}

fn gen_random_source_points(count: usize) -> impl Iterator<Item = SourcePointType> {
    let rng = thread_rng();
    rng.sample_iter::<SourcePointType, _>(PointDistribution)
        .take(count)
}

#[derive(Copy, Clone, PointType, bytemuck::AnyBitPattern, bytemuck::NoUninit, Debug)]
#[repr(C, packed)]
struct TargetPointType {
    #[pasture(BUILTIN_COLOR_RGB)]
    pub color: Vector3<u8>,
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
}

fn main() {
    // In this example, we will look at how we can convert buffers from one `PointLayout` into another `PointLayout` using
    // the `BufferLayoutConverter` type

    // First, let's generate some points in a source `PointLayout` (in this case the `SourcePointType`):
    let source_points = gen_random_source_points(64).collect::<Vec<_>>();
    let source_points_buffer = source_points.iter().copied().collect::<VectorBuffer>();

    // To make it a bit more explicit, here is the `PointLayout` of the source points:
    let source_layout = source_points_buffer.point_layout();
    // And the layout of the target points, which in this case will be the `TargetPointType`
    let target_layout = TargetPointType::layout();

    // Now we can create a `BufferLayoutConverter` for the two types that will convert points from the `source_layout` to
    // the `target_layout`. First, let's try to use default conversions, so `pasture` will figure out how to map attributes
    // in `SourcePointType` to attributes in `TargetPointType`
    {
        let default_converter = BufferLayoutConverter::for_layouts(source_layout, &target_layout);
        // Conversion is as simple as calling `convert` on the `BufferLayoutConverter`. We do have to tell it what kind of
        // buffer we want for our output data. Here, we request a `VectorBuffer`:
        let converted_points = default_converter.convert::<VectorBuffer, _>(&source_points_buffer);
        // Here are some guarantees that should hold:
        assert_eq!(converted_points.len(), source_points_buffer.len());
        assert_eq!(converted_points.point_layout(), &target_layout);

        // If you already have a buffer that you want to convert *into*, this also works:
        let mut into_buffer =
            VectorBuffer::with_capacity(source_points_buffer.len(), target_layout.clone());
        // `convert_into` expects that the target buffer has a length at least as big as the source buffer, so
        // we have to resize first (which fills `into_buffer` with default values):
        into_buffer.resize(source_points_buffer.len());
        default_converter.convert_into(&source_points_buffer, &mut into_buffer);

        assert_eq!(converted_points, into_buffer);

        // As an explanation of what happened: The `BufferLayoutConverter` essentially performed the same operation as the
        // following piece of code, but using runtime `PointLayout` information and the pasture buffer API:
        let _equivalent_conversion_with_vec = source_points
            .iter()
            .map(|source_point| {
                let source_color = source_point.color;
                let source_position = source_point.position;
                TargetPointType {
                    color: Vector3::new(
                        source_color.x as u8,
                        source_color.y as u8,
                        source_color.z as u8,
                    ),
                    position: Vector3::new(
                        source_position.x as f64,
                        source_position.y as f64,
                        source_position.z as f64,
                    ),
                }
            })
            .collect::<Vec<_>>();
    }

    // Sometimes we want more control over how the attributes are converted. For this, we can set custom mappings
    // on the `BufferLayoutConverter`. Let's try this:
    {
        let mut default_converter =
            BufferLayoutConverter::for_layouts(source_layout, &target_layout);
        // We can map any attribute onto any other attribute, as long as their dataytpes can be converted (so no
        // vector-to-scalar conversion, e.g. going from positions to classifications)
        default_converter.set_custom_mapping(
            // Make sure that the attributes and their datatypes exactly match the source and target layouts!
            &POSITION_3D.with_custom_datatype(PointAttributeDataType::Vec3i32),
            &COLOR_RGB.with_custom_datatype(PointAttributeDataType::Vec3u8),
        );

        // Custom mappings also support transformations. With them, we can apply some arbitrary transformation
        // to each value. A useful example is the conversion from local integer coordinates into world-space
        // floating-point coordinates when parsing an LAS file, which would work something like this:
        const OFFSET: Vector3<f64> = Vector3::new(10.0, 10.0, 10.0);
        const SCALE: Vector3<f64> = Vector3::new(0.001, 0.001, 0.001);
        default_converter.set_custom_mapping_with_transformation(
            &POSITION_3D.with_custom_datatype(PointAttributeDataType::Vec3i32),
            &POSITION_3D,
            |local_position: Vector3<f64>| local_position.component_mul(&SCALE) + OFFSET,
            // We also have to specify whether we want to transform the source attribute or the target attribute. Here, we
            // want to retain the precision of the f64 offset and scale values, so we first apply the conversion i32->f64
            // and then apply the transformation to the target attribute!
            false,
        );

        let converted_points = default_converter.convert::<VectorBuffer, _>(&source_points_buffer);
        for point in converted_points.view::<TargetPointType>().into_iter() {
            println!("{point:#?}");
        }
    }
}
