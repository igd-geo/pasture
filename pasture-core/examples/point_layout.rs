use std::borrow::Cow;

use pasture_core::containers::{BorrowedMutBufferExt, MakeBufferFromLayout, VectorBuffer};
use pasture_core::layout::{
    attributes, PointAttributeDataType, PointAttributeDefinition, PointLayout, PointType,
};
use pasture_core::nalgebra::Vector3;
use pasture_derive::PointType;

fn main() {
    // In this example, we will take a closer look at the `PointLayout` type. We will learn what a `PointLayout` is, how it is constructed
    // and how it is used throughout pasture. Let's dive right in:

    {
        // A `PointLayout` describes the data attributes of a single point in a point cloud. It is quite similar to what a
        // `struct` in Rust is: A description for a collection of data attributes in a single type.
        // Let's create an empty `PointLayout`:
        let empty_layout: PointLayout = Default::default();

        // The empty `PointLayout` is quite boring, as it describes a point without any attributes. Still, let's see what we can do with
        // this layout.

        // We can ask for the size in bytes that a single point in this layout takes in memory. For the empty layout, the size is of course
        // zero:
        assert_eq!(0, empty_layout.size_of_point_entry());

        // We can ask the layout if it contains a specific attribute. Attributes in pasture are identified by a unique name. Since names can
        // be confusing, pasture provides a large number of constants for default attributes that we can use. Let's look at a very common
        // attribute: POSITION_3D
        let position_attribute = attributes::POSITION_3D;
        // This is a `PointAttributeDefinition`, a sort of template that represents a specific attribute. Think of it like a member of a Rust
        // `struct`. As such, it has some expected parameters:
        println!(
            "The position attribute is called: {}",
            position_attribute.name()
        );
        println!(
            "It has the following datatype: {}",
            position_attribute.datatype()
        );
        // The `datatype` of an attribute is one of a series of default datatypes that correspond to several Rust types. All supported datatypes
        // by pasture are defined in the `PointAttributeDataType` enum. They include most of the Rust primitive types, as well as a few Vector
        // types from the `nalgebra` crate. The `POSITION_3D` attribute for example has the default datatype `PointAttributeDataType::Vec3f64`, which
        // corresponds to the Rust type `nalgebra::Vector3<f64>`.
        // Now we can ask the layout if it contains this attribute:
        assert!(!empty_layout.has_attribute(&position_attribute));

        // Let's move on to a more interesting example: A layout that contains some attributes!
    }

    {
        // We can create a custom `PointLayout` by telling it the attributes that it should contain
        let layout =
            PointLayout::from_attributes(&[attributes::POSITION_3D, attributes::INTENSITY]);

        // This layout roughly corresponds to the following Rust-type:
        // ```
        // struct Point {
        //   pub position: Vector3<f64>,
        //   pub intensity: u16,
        // }
        // ```
        assert!(layout.has_attribute(&attributes::POSITION_3D));
        assert!(layout.has_attribute(&attributes::INTENSITY));

        // Let's look at the size of a single point in this layout:
        println!(
            "A single point in this layout takes {} bytes",
            layout.size_of_point_entry()
        );
        // The result might be surprising. POSITION_3D has datatype `nalgebra::Vector3<f64>`, which takes
        // 24 bytes. INTENSITY has datatype `u16`, which takes 2 bytes. This would equal 26 bytes total, however
        // the layout tells us 32 bytes instead. This is due to alignment requirements: Vector3<f64> has an 8-byte
        // minimum alignment. Indeed, if you call `std::mem::size_of::<Point>()`, you will also get 32 bytes.
    }

    {
        // We don't have to use default attributes in a `PointLayout`. We can use custom datatypes for the builtin attributes
        // as well as custom attribute names:
        let custom_layout = PointLayout::from_attributes(&[
            attributes::POSITION_3D.with_custom_datatype(PointAttributeDataType::Vec3f32),
            PointAttributeDefinition::custom(Cow::Borrowed("Custom"), PointAttributeDataType::U64),
        ]);

        // We can ask the layout if it contains an attribute just by its name, ignoring the datatype:
        assert!(custom_layout.has_attribute_with_name(attributes::POSITION_3D.name()));
    }

    {
        // But what is a `PointLayout` really used for? In the `basic_point_buffers.rs` example, we already saw that a `PointLayout` is required
        // to create any type of `PointBuffer`. Indeed, point data in pasture is stored in an arbitrary format that gets determined at runtime.
        // This prevents using generics, as they are compile-time, so instead pasture uses the `PointLayout` type to figure out the memory layout
        // of points at runtime. To that end, you will rarely create `PointLayout`s manually. Instead, pasture provides a `derive` macro to create
        // a `PointLayout` for a specific type:
        #[derive(Copy, Clone, PointType, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
        #[repr(C, packed)]
        struct CustomPointType {
            #[pasture(BUILTIN_POSITION_3D)]
            pub position: Vector3<f64>,
            #[pasture(BUILTIN_INTENSITY)]
            pub intensity: u16,
            #[pasture(attribute = "CUSTOM_ATTRIBUTE")]
            pub custom_attribute: f32,
        }

        let layout = CustomPointType::layout();
        println!(
            "The CustomPointType has the following PointLayout: {}",
            layout
        );

        //With this, we can create a `PointBuffer` that stores `CustomPointType`s
        let mut buffer = VectorBuffer::new_from_layout(layout);
        buffer.view_mut().push_point(CustomPointType {
            position: Vector3::new(1.0, 2.0, 3.0),
            intensity: 42,
            custom_attribute: std::f32::consts::PI,
        });
    }
}
