use crate::layout::conversion::get_converter_for_attributes;
use crate::layout::{PointAttributeDefinition, PointLayout, PrimitiveType};
use anyhow::{bail, Context, Result};
use std::io::Cursor;
use std::mem::MaybeUninit;

/// A trait to handle points that layout can't be known to compile time.
pub trait UntypedPoint {
    /// Gets the data as byte slice for a attribute of the point.
    fn get_raw_attribute<'point>(
        &'point self,
        attribute: &PointAttributeDefinition,
    ) -> Result<&'point [u8]>;
    /// Gets the data as mut byte slice for a attribute of the point.
    fn get_raw_attribute_mut<'point>(
        &'point mut self,
        attribute: &PointAttributeDefinition,
    ) -> Result<&'point mut [u8]>;
    /// Sets the data for a attribute of the point.
    fn set_raw_attribute(
        &mut self,
        attribute: &PointAttributeDefinition,
        value_byte_slice: &[u8],
    ) -> Result<()>;
    /// Gets the data from an attribute and converts it to an `PrimitiveType`.
    fn get_attribute<'point, T: PrimitiveType>(
        &'point self,
        attribute: &PointAttributeDefinition,
    ) -> Result<T>;
    /// Sets the data from an attribute with an `PrimitiveType`.
    fn set_attribute<T: PrimitiveType>(
        &mut self,
        attribute: &PointAttributeDefinition,
        value: &T,
    ) -> Result<()>;
    /// Get the layout of the point.
    fn get_layout(&self) -> &PointLayout;
    // To handle byte manipulation.
    fn get_cursor(&mut self) -> Cursor<&mut [u8]>;
}

/// An implementaion of `UntypedPoint` trait that has an internal buffer.
pub struct UntypedPointBuffer<'layout> {
    layout: &'layout PointLayout,
    buffer: Vec<u8>,
}

impl<'layout> UntypedPointBuffer<'layout> {
    pub fn new(layout: &'layout PointLayout) -> Self {
        Self {
            layout,
            buffer: vec![0; layout.size_of_point_entry() as usize],
        }
    }
}
impl UntypedPoint for UntypedPointBuffer<'_> {
    fn get_raw_attribute<'point>(
        &'point self,
        attribute: &PointAttributeDefinition,
    ) -> Result<&'point [u8]> {
        let attribute = self
            .layout
            .get_attribute(attribute)
            .with_context(|| "Cannot find attribute.")?;
        let start = attribute.offset() as usize;
        let end = start + attribute.datatype().size() as usize;
        if self.buffer.len() < end {
            bail!("Buffer size to small.");
        }
        Ok(&self.buffer[start..end])
    }
    fn set_raw_attribute(
        &mut self,
        attribute: &PointAttributeDefinition,
        value_byte_slice: &[u8],
    ) -> Result<()> {
        let attribute = self
            .layout
            .get_attribute(attribute)
            .with_context(|| "Cannot find attribute.")?;
        let start = attribute.offset() as usize;
        let end = start + attribute.datatype().size() as usize;
        if self.buffer.len() < end {
            bail!("Buffer size to small.");
        }
        self.buffer[start..end].copy_from_slice(value_byte_slice);
        Ok(())
    }

    fn get_layout(&self) -> &PointLayout {
        self.layout
    }

    fn get_cursor(&mut self) -> Cursor<&mut [u8]> {
        Cursor::new(&mut self.buffer)
    }

    fn get_attribute<'point, T: PrimitiveType>(
        &'point self,
        attribute: &PointAttributeDefinition,
    ) -> Result<T> {
        let mut target_attribute = MaybeUninit::<T>::uninit();
        let source_attribute_byte_slice = self.get_raw_attribute(attribute)?;
        // access via [u8] slice
        let target_attribute_byte_slice = unsafe {
            std::slice::from_raw_parts_mut(
                target_attribute.as_mut_ptr() as *mut u8,
                std::mem::size_of::<T>(),
            )
        };
        if T::data_type() != attribute.datatype() {
            let target_attribute_definition =
                PointAttributeDefinition::with_custom_datatype(attribute, T::data_type());
            let converter = match get_converter_for_attributes(attribute, &target_attribute_definition ) {
                        Some(c) => c,
                        None => bail!("Can't convert from attribute {} to attribute {} because no valid conversion exists", attribute, target_attribute_definition),
                    };
            unsafe { converter(source_attribute_byte_slice, target_attribute_byte_slice) };
        } else {
            target_attribute_byte_slice.copy_from_slice(source_attribute_byte_slice);
        }
        Ok(unsafe { target_attribute.assume_init() })
    }

    fn set_attribute<T: PrimitiveType>(
        &mut self,
        attribute: &PointAttributeDefinition,
        value: &T,
    ) -> Result<()> {
        let ptr = value as *const _;
        let value_byte_slice =
            unsafe { std::slice::from_raw_parts(ptr as *mut u8, std::mem::size_of::<T>()) };
        if T::data_type() != attribute.datatype() {
            let source_attribute_definition =
                PointAttributeDefinition::with_custom_datatype(attribute, T::data_type());
            let converter = get_converter_for_attributes(&source_attribute_definition, attribute)
                    .ok_or_else(|| anyhow::anyhow!("Can't convert from attribute {} to attribute {} because no valid conversion exists", 
                    attribute, source_attribute_definition))?;
            unsafe { converter(value_byte_slice, self.get_raw_attribute_mut(attribute)?) };
        } else {
            self.set_raw_attribute(attribute, value_byte_slice)?;
        }
        Ok(())
    }

    fn get_raw_attribute_mut<'point>(
        &'point mut self,
        attribute: &PointAttributeDefinition,
    ) -> Result<&'point mut [u8]> {
        let attribute = self
            .layout
            .get_attribute(attribute)
            .with_context(|| "Cannot find attribute.")?;
        let start = attribute.offset() as usize;
        let end = start + attribute.datatype().size() as usize;
        if self.buffer.len() < end {
            bail!("Buffer size to small.");
        }
        Ok(&mut self.buffer[start..end])
    }
}

/// An implementaion of `UntypedPoint` trait that handles an external buffer.
pub struct UntypedPointSlice<'point> {
    layout: &'point PointLayout,
    slice: &'point mut [u8],
}

impl<'point> UntypedPointSlice<'point> {
    pub fn new(layout: &'point PointLayout, slice: &'point mut [u8]) -> Self {
        Self { layout, slice }
    }
}

impl UntypedPoint for UntypedPointSlice<'_> {
    fn get_attribute<'point, T: PrimitiveType>(
        &'point self,
        attribute: &PointAttributeDefinition,
    ) -> Result<T> {
        let mut target_attribute = MaybeUninit::<T>::uninit();
        let source_attribute_byte_slice = self.get_raw_attribute(attribute)?;
        // access via [u8] slice
        let target_attribute_byte_slice = unsafe {
            std::slice::from_raw_parts_mut(
                target_attribute.as_mut_ptr() as *mut u8,
                std::mem::size_of::<T>(),
            )
        };
        if T::data_type() != attribute.datatype() {
            let target_attribute_definition =
                PointAttributeDefinition::with_custom_datatype(attribute, T::data_type());
            let converter = match get_converter_for_attributes(attribute, &target_attribute_definition ) {
                        Some(c) => c,
                        None => bail!("Can't convert from attribute {} to attribute {} because no valid conversion exists", attribute, target_attribute_definition),
                    };
            unsafe { converter(source_attribute_byte_slice, target_attribute_byte_slice) };
        } else {
            target_attribute_byte_slice.copy_from_slice(source_attribute_byte_slice);
        }
        Ok(unsafe { target_attribute.assume_init() })
    }

    fn get_raw_attribute<'point>(
        &'point self,
        attribute: &PointAttributeDefinition,
    ) -> Result<&'point [u8]> {
        let attribute = self
            .layout
            .get_attribute(attribute)
            .with_context(|| "Cannot find attribute.")?;
        let start = attribute.offset() as usize;
        let end = start + attribute.datatype().size() as usize;
        if self.slice.len() < end {
            bail!("Buffer size to small.");
        }
        Ok(&self.slice[start..end])
    }

    fn get_raw_attribute_mut<'point>(
        &'point mut self,
        attribute: &PointAttributeDefinition,
    ) -> Result<&'point mut [u8]> {
        let attribute = self
            .layout
            .get_attribute(attribute)
            .with_context(|| "Cannot find attribute.")?;
        let start = attribute.offset() as usize;
        let end = start + attribute.datatype().size() as usize;
        if self.slice.len() < end {
            bail!("Buffer size to small.");
        }
        Ok(&mut self.slice[start..end])
    }

    fn set_raw_attribute(
        &mut self,
        attribute: &PointAttributeDefinition,
        value: &[u8],
    ) -> Result<()> {
        let attribute = self
            .layout
            .get_attribute(attribute)
            .with_context(|| "Cannot find attribute.")?;
        let start = attribute.offset() as usize;
        let end = start as usize + attribute.datatype().size() as usize;
        if self.slice.len() < end {
            bail!("Buffer size to small.");
        }
        self.slice[start..end].copy_from_slice(value);
        Ok(())
    }

    fn get_layout(&self) -> &PointLayout {
        self.layout
    }

    fn get_cursor(&mut self) -> Cursor<&mut [u8]> {
        Cursor::new(self.slice)
    }

    fn set_attribute<T: PrimitiveType>(
        &mut self,
        attribute: &PointAttributeDefinition,
        value: &T,
    ) -> Result<()> {
        let ptr = value as *const _;
        let value_byte_slice =
            unsafe { std::slice::from_raw_parts(ptr as *mut u8, std::mem::size_of::<T>()) };
        if T::data_type() != attribute.datatype() {
            let source_attribute_definition =
                PointAttributeDefinition::with_custom_datatype(attribute, T::data_type());
            let converter = get_converter_for_attributes(&source_attribute_definition, attribute)
                    .ok_or_else(|| anyhow::anyhow!("Can't convert from attribute {} to attribute {} because no valid conversion exists", 
                    attribute, source_attribute_definition))?;
            unsafe { converter(value_byte_slice, self.get_raw_attribute_mut(attribute)?) };
        } else {
            self.set_raw_attribute(attribute, value_byte_slice)?;
        }
        Ok(())
    }
}

// Test UntypedPointBuffer
// - test getter setter
// - test cursor
// Test UntypedPointSlice
// - test getter setter
// - test cursor
#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::attributes;
    use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
    use nalgebra::Vector3;
    use std::convert::TryInto;

    #[test]
    fn test_point_buffer_getter_setter() -> Result<()> {
        let layout =
            PointLayout::from_attributes(&[attributes::POSITION_3D, attributes::INTENSITY]);
        let mut point = UntypedPointBuffer::new(&layout);
        let intensity_value: u16 = 42;
        // Write
        point.set_raw_attribute(&attributes::INTENSITY, &u16::to_le_bytes(intensity_value))?;
        // Readback
        let buffer = point.get_raw_attribute(&attributes::INTENSITY)?;
        let intensity_from_point = u16::from_le_bytes(buffer.try_into()?);

        assert_eq!(intensity_value, intensity_from_point);
        Ok(())
    }
    #[test]
    fn test_point_buffer_cursor() -> Result<()> {
        let layout =
            PointLayout::from_attributes(&[attributes::POSITION_3D, attributes::INTENSITY]);
        let mut point = UntypedPointBuffer::new(&layout);
        let intensity_value: u16 = 42;

        let offset = layout.offset_of(&attributes::INTENSITY).unwrap();
        // Write
        let mut cursor = point.get_cursor();
        cursor.set_position(offset);
        cursor.write_u16::<LittleEndian>(intensity_value)?;
        // Readback
        cursor.set_position(offset);
        let intensity_from_point = cursor.read_u16::<LittleEndian>()?;

        assert_eq!(intensity_value, intensity_from_point);
        Ok(())
    }

    #[test]
    fn test_point_slice_getter_setter() -> Result<()> {
        let layout =
            PointLayout::from_attributes(&[attributes::POSITION_3D, attributes::INTENSITY]);
        let mut buffer: Vec<u8> = vec![0; layout.size_of_point_entry() as usize];
        let mut point = UntypedPointSlice::new(&layout, &mut buffer);
        let intensity_value: u16 = 42;
        // Write
        point.set_raw_attribute(&attributes::INTENSITY, &u16::to_le_bytes(intensity_value))?;
        // Readback
        let buffer = point.get_raw_attribute(&attributes::INTENSITY)?;
        let intensity_from_point = u16::from_le_bytes(buffer.try_into()?);
        assert_eq!(intensity_value, intensity_from_point);
        Ok(())
    }
    #[test]
    fn test_point_slice_cursor() -> Result<()> {
        let layout =
            PointLayout::from_attributes(&[attributes::POSITION_3D, attributes::INTENSITY]);
        let mut buffer: Vec<u8> = vec![0; layout.size_of_point_entry() as usize];
        let mut point = UntypedPointSlice::new(&layout, &mut buffer);
        let intensity_value: u16 = 42;

        let offset = layout.offset_of(&attributes::INTENSITY).unwrap();
        // Write
        let mut cursor = point.get_cursor();
        cursor.set_position(offset);
        cursor.write_u16::<LittleEndian>(intensity_value)?;
        // Readback
        cursor.set_position(offset);
        let intensity_from_point = cursor.read_u16::<LittleEndian>()?;

        assert_eq!(intensity_value, intensity_from_point);
        Ok(())
    }

    #[test]
    fn test_point_get_set_attribute() -> Result<()> {
        let layout =
            PointLayout::from_attributes(&[attributes::POSITION_3D, attributes::INTENSITY]);
        let mut point = UntypedPointBuffer::new(&layout);
        let intensity_value: u8 = 42;
        let position = Vector3::<f32>::new(1.1, 2.2, 3.3);
        // Write
        point.set_attribute(&attributes::INTENSITY, &intensity_value)?;
        point.set_attribute(&attributes::POSITION_3D, &position)?;
        // Readback
        let intencity_from_point = point.get_attribute::<u16>(&attributes::INTENSITY)?;
        let position_from_point = point.get_attribute::<Vector3<f32>>(&attributes::POSITION_3D)?;

        assert_eq!(intensity_value, intencity_from_point as u8);
        assert_eq!(position, position_from_point);
        Ok(())
    }

    #[test]
    fn test_point_slice_get_set_attribute() -> Result<()> {
        let layout =
            PointLayout::from_attributes(&[attributes::POSITION_3D, attributes::INTENSITY]);
        let mut buffer: Vec<u8> = vec![0; layout.size_of_point_entry() as usize];
        let mut point = UntypedPointSlice::new(&layout, &mut buffer);
        let intensity_value: u16 = 42;
        let position = Vector3::<f32>::new(1.1, 2.2, 3.3);
        // Write
        point.set_attribute(&attributes::INTENSITY, &intensity_value)?;
        point.set_attribute(&attributes::POSITION_3D, &position)?;
        // Readback
        let intencity_from_point = point.get_attribute::<u16>(&attributes::INTENSITY)?;
        let position_from_point = point.get_attribute::<Vector3<f32>>(&attributes::POSITION_3D)?;

        assert_eq!(intensity_value, intencity_from_point as u16);
        assert_eq!(position, position_from_point);
        Ok(())
    }
}
