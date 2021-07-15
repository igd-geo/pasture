use crate::containers::InterleavedPointView;
use crate::layout::{PointAttributeDefinition, PointLayout};
use anyhow::{bail, Context, Result};
use std::io::Cursor;

/// A trait to handle points that layout can't be known to compile time.
pub trait UntypedPoint {
    /// Gets the data for a attribute of the point.
    fn get_attribute<'point>(
        &'point self,
        attribute: &PointAttributeDefinition,
    ) -> Result<&'point [u8]>;
     /// Sets the data for a attribute of the point.
    fn set_attribute(&mut self, attribute: &PointAttributeDefinition, value: &[u8]) -> Result<()>;
    /// Get the layout of the point. 
    fn get_layout(&self) -> &PointLayout;
    // To handle byte manipulation.
    fn get_cursor(&mut self) -> Cursor<&mut [u8]>;
    /// Gets an `InterleavedPointView` of the point.
    /// This can help to create `PointBuffer`.
    fn get_interleaved_point_view(&self) -> InterleavedPointView;
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
    fn get_attribute<'point>(
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

    fn set_attribute(&mut self, attribute: &PointAttributeDefinition, value: &[u8]) -> Result<()> {
        let attribute = self
            .layout
            .get_attribute(attribute)
            .with_context(|| "Cannot find attribute.")?;
        let start = attribute.offset() as usize;
        let end = start + attribute.datatype().size() as usize;
        if self.buffer.len() < end {
            bail!("Buffer size to small.");
        }
        self.buffer[start..end].copy_from_slice(value);
        Ok(())
    }

    fn get_layout(&self) -> &PointLayout {
        self.layout
    }

    fn get_cursor(&mut self) -> Cursor<&mut [u8]> {
        Cursor::new(&mut self.buffer)
    }

    fn get_interleaved_point_view(&self) -> InterleavedPointView {
        InterleavedPointView::from_raw_slice(&self.buffer, self.layout.clone())
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
    fn get_attribute<'point>(
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

    fn set_attribute(&mut self, attribute: &PointAttributeDefinition, value: &[u8]) -> Result<()> {
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

    fn get_interleaved_point_view(&self) -> InterleavedPointView {
        InterleavedPointView::from_raw_slice(&self.slice, self.layout.clone())
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
    use std::convert::TryInto;

    #[test]
    fn test_point_buffer_getter_setter() -> Result<()> {
        let layout =
            PointLayout::from_attributes(&[attributes::POSITION_3D, attributes::INTENSITY]);
        let mut point = UntypedPointBuffer::new(&layout);
        let intensity_value: u16 = 42;
        // Write
        point.set_attribute(&attributes::INTENSITY, &u16::to_le_bytes(intensity_value))?;
        // Readback
        let buffer = point.get_attribute(&attributes::INTENSITY)?;
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

        let offset = layout
            .offset_of(&attributes::INTENSITY)
            .unwrap();
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
        point.set_attribute(&attributes::INTENSITY, &u16::to_le_bytes(intensity_value))?;
        // Readback
        let buffer = point.get_attribute(&attributes::INTENSITY)?;
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

        let offset = layout
            .offset_of(&attributes::INTENSITY)
            .unwrap();
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
}
