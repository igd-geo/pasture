use anyhow::{bail, Context, Result};
use crate::layout::{PointLayout, PointAttributeDefinition};
use crate::containers::InterleavedPointView;
use std::io::Cursor;

/// A struct to create and handle points that layout can't be known to compile time.
pub struct UntypedPoint<'layout> {
    layout: &'layout PointLayout,
    buffer: Vec<u8>,
}

impl<'layout> UntypedPoint<'layout> {
    pub fn new(layout: &'layout PointLayout) -> Self {
        Self {
            layout,
            buffer: vec![0; layout.size_of_point_entry() as usize],
        }
    }

    pub fn get_attribute<'point>(
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

    pub fn set_attribute(
        &mut self,
        attribute: &PointAttributeDefinition,
        value: &[u8],
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
        self.buffer[start..end].copy_from_slice(value);
        Ok(())
    }

    //To handle byte manipulation.
    pub fn get_buffer_cursor(&mut self) -> Cursor<&mut Vec<u8>> {
        return Cursor::new(&mut self.buffer);
    }

    /// Returns the offset from an attribute of the layout.
    /// If the attribute don't exist in the layout this function returns None.
    pub fn get_offset_from_attribute(&self, attribute: &PointAttributeDefinition) -> Option<u64> {
        self.layout
            .get_attribute(attribute)
            .map(|member| member.offset())
    }

    pub fn get_interlieved_point_view(&self) -> InterleavedPointView {
        InterleavedPointView::from_raw_slice(&self.buffer, self.layout.clone())
    }
}

// Test UntypedPoint
// - test getter setter
// - test cursor
#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
    use crate::layout::attributes;
    use std::convert::TryInto;

    #[test]
    fn test_point_getter_setter() -> Result<()> {
        let layout = PointLayout::from_attributes(&[attributes::POSITION_3D, attributes::INTENSITY]);
        let mut point = UntypedPoint::new(&layout);
        let intensity_value:u16 = 42;
        // Write
        point.set_attribute(&attributes::INTENSITY, &u16::to_le_bytes(intensity_value))?;
        // Readback
        let buffer = point.get_attribute(&attributes::INTENSITY)?;
        let intensity_from_point = u16::from_le_bytes(buffer.try_into()?);
        
        assert_eq!(intensity_value, intensity_from_point);
        Ok(())
    }
    #[test]
    fn test_point_cursor() -> Result<()> {
        let layout = PointLayout::from_attributes(&[attributes::POSITION_3D, attributes::INTENSITY]);
        let mut point = UntypedPoint::new(&layout);
        let intensity_value:u16 = 42;
        
        let offset = point.get_offset_from_attribute(&attributes::INTENSITY).unwrap();
        let mut cursor = point.get_buffer_cursor();
        // Write
        cursor.set_position(offset);
        cursor.write_u16::<LittleEndian>(intensity_value)?;
        // Readback
        cursor.set_position(offset);
        let intensity_from_point = cursor.read_u16::<LittleEndian>()?;

        assert_eq!(intensity_value, intensity_from_point);
        Ok(())
    }
}

