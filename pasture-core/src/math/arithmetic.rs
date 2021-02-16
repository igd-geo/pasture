/// Trait for aligning a numeric value to a given byte boundary
///
/// # Example
/// ```
/// # use pasture_core::math::*;
/// assert_eq!(8, 5_u32.align_to(8));
/// ```
pub trait Alignable {
    /// Align the associated value to an `alignment` bytes boundary
    fn align_to(&self, alignment: Self) -> Self;
}

impl Alignable for u8 {
    fn align_to(&self, alignment: Self) -> Self {
        if alignment == 0 {
            *self
        } else {
            ((self + alignment - 1) / alignment) * alignment
        }
    }
}

impl Alignable for u16 {
    fn align_to(&self, alignment: Self) -> Self {
        if alignment == 0 {
            *self
        } else {
            ((self + alignment - 1) / alignment) * alignment
        }
    }
}

impl Alignable for u32 {
    fn align_to(&self, alignment: Self) -> Self {
        if alignment == 0 {
            *self
        } else {
            ((self + alignment - 1) / alignment) * alignment
        }
    }
}

impl Alignable for u64 {
    fn align_to(&self, alignment: Self) -> Self {
        if alignment == 0 {
            *self
        } else {
            ((self + alignment - 1) / alignment) * alignment
        }
    }
}

impl Alignable for u128 {
    fn align_to(&self, alignment: Self) -> Self {
        if alignment == 0 {
            *self
        } else {
            ((self + alignment - 1) / alignment) * alignment
        }
    }
}

impl Alignable for usize {
    fn align_to(&self, alignment: Self) -> Self {
        if alignment == 0 {
            *self
        } else {
            ((self + alignment - 1) / alignment) * alignment
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Alignable;

    #[test]
    fn test_align_to() {
        assert_eq!(1_u32.align_to(0), 1);
        assert_eq!(1_u32.align_to(2), 2);
        assert_eq!(0_u32.align_to(2), 0);
        assert_eq!(4_u32.align_to(8), 8);
        assert_eq!(27_u32.align_to(8), 32);
        assert_eq!(8_u32.align_to(8), 8);
    }
}
