/// Inserts two zero-bits before any two bits of `val`
pub fn expand_bits_by_3(mut val: u64) -> u64 {
    val &= 0x1FFFFF; //Truncate to 21 bits
    val = (val | (val << 32)) & 0x00FF00000000FFFF;
    val = (val | (val << 16)) & 0x00FF0000FF0000FF;
    val = (val | (val << 8)) & 0xF00F00F00F00F00F;
    val = (val | (val << 4)) & 0x30C30C30C30C30C3;
    val = (val | (val << 2)) & 0x1249249249249249;
    val
}
