/// Returns a byte slice that points to the raw bytes of `val`.
///
/// # Safety
///
/// This is a thin wrapper around `std::slice::from_raw_parts` which internally uses an
/// unsafe cast from *const T to *const u8. This function is very unsafe but performs no
/// copy operations. It has the same safety requirements as [`std::slice::from_raw_parts`](std::slice::from_raw_parts).
///
/// ```
/// # use pasture_core::util::*;
/// let val : u64 = 128;
/// let val_raw_bytes = unsafe { view_raw_bytes(&val) };
/// assert_eq!(val_raw_bytes.len(), 8);
/// ```
pub unsafe fn view_raw_bytes<T>(val: &T) -> &[u8] {
    std::slice::from_raw_parts(val as *const T as *const u8, std::mem::size_of::<T>())
}
