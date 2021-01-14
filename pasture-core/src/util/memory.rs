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

/// Returns a mutable byte slice that points to the raw bytes of `val`.
///
/// # Safety
///
/// This is a thin wrapper around `std::slice::from_raw_parts_mut` which internally uses an
/// unsafe cast from *mut T to *mut u8. This function is very unsafe but performs no
/// copy operations. It has the same safety requirements as [`std::slice::from_raw_parts_mut`](std::slice::from_raw_parts_mut).
///
/// ```
/// # use pasture_core::util::*;
/// let mut val : u64 = 128;
/// let val_raw_bytes = unsafe { view_raw_bytes_mut(&mut val) };
/// assert_eq!(val_raw_bytes.len(), 8);
/// ```
pub unsafe fn view_raw_bytes_mut<T>(val: &mut T) -> &mut [u8] {
    std::slice::from_raw_parts_mut(val as *mut T as *mut u8, std::mem::size_of::<T>())
}

/// Sorts a slice containing binary untyped data as if it did contain typed data. Sorting is performed using the given `permutation`. `stride` defines the
/// offset between two consecutive elements within `slice`.
///
/// # Example
/// ```
/// # use pasture_core::util::*;
/// let mut typed_vec : Vec<u32> = vec![10,20,30,40,50,60];
/// let mut untyped_vec = unsafe { std::slice::from_raw_parts_mut(typed_vec.as_mut_ptr() as *mut u8, std::mem::size_of::<u32>() * typed_vec.len())};
/// let permutation = vec![1,0,3,2,5,4];
///
/// sort_untyped_slice_by_permutation(untyped_vec, permutation.as_slice(), std::mem::size_of::<u32>());
/// assert_eq!(vec![20,10,40,30,60,50], typed_vec);
/// ```
/// # Safety
///
/// Make sure that `stride` matches the size of the typed data that `slice` refers to, otherwise
/// the binary layout will get scrambled!
///
/// # Panics
///
/// If the number of entries in `permutation` times the `stride` does not equal the number of entries in `slice`.
/// If `permutation` contains any entry that is >= `slice.len() / stride`
pub fn sort_untyped_slice_by_permutation(slice: &mut [u8], permutation: &[usize], stride: usize) {
    if permutation.len() * stride != slice.len() {
        panic!("sort_untyped_slice_by_permutation: permutation.len() * stride did not equal slice.len()!");
    }

    let num_typed_elements = permutation.len();
    let mut done_indices = vec![false; num_typed_elements];
    for idx in 0..num_typed_elements {
        if done_indices[idx] {
            continue;
        }
        done_indices[idx] = true;

        let mut prev_idx = idx;
        let mut new_idx = permutation[idx];

        if new_idx >= num_typed_elements {
            panic!("sort_untyped_slice_by_permutation: Encountered out-of-bounds value in `permutation`!");
        }

        while idx != new_idx {
            unsafe {
                let old_region = slice.as_mut_ptr().offset((prev_idx * stride) as isize);
                let new_region = slice.as_mut_ptr().offset((new_idx * stride) as isize);
                std::ptr::swap_nonoverlapping(old_region, new_region, stride);
            }

            done_indices[new_idx] = true;
            prev_idx = new_idx;
            new_idx = permutation[new_idx];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};

    #[test]
    fn test_sort_untyped_slice_by_permutation() {
        let mut rng = thread_rng();

        let count = 4096;
        let mut rnd_values: Vec<u32> = (0..count).map(|_| rng.gen_range(0..count)).collect();

        let mut values_and_indices = rnd_values
            .iter()
            .enumerate()
            .map(|(idx, &val)| (idx, val))
            .collect::<Vec<_>>();
        values_and_indices.sort_by(|l, r| l.1.cmp(&r.1));

        let permutation = values_and_indices
            .iter()
            .map(|(idx, _)| *idx)
            .collect::<Vec<_>>();
        let reference_values = values_and_indices
            .iter()
            .map(|(_, val)| *val)
            .collect::<Vec<_>>();

        let stride = std::mem::size_of::<u32>();
        let rnd_values_untyped = unsafe {
            std::slice::from_raw_parts_mut(
                rnd_values.as_mut_ptr() as *mut u8,
                count as usize * stride,
            )
        };

        sort_untyped_slice_by_permutation(rnd_values_untyped, permutation.as_slice(), stride);

        assert_eq!(reference_values, rnd_values);
    }
}
