// === File: src/lib.rs ======================================================
// AGENT: PURPOSE    — Fast argmin using SIMD for positive f64 arrays
// AGENT: OWNER      — argmin-simd
// AGENT: INTERFACE  — exports: argmin_scalar, argmin_simd, argmin_par_scalar, argmin_par_simd
// AGENT: INVARIANTS — input values must be positive/zero, no NaN/infinity
// AGENT: RISK       — relies on nightly portable_simd API
// AGENT: PLAN-REF   — PX-2, PLAN.md
// AGENT: STATUS     — stable (2025-09-20)
// ==========================================================================

#![feature(portable_simd)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rayon::prelude::*;
use std::simd::{f64x8, prelude::*, LaneCount, Simd, SupportedLaneCount};

pub fn argmin_scalar(data: &[f64]) -> Option<usize> {
    data.iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(idx, _)| idx)
}

pub fn argmin_simd(data: &[f64]) -> Option<usize> {
    if data.is_empty() {
        return None;
    }

    const LANES: usize = 8;
    let chunks = data.chunks_exact(LANES);
    let remainder = chunks.remainder();

    // AGENT: DECISION — track 8 indices simultaneously to preserve argmin
    let mut min_vals = f64x8::splat(f64::MAX);
    let mut min_idxs = f64x8::from_array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let idx_increment = f64x8::splat(LANES as f64);
    let mut current_idx = f64x8::from_array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

    for chunk in chunks {
        let vals = f64x8::from_slice(chunk);
        let mask = vals.simd_lt(min_vals);
        min_vals = mask.select(vals, min_vals);
        min_idxs = mask.select(current_idx, min_idxs);
        current_idx += idx_increment;
    }

    // AGENT: REASONING — reduce SIMD lanes to single minimum
    let mut final_min_val = min_vals[0];
    let mut final_min_idx = min_idxs[0] as usize;

    for i in 1..LANES {
        if min_vals[i] < final_min_val {
            final_min_val = min_vals[i];
            final_min_idx = min_idxs[i] as usize;
        }
    }

    // Process remainder
    let remainder_start = data.len() - remainder.len();
    for (i, &val) in remainder.iter().enumerate() {
        if val < final_min_val {
            final_min_val = val;
            final_min_idx = remainder_start + i;
        }
    }

    Some(final_min_idx)
}

// AGENT: DECISION — const generic SIMD implementation for any lane width
pub fn argmin_simd_n<const N: usize>(data: &[f64]) -> Option<usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    if data.is_empty() {
        return None;
    }

    let chunks = data.chunks_exact(N);
    let remainder = chunks.remainder();

    // Initialize indices array
    let mut init_indices = [0.0f64; N];
    for (i, val) in init_indices.iter_mut().enumerate().take(N) {
        *val = i as f64;
    }

    let mut min_vals = Simd::<f64, N>::splat(f64::MAX);
    let mut min_idxs = Simd::<f64, N>::from_array(init_indices);
    let idx_increment = Simd::<f64, N>::splat(N as f64);
    let mut current_idx = Simd::<f64, N>::from_array(init_indices);

    for chunk in chunks {
        let vals = Simd::<f64, N>::from_slice(chunk);
        let mask = vals.simd_lt(min_vals);
        min_vals = mask.select(vals, min_vals);
        min_idxs = mask.select(current_idx, min_idxs);
        current_idx += idx_increment;
    }

    // Horizontal reduction
    let mut final_min_val = min_vals[0];
    let mut final_min_idx = min_idxs[0] as usize;

    for i in 1..N {
        if min_vals[i] < final_min_val {
            final_min_val = min_vals[i];
            final_min_idx = min_idxs[i] as usize;
        }
    }

    // Process remainder
    let remainder_start = data.len() - remainder.len();
    for (i, &val) in remainder.iter().enumerate() {
        if val < final_min_val {
            final_min_val = val;
            final_min_idx = remainder_start + i;
        }
    }

    Some(final_min_idx)
}

// AGENT: REASONING — specialized versions for common lane widths
pub fn argmin_simd_2(data: &[f64]) -> Option<usize> {
    argmin_simd_n::<2>(data)
}

pub fn argmin_simd_4(data: &[f64]) -> Option<usize> {
    argmin_simd_n::<4>(data)
}

pub fn argmin_simd_8(data: &[f64]) -> Option<usize> {
    argmin_simd_n::<8>(data)
}

pub fn argmin_simd_16(data: &[f64]) -> Option<usize> {
    argmin_simd_n::<16>(data)
}

// AGENT: DECISION — parallel scalar implementation using rayon
pub fn argmin_par_scalar(data: &[f64]) -> Option<usize> {
    if data.is_empty() {
        return None;
    }

    const CHUNK_SIZE: usize = 10_000;

    // Find local minima in parallel
    let local_mins: Vec<(usize, f64)> = data
        .par_chunks(CHUNK_SIZE)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let local_min = chunk
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.total_cmp(b.1))
                .map(|(idx, &val)| (chunk_idx * CHUNK_SIZE + idx, val))
                .unwrap();
            local_min
        })
        .collect();

    // Find global minimum from local minima
    local_mins
        .into_iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(idx, _)| idx)
}

// AGENT: REASONING — parallel SIMD implementation combines both techniques
pub fn argmin_par_simd(data: &[f64]) -> Option<usize> {
    argmin_par_simd_n::<8>(data)
}

// AGENT: DECISION — const generic parallel SIMD implementation
pub fn argmin_par_simd_n<const N: usize>(data: &[f64]) -> Option<usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    if data.is_empty() {
        return None;
    }

    const CHUNK_SIZE: usize = 100_000;

    // Process chunks in parallel, each using SIMD
    let local_mins: Vec<(usize, f64)> = data
        .par_chunks(CHUNK_SIZE)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let chunk_base_idx = chunk_idx * CHUNK_SIZE;

            let simd_chunks = chunk.chunks_exact(N);
            let remainder = simd_chunks.remainder();

            // Initialize indices array
            let mut init_indices = [0.0f64; N];
            for (i, val) in init_indices.iter_mut().enumerate().take(N) {
                *val = i as f64;
            }

            let mut min_vals = Simd::<f64, N>::splat(f64::MAX);
            let mut min_idxs = Simd::<f64, N>::from_array(init_indices);
            let idx_increment = Simd::<f64, N>::splat(N as f64);
            let mut current_idx = Simd::<f64, N>::from_array(init_indices);

            for simd_chunk in simd_chunks {
                let vals = Simd::<f64, N>::from_slice(simd_chunk);
                let mask = vals.simd_lt(min_vals);
                min_vals = mask.select(vals, min_vals);
                min_idxs = mask.select(current_idx, min_idxs);
                current_idx += idx_increment;
            }

            // Reduce SIMD lanes
            let mut final_min_val = min_vals[0];
            let mut final_min_idx = min_idxs[0] as usize;

            for i in 1..N {
                if min_vals[i] < final_min_val {
                    final_min_val = min_vals[i];
                    final_min_idx = min_idxs[i] as usize;
                }
            }

            // Process remainder
            let remainder_start = chunk.len() - remainder.len();
            for (i, &val) in remainder.iter().enumerate() {
                if val < final_min_val {
                    final_min_val = val;
                    final_min_idx = remainder_start + i;
                }
            }

            (chunk_base_idx + final_min_idx, final_min_val)
        })
        .collect();

    // Find global minimum from local minima
    local_mins
        .into_iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(idx, _)| idx)
}

// AGENT: REASONING — specialized parallel versions for common lane widths
pub fn argmin_par_simd_2(data: &[f64]) -> Option<usize> {
    argmin_par_simd_n::<2>(data)
}

pub fn argmin_par_simd_4(data: &[f64]) -> Option<usize> {
    argmin_par_simd_n::<4>(data)
}

pub fn argmin_par_simd_8(data: &[f64]) -> Option<usize> {
    argmin_par_simd_n::<8>(data)
}

pub fn argmin_par_simd_16(data: &[f64]) -> Option<usize> {
    argmin_par_simd_n::<16>(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        assert_eq!(argmin_scalar(&[]), None);
        assert_eq!(argmin_simd(&[]), None);
        assert_eq!(argmin_par_scalar(&[]), None);
        assert_eq!(argmin_par_simd(&[]), None);
    }

    #[test]
    fn test_single() {
        assert_eq!(argmin_scalar(&[1.0]), Some(0));
        assert_eq!(argmin_simd(&[1.0]), Some(0));
        assert_eq!(argmin_par_scalar(&[1.0]), Some(0));
        assert_eq!(argmin_par_simd(&[1.0]), Some(0));
    }

    #[test]
    fn test_basic() {
        let data = vec![5.0, 2.0, 8.0, 1.0, 9.0];
        assert_eq!(argmin_scalar(&data), Some(3));
        assert_eq!(argmin_simd(&data), Some(3));
        assert_eq!(argmin_par_scalar(&data), Some(3));
        assert_eq!(argmin_par_simd(&data), Some(3));
    }

    #[test]
    fn test_large_aligned() {
        let mut data = vec![1.0; 64];
        data[42] = 0.5;
        assert_eq!(argmin_scalar(&data), Some(42));
        assert_eq!(argmin_simd(&data), Some(42));
    }

    #[test]
    fn test_large_unaligned() {
        let mut data = vec![1.0; 67];
        data[65] = 0.5;
        assert_eq!(argmin_scalar(&data), Some(65));
        assert_eq!(argmin_simd(&data), Some(65));
    }

    #[test]
    fn test_million() {
        let mut data = vec![1.0; 1_000_000];
        data[500_000] = 0.1;
        assert_eq!(argmin_scalar(&data), Some(500_000));
        assert_eq!(argmin_simd(&data), Some(500_000));
    }

    #[test]
    fn test_various_positions() {
        for size in [100, 1000, 10_000, 100_000] {
            for pos in [0, 1, size / 2, size - 2, size - 1] {
                let mut data = vec![1.0; size];
                data[pos] = 0.1;

                let scalar_result = argmin_scalar(&data);
                let simd_result = argmin_simd(&data);
                let par_scalar_result = argmin_par_scalar(&data);
                let par_simd_result = argmin_par_simd(&data);

                assert_eq!(
                    scalar_result,
                    Some(pos),
                    "scalar failed at size={}, pos={}",
                    size,
                    pos
                );
                assert_eq!(
                    simd_result,
                    Some(pos),
                    "simd failed at size={}, pos={}",
                    size,
                    pos
                );
                assert_eq!(
                    par_scalar_result,
                    Some(pos),
                    "par_scalar failed at size={}, pos={}",
                    size,
                    pos
                );
                assert_eq!(
                    par_simd_result,
                    Some(pos),
                    "par_simd failed at size={}, pos={}",
                    size,
                    pos
                );
            }
        }
    }

    #[test]
    fn test_parallel_million() {
        let mut data = vec![1.0; 1_000_000];
        data[750_000] = 0.1;

        assert_eq!(argmin_par_scalar(&data), Some(750_000));
        assert_eq!(argmin_par_simd(&data), Some(750_000));
    }

    #[test]
    fn test_simd_lane_widths() {
        let mut data = vec![1.0; 1000];
        data[567] = 0.1;

        // Test all lane width variants
        assert_eq!(argmin_simd_2(&data), Some(567));
        assert_eq!(argmin_simd_4(&data), Some(567));
        assert_eq!(argmin_simd_8(&data), Some(567));
        assert_eq!(argmin_simd_16(&data), Some(567));

        // Test generic version with various widths
        assert_eq!(argmin_simd_n::<2>(&data), Some(567));
        assert_eq!(argmin_simd_n::<4>(&data), Some(567));
        assert_eq!(argmin_simd_n::<8>(&data), Some(567));
        assert_eq!(argmin_simd_n::<16>(&data), Some(567));
    }

    #[test]
    fn test_parallel_simd_lane_widths() {
        let mut data = vec![1.0; 100_000];
        data[45_678] = 0.1;

        // Test all parallel lane width variants
        assert_eq!(argmin_par_simd_2(&data), Some(45_678));
        assert_eq!(argmin_par_simd_4(&data), Some(45_678));
        assert_eq!(argmin_par_simd_8(&data), Some(45_678));
        assert_eq!(argmin_par_simd_16(&data), Some(45_678));

        // Test generic version
        assert_eq!(argmin_par_simd_n::<2>(&data), Some(45_678));
        assert_eq!(argmin_par_simd_n::<4>(&data), Some(45_678));
        assert_eq!(argmin_par_simd_n::<8>(&data), Some(45_678));
        assert_eq!(argmin_par_simd_n::<16>(&data), Some(45_678));
    }

    #[test]
    fn test_edge_cases_all_lane_widths() {
        // Test with sizes that aren't exact multiples of lane width
        for size in [7, 15, 31, 63, 127] {
            let mut data = vec![1.0; size];
            data[size - 1] = 0.1;

            assert_eq!(argmin_simd_2(&data), Some(size - 1));
            assert_eq!(argmin_simd_4(&data), Some(size - 1));
            assert_eq!(argmin_simd_8(&data), Some(size - 1));
            assert_eq!(argmin_simd_16(&data), Some(size - 1));
        }
    }

    #[test]
    fn test_large_arrays() {
        // Test with 5M elements
        let mut data = vec![1.0; 5_000_000];
        data[2_500_000] = 0.1;
        assert_eq!(argmin_scalar(&data), Some(2_500_000));
        assert_eq!(argmin_simd_8(&data), Some(2_500_000));
        assert_eq!(argmin_par_simd_8(&data), Some(2_500_000));

        // Test with 10M elements
        let mut data = vec![1.0; 10_000_000];
        data[7_500_000] = 0.1;
        assert_eq!(argmin_simd_8(&data), Some(7_500_000));
        assert_eq!(argmin_par_simd_8(&data), Some(7_500_000));
    }

    #[test]
    fn test_scaling_consistency() {
        // Verify consistency across different array sizes
        let sizes = vec![1_000, 10_000, 100_000, 1_000_000, 2_500_000];

        for size in sizes {
            let mut data = vec![1.0; size];
            let expected = size / 3;
            data[expected] = 0.1;

            let scalar_result = argmin_scalar(&data);
            let simd_result = argmin_simd_8(&data);
            let par_result = argmin_par_simd_8(&data);

            assert_eq!(
                scalar_result,
                Some(expected),
                "Scalar failed at size {}",
                size
            );
            assert_eq!(simd_result, Some(expected), "SIMD failed at size {}", size);
            assert_eq!(
                par_result,
                Some(expected),
                "Parallel SIMD failed at size {}",
                size
            );
        }
    }
}
