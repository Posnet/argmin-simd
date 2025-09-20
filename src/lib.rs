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

use std::simd::{f64x8, prelude::*};
use rayon::prelude::*;

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
    if data.is_empty() {
        return None;
    }

    const CHUNK_SIZE: usize = 100_000;
    const LANES: usize = 8;

    // Process chunks in parallel, each using SIMD
    let local_mins: Vec<(usize, f64)> = data
        .par_chunks(CHUNK_SIZE)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let chunk_base_idx = chunk_idx * CHUNK_SIZE;

            let simd_chunks = chunk.chunks_exact(LANES);
            let remainder = simd_chunks.remainder();

            let mut min_vals = f64x8::splat(f64::MAX);
            let mut min_idxs = f64x8::from_array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
            let idx_increment = f64x8::splat(LANES as f64);
            let mut current_idx = f64x8::from_array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

            for simd_chunk in simd_chunks {
                let vals = f64x8::from_slice(simd_chunk);
                let mask = vals.simd_lt(min_vals);
                min_vals = mask.select(vals, min_vals);
                min_idxs = mask.select(current_idx, min_idxs);
                current_idx += idx_increment;
            }

            // Reduce SIMD lanes
            let mut final_min_val = min_vals[0];
            let mut final_min_idx = min_idxs[0] as usize;

            for i in 1..LANES {
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
            for pos in [0, 1, size/2, size-2, size-1] {
                let mut data = vec![1.0; size];
                data[pos] = 0.1;

                let scalar_result = argmin_scalar(&data);
                let simd_result = argmin_simd(&data);
                let par_scalar_result = argmin_par_scalar(&data);
                let par_simd_result = argmin_par_simd(&data);

                assert_eq!(scalar_result, Some(pos), "scalar failed at size={}, pos={}", size, pos);
                assert_eq!(simd_result, Some(pos), "simd failed at size={}, pos={}", size, pos);
                assert_eq!(par_scalar_result, Some(pos), "par_scalar failed at size={}, pos={}", size, pos);
                assert_eq!(par_simd_result, Some(pos), "par_simd failed at size={}, pos={}", size, pos);
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
}