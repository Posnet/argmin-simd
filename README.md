# argmin-simd

Fast argmin (index of minimum) implementation using Rust's portable SIMD API with parallel processing support.

## Problem

Find the index of the smallest element in a large array of f64 values, with constraints:
- All values are positive or +0.0
- No NaN or infinity values
- Optimize for arrays of ~1 million elements

## Solution

Provides four implementations:
1. **Scalar**: Standard iterator-based approach
2. **SIMD**: Uses portable_simd to process 8 elements at once
3. **Parallel Scalar**: Uses rayon for parallel chunk processing
4. **Parallel SIMD**: Combines rayon parallelism with SIMD operations

## Performance

For 1 million f64 elements (with optimized compiler flags):
- **Scalar**: 1882 µs/iteration
- **SIMD**: 164 µs/iteration (11.5x speedup)
- **Parallel Scalar**: 310 µs/iteration (6.1x speedup)
- **Parallel SIMD**: 81 µs/iteration (23.2x speedup)

## Optimized Build Configuration

The project includes aggressive optimization flags for maximum performance:
- Single codegen unit for better inlining
- Native CPU target for optimal instruction selection
- Fat LTO for whole-program optimization
- Custom LLVM tuning for unrolling

## Usage

```rust
use argmin_simd::{argmin_scalar, argmin_simd, argmin_par_scalar, argmin_par_simd};

let data = vec![5.0, 2.0, 8.0, 1.0, 9.0];
assert_eq!(argmin_simd(&data), Some(3));
assert_eq!(argmin_par_simd(&data), Some(3));
```

## Building

Requires nightly Rust for portable_simd:

```bash
cargo +nightly test
cargo +nightly bench
cargo +nightly run --release --bin perf_test
```