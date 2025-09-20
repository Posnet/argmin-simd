# argmin-simd

Fast argmin (index of minimum) implementation using Rust's portable SIMD API.

## Problem

Find the index of the smallest element in a large array of f64 values, with constraints:
- All values are positive or +0.0
- No NaN or infinity values
- Optimize for arrays of ~1 million elements

## Solution

Uses Rust's nightly portable_simd feature to process 8 elements simultaneously:
- Maintains parallel tracking of both values and indices
- Efficient horizontal reduction at the end
- Handles remainder elements correctly

## Performance

For 1 million f64 elements:
- **Scalar**: 1862 µs/iteration
- **SIMD**: 181 µs/iteration
- **Speedup**: 10.26x

## Usage

```rust
use argmin_simd::{argmin_scalar, argmin_simd};

let data = vec![5.0, 2.0, 8.0, 1.0, 9.0];
assert_eq!(argmin_simd(&data), Some(3));
```

## Building

Requires nightly Rust for portable_simd:

```bash
cargo +nightly test
cargo +nightly bench
```