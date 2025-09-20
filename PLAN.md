# Plan — argmin-simd (2025-09-20) [COMPLETE]

## Goals
- Implement fast argmin for f64 arrays using portable_simd
- Leverage constraints: positive/zero values only, no NaN/infinity
- Benchmark against scalar implementation

## Non-Goals
- Handle NaN/infinity values
- Support negative numbers
- Generic over different float types

## Constraints
- Must use nightly Rust portable_simd API
- Process 1 million f64 values
- All values guaranteed positive or +0

## Interfaces Touched
- std::simd — SIMD operations
- criterion — benchmarking

## Risks
- R1: Nightly API instability — pin rustc version

## Breaking Changes
- None (new crate)

## Bench Targets
- SIMD speedup: >4x vs scalar (method: criterion)

## Next Actions
- PX-1: Create crate structure — files: Cargo.toml, src/lib.rs — exit: cargo check [DONE]
- PX-2: Implement SIMD argmin — files: src/lib.rs — exit: tests pass [DONE]
- PX-3: Add benchmarks — files: benches/argmin.rs — exit: cargo bench runs [DONE]
- PX-4: Optimize based on results — files: src/lib.rs — exit: >4x speedup [DONE]

## Evidence
- Build: cargo test — 7 tests pass
- Tests: All correctness tests pass including edge cases
- Bench: 9.30x speedup achieved (1609 µs scalar vs 173 µs SIMD for 1M elements)

## Log
- [2025-09-20] Initial implementation using portable_simd
- [2025-09-20] Achieved 9.30x speedup with basic SIMD approach
- [2025-09-20] Loop unrolling optimization (v2) tested but slower due to overhead