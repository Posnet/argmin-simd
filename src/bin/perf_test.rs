use argmin_simd::{argmin_scalar, argmin_simd};
use std::time::Instant;

fn main() {
    let size = 1_000_000;
    let mut data = vec![1.0; size];
    data[size / 2] = 0.1;

    // Warmup
    for _ in 0..10 {
        let _ = argmin_scalar(&data);
        let _ = argmin_simd(&data);
    }

    // Benchmark scalar
    let start = Instant::now();
    for _ in 0..100 {
        let result = argmin_scalar(&data);
        assert_eq!(result, Some(size / 2));
    }
    let scalar_time = start.elapsed();

    // Benchmark SIMD
    let start = Instant::now();
    for _ in 0..100 {
        let result = argmin_simd(&data);
        assert_eq!(result, Some(size / 2));
    }
    let simd_time = start.elapsed();

    println!("Results for {} elements:", size);
    println!("Scalar: {:?} ({:.2} µs/iter)", scalar_time, scalar_time.as_micros() as f64 / 100.0);
    println!("SIMD:   {:?} ({:.2} µs/iter)", simd_time, simd_time.as_micros() as f64 / 100.0);
    println!();
    println!("Speedup: {:.2}x", scalar_time.as_secs_f64() / simd_time.as_secs_f64());
}