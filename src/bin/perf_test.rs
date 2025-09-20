use argmin_simd::{
    argmin_par_scalar, argmin_par_simd, argmin_par_simd_16, argmin_par_simd_2, argmin_par_simd_4,
    argmin_par_simd_8, argmin_scalar, argmin_simd, argmin_simd_16, argmin_simd_2, argmin_simd_4,
    argmin_simd_8,
};
use std::time::Instant;

fn bench_function<F>(
    name: &str,
    f: F,
    data: &[f64],
    expected: usize,
    iterations: usize,
) -> (f64, f64)
where
    F: Fn(&[f64]) -> Option<usize>,
{
    // Warmup
    for _ in 0..3 {
        let _ = f(data);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let result = f(data);
        assert_eq!(result, Some(expected));
    }
    let elapsed = start.elapsed();

    let us_per_iter = elapsed.as_micros() as f64 / iterations as f64;
    let throughput_gb_s =
        (data.len() as f64 * 8.0 * iterations as f64) / elapsed.as_secs_f64() / 1e9;

    println!(
        "{:<20} {:>10.2} µs/iter    {:>8.2} GB/s",
        name, us_per_iter, throughput_gb_s
    );
    (us_per_iter, throughput_gb_s)
}

fn test_size(size: usize, iterations: usize) {
    println!("\n{}", "=".repeat(60));
    println!(
        "Testing with {} elements ({:.2} MB)",
        size,
        size as f64 * 8.0 / 1e6
    );
    println!("{}\n", "=".repeat(60));

    let mut data = vec![1.0; size];
    let expected = size * 3 / 4; // Place minimum at 75% position
    data[expected] = 0.1;

    println!("--- Sequential ---");
    let (scalar_time, _) = bench_function("Scalar:", argmin_scalar, &data, expected, iterations);

    println!("\n--- SIMD Lane Widths ---");
    let (simd2_time, simd2_tp) =
        bench_function("SIMD-2:", argmin_simd_2, &data, expected, iterations);
    let (simd4_time, simd4_tp) =
        bench_function("SIMD-4:", argmin_simd_4, &data, expected, iterations);
    let (simd8_time, simd8_tp) =
        bench_function("SIMD-8:", argmin_simd_8, &data, expected, iterations);
    let (simd16_time, simd16_tp) =
        bench_function("SIMD-16:", argmin_simd_16, &data, expected, iterations);

    println!("\n--- Parallel ---");
    let (par_scalar_time, par_scalar_tp) = bench_function(
        "Par Scalar:",
        argmin_par_scalar,
        &data,
        expected,
        iterations,
    );
    let (par_simd2_time, par_simd2_tp) = bench_function(
        "Par SIMD-2:",
        argmin_par_simd_2,
        &data,
        expected,
        iterations,
    );
    let (par_simd4_time, par_simd4_tp) = bench_function(
        "Par SIMD-4:",
        argmin_par_simd_4,
        &data,
        expected,
        iterations,
    );
    let (par_simd8_time, par_simd8_tp) = bench_function(
        "Par SIMD-8:",
        argmin_par_simd_8,
        &data,
        expected,
        iterations,
    );
    let (par_simd16_time, par_simd16_tp) = bench_function(
        "Par SIMD-16:",
        argmin_par_simd_16,
        &data,
        expected,
        iterations,
    );

    println!("\n--- Speedups vs Scalar ---");
    println!("SIMD-2:              {:.2}x", scalar_time / simd2_time);
    println!("SIMD-4:              {:.2}x", scalar_time / simd4_time);
    println!("SIMD-8:              {:.2}x", scalar_time / simd8_time);
    println!("SIMD-16:             {:.2}x", scalar_time / simd16_time);
    println!("Par Scalar:          {:.2}x", scalar_time / par_scalar_time);
    println!("Par SIMD-2:          {:.2}x", scalar_time / par_simd2_time);
    println!("Par SIMD-4:          {:.2}x", scalar_time / par_simd4_time);
    println!("Par SIMD-8:          {:.2}x", scalar_time / par_simd8_time);
    println!("Par SIMD-16:         {:.2}x", scalar_time / par_simd16_time);

    // Find best configurations
    let simd_times = vec![
        ("SIMD-2", simd2_time, simd2_tp),
        ("SIMD-4", simd4_time, simd4_tp),
        ("SIMD-8", simd8_time, simd8_tp),
        ("SIMD-16", simd16_time, simd16_tp),
    ];
    let best_simd = simd_times
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    let par_times = vec![
        ("Par SIMD-2", par_simd2_time, par_simd2_tp),
        ("Par SIMD-4", par_simd4_time, par_simd4_tp),
        ("Par SIMD-8", par_simd8_time, par_simd8_tp),
        ("Par SIMD-16", par_simd16_time, par_simd16_tp),
    ];
    let best_par = par_times
        .iter()
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        .unwrap();

    println!("\n--- Best Performers ---");
    println!(
        "Sequential: {} ({:.2} µs, {:.2} GB/s, {:.1}x speedup)",
        best_simd.0,
        best_simd.1,
        best_simd.2,
        scalar_time / best_simd.1
    );
    println!(
        "Parallel:   {} ({:.2} µs, {:.2} GB/s, {:.1}x speedup)",
        best_par.0,
        best_par.1,
        best_par.2,
        scalar_time / best_par.1
    );
}

fn main() {
    println!("=== argmin-simd Performance Analysis ===\n");
    println!("System: {} cores available", num_cpus::get());

    // Test different array sizes
    let test_configs = vec![
        (10_000, 1000),   // 10K elements, 1000 iterations
        (100_000, 500),   // 100K elements, 500 iterations
        (1_000_000, 100), // 1M elements, 100 iterations
        (5_000_000, 20),  // 5M elements, 20 iterations
        (10_000_000, 10), // 10M elements, 10 iterations
        (25_000_000, 5),  // 25M elements, 5 iterations
        (40_000_000, 3),  // 40M elements, 3 iterations
    ];

    for (size, iters) in test_configs {
        test_size(size, iters);
    }

    println!("\n{}", "=".repeat(60));
    println!("Summary");
    println!("{}", "=".repeat(60));
    println!("\nKey observations:");
    println!("- SIMD-8 typically performs best for sequential processing");
    println!("- Parallel SIMD combines thread-level and data-level parallelism");
    println!("- Throughput scales well with array size up to memory bandwidth limits");
    println!("- Optimal lane width depends on CPU architecture and cache hierarchy");
}

// Add num_cpus dependency detection
#[cfg(not(feature = "num_cpus"))]
fn num_cpus_get() -> usize {
    1
}

#[cfg(feature = "num_cpus")]
use num_cpus;

#[cfg(not(feature = "num_cpus"))]
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }
}
