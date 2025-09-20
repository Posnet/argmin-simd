use argmin_simd::{
    argmin_par_scalar, argmin_par_simd, argmin_scalar, argmin_simd, argmin_simd_16, argmin_simd_2,
    argmin_simd_4, argmin_simd_8,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{Rng, SeedableRng};

fn generate_data(size: usize) -> Vec<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    (0..size).map(|_| rng.gen_range(0.0..1000.0)).collect()
}

fn benchmark_argmin(c: &mut Criterion) {
    let sizes = [
        1_000, 10_000, 100_000,  // Reduced for debugging
        // 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_500_000, 5_000_000,
        // 10_000_000, 25_000_000, 40_000_000,
    ];

    let mut group = c.benchmark_group("argmin");

    for &size in sizes.iter() {
        // Test all sizes in the array
        let data = generate_data(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("scalar", size), &data, |b, data| {
            b.iter(|| argmin_scalar(black_box(data)))
        });

        group.bench_with_input(BenchmarkId::new("simd", size), &data, |b, data| {
            b.iter(|| argmin_simd(black_box(data)))
        });

        group.bench_with_input(BenchmarkId::new("par_scalar", size), &data, |b, data| {
            b.iter(|| argmin_par_scalar(black_box(data)))
        });

        group.bench_with_input(BenchmarkId::new("par_simd", size), &data, |b, data| {
            b.iter(|| argmin_par_simd(black_box(data)))
        });
    }

    group.finish();
}

fn benchmark_lane_widths(c: &mut Criterion) {
    let sizes = [100_000, 1_000_000, 10_000_000];

    let mut group = c.benchmark_group("lane_widths");

    for &size in &sizes {
        let data = generate_data(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("simd_2", size), &data, |b, data| {
            b.iter(|| argmin_simd_2(black_box(data)))
        });

        group.bench_with_input(BenchmarkId::new("simd_4", size), &data, |b, data| {
            b.iter(|| argmin_simd_4(black_box(data)))
        });

        group.bench_with_input(BenchmarkId::new("simd_8", size), &data, |b, data| {
            b.iter(|| argmin_simd_8(black_box(data)))
        });

        group.bench_with_input(BenchmarkId::new("simd_16", size), &data, |b, data| {
            b.iter(|| argmin_simd_16(black_box(data)))
        });
    }

    group.finish();
}

fn benchmark_large_scale(c: &mut Criterion) {
    let sizes = [1_000_000, 5_000_000, 10_000_000, 25_000_000, 40_000_000];

    let mut group = c.benchmark_group("large_scale");
    group.sample_size(20); // Reduce sample size for large arrays

    for &size in &sizes {
        let data = generate_data(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("scalar", size), &data, |b, data| {
            b.iter(|| argmin_scalar(black_box(data)))
        });

        group.bench_with_input(BenchmarkId::new("simd_8", size), &data, |b, data| {
            b.iter(|| argmin_simd_8(black_box(data)))
        });

        group.bench_with_input(BenchmarkId::new("par_scalar", size), &data, |b, data| {
            b.iter(|| argmin_par_scalar(black_box(data)))
        });

        group.bench_with_input(BenchmarkId::new("par_simd", size), &data, |b, data| {
            b.iter(|| argmin_par_simd(black_box(data)))
        });
    }

    group.finish();
}

fn benchmark_million_special(c: &mut Criterion) {
    let mut group = c.benchmark_group("million_elements");

    // Case 1: Random distribution
    let random_data = generate_data(1_000_000);
    group.bench_function("random_scalar", |b| {
        b.iter(|| argmin_scalar(black_box(&random_data)))
    });
    group.bench_function("random_simd", |b| {
        b.iter(|| argmin_simd(black_box(&random_data)))
    });
    group.bench_function("random_par_scalar", |b| {
        b.iter(|| argmin_par_scalar(black_box(&random_data)))
    });
    group.bench_function("random_par_simd", |b| {
        b.iter(|| argmin_par_simd(black_box(&random_data)))
    });

    // Case 2: Min at beginning
    let mut begin_min = vec![1.0; 1_000_000];
    begin_min[0] = 0.1;
    group.bench_function("begin_scalar", |b| {
        b.iter(|| argmin_scalar(black_box(&begin_min)))
    });
    group.bench_function("begin_simd", |b| {
        b.iter(|| argmin_simd(black_box(&begin_min)))
    });
    group.bench_function("begin_par_scalar", |b| {
        b.iter(|| argmin_par_scalar(black_box(&begin_min)))
    });
    group.bench_function("begin_par_simd", |b| {
        b.iter(|| argmin_par_simd(black_box(&begin_min)))
    });

    // Case 3: Min at end
    let mut end_min = vec![1.0; 1_000_000];
    end_min[999_999] = 0.1;
    group.bench_function("end_scalar", |b| {
        b.iter(|| argmin_scalar(black_box(&end_min)))
    });
    group.bench_function("end_simd", |b| b.iter(|| argmin_simd(black_box(&end_min))));
    group.bench_function("end_par_scalar", |b| {
        b.iter(|| argmin_par_scalar(black_box(&end_min)))
    });
    group.bench_function("end_par_simd", |b| {
        b.iter(|| argmin_par_simd(black_box(&end_min)))
    });

    // Case 4: Min in middle
    let mut mid_min = vec![1.0; 1_000_000];
    mid_min[500_000] = 0.1;
    group.bench_function("middle_scalar", |b| {
        b.iter(|| argmin_scalar(black_box(&mid_min)))
    });
    group.bench_function("middle_simd", |b| {
        b.iter(|| argmin_simd(black_box(&mid_min)))
    });
    group.bench_function("middle_par_scalar", |b| {
        b.iter(|| argmin_par_scalar(black_box(&mid_min)))
    });
    group.bench_function("middle_par_simd", |b| {
        b.iter(|| argmin_par_simd(black_box(&mid_min)))
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_argmin,
    // benchmark_lane_widths,
    // benchmark_large_scale,
    // benchmark_million_special
);
criterion_main!(benches);
