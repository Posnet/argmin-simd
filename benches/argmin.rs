use argmin_simd::{argmin_scalar, argmin_simd};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{Rng, SeedableRng};

fn generate_data(size: usize) -> Vec<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    (0..size).map(|_| rng.gen_range(0.0..1000.0)).collect()
}

fn benchmark_argmin(c: &mut Criterion) {
    let sizes = [1000, 10_000, 100_000, 1_000_000];

    let mut group = c.benchmark_group("argmin");

    for size in &sizes {
        let data = generate_data(*size);

        group.bench_with_input(BenchmarkId::new("scalar", size), &data, |b, data| {
            b.iter(|| argmin_scalar(black_box(data)))
        });

        group.bench_with_input(BenchmarkId::new("simd", size), &data, |b, data| {
            b.iter(|| argmin_simd(black_box(data)))
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

    // Case 2: Min at beginning
    let mut begin_min = vec![1.0; 1_000_000];
    begin_min[0] = 0.1;
    group.bench_function("begin_scalar", |b| {
        b.iter(|| argmin_scalar(black_box(&begin_min)))
    });
    group.bench_function("begin_simd", |b| {
        b.iter(|| argmin_simd(black_box(&begin_min)))
    });

    // Case 3: Min at end
    let mut end_min = vec![1.0; 1_000_000];
    end_min[999_999] = 0.1;
    group.bench_function("end_scalar", |b| {
        b.iter(|| argmin_scalar(black_box(&end_min)))
    });
    group.bench_function("end_simd", |b| {
        b.iter(|| argmin_simd(black_box(&end_min)))
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

    group.finish();
}

criterion_group!(benches, benchmark_million_special);
criterion_main!(benches);