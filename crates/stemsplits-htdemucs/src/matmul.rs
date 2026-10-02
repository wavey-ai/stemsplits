//! The single GEMM the port is built on.
//!
//! Every Linear, the attention matmuls and every convolution (through im2col)
//! reduce to `out = a @ b`. The Rust scalar loop does not vectorise — a
//! reduction cannot be reassociated without changing the result, and the
//! compiler will not do it — so the kernel is C with explicit AVX2/FMA or
//! NEON, compiled by `build.rs`, as in `encodec-rs`.
//!
//! The GEMM splits its rows across threads, since a Lambda's vCPU count
//! scales with its memory and the model is embarrassingly parallel over rows.

#[cfg(not(target_arch = "wasm32"))]
extern "C" {
    fn stemsplits_gemm(a: *const f32, b: *const f32, out: *mut f32, m: usize, k: usize, n: usize);
}

/// The kernel for wasm32. Like the C kernel, it works on blocks of four rows
/// by 16 columns, so each load of `b` serves four rows, and it sums each
/// output in `inner` order.
#[cfg(target_arch = "wasm32")]
#[allow(clippy::needless_range_loop)]
unsafe fn stemsplits_gemm(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    #[cfg(target_feature = "simd128")]
    use core::arch::wasm32::{f32x4_add, f32x4_mul, f32x4_splat, v128, v128_load, v128_store};
    let scalar = |i: usize, column: usize| {
        let mut sum = 0.0f32;
        for inner in 0..k {
            sum += *a.add(i * k + inner) * *b.add(inner * n + column);
        }
        *out.add(i * n + column) = sum;
    };
    let mut i = 0;
    while i + 4 <= m {
        let mut j = 0;
        #[cfg(target_feature = "simd128")]
        {
            let rows = [
                a.add(i * k),
                a.add((i + 1) * k),
                a.add((i + 2) * k),
                a.add((i + 3) * k),
            ];
            while j + 16 <= n {
                let mut c = [[f32x4_splat(0.0); 4]; 4];
                for inner in 0..k {
                    let row = b.add(inner * n + j);
                    let x = [
                        v128_load(row as *const v128),
                        v128_load(row.add(4) as *const v128),
                        v128_load(row.add(8) as *const v128),
                        v128_load(row.add(12) as *const v128),
                    ];
                    for r in 0..4 {
                        let v = f32x4_splat(*rows[r].add(inner));
                        for lane in 0..4 {
                            c[r][lane] = f32x4_add(c[r][lane], f32x4_mul(v, x[lane]));
                        }
                    }
                }
                for r in 0..4 {
                    for lane in 0..4 {
                        v128_store(out.add((i + r) * n + j + lane * 4) as *mut v128, c[r][lane]);
                    }
                }
                j += 16;
            }
            while j + 4 <= n {
                let mut c = [f32x4_splat(0.0); 4];
                for inner in 0..k {
                    let x = v128_load(b.add(inner * n + j) as *const v128);
                    for r in 0..4 {
                        c[r] = f32x4_add(c[r], f32x4_mul(f32x4_splat(*rows[r].add(inner)), x));
                    }
                }
                for r in 0..4 {
                    v128_store(out.add((i + r) * n + j) as *mut v128, c[r]);
                }
                j += 4;
            }
        }
        for r in i..i + 4 {
            for column in j..n {
                scalar(r, column);
            }
        }
        i += 4;
    }
    for r in i..m {
        for column in 0..n {
            scalar(r, column);
        }
    }
}

/// How many threads the kernels use: every vCPU by default, since a Lambda's
/// vCPU count scales with its memory. `STEMSPLITS_THREADS` overrides.
pub fn thread_count() -> usize {
    if let Ok(value) = std::env::var("STEMSPLITS_THREADS") {
        if let Ok(count) = value.parse::<usize>() {
            return count.max(1);
        }
    }
    std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
}

/// Below this much work, the threads cost more than they save.
const PARALLEL_THRESHOLD: usize = 1 << 20;

/// Runs `f` over disjoint chunks of `values`, across threads. For elementwise
/// work (GELU, softmax) that would otherwise be serial beside the matmuls.
pub fn parallel_chunks(values: &mut [f32], f: impl Fn(&mut [f32]) + Sync) {
    let threads = thread_count();
    if threads <= 1 || values.len() < 8192 {
        f(values);
        return;
    }
    let chunk = values.len().div_ceil(threads);
    std::thread::scope(|scope| {
        for part in values.chunks_mut(chunk) {
            let body = &f;
            scope.spawn(move || body(part));
        }
    });
}

/// Runs `f` over disjoint groups of `rows_per_call` rows of `values`.
pub fn parallel_rows(values: &mut [f32], row_len: usize, f: impl Fn(&mut [f32]) + Sync) {
    let rows = values.len() / row_len.max(1);
    let threads = thread_count();
    if threads <= 1 || rows < threads * 2 {
        f(values);
        return;
    }
    let group = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        for part in values.chunks_mut(group * row_len) {
            let body = &f;
            scope.spawn(move || body(part));
        }
    });
}

/// `out = a @ b`, with `a` `[m, k]`, `b` `[k, n]`, `out` `[m, n]`.
///
/// Rows are independent, so they split across threads without changing any
/// per-row summation order — a threaded run is bit-identical to a serial one.
/// Each thread calls the C kernel on its own row block.
pub fn matmul(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(out.len(), m * n);

    let threads = thread_count();
    let work = m.saturating_mul(k).saturating_mul(n);
    if threads <= 1 || m < threads * 2 || work < PARALLEL_THRESHOLD {
        // SAFETY: the lengths above are checked; the slices are disjoint by
        // Rust's aliasing rules and the kernel writes only `out`.
        unsafe {
            stemsplits_gemm(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), m, k, n);
        }
        return;
    }

    // Raw addresses so the disjoint row blocks can move into the threads.
    let a_address = a.as_ptr() as usize;
    let b_address = b.as_ptr() as usize;
    let out_address = out.as_mut_ptr() as usize;
    let rows_per_thread = m.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut start = 0;
        while start < m {
            let rows = rows_per_thread.min(m - start);
            let a_offset = start * k * std::mem::size_of::<f32>();
            let out_offset = start * n * std::mem::size_of::<f32>();
            scope.spawn(move || {
                // SAFETY: each thread writes a disjoint row block of `out` and
                // reads only `a` and `b`.
                unsafe {
                    stemsplits_gemm(
                        (a_address + a_offset) as *const f32,
                        b_address as *const f32,
                        (out_address + out_offset) as *mut f32,
                        rows,
                        k,
                        n,
                    );
                }
            });
            start += rows;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_matches_a_hand_computation() {
        // [2,3] @ [3,2]
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut out = [0.0; 4];
        matmul(&a, &b, &mut out, 2, 3, 2);
        assert_eq!(out, [58.0, 64.0, 139.0, 154.0]);
    }

    /// On aarch64 the kernel sums the blocks of four rows by four columns with
    /// a separate multiply and add and the rest with a fused multiply-add, in
    /// `k` order. The shape crosses each cache block of the kernel.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn matmul_keeps_the_bits_of_the_aarch64_sums() {
        let (m, k, n) = (70, 600, 531);
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 2654435761) % 1999) as f32 / 999.0 - 1.0)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i * 40503) % 997) as f32 / 497.0 - 1.0)
            .collect();
        let mut got = vec![0.0f32; m * n];
        // SAFETY: the lengths match the shape.
        unsafe { stemsplits_gemm(a.as_ptr(), b.as_ptr(), got.as_mut_ptr(), m, k, n) };
        let (m4, n4) = (m - m % 4, n - n % 4);
        for row in 0..m {
            for column in 0..n {
                let mut expected = 0.0f32;
                for inner in 0..k {
                    let (x, y) = (a[row * k + inner], b[inner * n + column]);
                    expected = if row < m4 && column < n4 {
                        expected + x * y
                    } else {
                        x.mul_add(y, expected)
                    };
                }
                assert_eq!(
                    got[row * n + column].to_bits(),
                    expected.to_bits(),
                    "at {row},{column}"
                );
            }
        }
    }

    #[test]
    fn matmul_agrees_with_a_direct_loop_past_the_simd_width() {
        let (m, k, n) = (37, 130, 19);
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 2654435761) % 1000) as f32 / 1000.0)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i * 40503) % 997) as f32 / 997.0)
            .collect();
        let mut got = vec![0.0f32; m * n];
        matmul(&a, &b, &mut got, m, k, n);
        for row in 0..m {
            for column in 0..n {
                let mut expected = 0.0f32;
                for inner in 0..k {
                    expected += a[row * k + inner] * b[inner * n + column];
                }
                assert!(
                    (got[row * n + column] - expected).abs() < 1e-3,
                    "at {row},{column}: {} vs {expected}",
                    got[row * n + column]
                );
            }
        }
    }
}
