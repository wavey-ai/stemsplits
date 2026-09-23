//! The single GEMM the port is built on.
//!
//! Every Linear, the attention matmuls and every convolution (through im2col)
//! reduce to `out = a @ b`. The Rust scalar loop does not vectorise — a
//! reduction cannot be reassociated without changing the result, and the
//! compiler will not do it — so the kernel is C with explicit AVX2/FMA or
//! NEON, compiled by `build.rs`, as in `encodec-rs`.
//!
//! Single-threaded by design for now. A Lambda at 1769 MB gets one vCPU, so
//! threading would only pay on a larger function; it is deferred until the
//! scalar kernel is done. `STEMSPLITS_THREADS` is reserved for that.

extern "C" {
    fn stemsplits_gemm(a: *const f32, b: *const f32, out: *mut f32, m: usize, k: usize, n: usize);
}

/// How many threads the kernels would use. One for now.
pub fn thread_count() -> usize {
    if let Ok(value) = std::env::var("STEMSPLITS_THREADS") {
        if let Ok(count) = value.parse::<usize>() {
            return count.max(1);
        }
    }
    1
}

/// `out = a @ b`, with `a` `[m, k]`, `b` `[k, n]`, `out` `[m, n]`.
pub fn matmul(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(out.len(), m * n);
    // SAFETY: the lengths above are checked; the slices are disjoint by Rust's
    // aliasing rules and the kernel writes only `out`.
    unsafe {
        stemsplits_gemm(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), m, k, n);
    }
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
