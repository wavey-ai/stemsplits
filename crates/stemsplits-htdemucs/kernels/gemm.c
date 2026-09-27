// Single-precision GEMM, `out = a @ b`, `a` [m, k], `b` [k, n], `out` [m, n].
//
// This is the kernel the port spends its time in: every Linear, the attention
// matmuls and every convolution (through im2col) call it. The Rust scalar
// version does not vectorise — a reduction cannot be reassociated without
// changing the result, and the compiler will not do it — so the SIMD is
// explicit, as in `encodec-rs`.
//
// The micro-kernel holds four rows' accumulators in registers across the
// whole `k` loop. That is what makes it fast: a version that reads and writes
// `out` once per `k` is memory-bound on the output, and a version that keeps
// one accumulator per output re-reads the weight for every row. Here each
// weight vector is loaded once and used for four rows, and the output is
// written once. The per-output accumulation order is `k` increasing, as in
// the scalar loop, so a SIMD lane never reorders a sum; only FMA contraction
// differs, which is inside the parity tolerance the tests use.

#include <stddef.h>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static float dot(const float *a_row, const float *b, size_t k, size_t n, size_t column) {
    float sum = 0.0f;
    for (size_t inner = 0; inner < k; inner++) {
        sum += a_row[inner] * b[inner * n + column];
    }
    return sum;
}

void stemsplits_gemm(const float *a, const float *b, float *out,
                     size_t m, size_t k, size_t n) {
    size_t i = 0;
    for (; i + 4 <= m; i += 4) {
        const float *a0 = a + (i + 0) * k;
        const float *a1 = a + (i + 1) * k;
        const float *a2 = a + (i + 2) * k;
        const float *a3 = a + (i + 3) * k;
        float *o0 = out + (i + 0) * n;
        float *o1 = out + (i + 1) * n;
        float *o2 = out + (i + 2) * n;
        float *o3 = out + (i + 3) * n;
        size_t j = 0;
#if defined(__AVX2__)
#if defined(__x86_64__) || defined(_M_X64)
        for (; j + 16 <= n; j += 16) {
            __m256 c0l = _mm256_setzero_ps(), c0h = _mm256_setzero_ps();
            __m256 c1l = _mm256_setzero_ps(), c1h = _mm256_setzero_ps();
            __m256 c2l = _mm256_setzero_ps(), c2h = _mm256_setzero_ps();
            __m256 c3l = _mm256_setzero_ps(), c3h = _mm256_setzero_ps();
            for (size_t inner = 0; inner < k; inner++) {
                const __m256 xl = _mm256_loadu_ps(b + inner * n + j);
                const __m256 xh = _mm256_loadu_ps(b + inner * n + j + 8);
                const __m256 v0 = _mm256_set1_ps(a0[inner]);
                const __m256 v1 = _mm256_set1_ps(a1[inner]);
                const __m256 v2 = _mm256_set1_ps(a2[inner]);
                const __m256 v3 = _mm256_set1_ps(a3[inner]);
                c0l = _mm256_fmadd_ps(v0, xl, c0l);
                c0h = _mm256_fmadd_ps(v0, xh, c0h);
                c1l = _mm256_fmadd_ps(v1, xl, c1l);
                c1h = _mm256_fmadd_ps(v1, xh, c1h);
                c2l = _mm256_fmadd_ps(v2, xl, c2l);
                c2h = _mm256_fmadd_ps(v2, xh, c2h);
                c3l = _mm256_fmadd_ps(v3, xl, c3l);
                c3h = _mm256_fmadd_ps(v3, xh, c3h);
            }
            _mm256_storeu_ps(o0 + j, c0l);
            _mm256_storeu_ps(o0 + j + 8, c0h);
            _mm256_storeu_ps(o1 + j, c1l);
            _mm256_storeu_ps(o1 + j + 8, c1h);
            _mm256_storeu_ps(o2 + j, c2l);
            _mm256_storeu_ps(o2 + j + 8, c2h);
            _mm256_storeu_ps(o3 + j, c3l);
            _mm256_storeu_ps(o3 + j + 8, c3h);
        }
#endif
        for (; j + 8 <= n; j += 8) {
            __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
            for (size_t inner = 0; inner < k; inner++) {
                const __m256 x = _mm256_loadu_ps(b + inner * n + j);
                c0 = _mm256_fmadd_ps(_mm256_set1_ps(a0[inner]), x, c0);
                c1 = _mm256_fmadd_ps(_mm256_set1_ps(a1[inner]), x, c1);
                c2 = _mm256_fmadd_ps(_mm256_set1_ps(a2[inner]), x, c2);
                c3 = _mm256_fmadd_ps(_mm256_set1_ps(a3[inner]), x, c3);
            }
            _mm256_storeu_ps(o0 + j, c0);
            _mm256_storeu_ps(o1 + j, c1);
            _mm256_storeu_ps(o2 + j, c2);
            _mm256_storeu_ps(o3 + j, c3);
        }
#elif defined(__aarch64__) || defined(__ARM_NEON)
#if defined(__aarch64__)
        for (; j + 16 <= n; j += 16) {
            float32x4_t c00 = vdupq_n_f32(0.0f), c01 = vdupq_n_f32(0.0f);
            float32x4_t c02 = vdupq_n_f32(0.0f), c03 = vdupq_n_f32(0.0f);
            float32x4_t c10 = vdupq_n_f32(0.0f), c11 = vdupq_n_f32(0.0f);
            float32x4_t c12 = vdupq_n_f32(0.0f), c13 = vdupq_n_f32(0.0f);
            float32x4_t c20 = vdupq_n_f32(0.0f), c21 = vdupq_n_f32(0.0f);
            float32x4_t c22 = vdupq_n_f32(0.0f), c23 = vdupq_n_f32(0.0f);
            float32x4_t c30 = vdupq_n_f32(0.0f), c31 = vdupq_n_f32(0.0f);
            float32x4_t c32 = vdupq_n_f32(0.0f), c33 = vdupq_n_f32(0.0f);
            for (size_t inner = 0; inner < k; inner++) {
                const float32x4_t x0 = vld1q_f32(b + inner * n + j);
                const float32x4_t x1 = vld1q_f32(b + inner * n + j + 4);
                const float32x4_t x2 = vld1q_f32(b + inner * n + j + 8);
                const float32x4_t x3 = vld1q_f32(b + inner * n + j + 12);
                const float v0 = a0[inner], v1 = a1[inner];
                const float v2 = a2[inner], v3 = a3[inner];
                c00 = vmlaq_n_f32(c00, x0, v0);
                c01 = vmlaq_n_f32(c01, x1, v0);
                c02 = vmlaq_n_f32(c02, x2, v0);
                c03 = vmlaq_n_f32(c03, x3, v0);
                c10 = vmlaq_n_f32(c10, x0, v1);
                c11 = vmlaq_n_f32(c11, x1, v1);
                c12 = vmlaq_n_f32(c12, x2, v1);
                c13 = vmlaq_n_f32(c13, x3, v1);
                c20 = vmlaq_n_f32(c20, x0, v2);
                c21 = vmlaq_n_f32(c21, x1, v2);
                c22 = vmlaq_n_f32(c22, x2, v2);
                c23 = vmlaq_n_f32(c23, x3, v2);
                c30 = vmlaq_n_f32(c30, x0, v3);
                c31 = vmlaq_n_f32(c31, x1, v3);
                c32 = vmlaq_n_f32(c32, x2, v3);
                c33 = vmlaq_n_f32(c33, x3, v3);
            }
            vst1q_f32(o0 + j, c00);
            vst1q_f32(o0 + j + 4, c01);
            vst1q_f32(o0 + j + 8, c02);
            vst1q_f32(o0 + j + 12, c03);
            vst1q_f32(o1 + j, c10);
            vst1q_f32(o1 + j + 4, c11);
            vst1q_f32(o1 + j + 8, c12);
            vst1q_f32(o1 + j + 12, c13);
            vst1q_f32(o2 + j, c20);
            vst1q_f32(o2 + j + 4, c21);
            vst1q_f32(o2 + j + 8, c22);
            vst1q_f32(o2 + j + 12, c23);
            vst1q_f32(o3 + j, c30);
            vst1q_f32(o3 + j + 4, c31);
            vst1q_f32(o3 + j + 8, c32);
            vst1q_f32(o3 + j + 12, c33);
        }
#endif
        for (; j + 8 <= n; j += 8) {
            float32x4_t c0l = vdupq_n_f32(0.0f), c0h = vdupq_n_f32(0.0f);
            float32x4_t c1l = vdupq_n_f32(0.0f), c1h = vdupq_n_f32(0.0f);
            float32x4_t c2l = vdupq_n_f32(0.0f), c2h = vdupq_n_f32(0.0f);
            float32x4_t c3l = vdupq_n_f32(0.0f), c3h = vdupq_n_f32(0.0f);
            for (size_t inner = 0; inner < k; inner++) {
                const float32x4_t xl = vld1q_f32(b + inner * n + j);
                const float32x4_t xh = vld1q_f32(b + inner * n + j + 4);
                const float v0 = a0[inner], v1 = a1[inner];
                const float v2 = a2[inner], v3 = a3[inner];
                c0l = vmlaq_n_f32(c0l, xl, v0);
                c0h = vmlaq_n_f32(c0h, xh, v0);
                c1l = vmlaq_n_f32(c1l, xl, v1);
                c1h = vmlaq_n_f32(c1h, xh, v1);
                c2l = vmlaq_n_f32(c2l, xl, v2);
                c2h = vmlaq_n_f32(c2h, xh, v2);
                c3l = vmlaq_n_f32(c3l, xl, v3);
                c3h = vmlaq_n_f32(c3h, xh, v3);
            }
            vst1q_f32(o0 + j, c0l);
            vst1q_f32(o0 + j + 4, c0h);
            vst1q_f32(o1 + j, c1l);
            vst1q_f32(o1 + j + 4, c1h);
            vst1q_f32(o2 + j, c2l);
            vst1q_f32(o2 + j + 4, c2h);
            vst1q_f32(o3 + j, c3l);
            vst1q_f32(o3 + j + 4, c3h);
        }
        for (; j + 4 <= n; j += 4) {
            float32x4_t c0 = vdupq_n_f32(0.0f), c1 = vdupq_n_f32(0.0f);
            float32x4_t c2 = vdupq_n_f32(0.0f), c3 = vdupq_n_f32(0.0f);
            for (size_t inner = 0; inner < k; inner++) {
                const float32x4_t x = vld1q_f32(b + inner * n + j);
                c0 = vmlaq_n_f32(c0, x, a0[inner]);
                c1 = vmlaq_n_f32(c1, x, a1[inner]);
                c2 = vmlaq_n_f32(c2, x, a2[inner]);
                c3 = vmlaq_n_f32(c3, x, a3[inner]);
            }
            vst1q_f32(o0 + j, c0);
            vst1q_f32(o1 + j, c1);
            vst1q_f32(o2 + j, c2);
            vst1q_f32(o3 + j, c3);
        }
#endif
        for (; j < n; j++) {
            o0[j] = dot(a0, b, k, n, j);
            o1[j] = dot(a1, b, k, n, j);
            o2[j] = dot(a2, b, k, n, j);
            o3[j] = dot(a3, b, k, n, j);
        }
    }
    for (; i < m; i++) {
        const float *a_row = a + i * k;
        float *out_row = out + i * n;
        for (size_t j = 0; j < n; j++) {
            out_row[j] = dot(a_row, b, k, n, j);
        }
    }
}
