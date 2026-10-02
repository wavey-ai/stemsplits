// Single-precision GEMM, `out = a @ b`, `a` [m, k], `b` [k, n], `out` [m, n].
//
// This is the kernel the port spends its time in: every Linear, the attention
// matmuls and every convolution (through im2col) call it. The Rust scalar
// version does not vectorise — a reduction cannot be reassociated without
// changing the result, and the compiler will not do it — so the SIMD is
// explicit, as in `encodec-rs`.
//
// Each output is summed in `k` increasing order, as in the scalar loop, so a
// SIMD lane never reorders a sum. On aarch64 the vector outputs use a separate
// multiply and add, and the outputs outside the blocks of four rows and four
// columns use a fused multiply-add. `build.rs` turns off FMA contraction on
// aarch64 so the compiler keeps that split.

#include <stddef.h>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#include <math.h>
#include <stdlib.h>
#endif

#if defined(__aarch64__)

// The blocks that keep the operands in cache: a panel of `b` of `KC` rows by
// 16 columns stays in L1, a block of `a` of `MC` rows by `KC` stays in L2, and
// the packed `b` of `KC` rows by `NC` columns stays in L2. Between `k` blocks
// the partial sums go to `out` and come back unchanged, so the blocking keeps
// every sum bit for bit.
#define KC 256
#define MC 64
#define NC 512
#define NR 16

static float dot(const float *a_row, const float *b, size_t k, size_t n, size_t column) {
    float sum = 0.0f;
    for (size_t inner = 0; inner < k; inner++) {
        sum = fmaf(a_row[inner], b[inner * n + column], sum);
    }
    return sum;
}

// Four rows by up to 16 columns over `kc` steps. `ap` holds the four rows
// interleaved, `bp` 16 columns per step. `width` is the number of valid
// columns, a multiple of four. With `first`, the sums start at zero;
// otherwise they continue from `c`.
static void kernel_4x16(size_t kc, const float *ap, const float *bp,
                        float *c, size_t ldc, size_t width, int first) {
    float32x4_t c0[4], c1[4], c2[4], c3[4];
    const size_t vectors = width / 4;
    for (size_t v = 0; v < 4; v++) {
        if (!first && v < vectors) {
            c0[v] = vld1q_f32(c + 0 * ldc + 4 * v);
            c1[v] = vld1q_f32(c + 1 * ldc + 4 * v);
            c2[v] = vld1q_f32(c + 2 * ldc + 4 * v);
            c3[v] = vld1q_f32(c + 3 * ldc + 4 * v);
        } else {
            c0[v] = c1[v] = c2[v] = c3[v] = vdupq_n_f32(0.0f);
        }
    }
    for (size_t p = 0; p < kc; p++) {
        const float32x4_t a = vld1q_f32(ap + 4 * p);
        const float32x4_t x0 = vld1q_f32(bp + NR * p);
        const float32x4_t x1 = vld1q_f32(bp + NR * p + 4);
        const float32x4_t x2 = vld1q_f32(bp + NR * p + 8);
        const float32x4_t x3 = vld1q_f32(bp + NR * p + 12);
        c0[0] = vaddq_f32(c0[0], vmulq_laneq_f32(x0, a, 0));
        c0[1] = vaddq_f32(c0[1], vmulq_laneq_f32(x1, a, 0));
        c0[2] = vaddq_f32(c0[2], vmulq_laneq_f32(x2, a, 0));
        c0[3] = vaddq_f32(c0[3], vmulq_laneq_f32(x3, a, 0));
        c1[0] = vaddq_f32(c1[0], vmulq_laneq_f32(x0, a, 1));
        c1[1] = vaddq_f32(c1[1], vmulq_laneq_f32(x1, a, 1));
        c1[2] = vaddq_f32(c1[2], vmulq_laneq_f32(x2, a, 1));
        c1[3] = vaddq_f32(c1[3], vmulq_laneq_f32(x3, a, 1));
        c2[0] = vaddq_f32(c2[0], vmulq_laneq_f32(x0, a, 2));
        c2[1] = vaddq_f32(c2[1], vmulq_laneq_f32(x1, a, 2));
        c2[2] = vaddq_f32(c2[2], vmulq_laneq_f32(x2, a, 2));
        c2[3] = vaddq_f32(c2[3], vmulq_laneq_f32(x3, a, 2));
        c3[0] = vaddq_f32(c3[0], vmulq_laneq_f32(x0, a, 3));
        c3[1] = vaddq_f32(c3[1], vmulq_laneq_f32(x1, a, 3));
        c3[2] = vaddq_f32(c3[2], vmulq_laneq_f32(x2, a, 3));
        c3[3] = vaddq_f32(c3[3], vmulq_laneq_f32(x3, a, 3));
    }
    for (size_t v = 0; v < vectors; v++) {
        vst1q_f32(c + 0 * ldc + 4 * v, c0[v]);
        vst1q_f32(c + 1 * ldc + 4 * v, c1[v]);
        vst1q_f32(c + 2 * ldc + 4 * v, c2[v]);
        vst1q_f32(c + 3 * ldc + 4 * v, c3[v]);
    }
}

void stemsplits_gemm(const float *a, const float *b, float *out,
                     size_t m, size_t k, size_t n) {
    // The blocked part covers whole blocks of four rows and four columns;
    // the scalar loop below covers the rest, as before.
    const size_t m4 = m - m % 4;
    const size_t n4 = n - n % 4;
    if (m4 > 0 && n4 > 0 && k > 0) {
        const size_t kc_max = k < KC ? k : KC;
        const size_t nc_max = n4 < NC ? n4 : NC;
        const size_t mc_max = m4 < MC ? m4 : MC;
        float *bpack = malloc(sizeof(float) * kc_max * ((nc_max + NR - 1) / NR) * NR);
        float *apack = malloc(sizeof(float) * kc_max * mc_max);
        for (size_t jc = 0; jc < n4; jc += NC) {
            const size_t nc = n4 - jc < NC ? n4 - jc : NC;
            const size_t panels = (nc + NR - 1) / NR;
            for (size_t pc = 0; pc < k; pc += KC) {
                const size_t kc = k - pc < KC ? k - pc : KC;
                for (size_t q = 0; q < panels; q++) {
                    const size_t width = nc - q * NR < NR ? nc - q * NR : NR;
                    float *panel = bpack + q * kc * NR;
                    for (size_t p = 0; p < kc; p++) {
                        const float *row = b + (pc + p) * n + jc + q * NR;
                        size_t column = 0;
                        for (; column < width; column++) panel[p * NR + column] = row[column];
                        for (; column < NR; column++) panel[p * NR + column] = 0.0f;
                    }
                }
                for (size_t ic = 0; ic < m4; ic += MC) {
                    const size_t mc = m4 - ic < MC ? m4 - ic : MC;
                    for (size_t r = 0; r < mc; r += 4) {
                        float *block = apack + r * kc;
                        for (size_t p = 0; p < kc; p++) {
                            for (size_t lane = 0; lane < 4; lane++) {
                                block[4 * p + lane] = a[(ic + r + lane) * k + pc + p];
                            }
                        }
                    }
                    for (size_t q = 0; q < panels; q++) {
                        const size_t width = nc - q * NR < NR ? nc - q * NR : NR;
                        for (size_t r = 0; r < mc; r += 4) {
                            kernel_4x16(kc, apack + r * kc, bpack + q * kc * NR,
                                        out + (ic + r) * n + jc + q * NR, n, width, pc == 0);
                        }
                    }
                }
            }
        }
        free(apack);
        free(bpack);
    }
    for (size_t i = 0; i < m4; i++) {
        const float *a_row = a + i * k;
        for (size_t j = n4; j < n; j++) {
            out[i * n + j] = dot(a_row, b, k, n, j);
        }
    }
    for (size_t i = m4; i < m; i++) {
        const float *a_row = a + i * k;
        for (size_t j = 0; j < n; j++) {
            out[i * n + j] = dot(a_row, b, k, n, j);
        }
    }
}

#else

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

#endif
