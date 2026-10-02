//! The primitive operations HTDemucs is made of, written to match PyTorch.
//!
//! Scalar and obvious first. A SIMD kernel replaces the hot ones later; the
//! reference activation tests are what make that safe.

// Numeric kernels read better with explicit index loops, and the convolutions
// carry their geometry as arguments rather than a config struct.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use crate::tensor::Tensor;

const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// `F.gelu`, the exact (`approximate = "none"`) form.
pub fn gelu(x: &Tensor) -> Tensor {
    let mut data = x.data.clone();
    crate::matmul::parallel_chunks(&mut data, gelu_in_place);
    Tensor {
        shape: x.shape.clone(),
        data,
    }
}

fn gelu_one(value: f32) -> f32 {
    0.5 * value * (1.0 + libm::erff(value * FRAC_1_SQRT_2))
}

#[cfg(not(target_arch = "aarch64"))]
fn gelu_in_place(values: &mut [f32]) {
    for value in values {
        *value = gelu_one(*value);
    }
}

/// `gelu_one` four values at a time with `neon_erf::erf4`.
#[cfg(target_arch = "aarch64")]
fn gelu_in_place(values: &mut [f32]) {
    use core::arch::aarch64::*;
    let mut chunks = values.chunks_exact_mut(4);
    for chunk in &mut chunks {
        if chunk.iter().any(|value| !value.is_finite()) {
            for value in chunk {
                *value = gelu_one(*value);
            }
            continue;
        }
        // SAFETY: NEON is part of the aarch64 base architecture, and the
        // chunk holds four values.
        unsafe {
            let v = vld1q_f32(chunk.as_ptr());
            let erf = neon_erf::erf4(vmulq_n_f32(v, FRAC_1_SQRT_2));
            let y = vmulq_f32(vmulq_n_f32(v, 0.5), vaddq_f32(vdupq_n_f32(1.0), erf));
            vst1q_f32(chunk.as_mut_ptr(), y);
        }
    }
    for value in chunks.into_remainder() {
        *value = gelu_one(*value);
    }
}

/// `libm::erff` and the `libm::expf` it calls, in NEON. Each lane computes
/// every branch of `erff` with the same f32 operations in the same order,
/// and keeps the branch of its value, so each lane gives the bits of
/// `libm::erff`. The lanes must be finite.
// The constants keep the digits of `libm`, so each f32 is the same.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::excessive_precision)]
pub(crate) mod neon_erf {
    use core::arch::aarch64::*;

    const ERX: f32 = 8.4506291151e-01;
    const EFX8: f32 = 1.0270333290e+00;
    const PP: [f32; 5] = [
        1.2837916613e-01,
        -3.2504209876e-01,
        -2.8481749818e-02,
        -5.7702702470e-03,
        -2.3763017452e-05,
    ];
    const QQ: [f32; 5] = [
        3.9791721106e-01,
        6.5022252500e-02,
        5.0813062117e-03,
        1.3249473704e-04,
        -3.9602282413e-06,
    ];
    const PA: [f32; 7] = [
        -2.3621185683e-03,
        4.1485610604e-01,
        -3.7220788002e-01,
        3.1834661961e-01,
        -1.1089469492e-01,
        3.5478305072e-02,
        -2.1663755178e-03,
    ];
    const QA: [f32; 6] = [
        1.0642088205e-01,
        5.4039794207e-01,
        7.1828655899e-02,
        1.2617121637e-01,
        1.3637083583e-02,
        1.1984500103e-02,
    ];
    const RA: [f32; 8] = [
        -9.8649440333e-03,
        -6.9385856390e-01,
        -1.0558626175e+01,
        -6.2375331879e+01,
        -1.6239666748e+02,
        -1.8460508728e+02,
        -8.1287437439e+01,
        -9.8143291473e+00,
    ];
    const SA: [f32; 8] = [
        1.9651271820e+01,
        1.3765776062e+02,
        4.3456588745e+02,
        6.4538726807e+02,
        4.2900814819e+02,
        1.0863500214e+02,
        6.5702495575e+00,
        -6.0424413532e-02,
    ];
    const RB: [f32; 7] = [
        -9.8649431020e-03,
        -7.9928326607e-01,
        -1.7757955551e+01,
        -1.6063638306e+02,
        -6.3756646729e+02,
        -1.0250950928e+03,
        -4.8351919556e+02,
    ];
    const SB: [f32; 7] = [
        3.0338060379e+01,
        3.2579251099e+02,
        1.5367296143e+03,
        3.1998581543e+03,
        2.5530502930e+03,
        4.7452853394e+02,
        -2.2440952301e+01,
    ];

    #[inline(always)]
    unsafe fn n(value: f32) -> float32x4_t {
        vdupq_n_f32(value)
    }

    #[inline(always)]
    unsafe fn add(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vaddq_f32(a, b)
    }

    #[inline(always)]
    unsafe fn mul(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vmulq_f32(a, b)
    }

    /// `c[0] + s * (c[1] + s * (... + s * c[last]))`, innermost first.
    #[inline(always)]
    unsafe fn horner(s: float32x4_t, c: &[f32]) -> float32x4_t {
        let mut acc = n(c[c.len() - 1]);
        for &coefficient in c[..c.len() - 1].iter().rev() {
            acc = add(n(coefficient), mul(s, acc));
        }
        acc
    }

    /// `1 + s * (c[0] + s * (... + s * c[last]))`.
    #[inline(always)]
    unsafe fn one_plus(s: float32x4_t, c: &[f32]) -> float32x4_t {
        add(n(1.0), mul(s, horner(s, c)))
    }

    /// `libm::expf` for `|x| < 87.33` with a result above 2^-126.
    #[inline(always)]
    unsafe fn expf4(x: float32x4_t) -> float32x4_t {
        const LN2_HI: f32 = 6.9314575195e-01;
        const LN2_LO: f32 = 1.4286067653e-06;
        const INV_LN2: f32 = 1.4426950216e+00;
        const P1: f32 = 1.6666625440e-1;
        const P2: f32 = -2.7667332906e-3;
        let bits = vreinterpretq_u32_f32(x);
        let sign = vshrq_n_u32::<31>(bits);
        let hx = vandq_u32(bits, vdupq_n_u32(0x7fff_ffff));
        // k for |x| > 1.5 ln 2: `(INV_LN2 * x + HALF[sign]) as i32`.
        let half = vbslq_f32(vceqq_u32(sign, vdupq_n_u32(0)), n(0.5), n(-0.5));
        let k_far = vcvtq_s32_f32(add(mul(n(INV_LN2), x), half));
        // k for 0.5 ln 2 < |x| <= 1.5 ln 2: `1 - sign - sign`.
        let sign_i = vreinterpretq_s32_u32(sign);
        let k_near = vsubq_s32(vsubq_s32(vdupq_n_s32(1), sign_i), sign_i);
        let far = vcgtq_u32(hx, vdupq_n_u32(0x3f85_1592));
        let reduce = vcgtq_u32(hx, vdupq_n_u32(0x3eb1_7218));
        let k = vbslq_s32(far, k_far, k_near);
        let k = vbslq_s32(reduce, k, vdupq_n_s32(0));
        let kf = vcvtq_f32_s32(k);
        let hi_reduced = vsubq_f32(x, mul(kf, n(LN2_HI)));
        let lo_reduced = mul(kf, n(LN2_LO));
        let hi = vbslq_f32(reduce, hi_reduced, x);
        let lo = vbslq_f32(reduce, lo_reduced, n(0.0));
        let r = vbslq_f32(reduce, vsubq_f32(hi_reduced, lo_reduced), x);
        let xx = mul(r, r);
        let c = vsubq_f32(r, mul(xx, add(n(P1), mul(xx, n(P2)))));
        let q = vdivq_f32(mul(r, c), vsubq_f32(n(2.0), c));
        let y = add(n(1.0), add(vsubq_f32(q, lo), hi));
        // `scalbnf(y, k)` for k in the normal range.
        let scale = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(k, vdupq_n_s32(0x7f))));
        let y = mul(y, scale);
        // |x| <= 2^-14: `1 + x`.
        let tiny = vcleq_u32(hx, vdupq_n_u32(0x3900_0000));
        vbslq_f32(tiny, add(n(1.0), x), y)
    }

    #[inline(always)]
    pub(crate) unsafe fn erf4(x: float32x4_t) -> float32x4_t {
        let bits = vreinterpretq_u32_f32(x);
        let ix = vandq_u32(bits, vdupq_n_u32(0x7fff_ffff));
        let sign_bit = vandq_u32(bits, vdupq_n_u32(0x8000_0000));
        let ax = vreinterpretq_f32_u32(ix);

        // |x| < 2^-28.
        let tiny = vmulq_n_f32(add(mul(n(8.0), x), mul(n(EFX8), x)), 0.125);
        // |x| < 0.84375.
        let z = mul(x, x);
        let r = horner(z, &PP);
        let s = one_plus(z, &QQ);
        let small = add(x, mul(x, vdivq_f32(r, s)));
        // |x| < 1.25: `1 - erfc1(x)`.
        let s1 = vsubq_f32(ax, n(1.0));
        let p = horner(s1, &PA);
        let q = one_plus(s1, &QA);
        let erfc1 = vsubq_f32(n(1.0 - ERX), vdivq_f32(p, q));
        // |x| < 6: `1 - erfc2(x)`.
        let s2 = vdivq_f32(n(1.0), mul(ax, ax));
        let near = vcltq_u32(ix, vdupq_n_u32(0x4036_db6d));
        let r2 = vbslq_f32(near, horner(s2, &RA), horner(s2, &RB));
        let big_s = vbslq_f32(near, one_plus(s2, &SA), one_plus(s2, &SB));
        let z2 = vreinterpretq_f32_u32(vandq_u32(ix, vdupq_n_u32(0xffff_e000)));
        let first = expf4(vsubq_f32(mul(vnegq_f32(z2), z2), n(0.5625)));
        let second = expf4(add(
            mul(vsubq_f32(z2, ax), add(z2, ax)),
            vdivq_f32(r2, big_s),
        ));
        let erfc2 = vdivq_f32(mul(first, second), ax);
        let erfc = vbslq_f32(vcltq_u32(ix, vdupq_n_u32(0x3fa0_0000)), erfc1, erfc2);
        let large = vsubq_f32(n(1.0), erfc);
        // |x| >= 6.
        let large = vbslq_f32(
            vcltq_u32(ix, vdupq_n_u32(0x40c0_0000)),
            large,
            n(1.0 - f32::from_bits(0x0380_0000)),
        );
        // `-y` for a negative x.
        let large = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(large), sign_bit));

        let result = vbslq_f32(vcltq_u32(ix, vdupq_n_u32(0x3f58_0000)), small, large);
        vbslq_f32(vcltq_u32(ix, vdupq_n_u32(0x3180_0000)), tiny, result)
    }
}

pub fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + exp(-value))
}

/// `e^x`. On wasm32, `f32::exp` is a slow software routine, so this is a
/// range reduction to `[-ln 2 / 2, ln 2 / 2]` and a degree-6 polynomial,
/// within about 1 ulp of it.
#[cfg(target_arch = "wasm32")]
#[inline]
pub fn exp(x: f32) -> f32 {
    if x < -87.33 {
        return 0.0;
    }
    if x > 88.72 {
        return f32::INFINITY;
    }
    if x.is_nan() {
        return x;
    }
    let n = (x * std::f32::consts::LOG2_E).round_ties_even();
    let r = (x - n * 0.693_145_75) - n * 1.428_606_8e-6;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0 + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0))))));
    // 2^n for n in [-126, 127]; below that, scale in two steps.
    let n = n as i32;
    if n < -126 {
        return p
            * f32::from_bits(((n + 127 + 64) as u32) << 23)
            * f32::from_bits(((127 - 64) as u32) << 23);
    }
    p * f32::from_bits(((n + 127) as u32) << 23)
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub fn exp(x: f32) -> f32 {
    x.exp()
}

/// `exp` of each value in place.
#[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
pub fn exp_in_place(values: &mut [f32]) {
    for value in values {
        *value = exp(*value);
    }
}

/// `exp` of each value in place, four values at a time. On Linux, `f32::exp`
/// is the glibc `expf` of ARM's optimized routines. This is the same
/// arithmetic in NEON: the same constants and table, and the same fused
/// multiply-adds. Values outside its fast range go to `f32::exp`.
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
pub fn exp_in_place(values: &mut [f32]) {
    let mut chunks = values.chunks_exact_mut(4);
    for chunk in &mut chunks {
        let input: [f32; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        if input.iter().all(|value| glibc_exp::in_fast_range(*value)) {
            // SAFETY: NEON is part of the aarch64 base architecture.
            chunk.copy_from_slice(&unsafe { glibc_exp::four(input) });
        } else {
            for value in chunk {
                *value = value.exp();
            }
        }
    }
    for value in chunks.into_remainder() {
        *value = value.exp();
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
pub(crate) mod glibc_exp {
    use core::arch::aarch64::*;

    /// `N / ln 2` with `N = 32`.
    const INV_LN2_N: f64 = f64::from_bits(0x4047_1547_652b_82fe);
    /// The polynomial, highest degree first.
    const C0: f64 = f64::from_bits(0x3ebc_6af8_4b91_2394);
    const C1: f64 = f64::from_bits(0x3f2e_bfce_50fa_c4f3);
    const C2: f64 = f64::from_bits(0x3f96_2e42_ff0c_52d6);
    /// `2^(i / 32)` with `i << 47` taken from the bits.
    const TABLE: [u64; 32] = [
        0x3ff0000000000000,
        0x3fefd9b0d3158574,
        0x3fefb5586cf9890f,
        0x3fef9301d0125b51,
        0x3fef72b83c7d517b,
        0x3fef54873168b9aa,
        0x3fef387a6e756238,
        0x3fef1e9df51fdee1,
        0x3fef06fe0a31b715,
        0x3feef1a7373aa9cb,
        0x3feedea64c123422,
        0x3feece086061892d,
        0x3feebfdad5362a27,
        0x3feeb42b569d4f82,
        0x3feeab07dd485429,
        0x3feea47eb03a5585,
        0x3feea09e667f3bcd,
        0x3fee9f75e8ec5f74,
        0x3feea11473eb0187,
        0x3feea589994cce13,
        0x3feeace5422aa0db,
        0x3feeb737b0cdc5e5,
        0x3feec49182a3f090,
        0x3feed503b23e255d,
        0x3feee89f995ad3ad,
        0x3feeff76f2fb5e47,
        0x3fef199bdd85529c,
        0x3fef3720dcef9069,
        0x3fef5818dcfba487,
        0x3fef7c97337b9b5f,
        0x3fefa4afa2a490da,
        0x3fefd0765b6e4540,
    ];

    /// glibc takes the fast path when the top 12 bits of `|x|` are at most
    /// those of 88.0.
    #[inline(always)]
    pub(crate) fn in_fast_range(x: f32) -> bool {
        (x.to_bits() >> 20) & 0x7ff <= 0x42a
    }

    #[inline(always)]
    unsafe fn two(x: float64x2_t) -> float64x2_t {
        let z = vmulq_f64(x, vdupq_n_f64(INV_LN2_N));
        let kd = vrndaq_f64(z);
        let ki = vcvtaq_s64_f64(z);
        let r = vsubq_f64(z, kd);
        let index = vandq_s64(ki, vdupq_n_s64(31));
        let table = vcombine_u64(
            vcreate_u64(TABLE[vgetq_lane_s64::<0>(index) as usize]),
            vcreate_u64(TABLE[vgetq_lane_s64::<1>(index) as usize]),
        );
        let s = vreinterpretq_f64_u64(vaddq_u64(
            table,
            vshlq_n_u64::<47>(vreinterpretq_u64_s64(ki)),
        ));
        let p = vfmaq_f64(vdupq_n_f64(C1), vdupq_n_f64(C0), r);
        let y = vfmaq_f64(vdupq_n_f64(1.0), vdupq_n_f64(C2), r);
        let y = vfmaq_f64(y, p, vmulq_f64(r, r));
        vmulq_f64(y, s)
    }

    /// `expf` of four values in the fast range.
    #[inline(always)]
    pub(crate) unsafe fn four(x: [f32; 4]) -> [f32; 4] {
        let x = vld1q_f32(x.as_ptr());
        let low = two(vcvt_f64_f32(vget_low_f32(x)));
        let high = two(vcvt_high_f64_f32(x));
        let y = vcvt_high_f32_f64(vcvt_f32_f64(low), high);
        let mut out = [0.0f32; 4];
        vst1q_f32(out.as_mut_ptr(), y);
        out
    }
}

/// `F.glu(x, dim = 1)`: `a * sigmoid(b)` where the channel axis is split in
/// half. Works for any rank; the channel axis is the second.
pub fn glu(x: &Tensor) -> Tensor {
    let batch = x.dim(0);
    let channels = x.dim(1);
    let half = channels / 2;
    let spatial: usize = x.shape[2..].iter().product();
    let mut shape = x.shape.clone();
    shape[1] = half;
    let mut output = Tensor::zeros(shape);
    for index in 0..batch {
        for channel in 0..half {
            let a = (index * channels + channel) * spatial;
            let b = (index * channels + channel + half) * spatial;
            let destination = (index * half + channel) * spatial;
            // `sigmoid` with its exponentials taken together.
            let gate = &mut output.data[destination..destination + spatial];
            for (value, &b) in gate.iter_mut().zip(&x.data[b..b + spatial]) {
                *value = -b;
            }
            exp_in_place(gate);
            for (value, &a) in gate.iter_mut().zip(&x.data[a..a + spatial]) {
                *value = a * (1.0 / (1.0 + *value));
            }
        }
    }
    output
}

/// `nn.GroupNorm`: normalise over each group's channels and all spatial
/// positions, then scale per channel. Used with one group by DConv, so the
/// whole channel axis and the time axis share a statistic.
pub fn group_norm(x: &Tensor, groups: usize, weight: &[f32], bias: &[f32], eps: f32) -> Tensor {
    let batch = x.dim(0);
    let channels = x.dim(1);
    assert_eq!(
        channels % groups,
        0,
        "channels {channels} not divisible by {groups}"
    );
    let per_group = channels / groups;
    let spatial: usize = x.shape[2..].iter().product();
    let count = (per_group * spatial) as f32;
    let mut output = x.clone();
    for index in 0..batch {
        for group in 0..groups {
            let base = index * channels + group * per_group;
            let mut sum = 0.0f64;
            for channel in 0..per_group {
                let start = (base + channel) * spatial;
                for offset in 0..spatial {
                    sum += x.data[start + offset] as f64;
                }
            }
            let mean = (sum / count as f64) as f32;
            let mut variance = 0.0f64;
            for channel in 0..per_group {
                let start = (base + channel) * spatial;
                for offset in 0..spatial {
                    let delta = x.data[start + offset] - mean;
                    variance += (delta * delta) as f64;
                }
            }
            let inverse = 1.0 / ((variance / count as f64) as f32 + eps).sqrt();
            for channel in 0..per_group {
                let global = group * per_group + channel;
                let scale = weight[global];
                let shift = bias[global];
                let start = (base + channel) * spatial;
                for offset in 0..spatial {
                    output.data[start + offset] =
                        (x.data[start + offset] - mean) * inverse * scale + shift;
                }
            }
        }
    }
    output
}

/// `MyGroupNorm(1, channels)` on a `[B, T, C]` sequence: transpose to
/// `[B, C, T]`, normalise over all channels and time per sample, scale per
/// channel, transpose back. The transformer's `norm_out` is this.
pub fn group_norm_sequence(x: &Tensor, weight: &[f32], bias: &[f32], eps: f32) -> Tensor {
    let batch = x.dim(0);
    let time = x.dim(1);
    let channels = x.dim(2);
    assert_eq!(weight.len(), channels);
    let count = (time * channels) as f64;
    let mut output = x.clone();
    for index in 0..batch {
        let base = index * time * channels;
        let mut sum = 0.0f64;
        for offset in 0..(time * channels) {
            sum += x.data[base + offset] as f64;
        }
        let mean = sum / count;
        let mut variance = 0.0f64;
        for offset in 0..(time * channels) {
            let delta = x.data[base + offset] as f64 - mean;
            variance += delta * delta;
        }
        variance /= count;
        let inverse = 1.0 / ((variance as f32) + eps).sqrt();
        for t in 0..time {
            for c in 0..channels {
                let offset = base + t * channels + c;
                let normalised = (x.data[offset] - mean as f32) * inverse;
                output.data[offset] = normalised * weight[c] + bias[c];
            }
        }
    }
    output
}

/// `nn.LayerNorm` over the last axis, with affine weight and bias.
pub fn layer_norm(x: &Tensor, weight: &[f32], bias: &[f32], eps: f32) -> Tensor {
    let features = *x.shape.last().unwrap();
    assert_eq!(weight.len(), features);
    let rows = x.numel() / features;
    let mut output = x.clone();
    crate::matmul::parallel_rows(&mut output.data, features, |chunk| {
        for row in chunk.chunks_mut(features) {
            let mut mean = 0.0f64;
            for value in row.iter() {
                mean += *value as f64;
            }
            mean /= features as f64;
            let mut variance = 0.0f64;
            for value in row.iter() {
                let delta = *value as f64 - mean;
                variance += delta * delta;
            }
            variance /= features as f64;
            let inverse = 1.0 / ((variance as f32) + eps).sqrt();
            for (feature, value) in row.iter_mut().enumerate() {
                *value = (*value - mean as f32) * inverse * weight[feature] + bias[feature];
            }
        }
    });
    let _ = rows;
    output
}

/// `nn.Linear` over the last axis: `x @ weight^T + bias`.
///
/// Transposes the weight to `[features, out]` before multiplication.
pub fn linear(x: &Tensor, weight: &[f32], bias: &[f32]) -> Tensor {
    let features = *x.shape.last().unwrap();
    let out_features = weight.len() / features;
    assert_eq!(weight.len() % features, 0, "linear weight shape");
    assert_eq!(bias.len(), out_features);
    let transposed = transpose_weight(weight, features, out_features);
    linear_transposed(x, &transposed, bias)
}

fn transpose_weight(weight: &[f32], features: usize, out_features: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; features * out_features];
    for out in 0..out_features {
        let row = &weight[out * features..(out + 1) * features];
        for (feature, &value) in row.iter().enumerate() {
            transposed[feature * out_features + out] = value;
        }
    }
    transposed
}

/// Stores fixed weights in the matrix kernel's input layout.
pub(crate) struct Linear {
    transposed: Vec<f32>,
    bias: Vec<f32>,
    features: usize,
}

impl Linear {
    pub(crate) fn new(weight: &[f32], bias: Vec<f32>) -> Self {
        assert!(!bias.is_empty(), "linear output features");
        assert_eq!(weight.len() % bias.len(), 0, "linear weight shape");
        let features = weight.len() / bias.len();
        Self {
            transposed: transpose_weight(weight, features, bias.len()),
            bias,
            features,
        }
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Tensor {
        assert_eq!(*x.shape.last().unwrap(), self.features);
        linear_transposed(x, &self.transposed, &self.bias)
    }
}

fn linear_transposed(x: &Tensor, transposed: &[f32], bias: &[f32]) -> Tensor {
    let features = *x.shape.last().unwrap();
    let out_features = bias.len();
    let rows = x.numel() / features;
    let mut shape = x.shape.clone();
    *shape.last_mut().unwrap() = out_features;
    let mut data = vec![0.0f32; rows * out_features];
    crate::matmul::matmul(&x.data, transposed, &mut data, rows, features, out_features);
    for row in 0..rows {
        let base = row * out_features;
        for out in 0..out_features {
            data[base + out] += bias[out];
        }
    }
    Tensor::new(shape, data)
}

/// LayerScale with `channel_last = True`: `x` is `[B, T, C]` and the scale
/// multiplies the last axis. The transformer's gamma uses this form.
pub fn layer_scale_last(x: &Tensor, scale: &[f32]) -> Tensor {
    let channels = *x.shape.last().unwrap();
    assert_eq!(scale.len(), channels);
    let rows = x.numel() / channels;
    let mut output = x.clone();
    for row in 0..rows {
        for channel in 0..channels {
            output.data[row * channels + channel] *= scale[channel];
        }
    }
    output
}

/// LayerScale with `channel_last = False`: `x` is `[B, C, ...]` and the scale
/// multiplies the second axis. DConv uses this form.
pub fn layer_scale(x: &Tensor, scale: &[f32]) -> Tensor {
    let channels = x.dim(1);
    assert_eq!(scale.len(), channels);
    let spatial: usize = x.shape[2..].iter().product();
    let batch = x.dim(0);
    let mut output = x.clone();
    for index in 0..batch {
        for channel in 0..channels {
            let start = (index * channels + channel) * spatial;
            let factor = scale[channel];
            for offset in 0..spatial {
                output.data[start + offset] *= factor;
            }
        }
    }
    output
}
/// `nn.Conv1d` over `[N, C, T]`, as im2col plus the blocked matmul. The
/// encoder's time convolutions and every DConv use this.
pub fn conv1d(
    x: &Tensor,
    weight: &[f32],
    bias: &[f32],
    out_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Tensor {
    let batch = x.dim(0);
    let in_channels = x.dim(1);
    let time = x.dim(2);
    let effective = dilation * (kernel - 1) + 1;
    assert!(time + 2 * padding >= effective, "conv1d input too short");
    let out_time = (time + 2 * padding - effective) / stride + 1;
    let columns = in_channels * kernel;
    let mut output = Tensor::zeros(vec![batch, out_channels, out_time]);
    let mut weight_transposed = vec![0.0f32; columns * out_channels];
    for out in 0..out_channels {
        for column in 0..columns {
            weight_transposed[column * out_channels + out] = weight[out * columns + column];
        }
    }

    let rows_per_tile = (2_000_000 / columns.max(1)).max(1).min(out_time);
    let mut patches = vec![0.0f32; rows_per_tile * columns];
    let mut tile_output = vec![0.0f32; rows_per_tile * out_channels];

    for index in 0..batch {
        let mut start = 0;
        while start < out_time {
            let end = (start + rows_per_tile).min(out_time);
            let rows = end - start;
            let patches = &mut patches[..rows * columns];
            patches.fill(0.0);
            for out in start..end {
                let row = out - start;
                for input_channel in 0..in_channels {
                    for tap in 0..kernel {
                        let position = out * stride + tap * dilation;
                        if position < padding {
                            continue;
                        }
                        let position = position - padding;
                        if position >= time {
                            continue;
                        }
                        patches[row * columns + input_channel * kernel + tap] =
                            x.data[(index * in_channels + input_channel) * time + position];
                    }
                }
            }
            let tile_output = &mut tile_output[..rows * out_channels];
            crate::matmul::matmul(
                patches,
                &weight_transposed,
                tile_output,
                rows,
                columns,
                out_channels,
            );
            for row in 0..rows {
                for channel in 0..out_channels {
                    output.data[(index * out_channels + channel) * out_time + start + row] =
                        tile_output[row * out_channels + channel] + bias[channel];
                }
            }
            start = end;
        }
    }
    output
}
/// `nn.Conv2d` over `[N, C, H, W]` with zero padding, as im2col plus the
/// blocked matmul. `kernel`, `stride` and `pad` are `(height, width)`.
///
/// The direct form is a serial accumulation over `cin * kh * kw` per output
/// pixel; the decoder's 3x3 rewrites alone are ~28 GFLOP of it. Laying the
/// patches out and calling `matmul_transposed` gets the same arithmetic with
/// independent accumulation and vectorisation. The patch buffer is tiled over
/// output rows so it stays a few MB.
pub fn conv2d(
    x: &Tensor,
    weight: &[f32],
    bias: &[f32],
    out_channels: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
) -> Tensor {
    let batch = x.dim(0);
    let in_channels = x.dim(1);
    let height = x.dim(2);
    let width = x.dim(3);
    let (kernel_h, kernel_w) = kernel;
    let (stride_h, stride_w) = stride;
    let (pad_h, pad_w) = pad;
    let out_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    let out_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    let columns = in_channels * kernel_h * kernel_w;
    let mut output = Tensor::zeros(vec![batch, out_channels, out_height, out_width]);
    // Transpose the weight [out, columns] to [columns, out] so the matmul's
    // inner loop is the vectorisable one. Cost is `out * columns` against
    // `rows * out * columns` of work.
    let mut weight_transposed = vec![0.0f32; columns * out_channels];
    for out in 0..out_channels {
        for column in 0..columns {
            weight_transposed[column * out_channels + out] = weight[out * columns + column];
        }
    }

    // Keep the patch tile near 8 MB of f32.
    let rows_per_tile = (2_000_000 / (out_width * columns).max(1))
        .max(1)
        .min(out_height);
    let mut patches = vec![0.0f32; rows_per_tile * out_width * columns];
    let mut tile_output = vec![0.0f32; rows_per_tile * out_width * out_channels];

    for index in 0..batch {
        let mut start_row = 0;
        while start_row < out_height {
            let end_row = (start_row + rows_per_tile).min(out_height);
            let rows = (end_row - start_row) * out_width;
            let patches = &mut patches[..rows * columns];
            patches.fill(0.0);
            for out_h in start_row..end_row {
                for out_w in 0..out_width {
                    let row = (out_h - start_row) * out_width + out_w;
                    for input_channel in 0..in_channels {
                        for tap_h in 0..kernel_h {
                            let source_h = out_h * stride_h + tap_h;
                            if source_h < pad_h {
                                continue;
                            }
                            let source_h = source_h - pad_h;
                            if source_h >= height {
                                continue;
                            }
                            for tap_w in 0..kernel_w {
                                let source_w = out_w * stride_w + tap_w;
                                if source_w < pad_w {
                                    continue;
                                }
                                let source_w = source_w - pad_w;
                                if source_w >= width {
                                    continue;
                                }
                                patches[row * columns
                                    + (input_channel * kernel_h + tap_h) * kernel_w
                                    + tap_w] = x.data[((index * in_channels + input_channel)
                                    * height
                                    + source_h)
                                    * width
                                    + source_w];
                            }
                        }
                    }
                }
            }

            let tile_output = &mut tile_output[..rows * out_channels];
            crate::matmul::matmul(
                patches,
                &weight_transposed,
                tile_output,
                rows,
                columns,
                out_channels,
            );

            for row in 0..rows {
                let out_h = start_row + row / out_width;
                let out_w = row % out_width;
                for channel in 0..out_channels {
                    output.data[((index * out_channels + channel) * out_height + out_h)
                        * out_width
                        + out_w] = tile_output[row * out_channels + channel] + bias[channel];
                }
            }
            start_row = end_row;
        }
    }
    output
}

/// `nn.ConvTranspose1d` over `[N, C, T]`. Weight is `[Cin, Cout, kernel]`.
/// Each input position's contribution is a matmul against the reshaped
/// weight, then scattered into the output.
pub fn conv_transpose1d(
    x: &Tensor,
    weight: &[f32],
    bias: &[f32],
    out_channels: usize,
    kernel: usize,
    stride: usize,
) -> Tensor {
    let batch = x.dim(0);
    let in_channels = x.dim(1);
    let time = x.dim(2);
    let out_time = (time - 1) * stride + kernel;
    let contributions = out_channels * kernel;
    let mut output = Tensor::zeros(vec![batch, out_channels, out_time]);

    let positions_per_tile = (2_000_000 / contributions.max(1)).max(1).min(time);
    let mut block = vec![0.0f32; positions_per_tile * contributions];
    let mut input_block = vec![0.0f32; positions_per_tile * in_channels];
    for index in 0..batch {
        let mut start = 0;
        while start < time {
            let end = (start + positions_per_tile).min(time);
            let rows = end - start;
            // The input is channel-major; gather it position-major for the matmul.
            let input = &mut input_block[..rows * in_channels];
            for row in 0..rows {
                for input_channel in 0..in_channels {
                    input[row * in_channels + input_channel] =
                        x.data[(index * in_channels + input_channel) * time + start + row];
                }
            }
            // `weight` is [Cin, Cout, kernel], i.e. already [Cin, Cout*kernel].
            let block = &mut block[..rows * contributions];
            crate::matmul::matmul(input, weight, block, rows, in_channels, contributions);
            for position in start..end {
                let row = position - start;
                for channel in 0..out_channels {
                    for tap in 0..kernel {
                        output.data[(index * out_channels + channel) * out_time
                            + position * stride
                            + tap] += block[row * contributions + channel * kernel + tap];
                    }
                }
            }
            start = end;
        }
        for channel in 0..out_channels {
            let base = (index * out_channels + channel) * out_time;
            for position in 0..out_time {
                output.data[base + position] += bias[channel];
            }
        }
    }
    output
}

/// `nn.ConvTranspose2d` over `[N, C, H, W]`, no padding. Weight is
/// `[Cin, Cout, kH, kW]`.
pub fn conv_transpose2d(
    x: &Tensor,
    weight: &[f32],
    bias: &[f32],
    out_channels: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
) -> Tensor {
    let batch = x.dim(0);
    let in_channels = x.dim(1);
    let height = x.dim(2);
    let width = x.dim(3);
    let (kernel_h, kernel_w) = kernel;
    let (stride_h, stride_w) = stride;
    let out_height = (height - 1) * stride_h + kernel_h;
    let out_width = (width - 1) * stride_w + kernel_w;
    let contributions = out_channels * kernel_h * kernel_w;
    let mut output = Tensor::zeros(vec![batch, out_channels, out_height, out_width]);

    let positions = height * width;
    let positions_per_tile = (2_000_000 / contributions.max(1)).max(1).min(positions);
    let mut block = vec![0.0f32; positions_per_tile * contributions];
    let mut input_block = vec![0.0f32; positions_per_tile * in_channels];
    for index in 0..batch {
        let mut start = 0;
        while start < positions {
            let end = (start + positions_per_tile).min(positions);
            let rows = end - start;
            // Gather the channel planes into a position-major block.
            let input = &mut input_block[..rows * in_channels];
            for row in 0..rows {
                let position = start + row;
                for input_channel in 0..in_channels {
                    input[row * in_channels + input_channel] =
                        x.data[index * in_channels * height * width
                            + input_channel * height * width
                            + position];
                }
            }
            // `weight` is [Cin, Cout, kH, kW], i.e. already [Cin, Cout*kH*kW].
            let block = &mut block[..rows * contributions];
            crate::matmul::matmul(input, weight, block, rows, in_channels, contributions);
            for position in start..end {
                let row = position - start;
                let source_h = position / width;
                let source_w = position % width;
                for channel in 0..out_channels {
                    for tap_h in 0..kernel_h {
                        for tap_w in 0..kernel_w {
                            output.data[((index * out_channels + channel) * out_height
                                + source_h * stride_h
                                + tap_h)
                                * out_width
                                + source_w * stride_w
                                + tap_w] += block[row * contributions
                                + (channel * kernel_h + tap_h) * kernel_w
                                + tap_w];
                        }
                    }
                }
            }
            start = end;
        }
        for channel in 0..out_channels {
            let base = (index * out_channels + channel) * out_height * out_width;
            for position in 0..(out_height * out_width) {
                output.data[base + position] += bias[channel];
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_in_place_equals_exp_at_the_edges_of_its_range() {
        let mut values: Vec<f32> = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            87.9,
            -87.9,
            88.0,
            -88.0,
            88.8,
            -103.5,
            -104.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::MIN_POSITIVE,
            -1e-30,
        ];
        values.extend((0..997).map(|i| i as f32 * 0.21 - 105.0));
        let expected: Vec<u32> = values.iter().map(|value| exp(*value).to_bits()).collect();
        exp_in_place(&mut values);
        let got: Vec<u32> = values.iter().map(|value| value.to_bits()).collect();
        assert_eq!(got, expected);
    }

    /// Every f32 in the fast range against glibc `expf`. Run on the Lambda
    /// processor: `cargo test --release -- --ignored exp_fast_range`.
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    #[test]
    #[ignore]
    fn exp_fast_range_equals_glibc_for_every_f32() {
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let span = (1u64 << 32) / threads as u64;
        let failures: usize = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads as u64)
                .map(|t| {
                    scope.spawn(move || {
                        let mut failures = 0;
                        let mut bits = t * span;
                        let end = if t + 1 == threads as u64 {
                            1 << 32
                        } else {
                            (t + 1) * span
                        };
                        while bits < end {
                            let x: [f32; 4] =
                                std::array::from_fn(|i| f32::from_bits((bits + i as u64) as u32));
                            bits += 4;
                            if !x.iter().all(|v| glibc_exp::in_fast_range(*v)) {
                                continue;
                            }
                            // SAFETY: NEON is part of the aarch64 base architecture.
                            let got = unsafe { glibc_exp::four(x) };
                            for i in 0..4 {
                                if got[i].to_bits() != x[i].exp().to_bits() {
                                    failures += 1;
                                }
                            }
                        }
                        failures
                    })
                })
                .collect();
            workers.into_iter().map(|w| w.join().unwrap()).sum()
        });
        assert_eq!(failures, 0);
    }

    #[test]
    fn gelu_in_place_equals_gelu_one() {
        let mut values: Vec<f32> = (0..4001).map(|i| (i as f32 - 2000.0) * 0.0047).collect();
        values.extend([
            0.0,
            -0.0,
            1e-30,
            -1e-30,
            8.0,
            -8.0,
            1e30,
            f32::NAN,
            f32::INFINITY,
        ]);
        let expected: Vec<u32> = values.iter().map(|v| gelu_one(*v).to_bits()).collect();
        gelu_in_place(&mut values);
        let got: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
        assert_eq!(got, expected);
    }

    /// Every finite f32 against `libm::erff`:
    /// `cargo test --release -- --ignored erf4`.
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore]
    fn erf4_equals_libm_erff_for_every_finite_f32() {
        use core::arch::aarch64::*;
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let span = (1u64 << 32) / threads as u64;
        let failures: Vec<u32> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads as u64)
                .map(|t| {
                    scope.spawn(move || {
                        let mut failures = Vec::new();
                        let end = if t + 1 == threads as u64 {
                            1 << 32
                        } else {
                            (t + 1) * span
                        };
                        let mut bits = t * span;
                        while bits < end {
                            let x: [f32; 4] =
                                std::array::from_fn(|i| f32::from_bits((bits + i as u64) as u32));
                            bits += 4;
                            if x.iter().any(|v| !v.is_finite()) {
                                continue;
                            }
                            let mut got = [0.0f32; 4];
                            // SAFETY: NEON is part of the aarch64 base architecture.
                            unsafe {
                                vst1q_f32(got.as_mut_ptr(), neon_erf::erf4(vld1q_f32(x.as_ptr())))
                            };
                            for i in 0..4 {
                                if got[i].to_bits() != libm::erff(x[i]).to_bits()
                                    && failures.len() < 8
                                {
                                    failures.push(x[i].to_bits());
                                }
                            }
                        }
                        failures
                    })
                })
                .collect();
            workers
                .into_iter()
                .flat_map(|w| w.join().unwrap())
                .collect()
        });
        assert!(failures.is_empty(), "{failures:08x?}");
    }

    #[test]
    fn group_norm_matches_a_hand_computation() {
        // One group, two channels, two positions: normalise all four values.
        let x = Tensor::new(vec![1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let out = group_norm(&x, 1, &[1.0, 1.0], &[0.0, 0.0], 0.0);
        let mean = 2.5;
        let std = (1.25f32).sqrt();
        assert!((out.data[0] - (1.0 - mean) / std).abs() < 1e-6);
        assert!((out.data[3] - (4.0 - mean) / std).abs() < 1e-6);
    }

    #[test]
    fn conv2d_matches_a_hand_computation() {
        let x = Tensor::new(vec![1, 1, 3, 3], (1..=9).map(|v| v as f32).collect());
        let weight = vec![1.0, 1.0, 1.0, 1.0]; // [1, 1, 2, 2]
        let out = conv2d(&x, &weight, &[0.0], 1, (2, 2), (1, 1), (0, 0));
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        assert_eq!(out.data, vec![12.0, 16.0, 24.0, 28.0]);
    }

    #[test]
    fn conv_transpose2d_matches_a_hand_computation() {
        let x = Tensor::new(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let weight = vec![1.0, 0.0, 0.0, 1.0]; // [1, 1, 2, 2]
        let out = conv_transpose2d(&x, &weight, &[0.0], 1, (2, 2), (1, 1));
        assert_eq!(out.shape, vec![1, 1, 3, 3]);
        assert_eq!(out.data, vec![1.0, 2.0, 0.0, 3.0, 5.0, 2.0, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn glu_splits_the_channel_axis() {
        let x = Tensor::new(vec![1, 4, 1], vec![1.0, 2.0, 0.0, 0.0]);
        let out = glu(&x);
        // sigmoid(0) = 0.5
        assert_eq!(out.shape, vec![1, 2, 1]);
        assert!((out.data[0] - 0.5).abs() < 1e-6);
        assert!((out.data[1] - 1.0).abs() < 1e-6);
    }
}
