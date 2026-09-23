//! Exercises the shipped contract geometry end to end. The reconstruction is
//! lossy by design — Demucs drops the Nyquist bin — so the assertion is the
//! signal-to-noise ratio the Swift oracle reports for the same input, not
//! equality.

use stemsplits_stft::{Geometry, Stft};

/// The deterministic input the Swift oracle generates, so both sides see the
/// same samples.
fn input_signal(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 11) as f64 / (1u64 << 53) as f64) as f32;
            unit * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn contract_forward_then_inverse_reconstructs() {
    let geometry = Geometry::CONTRACT;
    let left = input_signal(geometry.segment, 0x1234_5678);

    let mut stft = Stft::new(geometry);
    let spectrum = stft.forward(&left);
    let inverse = stft.inverse(&spectrum);

    assert_eq!(inverse.len(), geometry.segment);
    assert!(
        inverse.iter().all(|value| value.is_finite()),
        "finite output"
    );

    let mut signal_energy = 0.0f64;
    let mut error_energy = 0.0f64;
    for (expected, actual) in left.iter().zip(&inverse) {
        signal_energy += (*expected as f64).powi(2);
        error_energy += (*actual as f64 - *expected as f64).powi(2);
    }
    let snr = 10.0 * (signal_energy / error_energy.max(f64::MIN_POSITIVE)).log10();
    // The Swift oracle measures 33.58 dB for this input.
    assert!(snr > 30.0, "reconstruction SNR {snr:.2} dB is too low");
}
