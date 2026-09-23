//! Proves the Rust transform matches the on-device Swift/vDSP contract.
//!
//! The golden was written by `tools/oracle-stft/main.swift`, which is a copy
//! of `StemSeparator.swift`. Regenerate it there when the contract changes.

use std::fs;
use std::path::PathBuf;

use stemsplits_stft::{Geometry, Spectrum, Stft};

fn golden() -> Vec<f32> {
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "golden",
        "stft-small.bin",
    ]
    .iter()
    .collect();
    let bytes = fs::read(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "golden is f32 little-endian");
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn max_abs_diff(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len(), "length");
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
    let difference = max_abs_diff(actual, expected);
    assert!(
        difference <= tolerance,
        "{label}: max abs diff {difference} exceeds {tolerance}"
    );
}

#[test]
fn forward_spectral_input_and_inverse_match_swift() {
    let geometry = Geometry::SMALL;
    let data = golden();

    let mut cursor = 0;
    let left = data[cursor..(cursor + geometry.segment)].to_vec();
    cursor += geometry.segment;
    let right = data[cursor..(cursor + geometry.segment)].to_vec();
    cursor += geometry.segment;
    let plane = geometry.bins * geometry.frames;
    let golden_real = data[cursor..(cursor + plane)].to_vec();
    cursor += plane;
    let golden_imaginary = data[cursor..(cursor + plane)].to_vec();
    cursor += plane;
    let golden_planes = data[cursor..(cursor + 4 * plane)].to_vec();
    cursor += 4 * plane;
    let golden_inverse = data[cursor..(cursor + geometry.segment)].to_vec();
    cursor += geometry.segment;
    assert_eq!(cursor, data.len(), "golden layout");

    let mut stft = Stft::new(geometry);

    let spectrum = stft.forward(&left);
    assert_close(&spectrum.real, &golden_real, 1e-4, "forward real");
    assert_close(
        &spectrum.imaginary,
        &golden_imaginary,
        1e-4,
        "forward imaginary",
    );

    let planes = stft.spectral_input(&left, &right);
    assert_close(&planes, &golden_planes, 1e-4, "spectral input");

    // Feed the golden spectrum back so the inverse is tested on its own,
    // independent of any forward rounding difference.
    let golden_spectrum = Spectrum {
        real: golden_real,
        imaginary: golden_imaginary,
    };
    let inverse = stft.inverse(&golden_spectrum);
    assert_close(&inverse, &golden_inverse, 1e-4, "inverse");
}
