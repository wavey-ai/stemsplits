//! Times one HTDemucs segment in the Rust forward pass.
//!
//!   cargo run --release -p stemsplits-htdemucs --bin bench [bundle-dir] [runs]
//!
//! The input is the same deterministic segment the tests use, so runs compare
//! like for like. Reports the real-time factor: wall seconds per audio second.

use std::path::Path;
use std::time::Instant;

use stemsplits_htdemucs::model::HtDemucs;
use stemsplits_htdemucs::tensor::Tensor;
use stemsplits_model::Weights;
use stemsplits_stft::{Geometry, Stft};

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

fn main() {
    let bundle = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tools/reference/out/bundle".into());
    let runs: usize = std::env::args()
        .nth(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(3);

    let weights = Weights::open(Path::new(&bundle)).expect("bundle");
    println!(
        "parameters: {}, threads: {}",
        weights.parameter_count(),
        stemsplits_htdemucs::matmul::thread_count()
    );
    let model = HtDemucs::load(&weights).expect("model");

    let geometry = Geometry::CONTRACT;
    let mut stft = Stft::new(geometry);
    let left = input_signal(geometry.segment, 0x1234_5678);
    let right = input_signal(geometry.segment, 0x9ABC_DEF0);
    // CaC magnitude: complex-as-channels, so `2 * channels` planes.
    let magnitude = Tensor::new(
        vec![1, 2 * geometry.channels, geometry.bins, geometry.frames],
        stft.spectral_input(&left, &right),
    );
    let mut interleaved = Vec::with_capacity(2 * geometry.segment);
    interleaved.extend_from_slice(&left);
    interleaved.extend_from_slice(&right);
    let waveform = Tensor::new(vec![1, geometry.channels, geometry.segment], interleaved);

    let audio = geometry.segment as f64 / geometry.sample_rate as f64;

    // First pass, timed on its own so a slow run still reports.
    let start = Instant::now();
    let _ = model.forward(&magnitude, &waveform);
    let first = start.elapsed().as_secs_f64();
    println!(
        "first forward: {first:.3} s/segment, audio {audio:.3} s, RTF {:.3}",
        first / audio
    );

    if runs == 0 {
        return;
    }
    let start = Instant::now();
    for _ in 0..runs {
        let _ = model.forward(&magnitude, &waveform);
    }
    let seconds = start.elapsed().as_secs_f64() / runs as f64;
    println!(
        "steady: {seconds:.3} s/segment, audio {audio:.3} s, RTF {:.3}",
        seconds / audio
    );
}
