//! Separates one seeded segment natively and writes the output as f32, for a
//! parity check against the WASM build.
//!
//!   cargo run --release -p stemsplits-wasm --example segment -- BUNDLE_DIR OUT.f32
use std::path::Path;
use stemsplits_htdemucs::{model::HtDemucs, separate::separate_segment};
use stemsplits_model::Weights;
use stemsplits_stft::{Geometry, Stft};

fn signal(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let weights = Weights::open(Path::new(&args[0])).expect("bundle");
    let model = HtDemucs::load(&weights).expect("model");
    let geometry = Geometry::CONTRACT;
    let mut stft = Stft::new(geometry);
    let (left, right) = (
        signal(geometry.segment, 0x1234_5678),
        signal(geometry.segment, 0x9ABC_DEF0),
    );
    let started = std::time::Instant::now();
    let out: Vec<f32> = separate_segment(&model, &left, &right, &mut stft)
        .into_iter()
        .flat_map(|[l, r]| l.into_iter().chain(r))
        .collect();
    eprintln!("native segment {:.2} s", started.elapsed().as_secs_f64());
    std::fs::write(
        &args[1],
        out.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
    .unwrap();
}
