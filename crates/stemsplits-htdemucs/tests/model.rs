//! Checks the decoder and the full forward pass against the reference.
//!
//!   tools/reference/.venv/bin/python tools/reference/export.py
//!   tools/reference/.venv/bin/python tools/reference/reference_stems.py
//!   cargo test --release -p stemsplits-htdemucs -- --ignored --nocapture

mod common;
use common::{compare, load_bundle, load_reference};

use stemsplits_htdemucs::decoder::DecLayer;
use stemsplits_htdemucs::model::HtDemucs;
use stemsplits_htdemucs::tensor::Tensor;
use stemsplits_stft::{Geometry, Spectrum, Stft};

#[test]
#[ignore = "needs tools/reference export and dump"]
fn decoder_0_matches_the_reference() {
    let weights = load_bundle();
    let layer = DecLayer::load(&weights, "decoder.0", true, false, 384, 8, 2).unwrap();
    let output = layer.forward(
        &load_reference("input_decoder_0_x"),
        &load_reference("input_decoder_0_skip"),
        0,
    );
    compare(&output, &load_reference("decoder_0_z"), "decoder.0");
}

#[test]
#[ignore = "needs tools/reference export and dump"]
fn time_decoder_0_matches_the_reference() {
    let weights = load_bundle();
    let layer = DecLayer::load(&weights, "tdecoder.0", false, false, 384, 8, 2).unwrap();
    let output = layer.forward(
        &load_reference("input_tdecoder_0_x"),
        &load_reference("input_tdecoder_0_skip"),
        5_375,
    );
    compare(&output, &load_reference("tdecoder_0_z"), "tdecoder.0");
}

#[test]
#[ignore = "needs tools/reference export and dump"]
fn full_model_matches_the_reference_stems() {
    let weights = load_bundle();
    let model = HtDemucs::load(&weights).unwrap();

    let magnitude = load_reference("mag");
    let waveform = load_reference("input");
    let (frequency, time) = model.forward(&magnitude, &waveform);
    assert_eq!(frequency.shape, vec![1, 16, 2048, 336]);
    assert_eq!(time.shape, vec![1, 8, 343_980]);

    let stems = reconstruct_stems(&frequency, &time);
    compare(&stems, &load_reference("stems"), "stems");
}

/// `time + iSTFT(freq) * sqrt(fft_size)`, per stem and channel. The frequency
/// branch is `[16, Fr, T]` with channel `stem * 4 + channel * 2 + component`.
fn reconstruct_stems(frequency: &Tensor, time: &Tensor) -> Tensor {
    let geometry = Geometry::CONTRACT;
    let mut stft = Stft::new(geometry);
    let bins = geometry.bins;
    let frames = geometry.frames;
    let segment = geometry.segment;
    let scale = (geometry.fft_size as f32).sqrt();
    let mut stems = Tensor::zeros(vec![1, 4, 2, segment]);

    for stem in 0..4 {
        for channel in 0..2 {
            let real_channel = stem * 4 + channel * 2;
            let real = frequency.data
                [real_channel * bins * frames..(real_channel + 1) * bins * frames]
                .to_vec();
            let imaginary = frequency.data
                [(real_channel + 1) * bins * frames..(real_channel + 2) * bins * frames]
                .to_vec();
            let inverse = stft.inverse(&Spectrum { real, imaginary });
            let time_base = (stem * 2 + channel) * segment;
            let stem_base = (stem * 2 + channel) * segment;
            for (index, value) in inverse.iter().enumerate() {
                stems.data[stem_base + index] = time.data[time_base + index] + value * scale;
            }
        }
    }
    stems
}
