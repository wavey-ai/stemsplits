//! Checks the encoder stack against the reference activations.
//!
//! Needs the PyTorch export and the reference dump:
//!   tools/reference/.venv/bin/python tools/reference/export.py
//!   tools/reference/.venv/bin/python tools/reference/reference_stems.py
//! so it is ignored by default (the bundle is 168 MB and gitignored).
//!
//!   cargo test -p stemsplits-htdemucs -- --ignored --nocapture

mod common;
use common::{compare, load_bundle, load_reference};

use stemsplits_htdemucs::encoder::{EncLayer, FrequencyEmbedding};

#[test]
#[ignore = "needs tools/reference export and dump"]
fn encoder_0_matches_the_reference() {
    let weights = load_bundle();
    let input = load_reference("input_encoder_0");
    let expected = load_reference("encoder_0");

    let layer = EncLayer::load(&weights, "encoder.0", true, 48, 8, 2).unwrap();
    let output = layer.forward(&input);
    compare(&output, &expected, "encoder.0");
}

#[test]
#[ignore = "needs tools/reference export and dump"]
fn frequency_encoder_stack_matches_the_reference() {
    let weights = load_bundle();
    let embedding = FrequencyEmbedding::load(&weights, 10.0).unwrap();
    let channels = [48usize, 96, 192, 384];

    let mut x = load_reference("input_encoder_0");
    for (index, chout) in channels.into_iter().enumerate() {
        let layer =
            EncLayer::load(&weights, &format!("encoder.{index}"), true, chout, 8, 2).unwrap();
        x = layer.forward(&x);
        if index == 0 {
            x = embedding.apply(&x, 0.2);
            // saved[0], the input to the next layer.
            compare(
                &x,
                &load_reference("input_encoder_1"),
                "encoder.0 + freq_emb",
            );
        }
    }
    compare(&x, &load_reference("encoder_3"), "encoder.3");
}

#[test]
#[ignore = "needs tools/reference export and dump"]
fn time_encoder_stack_matches_the_reference() {
    let weights = load_bundle();
    let channels = [48usize, 96, 192, 384];
    let mut x = load_reference("input_tencoder_0");
    for (index, chout) in channels.into_iter().enumerate() {
        let layer =
            EncLayer::load(&weights, &format!("tencoder.{index}"), false, chout, 8, 2).unwrap();
        x = layer.forward(&x);
    }
    compare(&x, &load_reference("tencoder_3"), "tencoder.3");
}

#[test]
#[ignore = "needs tools/reference export and dump"]
fn time_encoder_0_matches_the_reference() {
    let weights = load_bundle();
    let input = load_reference("input_tencoder_0");
    let expected = load_reference("tencoder_0");
    let layer = EncLayer::load(&weights, "tencoder.0", false, 48, 8, 2).unwrap();
    let output = layer.forward(&input);
    compare(&output, &expected, "tencoder.0");
}
