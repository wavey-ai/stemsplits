//! Checks the encoder stack against the reference activations.
//!
//! Needs the PyTorch export and the reference dump:
//!   tools/reference/.venv/bin/python tools/reference/export.py
//!   tools/reference/.venv/bin/python tools/reference/reference_stems.py
//! so it is ignored by default (the bundle is 168 MB and gitignored).
//!
//!   cargo test -p stemsplits-htdemucs -- --ignored --nocapture

use std::path::{Path, PathBuf};

use stemsplits_htdemucs::encoder::{EncLayer, FrequencyEmbedding};
use stemsplits_htdemucs::tensor::Tensor;
use stemsplits_model::Weights;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn load_reference(name: &str) -> Tensor {
    let directory = root().join("tools/reference/out/reference");
    let index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(directory.join("index.json")).unwrap())
            .unwrap();
    let shape: Vec<usize> = index[name]["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap() as usize)
        .collect();
    let bytes = std::fs::read(directory.join(format!("{name}.f32"))).unwrap();
    let data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Tensor::new(shape, data)
}

fn compare(actual: &Tensor, expected: &Tensor, label: &str) {
    assert_eq!(actual.shape, expected.shape, "{label}: shape");
    let mut max_difference = 0.0f32;
    let mut max_expected = 0.0f32;
    for (a, b) in actual.data.iter().zip(&expected.data) {
        max_difference = max_difference.max((a - b).abs());
        max_expected = max_expected.max(b.abs());
    }
    let relative = max_difference / max_expected.max(1e-9);
    println!(
        "{label}: shape {:?} max abs diff {max_difference:.3e} (relative {relative:.3e})",
        actual.shape
    );
    assert!(
        relative < 1e-3,
        "{label}: relative error {relative} too large"
    );
}

#[test]
#[ignore = "needs tools/reference export and dump"]
fn encoder_0_matches_the_reference() {
    let weights = Weights::open(&root().join("tools/reference/out/bundle")).expect("bundle");
    let input = load_reference("input_encoder_0");
    let expected = load_reference("encoder_0");

    let layer = EncLayer::load(&weights, "encoder.0", true, 48, 8, 2).unwrap();
    let output = layer.forward(&input);
    compare(&output, &expected, "encoder.0");
}

#[test]
#[ignore = "needs tools/reference export and dump"]
fn frequency_encoder_stack_matches_the_reference() {
    let weights = Weights::open(&root().join("tools/reference/out/bundle")).expect("bundle");
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
    let weights = Weights::open(&root().join("tools/reference/out/bundle")).expect("bundle");
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
    let weights = Weights::open(&root().join("tools/reference/out/bundle")).expect("bundle");
    let input = load_reference("input_tencoder_0");
    let expected = load_reference("tencoder_0");
    let layer = EncLayer::load(&weights, "tencoder.0", false, 48, 8, 2).unwrap();
    let output = layer.forward(&input);
    compare(&output, &expected, "tencoder.0");
}
