//! Checks the cross-transformer against the reference.
//!
//! Needs the PyTorch export and reference dump (see `tests/encoder.rs`).
//!   cargo test --release -p stemsplits-htdemucs -- --ignored --nocapture

use std::path::{Path, PathBuf};

use stemsplits_htdemucs::tensor::Tensor;
use stemsplits_htdemucs::transformer::CrossTransformer;
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
fn cross_transformer_matches_the_reference() {
    let weights = Weights::open(&root().join("tools/reference/out/bundle")).expect("bundle");
    let transformer =
        CrossTransformer::load(&weights, "crosstransformer", 8, 10_000.0, 1.0).unwrap();

    let x = load_reference("input_crosstransformer_x");
    let xt = load_reference("input_crosstransformer_t");
    let (out_x, out_t) = transformer.forward(&x, &xt);

    compare(
        &out_x,
        &load_reference("crosstransformer_x"),
        "crosstransformer.x",
    );
    compare(
        &out_t,
        &load_reference("crosstransformer_t"),
        "crosstransformer.t",
    );
}
