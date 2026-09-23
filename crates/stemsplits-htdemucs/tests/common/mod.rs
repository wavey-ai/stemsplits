//! Shared helpers for the reference activation tests.

use std::path::{Path, PathBuf};

use stemsplits_htdemucs::tensor::Tensor;

pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn load_reference(name: &str) -> Tensor {
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

pub fn compare(actual: &Tensor, expected: &Tensor, label: &str) {
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

pub fn load_bundle() -> stemsplits_model::Weights {
    stemsplits_model::Weights::open(&root().join("tools/reference/out/bundle")).expect("bundle")
}
