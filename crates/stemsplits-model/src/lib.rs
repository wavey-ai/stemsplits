//! The HTDemucs weight bundle: what `tools/reference/export.py` writes and
//! what the Rust runtime reads.
//!
//! A bundle is a JSON manifest plus one flat little-endian weight blob. There
//! is no ONNX and no PyTorch in the shipping path; PyTorch is only the oracle
//! that produced this bundle. The manifest names every tensor, so the runtime
//! can address weights by the same names the reference uses
//! (`encoder.0.conv.weight`), which keeps a layer port honest.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// One tensor's place in the weight blob.
#[derive(Clone, Debug, Deserialize)]
pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub offset: usize,
    pub count: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct WeightsMeta {
    pub file: String,
    pub dtype: String,
    pub byte_length: usize,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Bundle {
    pub format: String,
    pub version: u32,
    pub model: String,
    pub checkpoint: String,
    pub config: serde_json::Value,
    pub weights: WeightsMeta,
    pub tensors: Vec<TensorMeta>,
}

impl Bundle {
    pub fn from_json(text: &str) -> Result<Self> {
        let bundle: Bundle = serde_json::from_str(text).context("bundle.json is not valid")?;
        if bundle.format != "stemsplits-bundle" {
            bail!("not a stemsplits bundle: {}", bundle.format);
        }
        if bundle.version != 1 {
            bail!("unsupported bundle version {}", bundle.version);
        }
        if bundle.weights.dtype != "f32" {
            bail!("unsupported weight dtype {}", bundle.weights.dtype);
        }
        Ok(bundle)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorMeta> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }
}

/// A bundle's manifest and its weights, resident.
pub struct Weights {
    bundle: Bundle,
    directory: PathBuf,
    bytes: Vec<u8>,
    index: HashMap<String, usize>,
}

impl Weights {
    /// Reads and verifies a bundle directory. Verification is not optional:
    /// a bundle that does not match its own digest is a porting fault, not a
    /// runtime condition.
    pub fn open(directory: &Path) -> Result<Self> {
        let manifest = std::fs::read_to_string(directory.join("bundle.json"))
            .with_context(|| format!("read {}", directory.join("bundle.json").display()))?;
        let bundle = Bundle::from_json(&manifest)?;
        let path = directory.join(&bundle.weights.file);
        let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        if bytes.len() != bundle.weights.byte_length {
            bail!(
                "weight blob is {} bytes, manifest says {}",
                bytes.len(),
                bundle.weights.byte_length
            );
        }
        let digest = Sha256::digest(&bytes);
        let digest = digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        if digest != bundle.weights.sha256 {
            bail!("weight blob sha256 {digest} does not match the manifest");
        }
        let index = bundle
            .tensors
            .iter()
            .enumerate()
            .map(|(position, tensor)| (tensor.name.clone(), position))
            .collect();
        Ok(Self {
            bundle,
            directory: directory.to_path_buf(),
            bytes,
            index,
        })
    }

    pub fn bundle(&self) -> &Bundle {
        &self.bundle
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// A tensor's weights by the reference's own name.
    pub fn tensor(&self, name: &str) -> Option<Tensor<'_>> {
        let position = *self.index.get(name)?;
        let meta = &self.bundle.tensors[position];
        let start = meta.offset;
        let end = start + meta.count * 4;
        let data = self.bytes.get(start..end)?;
        Some(Tensor {
            shape: meta.shape.clone(),
            data: bytemuck_f32(data),
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.bundle.tensors.iter().map(|tensor| tensor.count).sum()
    }
}

/// A borrowed view of one tensor's f32 weights.
pub struct Tensor<'a> {
    shape: Vec<usize>,
    data: &'a [f32],
}

impl<'a> Tensor<'a> {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &'a [f32] {
        self.data
    }
}

/// The blob is little-endian f32; on the supported targets that is the native
/// layout, so the bytes reinterpret without a copy. A big-endian target would
/// need a conversion here.
fn bytemuck_f32(bytes: &[u8]) -> &[f32] {
    assert_eq!(bytes.len() % 4, 0, "weight bytes are f32");
    let pointer = bytes.as_ptr() as *const f32;
    // SAFETY: `bytes` is aligned to 4 (it is a subslice of a Vec<u8> starting
    // at an f32-aligned offset) and the length is a multiple of 4. The bundle
    // is generated locally, not from untrusted input.
    unsafe { std::slice::from_raw_parts(pointer, bytes.len() / 4) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_bundle(directory: &Path, name: &str, shape: Vec<usize>, values: &[f32]) -> String {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let digest = Sha256::digest(&bytes);
        let digest = digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        std::fs::write(directory.join("weights.f32"), &bytes).unwrap();
        let manifest = serde_json::json!({
            "format": "stemsplits-bundle",
            "version": 1,
            "model": "htdemucs",
            "checkpoint": "test",
            "config": {},
            "weights": {
                "file": "weights.f32",
                "dtype": "f32",
                "byte_length": bytes.len(),
                "sha256": digest,
            },
            "tensors": [{
                "name": name,
                "shape": shape,
                "dtype": "f32",
                "offset": 0,
                "count": values.len(),
            }],
        });
        let text = serde_json::to_string(&manifest).unwrap();
        std::fs::write(directory.join("bundle.json"), &text).unwrap();
        text
    }

    #[test]
    fn loads_a_tensor_by_name() {
        let directory = tempfile::tempdir().unwrap();
        write_bundle(
            directory.path(),
            "encoder.0.conv.weight",
            vec![2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        let weights = Weights::open(directory.path()).unwrap();
        let tensor = weights.tensor("encoder.0.conv.weight").unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(weights.tensor("missing").is_none());
        assert_eq!(weights.parameter_count(), 6);
    }

    /// Requires `tools/reference/export.py` to have run. Ignored by default
    /// because the blob is 168 MB and gitignored.
    #[test]
    #[ignore = "needs tools/reference/export.py output"]
    fn opens_the_exported_bundle() {
        let directory =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/reference/out/bundle");
        let weights = Weights::open(&directory).expect("run tools/reference/export.py");
        assert_eq!(weights.parameter_count(), 41_984_456);
        assert_eq!(weights.bundle().model, "htdemucs");
        let conv = weights
            .tensor("encoder.0.conv.weight")
            .expect("encoder.0.conv.weight");
        assert_eq!(conv.shape(), &[48, 4, 8, 1]);
    }

    #[test]
    fn rejects_a_corrupted_blob() {
        let directory = tempfile::tempdir().unwrap();
        write_bundle(directory.path(), "x", vec![1], &[1.0]);
        let mut bytes = std::fs::read(directory.path().join("weights.f32")).unwrap();
        bytes[0] ^= 0xff;
        std::fs::write(directory.path().join("weights.f32"), &bytes).unwrap();
        assert!(Weights::open(directory.path()).is_err());
    }
}
