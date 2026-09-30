//! HTDemucs in the browser: the pure-Rust forward pass behind wasm-bindgen.
//!
//! The page loads the bundle manifest and weight blob, then separates one
//! 7.8-second segment per call. The page cuts the segments and sews them with
//! the pinned `ChunkPlan` and seam, as the cloud does.
use stemsplits_htdemucs::{model::HtDemucs, separate::separate_segment};
use stemsplits_model::Weights;
use stemsplits_stft::{Geometry, Stft};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Separator {
    model: HtDemucs,
    stft: Stft,
}

#[wasm_bindgen]
impl Separator {
    /// Verifies the bundle against `digest`, the weight blob's SHA-256 in
    /// hex, and builds the model.
    #[wasm_bindgen(constructor)]
    pub fn new(manifest: &str, weights: Vec<u8>, digest: &str) -> Result<Separator, JsError> {
        let weights = Weights::from_bytes(manifest, weights, digest)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let model = HtDemucs::load(&weights).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Separator {
            model,
            stft: Stft::new(Geometry::CONTRACT),
        })
    }

    /// Samples in one segment.
    pub fn segment() -> usize {
        Geometry::CONTRACT.segment
    }

    /// Separates one segment. The result is drums, bass, other and vocals,
    /// each as left then right, one after the other.
    pub fn separate(&mut self, left: &[f32], right: &[f32]) -> Vec<f32> {
        separate_segment(&self.model, left, right, &mut self.stft)
            .into_iter()
            .flat_map(|[l, r]| l.into_iter().chain(r))
            .collect()
    }
}

/// Bytes of WASM memory in use, for sizing the worker pool.
#[wasm_bindgen]
pub fn memory_bytes() -> usize {
    #[cfg(target_arch = "wasm32")]
    return core::arch::wasm32::memory_size(0) * 65536;
    #[cfg(not(target_arch = "wasm32"))]
    0
}
