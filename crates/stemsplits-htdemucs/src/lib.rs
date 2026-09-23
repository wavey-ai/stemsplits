//! A portable Rust forward pass of HTDemucs.
//!
//! Written against the reference in `demucs/htdemucs.py` and checked, layer
//! by layer, against activations dumped by `tools/reference/reference_stems.py`.
//! Scalar and obvious first; SIMD kernels come later for the hot ops.

pub mod dconv;
pub mod decoder;
pub mod encoder;
pub mod matmul;
pub mod model;
pub mod ops;
pub mod separate;
pub mod tensor;
pub mod transformer;
pub mod wav;
