//! Checks the cross-transformer against the reference.
//!
//! Needs the PyTorch export and reference dump (see `tests/encoder.rs`).
//!   cargo test --release -p stemsplits-htdemucs -- --ignored --nocapture

mod common;
use common::{compare, load_bundle, load_reference};

use stemsplits_htdemucs::transformer::CrossTransformer;

#[test]
#[ignore = "needs tools/reference export and dump"]
fn cross_transformer_matches_the_reference() {
    let weights = load_bundle();
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
