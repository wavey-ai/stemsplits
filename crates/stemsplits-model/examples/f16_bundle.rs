//! Writes an f16 bundle from an f32 bundle.
//!
//!   cargo run --release -p stemsplits-model --example f16_bundle -- F32_DIR F16_DIR
//!
//! The f16 blob holds the same tensors in the same order, rounded to the
//! nearest half float. Each tensor's offset is half its f32 offset. The
//! example prints the largest magnitude and the relative error of the
//! rounding over all weights.
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (from, to) = (Path::new(&args[0]), Path::new(&args[1]));
    let mut manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(from.join("bundle.json"))?)?;
    anyhow::ensure!(
        manifest["weights"]["dtype"] == "f32",
        "the input bundle is not f32"
    );
    let blob = std::fs::read(from.join(manifest["weights"]["file"].as_str().unwrap()))?;
    anyhow::ensure!(
        stemsplits_model::sha256_hex(&blob) == manifest["weights"]["sha256"].as_str().unwrap(),
        "the input blob does not match its manifest"
    );
    let (mut largest, mut error, mut power) = (0.0f64, 0.0f64, 0.0f64);
    let mut out = Vec::with_capacity(blob.len() / 2);
    for chunk in blob.chunks_exact(4) {
        let value = f32::from_le_bytes(chunk.try_into().unwrap());
        let half = half::f16::from_f32(value);
        anyhow::ensure!(half.is_finite(), "{value} is outside the f16 range");
        largest = largest.max(value.abs() as f64);
        error += (half.to_f32() as f64 - value as f64).powi(2);
        power += (value as f64).powi(2);
        out.extend_from_slice(&half.to_le_bytes());
    }
    std::fs::create_dir_all(to)?;
    std::fs::write(to.join("weights.f16"), &out)?;
    manifest["weights"]["file"] = "weights.f16".into();
    manifest["weights"]["dtype"] = "f16".into();
    manifest["weights"]["byte_length"] = out.len().into();
    manifest["weights"]["sha256"] = stemsplits_model::sha256_hex(&out).into();
    for tensor in manifest["tensors"].as_array_mut().unwrap() {
        let offset = tensor["offset"].as_u64().unwrap();
        anyhow::ensure!(offset % 4 == 0, "an f32 offset is not a whole value");
        tensor["offset"] = (offset / 2).into();
        tensor["dtype"] = "f16".into();
    }
    std::fs::write(
        to.join("bundle.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    println!(
        "{} weights, {} bytes, largest |w| {largest:.3}, rounding error {:.1} dB",
        out.len() / 2,
        out.len(),
        10.0 * (error / power).log10()
    );
    Ok(())
}
