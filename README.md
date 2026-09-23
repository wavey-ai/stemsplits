# stemsplits

Four-stem HTDemucs separation that can run on a phone or fan out across a
cloud, producing the same stems either way.

Wavey already separates tracks on device: `StemSeparator.swift` runs the
MIT-licensed Core ML HTDemucs package, cutting a track into 7.8-second
segments with 25% overlap and sewing them back with triangular overlap-add.
This repository is the same work in Rust, so a server can run it too — and so
the seam is defined once instead of once per backend.

## Why it exists

The ECDC service already proves the shape: one independent
segment per Lambda invocation, fan out, concatenate. EnCodec's segments are
fixed-context, so they are byte-exact and order-free. HTDemucs is not: its
segment is the model's own inference unit, and the output depends on the
overlap plan. That makes the plan part of the format. It lives here, pinned,
so the phone and the cloud cannot disagree on a seam.

## No ONNX at runtime

`encodec-rs` removed ONNX from the shipping path: it extracts weights and
drives its own C SIMD kernels, with a portable Rust reference beside them and
FP contraction pinned off so the backends agree. This repository follows that
rule. PyTorch/ONNX are **oracles for parity tests**, not runtime dependencies.
The plan is portable Rust first, then kernels for the hot ops.

## Layout

```
crates/
  stemsplits-stft/    the STFT/iSTFT contract, ported from StemSeparator.swift
  stemsplits-demucs/  StemKind, the chunk plan, and the overlap-add seam
  stemsplits-model/   the weight bundle format and loader
tools/
  oracle-stft/        the Swift/vDSP oracle and its golden vectors
  reference/          the PyTorch reference, the pinned config, and the export
bench/                real-time-factor and memory benchmarks
```

## The model

The on-device model is `htdemucs` (Demucs v4, checkpoint `955717e8`), the
MIT-licensed 4-stem hybrid transformer: a 48-channel convolutional encoder and
decoder over both a spectrogram branch (complex-as-channels) and a waveform
branch, joined by a 5-layer cross-domain transformer at 512 channels. The
Core ML package Wavey downloads is that same graph stored in **f16**; the
PyTorch checkpoint is the structured source of the weights and the oracle for
activations. The config is pinned in `tools/reference/demucs_config.json`.

## Status

Done:

- `stemsplits-stft` matches the on-device Swift/vDSP transform to within f32
  rounding, proven against a golden vector (`cargo test`).
- The Demucs seam (chunk plan + triangular overlap-add) is pinned and tested,
  including that streaming flush equals batch flush.
- The weight bundle is defined, exported from PyTorch (533 tensors, 41.98 M
  parameters), and loadable in Rust by the reference's own tensor names.
- The reference harness runs HTDemucs on a fixed segment and dumps the stems
  and encoder activations to check a layer port against.
- The Rust encoder stack (both branches) and the cross-transformer match the
  reference to ~2e-6.

Next:

- The decoder, then the full segment in Rust, checked against the reference
  activations and stems.
- RTF measurement on arm64 (Graviton) to decide whether cloud stems beat the
  phone.
- A segment API and fan-out, mirroring `bench/ecdc/aws`.

## Test

```sh
cargo test
```

Regenerate the Swift oracle golden (macOS only) after changing the contract:

```sh
swift tools/oracle-stft/main.swift tools/oracle-stft/out
cp tools/oracle-stft/out/stft-small.bin crates/stemsplits-stft/tests/golden/
```
