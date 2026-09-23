# stemsplits

Four-stem HTDemucs separation in pure Rust — the same work Wavey does on
device, ported so a server can run it too, with the seam defined once instead
of once per backend.

Wavey separates tracks with the MIT-licensed Core ML HTDemucs package,
cutting a track into 7.8-second segments with 25% overlap and sewing them back
with triangular overlap-add. This repository is that same pipeline in Rust:
the STFT contract, the chunk plan, the overlap-add seam, the model, and its
kernels. The shipping path is pure Rust, carrying its own kernels.

## Why it exists

The ECDC service already proves the shape: one independent
segment per Lambda invocation, fan out, concatenate. EnCodec's segments are
fixed-context, so they are byte-exact and order-free. HTDemucs differs: its
segment is the model's own inference unit, and the output depends on the
overlap plan. That makes the plan part of the format. It lives here, pinned,
so the phone and the cloud share one seam definition.

## A pure-Rust runtime

[`encodec-rs`](https://github.com/wavey-ai/encodec-rs) removed ONNX from the
shipping path: it extracts weights and drives its own C SIMD kernels, with a
portable Rust reference beside them. This repository follows that rule.
PyTorch and ONNX serve as **oracles for parity tests**. The port was written
scalar and obvious first, then the hot ops became kernels. One 7.8 s segment runs at **RTF 1.61** single-threaded — faster than
ONNX Runtime on one thread (2.61) and near its eight-thread run (1.13); the
tables and method are in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Layout

```
crates/
  stemsplits-stft/     the STFT/iSTFT contract, ported from StemSeparator.swift
  stemsplits-demucs/   StemKind, the chunk plan, and the overlap-add seam
  stemsplits-model/    the weight bundle format and loader
  stemsplits-htdemucs/ the forward pass and its C GEMM kernel
tools/
  oracle-stft/         the Swift/vDSP oracle and its golden vectors
  reference/           the PyTorch reference, pinned config, export, ONNX baseline
bench/                 real-time-factor and memory benchmarks
docs/                  the decision log and the benchmarks
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

- `stemsplits-stft` matches the on-device Swift/vDSP transform to within f32
  rounding, proven against a golden vector.
- The chunk plan and triangular overlap-add seam are pinned and tested,
  including that streaming flush equals batch flush.
- The weight bundle is defined, exported from PyTorch (533 tensors, 41.98 M
  parameters), and loadable in Rust by the reference's own tensor names.
- The full forward pass matches the reference stems to 3e-6, layer by layer
  against dumped activations.
- The single-threaded kernel pass reaches RTF 1.61.

Next: the arm64/Graviton measurement that decides cloud versus phone, f16
weights, a segment API and fan-out mirroring `bench/ecdc/aws`, and threading
once a multi-vCPU function is chosen (a 1769 MB Lambda provides one vCPU).

## Separate a track

```sh
cargo run --release -p stemsplits-htdemucs --bin separate -- <in.wav> <out-dir>
```

Input is 44.1 kHz stereo (prepare with `ffmpeg -ar 44100 -ac 2 -c:a pcm_s16le`);
output is four stem WAVs. It runs the whole track through the pinned chunk
plan and overlap-add seam. Sample output is described in `samples/README.md`.

## Build and test

Requires a Rust toolchain and a C compiler (for the GEMM kernel).

```sh
cargo test
```

The activation and stem tests need the PyTorch export and the reference dump
(the 168 MB bundle is gitignored); run them with:

```sh
cargo test --release -p stemsplits-htdemucs -- --ignored --nocapture
```

Regenerate the Swift oracle golden on macOS after changing the contract:

```sh
swift tools/oracle-stft/main.swift tools/oracle-stft/out
cp tools/oracle-stft/out/stft-small.bin crates/stemsplits-stft/tests/golden/
```

The full reasoning — the decisions, the model identification, the exact
forward, and the kernel journey — is in
[`docs/DECISION_LOG.md`](docs/DECISION_LOG.md).
