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

## Cloud service

`stems-prod` runs on arm64 Lambda in `eu-north-1`. The deployment uses the default AWS credentials.
Run `bash deploy/aws/deploy.sh` with Docker available and the exported weight bundle present.
Run `bash deploy/aws/deploy-regions.sh` to copy that image to all five production regions.

The API Gateway endpoint accepts one segment at `POST /prod/separate`.
Requests require `X-Api-Key` and `Content-Type: application/octet-stream`.
`X-Stems-Format: soundkit-v2-opus-192` is the only wire format. The request contains one SoundKit v2 stream of 192 kbps stereo Opus frames. The response contains four length-delimited SoundKit v2 streams in drums, bass, other, vocals order. Both directions remove the Opus codec delay and pad the tail so each decoded segment remains aligned before overlap-add.
The decoded input has 343,980 frames at 44.1 kHz. API Gateway and Lambda use response streaming.

`deploy/cloudflare` contains the `yl-vin-stems` Worker. It verifies a yl.vin session and forwards segment requests.
The Worker stores the upstream key in its `STEMS_API_KEY` secret. The browser receives no upstream credential.
Run `bash deploy/cloudflare/deploy.sh` after the AWS deployment.
The Worker limits each user to 360 segment requests per minute. This permits a ten-minute track and bounded retries.
API Gateway also applies the shared daily quota. Throttled proxy responses include `Retry-After`.
Neither service stores submitted audio. Request handlers do not log audio or credentials.

`stemsplits-web` prepares browser audio and reconstructs model responses with the shared chunk plan.
It accepts mono or stereo tracks up to ten minutes long. FFT resampling converts other sample rates to 44.1 kHz.
Reconstruction retains only the unfinished overlap. The browser dispatches all planned segments concurrently.
The browser stores completed responses in temporary OPFS files and reads them in plan order.
It removes these files after completion or cancellation. Web Locks protect active jobs during abandoned-file cleanup.
The model segment remains 7.8 seconds with 25% overlap.

`deploy/aws/wholetrack` runs as a separate ARM64 Lambda in the US East 1
mastering stack. When a mastering job has no saved stems, it reads the
uploaded lossless original, uses this repository's `SplitSession` for chunk
planning and overlap-add, calls the deployed segment endpoint in batches of
eight, and writes four aligned 192 kb/s SoundKit Opus estimates. It then
invokes the mastering planner. The mastering deployment builds this worker
with `deploy/aws/Wholetrack.Dockerfile`; its API key comes from the existing
gitignored `deploy/aws/.api-key`. The worker stores estimates only in the
private, short-lived mastering job bucket. The original mix remains the
mastering render input.

Each segment attempt has a 150-second deadline, including response-body reads.
Transient network failures, timeouts, incomplete bodies, and HTTP 408, 425, 429, 500, 502, 503, and 504 can retry.
Each segment has at most three attempts. Retries use exponential backoff, jitter, and `Retry-After`.
Permanent errors cancel the remaining requests. Completed segments do not repeat.

The app builds this package with `node scripts/build-wasm.mjs stemsplits` in `../bitneedle-app`.
That command also installs `browser/client.mjs` and `browser/http.mjs`.
The app then imports four WAV files through its normal library pipeline.

## Cloud checks

Deployment details and measured results are in [Cloud validation](docs/CLOUD_VALIDATION.md).

```sh
cargo test --workspace --lib
cargo test --manifest-path deploy/aws/lambda/Cargo.toml
cargo test --manifest-path deploy/aws/wholetrack/Cargo.toml
node --test browser/client.test.mjs
```

In `../bitneedle-app`, run `node scripts/stems.browser.mjs` for isolated browser checks.
Run `node --test scripts/stems-wasm.test.mjs` to check the compiled browser module.
Run `node scripts/stems-edge.test.mjs` to check the compiled proxy with isolated service fixtures.
Run `node scripts/stems.browser.mjs --live` for one 16-second track through yl.vin.
This test creates and removes one temporary test identity.

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
