# Benchmarks

One 7.8-second segment (343,980 frames) on an Apple Silicon machine. Lower RTF
is faster; the real-time factor is wall seconds per audio second.

## Against ONNX Runtime and PyTorch

| implementation | threads | s/segment | RTF | artifact |
| --- | ---: | ---: | ---: | ---: |
| Rust, naive scalar (first port) | 1 | 343 | 44.0 | ~1 MB binary |
| Rust, after the kernel pass | 1 | 12.6 | **1.61** | ~1 MB binary |
| ONNX Runtime | 1 | 20.4 | 2.61 | 174.3 MB onnx + 58.4 MB lib |
| ONNX Runtime | 8 | 8.79 | 1.13 | |
| PyTorch CPU | 8 | 8.49 | 1.09 | |

Accuracy against PyTorch on the same input:

| implementation | freq_output | time_output | stems (end to end) |
| --- | ---: | ---: | ---: |
| Rust port | | | **2.99e-6** |
| ONNX Runtime | 5.6e-4 | 5.0e-5 | |

The Rust port is currently the more accurate of the two against PyTorch; the
ONNX export folds constants more aggressively.

Artifact size: the Rust binary is ~1 MB and self-contained; ONNX Runtime adds
58.4 MB here as a Python extension (~15–20 MB for the bare shared library).
The f32 weights are 168 MB either way. The f16 bundle is 84 MB and gives
the same weights.

## The kernel pass

Each step was measured with `cargo run --release -p stemsplits-htdemucs --bin
bench`. The Rust scalar loop stays scalar: reassociating a reduction changes
the result, so the compiler preserves the accumulation order, and blocking
with `target-cpu=native` held performance steady until the SIMD was explicit.

| change | RTF |
| --- | ---: |
| naive scalar | 44.0 |
| blocked matmul, four independent accumulation chains | 20.7 |
| conv2d as im2col + matmul (the decoder's 3×3 rewrites were ~28 GFLOP of serial accumulation) | 13.9 |
| conv_transpose1d/2d as matmul + scatter | 7.65 |
| `Linear` transposes its weight so the matmul inner loop vectorises | 3.28 |
| C NEON micro-kernel, four rows × one vector, k innermost | 2.26 |
| micro-kernel widened to eight columns with `vmlaq_n_f32` | **1.61** |

The GEMM is `crates/stemsplits-htdemucs/kernels/gemm.c`, compiled by
`build.rs` with AVX2/FMA on x86 and NEON on aarch64. Its per-output
accumulation order is preserved, so FMA contraction alone moves the result,
inside the parity tolerance (stems stay at 2.99e-6). The approach mirrors
[encodec-rs](https://github.com/wavey-ai/encodec-rs).

## WASM

One 7.8 s segment on an Apple M-series Mac with 8 cores, in Node 26 with one
thread. The input is the seeded segment of the `segment` example.

| change | seconds | RTF |
| --- | ---: | ---: |
| native build, one thread | 9.25 | 1.19 |
| WASM SIMD128, one row by 16 columns | 25.7 | 3.29 |
| WASM SIMD128, four rows by 16 columns | 15.3 | 1.96 |
| polynomial `exp` in the softmax, `f32.nearest` rounding | 13.8 | 1.77 |

The WASM output is −121 dB from the native output. The largest sample
difference is 1.8e-6.

Model load in WASM took 9.7 s with the f32 bundle. SHA-256 in WASM used 9.5 s
of that time. With the f16 bundle and a WebCrypto digest, the load takes
0.33 s.

Workers in Node, two segments for each worker:

| workers | seconds per segment |
| ---: | ---: |
| 1 | 14.1 |
| 2 | 7.8 |
| 4 | 5.7 |
| 6 | 5.9 |

A 193 s track (33 segments) through `splitSegmentsLocal` with four workers:

| browser | seconds | RTF |
| --- | ---: | ---: |
| Chromium, headless | 173 | 0.90 |
| WebKit, headless | 183 | 0.95 |

The two browsers gave the same 16-bit stems, byte for byte. The sum of the
browser stems was −31.4 dB from the mix. The sum of the cloud stems for the
same mix, with Opus in both directions, was −17.6 dB from the mix.

## Reproduce

The bench needs the weight bundle; see `tools/reference/README.md`.

```sh
cargo run --release -p stemsplits-htdemucs --bin bench
```

The ONNX baseline:

```sh
tools/reference/.venv/bin/python tools/reference/export_onnx.py
tools/reference/.venv/bin/python tools/reference/bench_onnx.py 3 1
```
