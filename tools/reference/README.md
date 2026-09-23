# The PyTorch/ONNX reference

Nothing in this directory ships. It exists to answer one question per model
change: does our Rust forward pass produce the same stems as the reference
HTDemucs?

The reference is the original PyTorch HTDemucs (Meta), optionally exported to
ONNX. It is the oracle for parity tests, not a runtime dependency — see
`AGENTS.md`.

The model is `htdemucs` checkpoint `955717e8` (Demucs v4, MIT). The config is
pinned in `demucs_config.json`; the checkpoint is cached by `demucs.pretrained`
under `~/.cache/torch/hub/checkpoints/`.

## Environment

The venv inherits the system PyTorch but must shadow the system NumPy with
`numpy<2`, because the installed PyTorch was built against NumPy 1.x:

```sh
python3 -m venv --system-site-packages .venv
.venv/bin/pip install demucs --no-deps
.venv/bin/pip install einops julius lameenc openunmix dora-search omegaconf pyyaml tqdm "numpy<2"
```

## Scripts

- `export.py` — emit the weight bundle (`bundle.json` + `weights.f32`) that
  `stemsplits-model` loads. 533 tensors, 41.98 M parameters, 168 MB f32.
- `reference_stems.py` — run the reference on the fixed deterministic segment
  and dump the stems plus encoder activations for layer-by-layer parity.

```sh
.venv/bin/python export.py
.venv/bin/python reference_stems.py
```

Both write under `out/`, which is gitignored.

## The device model

The Core ML package Wavey ships is this same graph converted to an ML
Program with **f16** weights (`tencoder_0_conv_weight_to_fp16` in the graph),
104.6 MB. It is not a different model, and it is not a different size in
parameters — only in storage. Our Rust runtime starts in f32; matching the
device exactly means f16 somewhere later, with a tolerance-based parity test.

The first number to get is the arm64 CPU real-time factor for one 7.8-second
segment. That single measurement decides whether cloud stems are worth
building. See `bench/README.md`.
