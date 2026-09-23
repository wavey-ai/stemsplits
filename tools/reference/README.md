# The PyTorch/ONNX reference

Nothing in this directory ships. It exists to answer one question per model
change: does our Rust forward pass produce the same stems as the reference
HTDemucs?

The reference is the original PyTorch HTDemucs (Meta), optionally exported to
ONNX. It is the oracle for parity tests, not a runtime dependency — see
`AGENTS.md`.

Planned contents:

- `export.py` — pull the HTDemucs checkpoint, emit a weight bundle in our
  format, and (for the oracle only) an ONNX graph.
- `reference_stems.py` — run the reference on a fixed input and write stems.
- `compare.py` — compare our Rust stems to the reference: max abs difference
  and SNR per stem.

The first number to get is the arm64 CPU real-time factor for one 7.8-second
segment. That single measurement decides whether cloud stems are worth
building. See `bench/README.md`.
