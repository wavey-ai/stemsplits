"""Measure ONNX Runtime on one HTDemucs segment, and check it against PyTorch.

    tools/reference/.venv/bin/python tools/reference/bench_onnx.py [runs] [threads]

Prints the real-time factor (wall seconds per audio second, lower is faster).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from export_onnx import CONFIG, SEGMENT, StemOnnx

AUDIO_SECONDS = SEGMENT / 44_100


def input_signal(count: int, seed: int) -> np.ndarray:
    state = seed
    out = np.empty(count, dtype=np.float32)
    for index in range(count):
        state = (state * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) & ((1 << 64) - 1)
        out[index] = np.float32((state >> 11) / float(1 << 53) * 2 - 1)
    return out


def main() -> None:
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    import onnxruntime as ort

    model_path = Path(__file__).parent / "out" / "htdemucs.onnx"

    import demucs.pretrained as pretrained

    model = pretrained.get_model(CONFIG["model"]).models[0]
    model.eval()
    wrapper = StemOnnx(model).eval()

    left = input_signal(SEGMENT, 0x1234_5678)
    right = input_signal(SEGMENT, 0x9ABC_DEF0)
    mix = torch.from_numpy(np.stack([left, right])[None])
    with torch.no_grad():
        mag = model._magnitude(model._spec(mix))

    options = ort.SessionOptions()
    if threads > 0:
        options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    print("onnxruntime", ort.__version__, "threads", options.intra_op_num_threads)

    feeds = {
        "spectral_magnitude": mag.numpy(),
        "audio_waveform": mix.numpy(),
    }
    outputs = session.run(None, feeds)

    with torch.no_grad():
        reference = wrapper(mag, mix)
    for name, got, expected in zip(
        ["freq_output", "time_output"], outputs, reference
    ):
        difference = np.abs(got - expected.numpy()).max()
        scale = np.abs(expected.numpy()).max()
        print(f"{name}: max abs diff {difference:.3e} (relative {difference / scale:.3e})")

    session.run(None, feeds)  # warm
    start = time.time()
    for _ in range(runs):
        session.run(None, feeds)
    seconds = (time.time() - start) / runs
    print(
        f"onnxruntime: {seconds:.3f} s/segment, audio {AUDIO_SECONDS:.3f} s, "
        f"RTF {seconds / AUDIO_SECONDS:.3f}"
    )

    size = model_path.stat().st_size
    print(f"model file: {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
