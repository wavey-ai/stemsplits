"""Run the reference HTDemucs and dump stems and activations.

The oracle for the Rust forward pass: same fixed input, same weights, so a
layer port can be checked one tensor at a time.

    tools/reference/.venv/bin/python tools/reference/reference_stems.py [out-dir]

Default out-dir is tools/reference/out/reference, which is gitignored.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

CONFIG = json.loads((Path(__file__).parent / "demucs_config.json").read_text())
SEGMENT = 343_980
SEED_LEFT = 0x1234_5678
SEED_RIGHT = 0x9ABC_DEF0


def input_signal(count: int, seed: int) -> np.ndarray:
    """The same deterministic generator the Rust tests use."""
    state = seed
    out = np.empty(count, dtype=np.float32)
    for index in range(count):
        state = (state * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) & ((1 << 64) - 1)
        unit = (state >> 11) / float(1 << 53)
        out[index] = np.float32(unit * 2 - 1)
    return out


def write(name: str, array: np.ndarray, out: Path, index: dict) -> None:
    array = np.ascontiguousarray(array, dtype="<f4")
    path = out / f"{name}.f32"
    path.write_bytes(array.tobytes())
    index[name] = {"shape": list(array.shape), "count": int(array.size)}


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "out" / "reference"
    out.mkdir(parents=True, exist_ok=True)

    import demucs.pretrained as pretrained

    model = pretrained.get_model(CONFIG["model"]).models[0]
    model.eval()

    left = input_signal(SEGMENT, SEED_LEFT)
    right = input_signal(SEGMENT, SEED_RIGHT)
    mix = torch.from_numpy(np.stack([left, right])[None])  # [1, 2, N]

    index: dict = {}
    write("input", mix.numpy(), out, index)

    # Capture the named tensors a layer port needs to be checked against.
    captured: dict[str, np.ndarray] = {}

    def hook(name):
        def run(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                captured[name] = output.detach().cpu().numpy()
        return run

    def pre_hook(name):
        def run(_module, inputs):
            captured[name] = inputs[0].detach().cpu().numpy()
        return run

    for name, module in model.named_modules():
        if name in {
            "encoder.0",
            "encoder.1",
            "encoder.2",
            "encoder.3",
            "crosstransformer",
            "decoder.0",
            "decoder.1",
            "decoder.2",
            "decoder.3",
            "channel_upsampler",
            "channel_upsampler_t",
        }:
            module.register_forward_hook(hook(name))
        if name in {"encoder.0", "crosstransformer"}:
            module.register_forward_pre_hook(pre_hook(f"input_{name}"))

    with torch.no_grad():
        stems = model(mix)  # [1, 4, 2, N]

    write("stems", stems.numpy(), out, index)
    for name, array in captured.items():
        write(name.replace(".", "_"), array, out, index)

    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"stems {tuple(stems.shape)}")
    for name, array in captured.items():
        print(f"  {name} {tuple(array.shape)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
