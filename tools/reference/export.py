"""Extract HTDemucs weights into the stemsplits bundle format.

This runs once, offline. The output is what the Rust runtime reads: a JSON
manifest and a flat little-endian f32 weight blob. There is no ONNX and no
PyTorch in the shipping path — see AGENTS.md.

    tools/reference/.venv/bin/python tools/reference/export.py [out-dir]

Default out-dir is tools/reference/out/bundle, which is gitignored.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

CONFIG = json.loads(
    (Path(__file__).parent / "demucs_config.json").read_text()
)


def build_model():
    # Imported lazily so the manifest can be inspected without demucs.
    import demucs.pretrained as pretrained

    bag = pretrained.get_model(CONFIG["model"])
    model = bag.models[0]
    model.eval()
    return model


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "out" / "bundle"
    out.mkdir(parents=True, exist_ok=True)

    model = build_model()
    state = model.state_dict()

    tensors = []
    blob = bytearray()
    for name, tensor in state.items():
        array = tensor.detach().to(torch.float32).cpu().numpy()
        flat = np.ascontiguousarray(array.reshape(-1), dtype="<f4")
        offset = len(blob)
        blob.extend(flat.tobytes())
        tensors.append(
            {
                "name": name,
                "shape": list(array.shape),
                "dtype": "f32",
                "offset": offset,
                "count": int(flat.size),
            }
        )

    weight_bytes = bytes(blob)
    digest = hashlib.sha256(weight_bytes).hexdigest()
    (out / "weights.f32").write_bytes(weight_bytes)

    manifest = {
        "format": "stemsplits-bundle",
        "version": 1,
        "model": CONFIG["model"],
        "checkpoint": CONFIG["checkpoint"],
        "config": CONFIG,
        "weights": {
            "file": "weights.f32",
            "dtype": "f32",
            "byte_length": len(weight_bytes),
            "sha256": digest,
        },
        "tensors": tensors,
    }
    (out / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")

    parameters = sum(t["count"] for t in tensors)
    print(f"{len(tensors)} tensors, {parameters} parameters")
    print(f"{len(weight_bytes)} bytes -> {out / 'weights.f32'}")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
