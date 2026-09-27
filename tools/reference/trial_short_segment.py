"""Trial: whole-track separation at different segment lengths.

Runs the reference HTDemucs over the whole track, chunked with the 25%
overlap and triangular overlap-add exactly as the service would, once per
segment length. Writes all four stems for each so they can be compared by ear,
and reports how far each stem moves from the default run (vocals first).

    tools/reference/.venv/bin/python tools/reference/trial_short_segment.py \
        <mix.wav> [segment-seconds ...]
"""

import sys
import wave
from pathlib import Path

import numpy as np
import torch

MIX = sys.argv[1]
SEGMENTS = [float(value) for value in sys.argv[2:]] or [7.8, 1.0]
OUT = Path(__file__).parent.parent.parent / "samples" / "westside-trial"
STEM_NAMES = ["drums", "bass", "other", "vocals"]
OVERLAP = 0.25


def read(path: str) -> np.ndarray:
    with wave.open(path, "rb") as handle:
        if handle.getsampwidth() != 2:
            raise SystemExit(f"{path} must be 16-bit PCM (ffmpeg -c:a pcm_s16le)")
        data = handle.readframes(handle.getnframes())
    samples = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32_768.0
    return samples.reshape(-1, 2).T


def triangular(frames: int) -> np.ndarray:
    half = frames // 2
    values = np.empty(frames)
    values[:half] = np.arange(1, half + 1)
    values[half:] = np.arange(frames - half, 0, -1)
    return values / values.max()


def separate(model, mix: np.ndarray, segment_frames: int) -> np.ndarray:
    total = mix.shape[-1]
    stride = int(segment_frames * (1 - OVERLAP))
    offsets = [0] if total <= segment_frames else list(range(0, total - segment_frames + 1, stride))
    if total > segment_frames and offsets[-1] != total - segment_frames:
        offsets.append(total - segment_frames)
    window = triangular(segment_frames)
    out = np.zeros((4, 2, total))
    weights = np.zeros(total)
    for offset in offsets:
        segment = mix[:, offset : offset + segment_frames]
        if segment.shape[-1] < segment_frames:
            segment = np.pad(segment, ((0, 0), (0, segment_frames - segment.shape[-1])))
        with torch.no_grad():
            stems = model(torch.from_numpy(segment[None]).float()).numpy()[0]
        out[:, :, offset : offset + segment_frames] += stems * window
        weights[offset : offset + segment_frames] += window
    out /= np.maximum(weights, 1e-8)
    return out


def write_wav(path: Path, stereo: np.ndarray) -> None:
    pcm = (np.clip(stereo, -1.0, 1.0) * 32_767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(44_100)
        handle.writeframes(pcm.tobytes())


def main() -> None:
    import demucs.pretrained as pretrained

    OUT.mkdir(parents=True, exist_ok=True)
    model = pretrained.get_model("htdemucs").models[0]
    model.eval()
    model.use_train_segment = False

    mix = read(MIX)
    print(f"track {mix.shape[-1] / 44_100:.1f} s, segments {SEGMENTS}")

    results = {}
    for seconds in SEGMENTS:
        stems = separate(model, mix, int(seconds * 44_100))
        results[seconds] = stems
        residual = stems.sum(axis=0) - mix
        print(
            f"{seconds:>5.1f}s  reconstruction "
            f"{np.sqrt((residual**2).mean()) / np.sqrt((mix**2).mean()):.4f}"
        )
        for stem, name in enumerate(STEM_NAMES):
            write_wav(OUT / f"{name}-{seconds}s.wav", stems[stem].T)

    reference = results[SEGMENTS[0]]
    print(f"\ndifference from the {SEGMENTS[0]:.1f}s run, whole track:")
    print("  segment  " + "  ".join(f"{name:>8}" for name in STEM_NAMES))
    for seconds in SEGMENTS:
        parts = []
        for stem in range(4):
            difference = results[seconds][stem] - reference[stem]
            parts.append(
                np.sqrt((difference**2).mean()) / np.sqrt((reference[stem] ** 2).mean())
            )
        print(f"  {seconds:>5.1f}s  " + "  ".join(f"{value:>8.3f}" for value in parts))

    print(f"\nwrote stems to {OUT}")


if __name__ == "__main__":
    main()
