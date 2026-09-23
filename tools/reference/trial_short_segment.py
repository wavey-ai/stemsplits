"""Trial: what does a shorter model segment cost in quality?

Runs the reference HTDemucs at several segment lengths on the same audio and
reports, per stem, how far each result is from the 7.8 s-segment result over
the same window. Writes WAVs of the 7.8 s and 2 s cases for listening.

    tools/reference/.venv/bin/python tools/reference/trial_short_segment.py
"""

import sys
import wave
from pathlib import Path

import numpy as np
import torch

WAV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/westside-30s-44k.wav"
START = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
OUT = Path(__file__).parent.parent.parent / "samples" / "short-segment-trial"
SEGMENTS = [7.8, 4.0, 2.0, 1.0]
STEM_NAMES = ["drums", "bass", "other", "vocals"]


def read(seconds: float) -> torch.Tensor:
    with wave.open(WAV, "rb") as handle:
        if handle.getsampwidth() != 2:
            raise SystemExit(
                f"{WAV} is {handle.getsampwidth() * 8}-bit; convert to 16-bit PCM first "
                "(ffmpeg -c:a pcm_s16le)"
            )
        handle.setpos(int(START * 44_100))
        data = handle.readframes(int(seconds * 44_100))
    samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32_768.0
    # interleaved (N, 2) -> (1, 2, N)
    return torch.from_numpy(samples.reshape(-1, 2).T[None])


def write_wav(path: Path, stereo: np.ndarray) -> None:
    pcm = (np.clip(stereo, -1.0, 1.0) * 32_767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(44_100)
        handle.writeframes(pcm.tobytes())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    import demucs.pretrained as pretrained

    model = pretrained.get_model("htdemucs").models[0]
    model.eval()
    # Process exactly the audio given, rather than padding up to the 7.8 s
    # training segment.
    model.use_train_segment = False

    results = {}
    mixes = {}
    for seconds in SEGMENTS:
        mix = read(seconds)
        mixes[seconds] = mix
        with torch.no_grad():
            stems = model(mix)
        results[seconds] = stems
        residual = stems.sum(dim=1) - mix
        print(
            f"{seconds:>4.1f}s: stems {tuple(stems.shape)}  "
            f"sum-vs-mix rel {residual.pow(2).mean().sqrt() / mix.pow(2).mean().sqrt():.4f}"
        )

    reference = results[7.8]
    print("\nrelative difference from the 7.8 s result, over the same window:")
    header = "seg   " + "  ".join(f"{name:>8}" for name in STEM_NAMES) + "   mean"
    print(header)
    for seconds in SEGMENTS:
        length = results[seconds].shape[-1]
        parts = []
        for stem in range(4):
            short = results[seconds][0, stem, :, :length]
            long = reference[0, stem, :, :length]
            difference = (short - long).pow(2).mean().sqrt()
            base = long.pow(2).mean().sqrt()
            parts.append((difference / base).item())
        mean = sum(parts) / len(parts)
        print(f"{seconds:>4.1f}  " + "  ".join(f"{value:>8.3f}" for value in parts) + f"   {mean:.3f}")

    # A/B WAVs for the 7.8 s and 2.0 s cases, over the first 2 seconds.
    two = int(2.0 * 44_100)
    for seconds in (7.8, 2.0):
        for stem, name in enumerate(STEM_NAMES):
            stereo = results[seconds][0, stem, :, :two].numpy().T
            write_wav(OUT / f"{name}-{seconds}s.wav", stereo)
    print(f"\nwrote A/B stems to {OUT}")


if __name__ == "__main__":
    main()
