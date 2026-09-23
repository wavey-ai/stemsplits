"""Export HTDemucs to ONNX with the same I/O contract as the Core ML model.

Inputs:  spectral_magnitude [1, 4, 2048, 336], audio_waveform [1, 2, 343980]
Outputs: freq_output [1, 16, 2048, 336], time_output [1, 8, 343980]

This is the ONNX Runtime baseline the Rust port is measured against. It is an
oracle only; it does not ship (see AGENTS.md).

    tools/reference/.venv/bin/python tools/reference/export_onnx.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

CONFIG = json.loads((Path(__file__).parent / "demucs_config.json").read_text())
SEGMENT = 343_980


class StemOnnx(torch.nn.Module):
    """`HTDemucs.forward` with the STFT lifted out: mag and waveform come in,
    the denormalised frequency and waveform branches go out."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, mag, mix):
        m = self.m
        batch, _, frequency, time = mag.shape

        mean = mag.mean(dim=(1, 2, 3), keepdim=True)
        std = mag.std(dim=(1, 2, 3), keepdim=True)
        x = (mag - mean) / (1e-5 + std)

        meant = mix.mean(dim=(1, 2), keepdim=True)
        stdt = mix.std(dim=(1, 2), keepdim=True)
        xt = (mix - meant) / (1e-5 + stdt)

        saved, saved_t, lengths, lengths_t = [], [], [], []
        for idx, encode in enumerate(m.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(m.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = m.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and m.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = m.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + m.freq_emb_scale * emb
            saved.append(x)

        if m.crosstransformer:
            b, c, f, t = x.shape
            x = rearrange(x, "b c f t-> b c (f t)")
            x = m.channel_upsampler(x)
            x = rearrange(x, "b c (f t)-> b c f t", f=f)
            xt = m.channel_upsampler_t(xt)
            x, xt = m.crosstransformer(x, xt)
            x = rearrange(x, "b c f t-> b c (f t)")
            x = m.channel_downsampler(x)
            x = rearrange(x, "b c (f t)-> b c f t", f=f)
            xt = m.channel_downsampler_t(xt)

        for idx, decode in enumerate(m.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = m.depth - len(m.tdecoder)
            if idx >= offset:
                tdec = m.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    xt, _ = tdec(pre[:, :, 0], None, length_t)
                else:
                    xt, _ = tdec(xt, saved_t.pop(-1), length_t)

        sources = len(m.sources)
        x = x.view(batch, sources, -1, frequency, time)
        x = x * std[:, None] + mean[:, None]
        x = x.reshape(batch, sources * x.shape[2], frequency, time)

        xt = xt.view(batch, sources, -1, mix.shape[-1])
        xt = xt * stdt[:, None] + meant[:, None]
        xt = xt.reshape(batch, sources * xt.shape[2], mix.shape[-1])
        return x, xt


def main() -> None:
    out = Path(__file__).parent / "out"
    out.mkdir(parents=True, exist_ok=True)

    import demucs.pretrained as pretrained

    model = pretrained.get_model(CONFIG["model"]).models[0]
    model.eval()
    wrapper = StemOnnx(model).eval()

    magnitude = torch.zeros(1, 4, 2048, 336)
    waveform = torch.zeros(1, 2, SEGMENT)
    path = out / "htdemucs.onnx"
    torch.onnx.export(
        wrapper,
        (magnitude, waveform),
        str(path),
        input_names=["spectral_magnitude", "audio_waveform"],
        output_names=["freq_output", "time_output"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
