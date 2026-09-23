//! The full HTDemucs forward pass.
//!
//! Mirrors `HTDemucs.forward` with the STFT moved outside: the caller supplies
//! the CaC magnitude and the waveform, exactly as the Core ML model does. The
//! output is the denormalised frequency branch (16 channels: 4 stems × 2
//! audio channels × real/imag) and the denormalised waveform branch (8
//! channels). The caller does `time + iSTFT(freq) * sqrt(fft_size)` per stem.

use anyhow::{Context, Result};

use crate::decoder::DecLayer;
use crate::encoder::{EncLayer, FrequencyEmbedding};
use crate::ops::conv1d;
use crate::tensor::Tensor;
use crate::transformer::CrossTransformer;
use stemsplits_model::Weights;

struct ChannelConv {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl ChannelConv {
    fn load(weights: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: weights
                .tensor(&format!("{prefix}.weight"))
                .with_context(|| format!("missing {prefix}.weight"))?
                .data()
                .to_vec(),
            bias: weights
                .tensor(&format!("{prefix}.bias"))
                .with_context(|| format!("missing {prefix}.bias"))?
                .data()
                .to_vec(),
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let out_channels = self.bias.len();
        conv1d(x, &self.weight, &self.bias, out_channels, 1, 1, 0, 1)
    }
}

pub struct HtDemucs {
    encoders: Vec<EncLayer>,
    tencoders: Vec<EncLayer>,
    frequency_embedding: FrequencyEmbedding,
    channel_upsampler: ChannelConv,
    channel_upsampler_t: ChannelConv,
    channel_downsampler: ChannelConv,
    channel_downsampler_t: ChannelConv,
    transformer: CrossTransformer,
    decoders: Vec<DecLayer>,
    tdecoders: Vec<DecLayer>,
}

const CHANNELS: [usize; 4] = [48, 96, 192, 384];

impl HtDemucs {
    pub fn load(weights: &Weights) -> Result<Self> {
        let mut encoders = Vec::new();
        let mut tencoders = Vec::new();
        let mut decoders = Vec::new();
        let mut tdecoders = Vec::new();
        for (index, &chout) in CHANNELS.iter().enumerate() {
            encoders.push(EncLayer::load(
                weights,
                &format!("encoder.{index}"),
                true,
                chout,
                8,
                2,
            )?);
            tencoders.push(EncLayer::load(
                weights,
                &format!("tencoder.{index}"),
                false,
                chout,
                8,
                2,
            )?);
            // Decoder modules are stored in reverse: decoder.0 is the deepest,
            // so its input channels are the last encoder's output.
            let chin = CHANNELS[CHANNELS.len() - 1 - index];
            let last = index == 3;
            decoders.push(DecLayer::load(
                weights,
                &format!("decoder.{index}"),
                true,
                last,
                chin,
                8,
                2,
            )?);
            tdecoders.push(DecLayer::load(
                weights,
                &format!("tdecoder.{index}"),
                false,
                last,
                chin,
                8,
                2,
            )?);
        }
        Ok(Self {
            encoders,
            tencoders,
            frequency_embedding: FrequencyEmbedding::load(weights, 10.0)?,
            channel_upsampler: ChannelConv::load(weights, "channel_upsampler")?,
            channel_upsampler_t: ChannelConv::load(weights, "channel_upsampler_t")?,
            channel_downsampler: ChannelConv::load(weights, "channel_downsampler")?,
            channel_downsampler_t: ChannelConv::load(weights, "channel_downsampler_t")?,
            transformer: CrossTransformer::load(weights, "crosstransformer", 8, 10_000.0, 1.0)?,
            decoders,
            tdecoders,
        })
    }

    /// `magnitude` is `[B, 4, Fr, T]` (CaC), `waveform` is `[B, 2, N]`.
    /// Returns `(freq [B, 16, Fr, T], time [B, 8, N])`, both denormalised.
    pub fn forward(&self, magnitude: &Tensor, waveform: &Tensor) -> (Tensor, Tensor) {
        let profile = std::env::var_os("STEMSPLITS_PROFILE").is_some();
        let mut mark = std::time::Instant::now();
        let mut lap = |name: &str| {
            if profile {
                eprintln!("  {name}: {:.1} ms", mark.elapsed().as_secs_f64() * 1e3);
                mark = std::time::Instant::now();
            }
        };

        let (mut x, mean, std) = normalise(magnitude);
        let (mut xt, mean_t, std_t) = normalise(waveform);
        lap("normalise");

        let mut saved = Vec::new();
        let mut saved_t = Vec::new();
        let mut lengths = Vec::new();
        let mut lengths_t = Vec::new();

        for index in 0..self.encoders.len() {
            lengths.push(x.dim(2));
            lengths_t.push(xt.dim(2));
            xt = self.tencoders[index].forward(&xt);
            saved_t.push(xt.clone());
            x = self.encoders[index].forward(&x);
            if index == 0 {
                x = self.frequency_embedding.apply(&x, 0.2);
            }
            saved.push(x.clone());
        }
        lap("encode");

        x = upsample(&self.channel_upsampler, &x);
        xt = self.channel_upsampler_t.forward(&xt);
        let (transformed_x, transformed_t) = self.transformer.forward(&x, &xt);
        lap("transformer");
        let mut x = downsample(&self.channel_downsampler, &transformed_x);
        let mut xt = self.channel_downsampler_t.forward(&transformed_t);
        for index in 0..self.decoders.len() {
            let skip = saved.pop().expect("freq skip");
            let length = lengths.pop().expect("freq length");
            x = self.decoders[index].forward(&x, &skip, length);
            let skip_t = saved_t.pop().expect("time skip");
            let length_t = lengths_t.pop().expect("time length");
            xt = self.tdecoders[index].forward(&xt, &skip_t, length_t);
        }
        lap("decode");

        (denormalise(&x, mean, std), denormalise(&xt, mean_t, std_t))
    }
}

/// Per-sample mean/std over every element, matching `x.mean(dim=(1,2,3))` and
/// `x.std(...)` (unbiased) in the reference.
fn normalise(x: &Tensor) -> (Tensor, f32, f32) {
    let count = x.numel() as f64;
    let sum: f64 = x.data.iter().map(|v| *v as f64).sum();
    let mean = sum / count;
    let variance: f64 = x
        .data
        .iter()
        .map(|v| (*v as f64 - mean).powi(2))
        .sum::<f64>()
        / (count - 1.0);
    let std = variance.sqrt() as f32;
    let mean = mean as f32;
    let normalised = Tensor {
        shape: x.shape.clone(),
        data: x.data.iter().map(|v| (v - mean) / (1e-5 + std)).collect(),
    };
    (normalised, mean, std)
}

fn denormalise(x: &Tensor, mean: f32, std: f32) -> Tensor {
    Tensor {
        shape: x.shape.clone(),
        data: x.data.iter().map(|v| v * std + mean).collect(),
    }
}

/// `[B, C, Fr, T] -> [B, C, Fr*T]` (frequency-major, which row-major already
/// is), conv, back.
fn upsample(conv: &ChannelConv, x: &Tensor) -> Tensor {
    let batch = x.dim(0);
    let channels = x.dim(1);
    let frequency = x.dim(2);
    let time = x.dim(3);
    let flat = Tensor::new(vec![batch, channels, frequency * time], x.data.clone());
    let out = conv.forward(&flat);
    Tensor::new(vec![batch, conv.bias.len(), frequency, time], out.data)
}

fn downsample(conv: &ChannelConv, x: &Tensor) -> Tensor {
    upsample(conv, x)
}
