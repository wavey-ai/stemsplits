//! `HEncLayer` from `demucs/hdemucs.py`, both branches.
//!
//! The frequency branch is `[B, C, Fr, T]` and convolves over both axes with
//! a `(8, 1)` kernel; the time branch is `[B, C, T]` and convolves over time.
//! With `norm_starts = 4` and `depth = 4` the layer's own GroupNorms are
//! identities, so only GELU, DConv and the GLU rewrite are active.

use anyhow::{Context, Result};

use crate::dconv::DConv;
use crate::ops::{conv1d, conv2d, gelu, glu};
use crate::tensor::Tensor;
use stemsplits_model::Weights;

pub struct EncLayer {
    freq: bool,
    conv_weight: Vec<f32>,
    conv_bias: Vec<f32>,
    dconv: DConv,
    rewrite_weight: Vec<f32>,
    rewrite_bias: Vec<f32>,
}

fn load(weights: &Weights, name: &str) -> Result<Vec<f32>> {
    weights
        .tensor(name)
        .with_context(|| format!("missing weight {name}"))
        .map(|tensor| tensor.data().to_vec())
}

impl EncLayer {
    /// `prefix` is `encoder.i` or `tencoder.i`. `chout` is the output
    /// channels; `context_enc` is 0 for HTDemucs, so the rewrite is 1x1.
    pub fn load(
        weights: &Weights,
        prefix: &str,
        freq: bool,
        chout: usize,
        compress: usize,
        depth: usize,
    ) -> Result<Self> {
        Ok(Self {
            freq,
            conv_weight: load(weights, &format!("{prefix}.conv.weight"))?,
            conv_bias: load(weights, &format!("{prefix}.conv.bias"))?,
            dconv: DConv::load(weights, &format!("{prefix}.dconv"), chout, compress, depth)?,
            rewrite_weight: load(weights, &format!("{prefix}.rewrite.weight"))?,
            rewrite_bias: load(weights, &format!("{prefix}.rewrite.bias"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        if self.freq {
            self.forward_freq(x)
        } else {
            self.forward_time(x)
        }
    }

    fn forward_freq(&self, x: &Tensor) -> Tensor {
        let channels = self.conv_bias.len();
        let y = conv2d(
            x,
            &self.conv_weight,
            &self.conv_bias,
            channels,
            (8, 1),
            (4, 1),
            (2, 0),
        );
        let y = gelu(&y);

        // Flatten frequency into the batch for the DConv, exactly as
        // `y.permute(0, 2, 1, 3).reshape(-1, C, T)`.
        let batch = y.dim(0);
        let frequency = y.dim(2);
        let time = y.dim(3);
        let flattened = Tensor::new(
            vec![batch * frequency, channels, time],
            permute_0213(&y).data,
        );
        let convolved = self.dconv.forward(&flattened);
        let restored = Tensor::new(vec![batch, frequency, channels, time], convolved.data);
        let y = unpermute_0213(&restored);

        let z = conv2d(
            &y,
            &self.rewrite_weight,
            &self.rewrite_bias,
            2 * channels,
            (1, 1),
            (1, 1),
            (0, 0),
        );
        glu(&z)
    }

    fn forward_time(&self, x: &Tensor) -> Tensor {
        let channels = self.conv_bias.len();
        // `le % stride` is padded away so the output is exactly length/stride.
        let time = x.dim(2);
        let stride = 4;
        let x = if time.is_multiple_of(stride) {
            x.clone()
        } else {
            pad_time(x, stride - (time % stride))
        };
        let y = conv1d(
            &x,
            &self.conv_weight,
            &self.conv_bias,
            channels,
            8,
            stride,
            2,
            1,
        );
        let y = gelu(&y);
        let y = self.dconv.forward(&y);
        let z = conv1d(
            &y,
            &self.rewrite_weight,
            &self.rewrite_bias,
            2 * channels,
            1,
            1,
            0,
            1,
        );
        glu(&z)
    }
}

/// `ScaledEmbedding` added after the first encoder layer: a learned vector per
/// frequency, scaled by `freq_emb_scale`. The stored parameter is the raw
/// embedding weight, which `forward` multiplies by `scale`.
pub struct FrequencyEmbedding {
    weight: Vec<f32>,
    dim: usize,
    scale: f32,
}

impl FrequencyEmbedding {
    pub fn load(weights: &Weights, scale: f32) -> Result<Self> {
        let tensor = weights
            .tensor("freq_emb.embedding.weight")
            .context("missing freq_emb.embedding.weight")?;
        let shape = tensor.shape();
        Ok(Self {
            weight: tensor.data().to_vec(),
            dim: shape[1],
            scale,
        })
    }

    /// Adds `weight_scale * scale * weight[freq, channel]` to `[B, C, Fr, T]`.
    pub fn apply(&self, x: &Tensor, weight_scale: f32) -> Tensor {
        let batch = x.dim(0);
        let channels = x.dim(1);
        let frequency = x.dim(2);
        let time = x.dim(3);
        assert_eq!(channels, self.dim, "frequency embedding channels");
        assert_eq!(
            frequency * self.dim,
            self.weight.len(),
            "frequency embedding length"
        );
        let mut output = x.clone();
        let factor = weight_scale * self.scale;
        for b in 0..batch {
            for c in 0..channels {
                for f in 0..frequency {
                    let add = factor * self.weight[f * self.dim + c];
                    let base = (b * channels + c) * frequency * time + f * time;
                    for t in 0..time {
                        output.data[base + t] += add;
                    }
                }
            }
        }
        output
    }
}

/// `[B, C, F, T] -> [B, F, C, T]`.
fn permute_0213(x: &Tensor) -> Tensor {
    let batch = x.dim(0);
    let channels = x.dim(1);
    let frequency = x.dim(2);
    let time = x.dim(3);
    let mut output = Tensor::zeros(vec![batch, frequency, channels, time]);
    for b in 0..batch {
        for c in 0..channels {
            for f in 0..frequency {
                for t in 0..time {
                    output.data[((b * frequency + f) * channels + c) * time + t] =
                        x.data[((b * channels + c) * frequency + f) * time + t];
                }
            }
        }
    }
    output
}

/// `[B, F, C, T] -> [B, C, F, T]`.
fn unpermute_0213(x: &Tensor) -> Tensor {
    let batch = x.dim(0);
    let frequency = x.dim(1);
    let channels = x.dim(2);
    let time = x.dim(3);
    let mut output = Tensor::zeros(vec![batch, channels, frequency, time]);
    for b in 0..batch {
        for f in 0..frequency {
            for c in 0..channels {
                for t in 0..time {
                    output.data[((b * channels + c) * frequency + f) * time + t] =
                        x.data[((b * frequency + f) * channels + c) * time + t];
                }
            }
        }
    }
    output
}

fn pad_time(x: &Tensor, right: usize) -> Tensor {
    let batch = x.dim(0);
    let channels = x.dim(1);
    let time = x.dim(2);
    let mut output = Tensor::zeros(vec![batch, channels, time + right]);
    for b in 0..batch {
        for c in 0..channels {
            let source = (b * channels + c) * time;
            let destination = (b * channels + c) * (time + right);
            output.data[destination..(destination + time)]
                .copy_from_slice(&x.data[source..(source + time)]);
        }
    }
    output
}
