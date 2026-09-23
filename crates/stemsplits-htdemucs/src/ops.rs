//! The primitive operations HTDemucs is made of, written to match PyTorch.
//!
//! Scalar and obvious first. A SIMD kernel replaces the hot ones later; the
//! reference activation tests are what make that safe.

// Numeric kernels read better with explicit index loops, and the convolutions
// carry their geometry as arguments rather than a config struct.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use crate::tensor::Tensor;

const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// `F.gelu`, the exact (`approximate = "none"`) form.
pub fn gelu(x: &Tensor) -> Tensor {
    Tensor {
        shape: x.shape.clone(),
        data: x
            .data
            .iter()
            .map(|&value| 0.5 * value * (1.0 + libm::erff(value * FRAC_1_SQRT_2)))
            .collect(),
    }
}

pub fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

/// `F.glu(x, dim = 1)`: `a * sigmoid(b)` where the channel axis is split in
/// half. Works for any rank; the channel axis is the second.
pub fn glu(x: &Tensor) -> Tensor {
    let batch = x.dim(0);
    let channels = x.dim(1);
    let half = channels / 2;
    let spatial: usize = x.shape[2..].iter().product();
    let mut shape = x.shape.clone();
    shape[1] = half;
    let mut output = Tensor::zeros(shape);
    for index in 0..batch {
        for channel in 0..half {
            let a = (index * channels + channel) * spatial;
            let b = (index * channels + channel + half) * spatial;
            let destination = (index * half + channel) * spatial;
            for offset in 0..spatial {
                output.data[destination + offset] =
                    x.data[a + offset] * sigmoid(x.data[b + offset]);
            }
        }
    }
    output
}

/// `nn.GroupNorm`: normalise over each group's channels and all spatial
/// positions, then scale per channel. Used with one group by DConv, so the
/// whole channel axis and the time axis share a statistic.
pub fn group_norm(x: &Tensor, groups: usize, weight: &[f32], bias: &[f32], eps: f32) -> Tensor {
    let batch = x.dim(0);
    let channels = x.dim(1);
    assert_eq!(
        channels % groups,
        0,
        "channels {channels} not divisible by {groups}"
    );
    let per_group = channels / groups;
    let spatial: usize = x.shape[2..].iter().product();
    let count = (per_group * spatial) as f32;
    let mut output = x.clone();
    for index in 0..batch {
        for group in 0..groups {
            let base = index * channels + group * per_group;
            let mut sum = 0.0f64;
            for channel in 0..per_group {
                let start = (base + channel) * spatial;
                for offset in 0..spatial {
                    sum += x.data[start + offset] as f64;
                }
            }
            let mean = (sum / count as f64) as f32;
            let mut variance = 0.0f64;
            for channel in 0..per_group {
                let start = (base + channel) * spatial;
                for offset in 0..spatial {
                    let delta = x.data[start + offset] - mean;
                    variance += (delta * delta) as f64;
                }
            }
            let inverse = 1.0 / ((variance / count as f64) as f32 + eps).sqrt();
            for channel in 0..per_group {
                let global = group * per_group + channel;
                let scale = weight[global];
                let shift = bias[global];
                let start = (base + channel) * spatial;
                for offset in 0..spatial {
                    output.data[start + offset] =
                        (x.data[start + offset] - mean) * inverse * scale + shift;
                }
            }
        }
    }
    output
}

/// `MyGroupNorm(1, channels)` on a `[B, T, C]` sequence: transpose to
/// `[B, C, T]`, normalise over all channels and time per sample, scale per
/// channel, transpose back. The transformer's `norm_out` is this.
pub fn group_norm_sequence(x: &Tensor, weight: &[f32], bias: &[f32], eps: f32) -> Tensor {
    let batch = x.dim(0);
    let time = x.dim(1);
    let channels = x.dim(2);
    assert_eq!(weight.len(), channels);
    let count = (time * channels) as f64;
    let mut output = x.clone();
    for index in 0..batch {
        let base = index * time * channels;
        let mut sum = 0.0f64;
        for offset in 0..(time * channels) {
            sum += x.data[base + offset] as f64;
        }
        let mean = sum / count;
        let mut variance = 0.0f64;
        for offset in 0..(time * channels) {
            let delta = x.data[base + offset] as f64 - mean;
            variance += delta * delta;
        }
        variance /= count;
        let inverse = 1.0 / ((variance as f32) + eps).sqrt();
        for t in 0..time {
            for c in 0..channels {
                let offset = base + t * channels + c;
                let normalised = (x.data[offset] - mean as f32) * inverse;
                output.data[offset] = normalised * weight[c] + bias[c];
            }
        }
    }
    output
}

/// `nn.LayerNorm` over the last axis, with affine weight and bias.
pub fn layer_norm(x: &Tensor, weight: &[f32], bias: &[f32], eps: f32) -> Tensor {
    let features = *x.shape.last().unwrap();
    assert_eq!(weight.len(), features);
    let rows = x.numel() / features;
    let mut output = x.clone();
    for row in 0..rows {
        let base = row * features;
        let mut mean = 0.0f64;
        for feature in 0..features {
            mean += x.data[base + feature] as f64;
        }
        mean /= features as f64;
        let mut variance = 0.0f64;
        for feature in 0..features {
            let delta = x.data[base + feature] as f64 - mean;
            variance += delta * delta;
        }
        variance /= features as f64;
        let inverse = 1.0 / ((variance as f32) + eps).sqrt();
        for feature in 0..features {
            let normalised = (x.data[base + feature] - mean as f32) * inverse;
            output.data[base + feature] = normalised * weight[feature] + bias[feature];
        }
    }
    output
}

/// `nn.Linear` over the last axis.
pub fn linear(x: &Tensor, weight: &[f32], bias: &[f32]) -> Tensor {
    let features = *x.shape.last().unwrap();
    let out_features = weight.len() / features;
    assert_eq!(weight.len() % features, 0, "linear weight shape");
    assert_eq!(bias.len(), out_features);
    let rows = x.numel() / features;
    let mut shape = x.shape.clone();
    *shape.last_mut().unwrap() = out_features;
    let mut output = Tensor::zeros(shape);
    for row in 0..rows {
        let base = row * features;
        let destination = row * out_features;
        for out in 0..out_features {
            let mut sum = bias[out];
            let weight_row = out * features;
            for feature in 0..features {
                sum += x.data[base + feature] * weight[weight_row + feature];
            }
            output.data[destination + out] = sum;
        }
    }
    output
}

/// LayerScale with `channel_last = True`: `x` is `[B, T, C]` and the scale
/// multiplies the last axis. The transformer's gamma uses this form.
pub fn layer_scale_last(x: &Tensor, scale: &[f32]) -> Tensor {
    let channels = *x.shape.last().unwrap();
    assert_eq!(scale.len(), channels);
    let rows = x.numel() / channels;
    let mut output = x.clone();
    for row in 0..rows {
        for channel in 0..channels {
            output.data[row * channels + channel] *= scale[channel];
        }
    }
    output
}

/// LayerScale with `channel_last = False`: `x` is `[B, C, ...]` and the scale
/// multiplies the second axis. DConv uses this form.
pub fn layer_scale(x: &Tensor, scale: &[f32]) -> Tensor {
    let channels = x.dim(1);
    assert_eq!(scale.len(), channels);
    let spatial: usize = x.shape[2..].iter().product();
    let batch = x.dim(0);
    let mut output = x.clone();
    for index in 0..batch {
        for channel in 0..channels {
            let start = (index * channels + channel) * spatial;
            let factor = scale[channel];
            for offset in 0..spatial {
                output.data[start + offset] *= factor;
            }
        }
    }
    output
}

/// `nn.Conv1d` over `[N, C, T]`.
pub fn conv1d(
    x: &Tensor,
    weight: &[f32],
    bias: &[f32],
    out_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Tensor {
    let batch = x.dim(0);
    let in_channels = x.dim(1);
    let time = x.dim(2);
    let effective = dilation * (kernel - 1) + 1;
    assert!(time + 2 * padding >= effective, "conv1d input too short");
    let out_time = (time + 2 * padding - effective) / stride + 1;
    let mut output = Tensor::zeros(vec![batch, out_channels, out_time]);
    for index in 0..batch {
        for channel in 0..out_channels {
            for out in 0..out_time {
                let mut sum = bias[channel];
                for input_channel in 0..in_channels {
                    for tap in 0..kernel {
                        let position = out * stride + tap * dilation;
                        if position < padding {
                            continue;
                        }
                        let position = position - padding;
                        if position >= time {
                            continue;
                        }
                        let weight_index = ((channel * in_channels + input_channel) * kernel) + tap;
                        sum += x.data[(index * in_channels + input_channel) * time + position]
                            * weight[weight_index];
                    }
                }
                output.data[(index * out_channels + channel) * out_time + out] = sum;
            }
        }
    }
    output
}

/// `nn.Conv2d` over `[N, C, H, W]` with zero padding. `kernel`, `stride` and
/// `pad` are `(height, width)`.
pub fn conv2d(
    x: &Tensor,
    weight: &[f32],
    bias: &[f32],
    out_channels: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
) -> Tensor {
    let batch = x.dim(0);
    let in_channels = x.dim(1);
    let height = x.dim(2);
    let width = x.dim(3);
    let (kernel_h, kernel_w) = kernel;
    let (stride_h, stride_w) = stride;
    let (pad_h, pad_w) = pad;
    let out_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    let out_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    let mut output = Tensor::zeros(vec![batch, out_channels, out_height, out_width]);
    for index in 0..batch {
        for channel in 0..out_channels {
            for out_h in 0..out_height {
                for out_w in 0..out_width {
                    let mut sum = bias[channel];
                    for input_channel in 0..in_channels {
                        for tap_h in 0..kernel_h {
                            let row = out_h * stride_h + tap_h;
                            if row < pad_h {
                                continue;
                            }
                            let row = row - pad_h;
                            if row >= height {
                                continue;
                            }
                            for tap_w in 0..kernel_w {
                                let column = out_w * stride_w + tap_w;
                                if column < pad_w {
                                    continue;
                                }
                                let column = column - pad_w;
                                if column >= width {
                                    continue;
                                }
                                let weight_index =
                                    ((channel * in_channels + input_channel) * kernel_h + tap_h)
                                        * kernel_w
                                        + tap_w;
                                sum += x.data[((index * in_channels + input_channel) * height
                                    + row)
                                    * width
                                    + column]
                                    * weight[weight_index];
                            }
                        }
                    }
                    output.data[((index * out_channels + channel) * out_height + out_h)
                        * out_width
                        + out_w] = sum;
                }
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_norm_matches_a_hand_computation() {
        // One group, two channels, two positions: normalise all four values.
        let x = Tensor::new(vec![1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let out = group_norm(&x, 1, &[1.0, 1.0], &[0.0, 0.0], 0.0);
        let mean = 2.5;
        let std = (1.25f32).sqrt();
        assert!((out.data[0] - (1.0 - mean) / std).abs() < 1e-6);
        assert!((out.data[3] - (4.0 - mean) / std).abs() < 1e-6);
    }

    #[test]
    fn glu_splits_the_channel_axis() {
        let x = Tensor::new(vec![1, 4, 1], vec![1.0, 2.0, 0.0, 0.0]);
        let out = glu(&x);
        // sigmoid(0) = 0.5
        assert_eq!(out.shape, vec![1, 2, 1]);
        assert!((out.data[0] - 0.5).abs() < 1e-6);
        assert!((out.data[1] - 1.0).abs() < 1e-6);
    }
}
