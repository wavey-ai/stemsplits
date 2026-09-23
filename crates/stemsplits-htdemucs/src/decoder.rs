//! `HDecLayer` from `demucs/hdemucs.py`.
//!
//! Add the skip, rewrite through a GLU, run the DConv residual, then a
//! transposed convolution. The frequency branch crops the padded frequency
//! axis symmetrically; the time branch crops to the encoder's length. GELU is
//! applied except in the final layer.

use anyhow::{Context, Result};

use crate::dconv::DConv;
use crate::ops::{conv_transpose1d, conv_transpose2d, gelu, glu};
use crate::tensor::Tensor;
use stemsplits_model::Weights;

pub struct DecLayer {
    freq: bool,
    last: bool,
    conv_tr_weight: Vec<f32>,
    conv_tr_bias: Vec<f32>,
    dconv: DConv,
    rewrite_weight: Vec<f32>,
    rewrite_bias: Vec<f32>,
    pad: usize,
}

fn load(weights: &Weights, name: &str) -> Result<Vec<f32>> {
    weights
        .tensor(name)
        .with_context(|| format!("missing weight {name}"))
        .map(|tensor| tensor.data().to_vec())
}

impl DecLayer {
    pub fn load(
        weights: &Weights,
        prefix: &str,
        freq: bool,
        last: bool,
        chin: usize,
        compress: usize,
        depth: usize,
    ) -> Result<Self> {
        Ok(Self {
            freq,
            last,
            conv_tr_weight: load(weights, &format!("{prefix}.conv_tr.weight"))?,
            conv_tr_bias: load(weights, &format!("{prefix}.conv_tr.bias"))?,
            dconv: DConv::load(weights, &format!("{prefix}.dconv"), chin, compress, depth)?,
            rewrite_weight: load(weights, &format!("{prefix}.rewrite.weight"))?,
            rewrite_bias: load(weights, &format!("{prefix}.rewrite.bias"))?,
            pad: 2,
        })
    }

    /// `x` is the decoder input, `skip` the matching encoder output. `length`
    /// is the time branch's target; the frequency branch ignores it.
    pub fn forward(&self, x: &Tensor, skip: &Tensor, length: usize) -> Tensor {
        let x = add(x, skip);
        let y = glu(&conv_rewrite(
            &x,
            &self.rewrite_weight,
            &self.rewrite_bias,
            self.freq,
        ));

        let y = if self.freq {
            let batch = y.dim(0);
            let channels = y.dim(1);
            let frequency = y.dim(2);
            let time = y.dim(3);
            let flattened = Tensor::new(
                vec![batch * frequency, channels, time],
                permute_0213(&y).data,
            );
            let convolved = self.dconv.forward(&flattened);
            unpermute_0213(&Tensor::new(
                vec![batch, frequency, channels, time],
                convolved.data,
            ))
        } else {
            self.dconv.forward(&y)
        };

        let out_channels = self.conv_tr_bias.len();
        let z = if self.freq {
            let z = conv_transpose2d(
                &y,
                &self.conv_tr_weight,
                &self.conv_tr_bias,
                out_channels,
                (8, 1),
                (4, 1),
            );
            crop_frequency(&z, self.pad)
        } else {
            let z = conv_transpose1d(
                &y,
                &self.conv_tr_weight,
                &self.conv_tr_bias,
                out_channels,
                8,
                4,
            );
            crop_time(&z, self.pad, length)
        };
        if self.last {
            z
        } else {
            gelu(&z)
        }
    }
}

/// The 1x1 or 3x3 rewrite convolution, `norm1` being an identity.
fn conv_rewrite(x: &Tensor, weight: &[f32], bias: &[f32], freq: bool) -> Tensor {
    let out_channels = bias.len();
    if freq {
        crate::ops::conv2d(x, weight, bias, out_channels, (3, 3), (1, 1), (1, 1))
    } else {
        crate::ops::conv1d(x, weight, bias, out_channels, 3, 1, 1, 1)
    }
}

fn crop_frequency(x: &Tensor, pad: usize) -> Tensor {
    let batch = x.dim(0);
    let channels = x.dim(1);
    let height = x.dim(2);
    let width = x.dim(3);
    let out_height = height - 2 * pad;
    let mut output = Tensor::zeros(vec![batch, channels, out_height, width]);
    for index in 0..batch {
        for channel in 0..channels {
            for row in 0..out_height {
                let source = ((index * channels + channel) * height + row + pad) * width;
                let destination = ((index * channels + channel) * out_height + row) * width;
                output.data[destination..(destination + width)]
                    .copy_from_slice(&x.data[source..(source + width)]);
            }
        }
    }
    output
}

fn crop_time(x: &Tensor, pad: usize, length: usize) -> Tensor {
    let batch = x.dim(0);
    let channels = x.dim(1);
    let time = x.dim(2);
    let mut output = Tensor::zeros(vec![batch, channels, length]);
    for index in 0..batch {
        for channel in 0..channels {
            let source = (index * channels + channel) * time + pad;
            let destination = (index * channels + channel) * length;
            output.data[destination..(destination + length)]
                .copy_from_slice(&x.data[source..(source + length)]);
        }
    }
    output
}

fn add(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape, b.shape, "decoder skip shape");
    Tensor {
        shape: a.shape.clone(),
        data: a.data.iter().zip(&b.data).map(|(l, r)| l + r).collect(),
    }
}

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
