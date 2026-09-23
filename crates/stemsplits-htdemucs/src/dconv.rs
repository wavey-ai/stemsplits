//! The DConv residual branch from `demucs/demucs.py`.
//!
//! Two dilated convolutions with GroupNorm(1) and GLU, each wrapped in a
//! LayerScale residual. It runs on `[N, C, T]`; the frequency branch flattens
//! its frequency axis into the batch before calling it.

use anyhow::{Context, Result};

use crate::ops::{conv1d, gelu, glu, group_norm, layer_scale};
use crate::tensor::Tensor;
use stemsplits_model::Weights;

const NORM_EPS: f32 = 1e-5;

struct DConvLayer {
    conv1_weight: Vec<f32>,
    conv1_bias: Vec<f32>,
    norm1_weight: Vec<f32>,
    norm1_bias: Vec<f32>,
    conv2_weight: Vec<f32>,
    conv2_bias: Vec<f32>,
    norm2_weight: Vec<f32>,
    norm2_bias: Vec<f32>,
    scale: Vec<f32>,
    dilation: usize,
    hidden: usize,
    channels: usize,
}

pub struct DConv {
    layers: Vec<DConvLayer>,
}

fn load(weights: &Weights, name: &str) -> Result<Vec<f32>> {
    weights
        .tensor(name)
        .with_context(|| format!("missing weight {name}"))
        .map(|tensor| tensor.data().to_vec())
}

impl DConv {
    /// `prefix` is the module path, e.g. `encoder.0.dconv`.
    pub fn load(
        weights: &Weights,
        prefix: &str,
        channels: usize,
        compress: usize,
        depth: usize,
    ) -> Result<Self> {
        let hidden = channels / compress;
        let mut layers = Vec::with_capacity(depth);
        for index in 0..depth {
            let base = format!("{prefix}.layers.{index}");
            layers.push(DConvLayer {
                conv1_weight: load(weights, &format!("{base}.0.weight"))?,
                conv1_bias: load(weights, &format!("{base}.0.bias"))?,
                norm1_weight: load(weights, &format!("{base}.1.weight"))?,
                norm1_bias: load(weights, &format!("{base}.1.bias"))?,
                conv2_weight: load(weights, &format!("{base}.3.weight"))?,
                conv2_bias: load(weights, &format!("{base}.3.bias"))?,
                norm2_weight: load(weights, &format!("{base}.4.weight"))?,
                norm2_bias: load(weights, &format!("{base}.4.bias"))?,
                scale: load(weights, &format!("{base}.6.scale"))?,
                dilation: 1 << index,
                hidden,
                channels,
            });
        }
        Ok(Self { layers })
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut current = x.clone();
        for layer in &self.layers {
            let residual = layer.forward(&current);
            current = add(&current, &residual);
        }
        current
    }
}

impl DConvLayer {
    fn forward(&self, x: &Tensor) -> Tensor {
        // kernel 3, so `dilation * (kernel // 2)` is just the dilation.
        let padding = self.dilation;
        let y = conv1d(
            x,
            &self.conv1_weight,
            &self.conv1_bias,
            self.hidden,
            3,
            1,
            padding,
            self.dilation,
        );
        let y = group_norm(&y, 1, &self.norm1_weight, &self.norm1_bias, NORM_EPS);
        let y = gelu(&y);
        let y = conv1d(
            &y,
            &self.conv2_weight,
            &self.conv2_bias,
            2 * self.channels,
            1,
            1,
            0,
            1,
        );
        let y = group_norm(&y, 1, &self.norm2_weight, &self.norm2_bias, NORM_EPS);
        let y = glu(&y);
        layer_scale(&y, &self.scale)
    }
}

fn add(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape, b.shape, "residual shape");
    Tensor {
        shape: a.shape.clone(),
        data: a
            .data
            .iter()
            .zip(&b.data)
            .map(|(left, right)| left + right)
            .collect(),
    }
}
