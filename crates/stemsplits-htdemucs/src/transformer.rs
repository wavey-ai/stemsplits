//! `CrossTransformerEncoder` from `demucs/transformer.py`.
//!
//! Five layers on each branch. Even layers self-attend; odd layers
//! cross-attend, the frequency branch querying the time branch and the time
//! branch querying the *pre-update* frequency branch. The frequency branch
//! carries a 2D (frequency × time) sinusoidal positional embedding, the time
//! branch a 1D one.
//!
//! Sequences are `[B, T, C]` here, matching `batch_first = True`.

use anyhow::{Context, Result};

use crate::ops::{gelu, group_norm_sequence, layer_norm, layer_scale_last, linear};
use crate::tensor::Tensor;
use stemsplits_model::Weights;

const LAYER_NORM_EPS: f32 = 1e-5;
const NORM_OUT_EPS: f32 = 1e-5;

struct AttentionWeights {
    in_proj_weight: Vec<f32>,
    in_proj_bias: Vec<f32>,
    out_proj_weight: Vec<f32>,
    out_proj_bias: Vec<f32>,
}

struct FeedForward {
    linear1_weight: Vec<f32>,
    linear1_bias: Vec<f32>,
    linear2_weight: Vec<f32>,
    linear2_bias: Vec<f32>,
}

struct Norms {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

struct ClassicLayer {
    norm1: Norms,
    norm2: Norms,
    norm_out: Norms,
    gamma1: Vec<f32>,
    gamma2: Vec<f32>,
    attention: AttentionWeights,
    feed_forward: FeedForward,
}

struct CrossLayer {
    norm1: Norms,
    norm2: Norms,
    norm3: Norms,
    norm_out: Norms,
    gamma1: Vec<f32>,
    gamma2: Vec<f32>,
    attention: AttentionWeights,
    feed_forward: FeedForward,
}

enum Layer {
    Classic(ClassicLayer),
    Cross(CrossLayer),
}

impl Layer {
    fn forward(&self, x: &Tensor, other: Option<&Tensor>, heads: usize) -> Tensor {
        match self {
            Layer::Classic(layer) => layer.forward(x, heads),
            Layer::Cross(layer) => {
                layer.forward(x, other.expect("cross layer needs its pair"), heads)
            }
        }
    }
}

impl ClassicLayer {
    fn forward(&self, x: &Tensor, heads: usize) -> Tensor {
        let normed = layer_norm(x, &self.norm1.weight, &self.norm1.bias, LAYER_NORM_EPS);
        let attended = attention(&normed, &normed, &normed, &self.attention, heads);
        let x = add(x, &layer_scale_last(&attended, &self.gamma1));

        let normed = layer_norm(&x, &self.norm2.weight, &self.norm2.bias, LAYER_NORM_EPS);
        let forwarded = feed_forward(&normed, &self.feed_forward);
        let x = add(&x, &layer_scale_last(&forwarded, &self.gamma2));
        group_norm_sequence(&x, &self.norm_out.weight, &self.norm_out.bias, NORM_OUT_EPS)
    }
}

impl CrossLayer {
    fn forward(&self, q: &Tensor, k: &Tensor, heads: usize) -> Tensor {
        let query = layer_norm(q, &self.norm1.weight, &self.norm1.bias, LAYER_NORM_EPS);
        let key = layer_norm(k, &self.norm2.weight, &self.norm2.bias, LAYER_NORM_EPS);
        let attended = attention(&query, &key, &key, &self.attention, heads);
        let q = add(q, &layer_scale_last(&attended, &self.gamma1));

        let normed = layer_norm(&q, &self.norm3.weight, &self.norm3.bias, LAYER_NORM_EPS);
        let forwarded = feed_forward(&normed, &self.feed_forward);
        let q = add(&q, &layer_scale_last(&forwarded, &self.gamma2));
        group_norm_sequence(&q, &self.norm_out.weight, &self.norm_out.bias, NORM_OUT_EPS)
    }
}

pub struct CrossTransformer {
    dim: usize,
    heads: usize,
    max_period: f32,
    weight_pos_embed: f32,
    norm_in: Norms,
    norm_in_t: Norms,
    layers: Vec<Layer>,
    layers_t: Vec<Layer>,
}

fn load(weights: &Weights, name: &str) -> Result<Vec<f32>> {
    weights
        .tensor(name)
        .with_context(|| format!("missing weight {name}"))
        .map(|tensor| tensor.data().to_vec())
}

fn load_norms(weights: &Weights, prefix: &str) -> Result<Norms> {
    Ok(Norms {
        weight: load(weights, &format!("{prefix}.weight"))?,
        bias: load(weights, &format!("{prefix}.bias"))?,
    })
}

fn load_attention(weights: &Weights, prefix: &str) -> Result<AttentionWeights> {
    Ok(AttentionWeights {
        in_proj_weight: load(weights, &format!("{prefix}.in_proj_weight"))?,
        in_proj_bias: load(weights, &format!("{prefix}.in_proj_bias"))?,
        out_proj_weight: load(weights, &format!("{prefix}.out_proj.weight"))?,
        out_proj_bias: load(weights, &format!("{prefix}.out_proj.bias"))?,
    })
}

fn load_feed_forward(weights: &Weights, prefix: &str) -> Result<FeedForward> {
    Ok(FeedForward {
        linear1_weight: load(weights, &format!("{prefix}.linear1.weight"))?,
        linear1_bias: load(weights, &format!("{prefix}.linear1.bias"))?,
        linear2_weight: load(weights, &format!("{prefix}.linear2.weight"))?,
        linear2_bias: load(weights, &format!("{prefix}.linear2.bias"))?,
    })
}

impl CrossTransformer {
    pub fn load(
        weights: &Weights,
        prefix: &str,
        heads: usize,
        max_period: f32,
        weight_pos_embed: f32,
    ) -> Result<Self> {
        let dim = weights
            .tensor(&format!("{prefix}.norm_in.weight"))
            .context("missing transformer norm_in")?
            .shape()[0];
        let mut layers = Vec::new();
        let mut layers_t = Vec::new();
        for index in 0..5 {
            let classic = index % 2 == 0;
            layers.push(load_layer(
                weights,
                &format!("{prefix}.layers.{index}"),
                classic,
            )?);
            layers_t.push(load_layer(
                weights,
                &format!("{prefix}.layers_t.{index}"),
                classic,
            )?);
        }
        Ok(Self {
            dim,
            heads,
            max_period,
            weight_pos_embed,
            norm_in: load_norms(weights, &format!("{prefix}.norm_in"))?,
            norm_in_t: load_norms(weights, &format!("{prefix}.norm_in_t"))?,
            layers,
            layers_t,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// `x` is `[B, C, Fr, T1]`, `xt` is `[B, C, T2]`; returns the same shapes.
    pub fn forward(&self, x: &Tensor, xt: &Tensor) -> (Tensor, Tensor) {
        let batch = x.dim(0);
        let frequency = x.dim(2);
        let time = x.dim(3);
        let time_t = xt.dim(2);

        let pos_2d = sin_embedding_2d(self.dim, frequency, time, self.max_period);
        let mut x_tokens = arrange_freq(x);
        x_tokens = layer_norm(
            &x_tokens,
            &self.norm_in.weight,
            &self.norm_in.bias,
            LAYER_NORM_EPS,
        );
        add_scaled_in_place(&mut x_tokens, &pos_2d, self.weight_pos_embed);

        let pos_1d = sin_embedding_1d(time_t, self.dim, self.max_period);
        let mut xt_tokens = arrange_time(xt);
        xt_tokens = layer_norm(
            &xt_tokens,
            &self.norm_in_t.weight,
            &self.norm_in_t.bias,
            LAYER_NORM_EPS,
        );
        add_scaled_in_place(&mut xt_tokens, &pos_1d, self.weight_pos_embed);

        for (index, layer) in self.layers.iter().enumerate() {
            let layer_t = &self.layers_t[index];
            match layer {
                Layer::Classic(_) => {
                    x_tokens = layer.forward(&x_tokens, None, self.heads);
                    xt_tokens = layer_t.forward(&xt_tokens, None, self.heads);
                }
                Layer::Cross(_) => {
                    let previous_x = x_tokens.clone();
                    x_tokens = layer.forward(&x_tokens, Some(&xt_tokens), self.heads);
                    xt_tokens = layer_t.forward(&xt_tokens, Some(&previous_x), self.heads);
                }
            }
        }

        let x_out = unarrange_freq(&x_tokens, batch, self.dim, frequency, time);
        let xt_out = unarrange_time(&xt_tokens, batch, self.dim, time_t);
        (x_out, xt_out)
    }
}

fn load_layer(weights: &Weights, prefix: &str, classic: bool) -> Result<Layer> {
    let feed_forward = load_feed_forward(weights, prefix)?;
    let gamma1 = load(weights, &format!("{prefix}.gamma_1.scale"))?;
    let gamma2 = load(weights, &format!("{prefix}.gamma_2.scale"))?;
    if classic {
        Ok(Layer::Classic(ClassicLayer {
            norm1: load_norms(weights, &format!("{prefix}.norm1"))?,
            norm2: load_norms(weights, &format!("{prefix}.norm2"))?,
            norm_out: load_norms(weights, &format!("{prefix}.norm_out"))?,
            gamma1,
            gamma2,
            attention: load_attention(weights, &format!("{prefix}.self_attn"))?,
            feed_forward,
        }))
    } else {
        Ok(Layer::Cross(CrossLayer {
            norm1: load_norms(weights, &format!("{prefix}.norm1"))?,
            norm2: load_norms(weights, &format!("{prefix}.norm2"))?,
            norm3: load_norms(weights, &format!("{prefix}.norm3"))?,
            norm_out: load_norms(weights, &format!("{prefix}.norm_out"))?,
            gamma1,
            gamma2,
            attention: load_attention(weights, &format!("{prefix}.cross_attn"))?,
            feed_forward,
        }))
    }
}

/// `nn.MultiheadAttention` with `batch_first = True`, `need_weights = False`.
fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    weights: &AttentionWeights,
    heads: usize,
) -> Tensor {
    let batch = q.dim(0);
    let dim = q.dim(2);
    let queries = q.dim(1);
    let keys = k.dim(1);
    let head_dim = dim / heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // in_proj_weight is [3 * dim, dim], split into q, k, v.
    let project = |x: &Tensor, offset: usize| -> Vec<f32> {
        let w = &weights.in_proj_weight[offset * dim * dim..(offset + 1) * dim * dim];
        let b = &weights.in_proj_bias[offset * dim..(offset + 1) * dim];
        linear(x, w, b).data
    };
    // Head-major, so each head's queries and keys are contiguous.
    let mut q_proj = to_heads(&project(q, 0), batch, queries, dim, heads);
    let k_proj = to_heads(&project(k, 1), batch, keys, dim, heads);
    let v_proj = to_heads(&project(v, 2), batch, keys, dim, heads);
    // The reference scales the queries once: `q / sqrt(head_dim)`.
    for value in q_proj.iter_mut() {
        *value *= scale;
    }

    let mut head_output = vec![0.0f32; batch * heads * queries * head_dim];
    let mut scores = vec![0.0f32; queries * keys];
    for head in 0..batch * heads {
        let q_head = &q_proj[head * queries * head_dim..(head + 1) * queries * head_dim];
        let k_head = &k_proj[head * keys * head_dim..(head + 1) * keys * head_dim];
        let v_head = &v_proj[head * keys * head_dim..(head + 1) * keys * head_dim];
        let out = &mut head_output[head * queries * head_dim..(head + 1) * queries * head_dim];

        // Transpose the keys so the scores matmul is the vectorisable form.
        let mut keys_transposed = vec![0.0f32; head_dim * keys];
        for key in 0..keys {
            for d in 0..head_dim {
                keys_transposed[d * keys + key] = k_head[key * head_dim + d];
            }
        }
        crate::matmul::matmul(
            q_head,
            &keys_transposed,
            &mut scores,
            queries,
            head_dim,
            keys,
        );
        for query in 0..queries {
            let row = &mut scores[query * keys..(query + 1) * keys];
            let maximum = row.iter().copied().fold(f32::MIN, f32::max);
            let mut sum = 0.0f32;
            for value in row.iter_mut() {
                *value = (*value - maximum).exp();
                sum += *value;
            }
            for value in row.iter_mut() {
                *value /= sum;
            }
        }
        crate::matmul::matmul(&scores, v_head, out, queries, keys, head_dim);
    }

    let merged = from_heads(&head_output, batch, heads, queries, head_dim);
    let attended = Tensor::new(vec![batch, queries, dim], merged);
    linear(&attended, &weights.out_proj_weight, &weights.out_proj_bias)
}

/// `[B, T, C]` -> `[B * H, T, D]`, head-major and contiguous.
fn to_heads(x: &[f32], batch: usize, time: usize, dim: usize, heads: usize) -> Vec<f32> {
    let head_dim = dim / heads;
    let mut output = vec![0.0f32; batch * heads * time * head_dim];
    for b in 0..batch {
        for head in 0..heads {
            for t in 0..time {
                let source = (b * time + t) * dim + head * head_dim;
                let destination = ((b * heads + head) * time + t) * head_dim;
                output[destination..destination + head_dim]
                    .copy_from_slice(&x[source..source + head_dim]);
            }
        }
    }
    output
}

/// `[B * H, T, D]` -> `[B, T, C]`.
fn from_heads(x: &[f32], batch: usize, heads: usize, time: usize, head_dim: usize) -> Vec<f32> {
    let dim = heads * head_dim;
    let mut output = vec![0.0f32; batch * time * dim];
    for b in 0..batch {
        for head in 0..heads {
            for t in 0..time {
                let source = ((b * heads + head) * time + t) * head_dim;
                let destination = (b * time + t) * dim + head * head_dim;
                output[destination..destination + head_dim]
                    .copy_from_slice(&x[source..source + head_dim]);
            }
        }
    }
    output
}

fn feed_forward(x: &Tensor, ff: &FeedForward) -> Tensor {
    let hidden = linear(x, &ff.linear1_weight, &ff.linear1_bias);
    let hidden = gelu(&hidden);
    linear(&hidden, &ff.linear2_weight, &ff.linear2_bias)
}

fn add(a: &Tensor, b: &Tensor) -> Tensor {
    Tensor {
        shape: a.shape.clone(),
        data: a.data.iter().zip(&b.data).map(|(l, r)| l + r).collect(),
    }
}

fn add_scaled_in_place(x: &mut Tensor, addend: &Tensor, scale: f32) {
    for (value, extra) in x.data.iter_mut().zip(&addend.data) {
        *value += scale * extra;
    }
}

/// `[B, C, Fr, T] -> [B, T * Fr, C]`, token index `t * Fr + fr`.
fn arrange_freq(x: &Tensor) -> Tensor {
    let batch = x.dim(0);
    let channels = x.dim(1);
    let frequency = x.dim(2);
    let time = x.dim(3);
    let mut output = Tensor::zeros(vec![batch, time * frequency, channels]);
    for b in 0..batch {
        for t in 0..time {
            for f in 0..frequency {
                let token = t * frequency + f;
                for c in 0..channels {
                    output.data[(b * time * frequency + token) * channels + c] =
                        x.data[((b * channels + c) * frequency + f) * time + t];
                }
            }
        }
    }
    output
}

fn unarrange_freq(
    tokens: &Tensor,
    batch: usize,
    channels: usize,
    frequency: usize,
    time: usize,
) -> Tensor {
    let mut output = Tensor::zeros(vec![batch, channels, frequency, time]);
    for b in 0..batch {
        for t in 0..time {
            for f in 0..frequency {
                let token = t * frequency + f;
                for c in 0..channels {
                    output.data[((b * channels + c) * frequency + f) * time + t] =
                        tokens.data[(b * time * frequency + token) * channels + c];
                }
            }
        }
    }
    output
}

/// `[B, C, T] -> [B, T, C]`.
fn arrange_time(x: &Tensor) -> Tensor {
    let batch = x.dim(0);
    let channels = x.dim(1);
    let time = x.dim(2);
    let mut output = Tensor::zeros(vec![batch, time, channels]);
    for b in 0..batch {
        for t in 0..time {
            for c in 0..channels {
                output.data[(b * time + t) * channels + c] = x.data[(b * channels + c) * time + t];
            }
        }
    }
    output
}

fn unarrange_time(tokens: &Tensor, batch: usize, channels: usize, time: usize) -> Tensor {
    let mut output = Tensor::zeros(vec![batch, channels, time]);
    for b in 0..batch {
        for t in 0..time {
            for c in 0..channels {
                output.data[(b * channels + c) * time + t] =
                    tokens.data[(b * time + t) * channels + c];
            }
        }
    }
    output
}

/// `create_sin_embedding`: `[T, 1, C]`, cosine half then sine half.
fn sin_embedding_1d(time: usize, dim: usize, max_period: f32) -> Tensor {
    let half = dim / 2;
    let mut output = Tensor::zeros(vec![1, time, dim]);
    for t in 0..time {
        for i in 0..half {
            let angle = t as f32 / max_period.powf(i as f32 / (half - 1) as f32);
            output.data[t * dim + i] = angle.cos();
            output.data[t * dim + half + i] = angle.sin();
        }
    }
    output
}

/// `create_2d_sin_embedding`: `[1, T1 * Fr, C]`, frequency half then height
/// half, as the reference's `b c fr t1 -> b (t1 fr) c`.
fn sin_embedding_2d(dim: usize, frequency: usize, time: usize, max_period: f32) -> Tensor {
    let half = dim / 2;
    let frequencies = half / 2;
    let log_period = max_period.ln() / half as f32;
    let div_term: Vec<f32> = (0..frequencies)
        .map(|i| (-((2 * i) as f32) * log_period).exp())
        .collect();

    let mut output = Tensor::zeros(vec![1, time * frequency, dim]);
    for t in 0..time {
        for f in 0..frequency {
            let token = t * frequency + f;
            for (i, &term) in div_term.iter().enumerate() {
                // time axis occupies the first `half` channels
                let angle = t as f32 * term;
                output.data[token * dim + 2 * i] = angle.sin();
                output.data[token * dim + 2 * i + 1] = angle.cos();
                // frequency axis occupies the second `half`
                let angle = f as f32 * term;
                output.data[token * dim + half + 2 * i] = angle.sin();
                output.data[token * dim + half + 2 * i + 1] = angle.cos();
            }
        }
    }
    output
}
