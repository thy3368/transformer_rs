use super::attention::causal_attention;
use super::layers::{layer_norm, layer_norm_backward, linear, linear_backward};
use super::{
    HiddenState, Logits, ModelConfig, ModelSnapshot, ParameterSnapshot, Tensor, TokenizerConfig,
    TransformerError,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Clone, Debug)]
struct Parameter {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}
#[derive(Clone, Debug)]
struct LayerCache {
    x: Vec<f32>,
    qkv: Vec<f32>,
    probs: Vec<f32>,
    context: Vec<f32>,
    res1: Vec<f32>,
    m1: Vec<f32>,
    i1: Vec<f32>,
    ff1: Vec<f32>,
    relu: Vec<f32>,
    res2: Vec<f32>,
    m2: Vec<f32>,
    i2: Vec<f32>,
}
#[derive(Clone, Debug)]
struct ForwardCache {
    ids: Vec<u32>,
    layers: Vec<LayerCache>,
    hidden: Vec<f32>,
}
#[derive(Clone, Debug)]
pub struct ForwardOutput {
    pub logits: Logits,
    cache: ForwardCache,
}
#[derive(Clone, Debug)]
pub struct Gradients {
    pub(crate) values: Vec<Vec<f32>>,
}
#[derive(Clone, Debug)]
pub struct TransformerModel {
    config: ModelConfig,
    parameters: Vec<Parameter>,
}

impl TransformerModel {
    pub fn new(config: ModelConfig, seed: u64) -> Result<Self, TransformerError> {
        config.validate()?;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut p = Vec::new();
        let mut add = |name: String, shape: Vec<usize>, fan_in: usize, identity: bool| {
            let n = shape.iter().product();
            let lim = (6.0 / (fan_in + shape.last().copied().unwrap_or(1)) as f32).sqrt();
            let values = if identity {
                vec![1.0; n]
            } else {
                (0..n).map(|_| rng.gen_range(-lim..=lim)).collect()
            };
            p.push(Parameter {
                name,
                shape,
                values,
            });
        };
        add(
            "embedding.weight".into(),
            vec![config.vocab_size, config.d_model],
            config.vocab_size,
            false,
        );
        for l in 0..config.num_layers {
            let n = format!("decoder.layers.{l}");
            add(
                format!("{n}.qkv.weight"),
                vec![config.d_model, 3 * config.d_model],
                config.d_model,
                false,
            );
            add(format!("{n}.qkv.bias"), vec![3 * config.d_model], 1, false);
            add(
                format!("{n}.attention_output.weight"),
                vec![config.d_model, config.d_model],
                config.d_model,
                false,
            );
            add(
                format!("{n}.attention_output.bias"),
                vec![config.d_model],
                1,
                false,
            );
            add(format!("{n}.norm1.weight"), vec![config.d_model], 1, true);
            add(format!("{n}.norm1.bias"), vec![config.d_model], 1, false);
            add(
                format!("{n}.ffn.linear1.weight"),
                vec![config.d_model, config.d_ff],
                config.d_model,
                false,
            );
            add(format!("{n}.ffn.linear1.bias"), vec![config.d_ff], 1, false);
            add(
                format!("{n}.ffn.linear2.weight"),
                vec![config.d_ff, config.d_model],
                config.d_ff,
                false,
            );
            add(
                format!("{n}.ffn.linear2.bias"),
                vec![config.d_model],
                1,
                false,
            );
            add(format!("{n}.norm2.weight"), vec![config.d_model], 1, true);
            add(format!("{n}.norm2.bias"), vec![config.d_model], 1, false);
        }
        add(
            "output_projection.weight".into(),
            vec![config.d_model, config.vocab_size],
            config.d_model,
            false,
        );
        add(
            "output_projection.bias".into(),
            vec![config.vocab_size],
            1,
            false,
        );
        Ok(Self {
            config,
            parameters: p,
        })
    }
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
    fn base(&self, l: usize) -> usize {
        1 + l * 12
    }
    pub fn embed_token_ids(&self, ids: &[u32]) -> Result<HiddenState, TransformerError> {
        if ids.iter().any(|&id| id as usize >= self.config.vocab_size) {
            return Err(TransformerError::InvalidData(
                "token outside vocabulary".into(),
            ));
        }
        let d = self.config.d_model;
        let embedding = &self.parameters[0].values;
        let mut values = Vec::with_capacity(ids.len() * d);
        for &id in ids {
            let start = id as usize * d;
            values.extend_from_slice(&embedding[start..start + d]);
        }
        Ok(HiddenState(Tensor::new(values, vec![ids.len(), d])?))
    }
    pub fn forward(&self, ids: &[u32]) -> Result<ForwardOutput, TransformerError> {
        let s = ids.len();
        let d = self.config.d_model;
        if s == 0 || s > self.config.max_seq_len {
            return Err(TransformerError::InvalidData(
                "sequence length outside model context".into(),
            ));
        }
        let mut x = self.embed_token_ids(ids)?.0.data;
        let scale = (d as f32).sqrt();
        for r in 0..s {
            for c in 0..d {
                let angle = r as f32 / 10000f32.powf((2 * (c / 2)) as f32 / d as f32);
                let pe = if c % 2 == 0 { angle.sin() } else { angle.cos() };
                x[r * d + c] = x[r * d + c] * scale + pe;
            }
        }
        let mut caches = Vec::new();
        for l in 0..self.config.num_layers {
            let b = self.base(l);
            let qkv = linear(
                &x,
                s,
                d,
                &self.parameters[b].values,
                3 * d,
                &self.parameters[b + 1].values,
            );
            let (q, k, v) = split_qkv(&qkv, s, d);
            let (context, probs) = causal_attention(
                &q,
                &k,
                &v,
                s,
                self.config.num_heads,
                d / self.config.num_heads,
            );
            let attn = linear(
                &context,
                s,
                d,
                &self.parameters[b + 2].values,
                d,
                &self.parameters[b + 3].values,
            );
            let res1: add::Output = add::sum(&x, &attn);
            let (n1, m1, i1) = layer_norm(
                &res1,
                s,
                d,
                &self.parameters[b + 4].values,
                &self.parameters[b + 5].values,
            );
            let ff1 = linear(
                &n1,
                s,
                d,
                &self.parameters[b + 6].values,
                self.config.d_ff,
                &self.parameters[b + 7].values,
            );
            let relu: Vec<f32> = ff1.iter().map(|v| v.max(0.0)).collect();
            let ff2 = linear(
                &relu,
                s,
                self.config.d_ff,
                &self.parameters[b + 8].values,
                d,
                &self.parameters[b + 9].values,
            );
            let res2 = add::sum(&n1, &ff2);
            let (n2, m2, i2) = layer_norm(
                &res2,
                s,
                d,
                &self.parameters[b + 10].values,
                &self.parameters[b + 11].values,
            );
            caches.push(LayerCache {
                x: x.clone(),
                qkv,
                probs,
                context,
                res1,
                m1,
                i1,
                ff1,
                relu,
                res2,
                m2,
                i2,
            });
            x = n2;
        }
        let b = 1 + self.config.num_layers * 12;
        let logits = linear(
            &x,
            s,
            d,
            &self.parameters[b].values,
            self.config.vocab_size,
            &self.parameters[b + 1].values,
        );
        Ok(ForwardOutput {
            logits: Logits(Tensor::new(logits, vec![s, self.config.vocab_size])?),
            cache: ForwardCache {
                ids: ids.to_vec(),
                layers: caches,
                hidden: x,
            },
        })
    }
    pub fn backward(
        &self,
        out: &ForwardOutput,
        dlogits: &[f32],
    ) -> Result<Gradients, TransformerError> {
        let s = out.cache.ids.len();
        let d = self.config.d_model;
        let v = self.config.vocab_size;
        if dlogits.len() != s * v {
            return Err(TransformerError::InvalidData(
                "loss gradient shape mismatch".into(),
            ));
        }
        let mut grads: Vec<Vec<f32>> = self
            .parameters
            .iter()
            .map(|p| vec![0.0; p.values.len()])
            .collect();
        let ob = 1 + self.config.num_layers * 12;
        let (mut dx, dw, db) = linear_backward(
            &out.cache.hidden,
            dlogits,
            s,
            d,
            &self.parameters[ob].values,
            v,
        );
        grads[ob] = dw;
        grads[ob + 1] = db;
        for l in (0..self.config.num_layers).rev() {
            let b = self.base(l);
            let c = &out.cache.layers[l];
            let (dres2, dg2, db2) = layer_norm_backward(
                &c.res2,
                &dx,
                s,
                d,
                &self.parameters[b + 10].values,
                &c.m2,
                &c.i2,
            );
            grads[b + 10] = dg2;
            grads[b + 11] = db2;
            let (drelu, dw2, dbf2) = linear_backward(
                &c.relu,
                &dres2,
                s,
                self.config.d_ff,
                &self.parameters[b + 8].values,
                d,
            );
            grads[b + 8] = dw2;
            grads[b + 9] = dbf2;
            let mut dff = drelu;
            for (i, value) in dff.iter_mut().enumerate() {
                if c.ff1[i] <= 0.0 {
                    *value = 0.0;
                }
            }
            let n1 = layer_norm(
                &c.res1,
                s,
                d,
                &self.parameters[b + 4].values,
                &self.parameters[b + 5].values,
            )
            .0;
            let (dn_ff, dw1, dbf1) = linear_backward(
                &n1,
                &dff,
                s,
                d,
                &self.parameters[b + 6].values,
                self.config.d_ff,
            );
            grads[b + 6] = dw1;
            grads[b + 7] = dbf1;
            let mut dn1 = dres2.clone();
            for i in 0..dn1.len() {
                dn1[i] += dn_ff[i];
            }
            let (dres1, dg1, db1) = layer_norm_backward(
                &c.res1,
                &dn1,
                s,
                d,
                &self.parameters[b + 4].values,
                &c.m1,
                &c.i1,
            );
            grads[b + 4] = dg1;
            grads[b + 5] = db1;
            let (dcontext, dwo, dbo) =
                linear_backward(&c.context, &dres1, s, d, &self.parameters[b + 2].values, d);
            grads[b + 2] = dwo;
            grads[b + 3] = dbo;
            let dqkv = attention_backward(
                &c.qkv,
                &c.probs,
                &dcontext,
                s,
                self.config.num_heads,
                d / self.config.num_heads,
            );
            let (dxin, dwq, dbq) =
                linear_backward(&c.x, &dqkv, s, d, &self.parameters[b].values, 3 * d);
            grads[b] = dwq;
            grads[b + 1] = dbq;
            for i in 0..dx.len() {
                dx[i] = dres1[i] + dxin[i];
            }
        }
        let scale = (d as f32).sqrt();
        for (r, &id) in out.cache.ids.iter().enumerate() {
            for c in 0..d {
                grads[0][id as usize * d + c] += dx[r * d + c] * scale;
            }
        }
        Ok(Gradients { values: grads })
    }
    pub fn snapshot(&self, tokenizer_config: TokenizerConfig) -> ModelSnapshot {
        ModelSnapshot {
            model_config: self.config.clone(),
            tokenizer_config,
            parameters: self
                .parameters
                .iter()
                .map(|p| ParameterSnapshot {
                    name: p.name.clone(),
                    shape: p.shape.clone(),
                    values: p.values.clone(),
                })
                .collect(),
        }
    }
    pub fn from_snapshot(s: ModelSnapshot) -> Result<Self, TransformerError> {
        if s.tokenizer_config.vocab_size != s.model_config.vocab_size {
            return Err(TransformerError::Checkpoint(
                "tokenizer and model vocabulary sizes differ".into(),
            ));
        }
        let expected = Self::new(s.model_config.clone(), 0)?;
        if s.parameters.len() != expected.parameters.len() {
            return Err(TransformerError::Checkpoint(
                "parameter count mismatch".into(),
            ));
        }
        let mut seen = std::collections::HashSet::new();
        for (e, p) in expected.parameters.iter().zip(&s.parameters) {
            if !seen.insert(&p.name) {
                return Err(TransformerError::Checkpoint(format!(
                    "duplicate parameter {}",
                    p.name
                )));
            }
            if e.name != p.name
                || e.shape != p.shape
                || p.values.len() != p.shape.iter().product::<usize>()
            {
                return Err(TransformerError::Checkpoint(format!(
                    "parameter mismatch for {}",
                    p.name
                )));
            }
            if p.values.iter().any(|x| !x.is_finite()) {
                return Err(TransformerError::Checkpoint(format!(
                    "non-finite parameter {}",
                    p.name
                )));
            }
        }
        Ok(Self {
            config: s.model_config,
            parameters: s
                .parameters
                .into_iter()
                .map(|p| Parameter {
                    name: p.name,
                    shape: p.shape,
                    values: p.values,
                })
                .collect(),
        })
    }
    pub(crate) fn apply(
        &mut self,
        grads: &Gradients,
        m: &mut [Vec<f32>],
        v: &mut [Vec<f32>],
        step: usize,
        lr: f32,
    ) {
        for (pi, p) in self.parameters.iter_mut().enumerate() {
            for i in 0..p.values.len() {
                m[pi][i] = 0.9 * m[pi][i] + 0.1 * grads.values[pi][i];
                v[pi][i] = 0.999 * v[pi][i] + 0.001 * grads.values[pi][i] * grads.values[pi][i];
                let mh = m[pi][i] / (1.0 - 0.9f32.powi(step as i32));
                let vh = v[pi][i] / (1.0 - 0.999f32.powi(step as i32));
                p.values[i] -= lr * mh / (vh.sqrt() + 1e-8);
            }
        }
    }
    pub(crate) fn parameter_sizes(&self) -> Vec<usize> {
        self.parameters.iter().map(|p| p.values.len()).collect()
    }
}
fn split_qkv(qkv: &[f32], s: usize, d: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut q = vec![0.0; s * d];
    let mut k = q.clone();
    let mut v = q.clone();
    for r in 0..s {
        q[r * d..(r + 1) * d].copy_from_slice(&qkv[r * 3 * d..r * 3 * d + d]);
        k[r * d..(r + 1) * d].copy_from_slice(&qkv[r * 3 * d + d..r * 3 * d + 2 * d]);
        v[r * d..(r + 1) * d].copy_from_slice(&qkv[r * 3 * d + 2 * d..(r + 1) * 3 * d]);
    }
    (q, k, v)
}
fn attention_backward(
    qkv: &[f32],
    p: &[f32],
    dy: &[f32],
    s: usize,
    h: usize,
    hd: usize,
) -> Vec<f32> {
    let d = h * hd;
    let (q, k, v) = split_qkv(qkv, s, d);
    let mut dq = vec![0.0; s * d];
    let mut dk = dq.clone();
    let mut dv = dq.clone();
    let scale = (hd as f32).sqrt().recip();
    for head in 0..h {
        for i in 0..s {
            let mut dp = vec![0.0; i + 1];
            for j in 0..=i {
                for z in 0..hd {
                    dp[j] += dy[i * d + head * hd + z] * v[j * d + head * hd + z];
                    dv[j * d + head * hd + z] +=
                        p[(head * s + i) * s + j] * dy[i * d + head * hd + z];
                }
            }
            let dot: f32 = (0..=i).map(|j| dp[j] * p[(head * s + i) * s + j]).sum();
            for j in 0..=i {
                let ds = p[(head * s + i) * s + j] * (dp[j] - dot) * scale;
                for z in 0..hd {
                    dq[i * d + head * hd + z] += ds * k[j * d + head * hd + z];
                    dk[j * d + head * hd + z] += ds * q[i * d + head * hd + z];
                }
            }
        }
    }
    let mut out = vec![0.0; s * 3 * d];
    for r in 0..s {
        out[r * 3 * d..r * 3 * d + d].copy_from_slice(&dq[r * d..(r + 1) * d]);
        out[r * 3 * d + d..r * 3 * d + 2 * d].copy_from_slice(&dk[r * d..(r + 1) * d]);
        out[r * 3 * d + 2 * d..(r + 1) * 3 * d].copy_from_slice(&dv[r * d..(r + 1) * d]);
    }
    out
}
mod add {
    pub type Output = Vec<f32>;
    pub fn sum(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    }
}
