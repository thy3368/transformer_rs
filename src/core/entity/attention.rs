use super::TransformerError;

pub fn softmax(values: &[f32]) -> Result<Vec<f32>, TransformerError> {
    if values.is_empty() || values.iter().any(|x| !x.is_finite()) {
        return Err(TransformerError::InvalidData(
            "softmax requires finite non-empty input".into(),
        ));
    }
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = values.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = out.iter().sum();
    for x in &mut out {
        *x /= sum;
    }
    Ok(out)
}

pub fn causal_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let d = heads * head_dim;
    let mut out = vec![0.0; seq * d];
    let mut probs = vec![0.0; heads * seq * seq];
    let scale = (head_dim as f32).sqrt().recip();
    for h in 0..heads {
        for i in 0..seq {
            let mut scores = vec![0.0; i + 1];
            for j in 0..=i {
                for z in 0..head_dim {
                    scores[j] += q[i * d + h * head_dim + z] * k[j * d + h * head_dim + z];
                }
                scores[j] *= scale;
            }
            let p = softmax(&scores).expect("finite model activations");
            for j in 0..=i {
                probs[(h * seq + i) * seq + j] = p[j];
                for z in 0..head_dim {
                    out[i * d + h * head_dim + z] += p[j] * v[j * d + h * head_dim + z];
                }
            }
        }
    }
    (out, probs)
}
