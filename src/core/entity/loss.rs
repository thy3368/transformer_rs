use super::{PAD_TOKEN, TransformerError, softmax};
#[derive(Clone, Debug)]
pub struct LossResult {
    pub loss: f32,
    pub accuracy: f32,
    pub token_count: usize,
    pub gradient: Vec<f32>,
}
pub fn next_token_loss(
    logits: &[f32],
    targets: &[u32],
    vocab: usize,
) -> Result<LossResult, TransformerError> {
    if logits.len() != targets.len() * vocab {
        return Err(TransformerError::InvalidData(
            "logit/target shape mismatch".into(),
        ));
    }
    let mut grad = vec![0.0; logits.len()];
    let mut loss = 0.0;
    let mut correct = 0;
    let mut count = 0;
    for (r, &target) in targets.iter().enumerate() {
        if target == PAD_TOKEN {
            continue;
        }
        if target as usize >= vocab {
            return Err(TransformerError::InvalidData(
                "target outside vocabulary".into(),
            ));
        }
        let p = softmax(&logits[r * vocab..(r + 1) * vocab])?;
        loss -= p[target as usize].max(1e-12).ln();
        let pred = p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        if pred == target as usize {
            correct += 1;
        }
        for j in 0..vocab {
            grad[r * vocab + j] = p[j] - (j == target as usize) as u8 as f32;
        }
        count += 1;
    }
    if count == 0 {
        return Err(TransformerError::InvalidData("no target tokens".into()));
    }
    for g in &mut grad {
        *g /= count as f32;
    }
    Ok(LossResult {
        loss: loss / count as f32,
        accuracy: correct as f32 / count as f32,
        token_count: count,
        gradient: grad,
    })
}
