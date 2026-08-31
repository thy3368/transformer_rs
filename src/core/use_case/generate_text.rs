use super::CheckpointStore;
use crate::core::entity::{
    EOS_TOKEN, GenerationConfig, Text, TokenIds, Tokenizer, TransformerError, TransformerModel,
    softmax,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
#[derive(Clone, Debug)]
pub struct GenerateTextQuery {
    pub checkpoint: String,
    pub prompt: Text,
    pub config: GenerationConfig,
}
pub struct GenerateTextUseCase<'a> {
    pub checkpoints: &'a dyn CheckpointStore,
    pub tokenizer: &'a dyn Tokenizer,
}
impl GenerateTextUseCase<'_> {
    pub fn execute(&self, q: GenerateTextQuery) -> Result<Text, TransformerError> {
        let snapshot = self.checkpoints.load(&q.checkpoint)?;
        if snapshot.tokenizer_config != self.tokenizer.config() {
            return Err(TransformerError::Checkpoint(
                "tokenizer configuration mismatch".into(),
            ));
        }
        let model = TransformerModel::from_snapshot(snapshot)?;
        let mut ids = self.tokenizer.encode(&q.prompt, false)?.0;
        ids.insert(0, crate::core::entity::BOS_TOKEN);
        let mut rng = StdRng::seed_from_u64(q.config.sampling.seed);
        for _ in 0..q.config.max_tokens {
            if ids.len() >= model.config().max_seq_len {
                break;
            }
            let out = model.forward(&ids)?;
            let vocab = model.config().vocab_size;
            let row = &out.logits.0.data[(ids.len() - 1) * vocab..ids.len() * vocab];
            let next = select(
                row,
                q.config.sampling.temperature,
                q.config.sampling.top_k,
                &mut rng,
            )? as u32;
            ids.push(next);
            if next == EOS_TOKEN {
                break;
            }
        }
        self.tokenizer.decode(&TokenIds(ids))
    }
}
fn select(
    logits: &[f32],
    temperature: f32,
    top_k: Option<usize>,
    rng: &mut StdRng,
) -> Result<usize, TransformerError> {
    if temperature <= 0.0 {
        return Ok(logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .ok_or_else(|| TransformerError::InvalidData("empty logits".into()))?
            .0);
    }
    let mut candidates: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &x)| (i, x / temperature))
        .collect();
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    if let Some(k) = top_k {
        candidates.truncate(k.max(1).min(candidates.len()));
    }
    let p = softmax(&candidates.iter().map(|x| x.1).collect::<Vec<_>>())?;
    let pick = rng.r#gen::<f32>();
    let mut sum = 0.0;
    for (i, prob) in p.into_iter().enumerate() {
        sum += prob;
        if pick <= sum {
            return Ok(candidates[i].0);
        }
    }
    Ok(candidates.last().unwrap().0)
}
