use super::{CheckpointStore, DatasetReader};
use crate::core::entity::{
    Metrics, Tokenizer, TransformerError, TransformerModel, next_token_loss,
};
#[derive(Clone, Debug)]
pub struct EvaluateTextQuery {
    pub dataset: String,
    pub checkpoint: String,
}
pub struct EvaluateTextUseCase<'a> {
    pub dataset: &'a dyn DatasetReader,
    pub checkpoints: &'a dyn CheckpointStore,
    pub tokenizer: &'a dyn Tokenizer,
}
impl EvaluateTextUseCase<'_> {
    pub fn execute(&self, q: EvaluateTextQuery) -> Result<Metrics, TransformerError> {
        let snapshot = self.checkpoints.load(&q.checkpoint)?;
        if snapshot.tokenizer_config != self.tokenizer.config() {
            return Err(TransformerError::Checkpoint(
                "tokenizer configuration mismatch".into(),
            ));
        }
        let model = TransformerModel::from_snapshot(snapshot)?;
        let text = self.dataset.read(&q.dataset)?;
        let tokens = self.tokenizer.encode(&text, true)?.0;
        let mut loss = 0.0;
        let mut acc = 0.0;
        let mut count = 0;
        for window in tokens.chunks(model.config().max_seq_len + 1) {
            if window.len() < 2 {
                continue;
            }
            let out = model.forward(&window[..window.len() - 1])?;
            let r = next_token_loss(&out.logits.0.data, &window[1..], model.config().vocab_size)?;
            loss += r.loss * r.token_count as f32;
            acc += r.accuracy * r.token_count as f32;
            count += r.token_count;
        }
        if count == 0 {
            return Err(TransformerError::InvalidData(
                "dataset produced no evaluation windows".into(),
            ));
        }
        let average_loss = loss / count as f32;
        Ok(Metrics {
            average_loss,
            perplexity: average_loss.exp(),
            next_token_accuracy: acc / count as f32,
            token_count: count,
        })
    }
}
