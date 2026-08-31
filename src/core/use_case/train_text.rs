use super::{CheckpointStore, DatasetReader};
use crate::core::entity::{
    Adam, Metrics, ModelConfig, Tokenizer, TransformerError, TransformerModel, next_token_loss,
};

#[derive(Clone, Debug)]
pub struct TrainTextCommand {
    pub dataset: String,
    pub checkpoint: String,
    pub model_config: ModelConfig,
    pub epochs: usize,
    pub learning_rate: f32,
    pub seed: u64,
}
pub struct TrainTextUseCase<'a> {
    pub dataset: &'a dyn DatasetReader,
    pub checkpoints: &'a dyn CheckpointStore,
    pub tokenizer: &'a dyn Tokenizer,
}
impl TrainTextUseCase<'_> {
    pub fn execute(&self, c: TrainTextCommand) -> Result<Metrics, TransformerError> {
        if c.epochs == 0 {
            return Err(TransformerError::InvalidConfig(
                "epochs must be positive".into(),
            ));
        }
        let text = self.dataset.read(&c.dataset)?;
        let tokens = self.tokenizer.encode(&text, true)?.0;
        if tokens.len() < 2 {
            return Err(TransformerError::InvalidData(
                "training text needs at least two tokens".into(),
            ));
        }
        let mut model = TransformerModel::new(c.model_config, c.seed)?;
        let mut adam = Adam::new(&model, c.learning_rate);
        let mut final_metrics = Metrics::default();
        for _ in 0..c.epochs {
            let mut loss = 0.0;
            let mut acc = 0.0;
            let mut count = 0;
            for window in tokens.chunks(model.config().max_seq_len + 1) {
                if window.len() < 2 {
                    continue;
                }
                let input = &window[..window.len() - 1];
                let target = &window[1..];
                let out = model.forward(input)?;
                let result =
                    next_token_loss(&out.logits.0.data, target, model.config().vocab_size)?;
                let mut gradients = model.backward(&out, &result.gradient)?;
                adam.step(&mut model, &mut gradients, 1.0);
                loss += result.loss * result.token_count as f32;
                acc += result.accuracy * result.token_count as f32;
                count += result.token_count;
            }
            if count == 0 {
                return Err(TransformerError::InvalidData(
                    "dataset produced no training windows".into(),
                ));
            }
            let avg = loss / count as f32;
            final_metrics = Metrics {
                average_loss: avg,
                perplexity: avg.exp(),
                next_token_accuracy: acc / count as f32,
                token_count: count,
            };
        }
        self.checkpoints
            .save(&c.checkpoint, &model.snapshot(self.tokenizer.config()))?;
        Ok(final_metrics)
    }
}
