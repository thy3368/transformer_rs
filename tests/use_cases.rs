use rstest::{fixture, rstest};
use tempfile::TempDir;
use transformer_rs::adapter::outbound::{
    BinaryCheckpointStore, ByteLevelTokenizer, TextDatasetReader,
};
use transformer_rs::core::entity::{GenerationConfig, ModelConfig, SamplingConfig, Text};
use transformer_rs::core::use_case::{
    EvaluateTextQuery, EvaluateTextUseCase, GenerateTextQuery, GenerateTextUseCase,
    TrainTextCommand, TrainTextUseCase,
};

#[fixture]
fn workspace() -> (TempDir, String, String) {
    let dir = tempfile::tempdir().unwrap();
    let dataset = dir.path().join("train.txt");
    let checkpoint = dir.path().join("model.trrs");
    std::fs::write(&dataset, "hello transformer hello transformer\n").unwrap();
    (
        dir,
        dataset.to_string_lossy().into_owned(),
        checkpoint.to_string_lossy().into_owned(),
    )
}
fn train(dataset: &str, checkpoint: &str, epochs: usize) -> transformer_rs::core::entity::Metrics {
    let reader = TextDatasetReader;
    let store = BinaryCheckpointStore;
    let tokenizer = ByteLevelTokenizer;
    TrainTextUseCase {
        dataset: &reader,
        checkpoints: &store,
        tokenizer: &tokenizer,
    }
    .execute(TrainTextCommand {
        dataset: dataset.into(),
        checkpoint: checkpoint.into(),
        model_config: ModelConfig {
            vocab_size: 259,
            max_seq_len: 16,
            d_model: 8,
            num_heads: 2,
            num_layers: 1,
            d_ff: 16,
        },
        epochs,
        learning_rate: 0.01,
        seed: 7,
    })
    .unwrap()
}

#[rstest]
fn train_text_use_case_writes_a_finite_checkpoint(workspace: (TempDir, String, String)) {
    let (_dir, dataset, checkpoint) = workspace;
    let metrics = train(&dataset, &checkpoint, 2);
    assert!(metrics.average_loss.is_finite());
    assert!(metrics.token_count > 0);
    assert!(std::fs::metadata(checkpoint).unwrap().len() > 8);
}

#[rstest]
fn evaluate_text_use_case_reads_checkpoint_without_training(workspace: (TempDir, String, String)) {
    let (_dir, dataset, checkpoint) = workspace;
    train(&dataset, &checkpoint, 1);
    let reader = TextDatasetReader;
    let store = BinaryCheckpointStore;
    let tokenizer = ByteLevelTokenizer;
    let metrics = EvaluateTextUseCase {
        dataset: &reader,
        checkpoints: &store,
        tokenizer: &tokenizer,
    }
    .execute(EvaluateTextQuery {
        dataset,
        checkpoint,
    })
    .unwrap();
    assert!(metrics.average_loss.is_finite());
    assert!(metrics.perplexity > 0.0);
}

#[rstest]
fn generate_text_use_case_returns_prompt_and_generated_text(workspace: (TempDir, String, String)) {
    let (_dir, dataset, checkpoint) = workspace;
    train(&dataset, &checkpoint, 1);
    let store = BinaryCheckpointStore;
    let tokenizer = ByteLevelTokenizer;
    let text = GenerateTextUseCase {
        checkpoints: &store,
        tokenizer: &tokenizer,
    }
    .execute(GenerateTextQuery {
        checkpoint,
        prompt: Text("hello".into()),
        config: GenerationConfig {
            max_tokens: 3,
            sampling: SamplingConfig::default(),
        },
    })
    .unwrap();
    assert!(text.0.starts_with("hello"));
}
