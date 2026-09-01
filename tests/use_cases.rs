use rstest::{fixture, rstest};
use tempfile::TempDir;
use transformer_rs::adapter::outbound::{
    BinaryCheckpointStore, BpeTokenizer, ByteLevelTokenizer, TextDatasetReader,
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

#[rstest]
fn checkpoint_rejects_a_different_bpe_with_the_same_vocab_size() {
    let dir = tempfile::tempdir().unwrap();
    let first_corpus = dir.path().join("first.txt");
    let second_corpus = dir.path().join("second.txt");
    std::fs::write(&first_corpus, "lower lower newer newer lowest widest\n").unwrap();
    std::fs::write(&second_corpus, "alpha alpha beta beta gamma gamma delta\n").unwrap();
    let trainer = || {
        bpe::BpeTrainer::new(bpe::BpeTrainerConfig {
            vocab_size: 20,
            min_frequency: 1,
            show_progress: false,
        })
        .unwrap()
    };
    let (vocab_a, merges_a) = trainer()
        .train_file(&first_corpus, dir.path().join("a"), false)
        .unwrap();
    let (vocab_b, merges_b) = trainer()
        .train_file(&second_corpus, dir.path().join("b"), false)
        .unwrap();
    let tokenizer_a = BpeTokenizer::from_files(vocab_a, merges_a).unwrap();
    let tokenizer_b = BpeTokenizer::from_files(vocab_b, merges_b).unwrap();
    assert_eq!(
        transformer_rs::core::entity::Tokenizer::config(&tokenizer_a).vocab_size,
        transformer_rs::core::entity::Tokenizer::config(&tokenizer_b).vocab_size
    );

    let checkpoint = dir.path().join("model.trrs").to_string_lossy().into_owned();
    let reader = TextDatasetReader;
    let store = BinaryCheckpointStore;
    TrainTextUseCase {
        dataset: &reader,
        checkpoints: &store,
        tokenizer: &tokenizer_a,
    }
    .execute(TrainTextCommand {
        dataset: first_corpus.to_string_lossy().into_owned(),
        checkpoint: checkpoint.clone(),
        model_config: ModelConfig {
            vocab_size: transformer_rs::core::entity::Tokenizer::config(&tokenizer_a).vocab_size,
            max_seq_len: 8,
            d_model: 4,
            num_heads: 2,
            num_layers: 1,
            d_ff: 8,
        },
        epochs: 1,
        learning_rate: 0.01,
        seed: 7,
    })
    .unwrap();
    let error = EvaluateTextUseCase {
        dataset: &reader,
        checkpoints: &store,
        tokenizer: &tokenizer_b,
    }
    .execute(EvaluateTextQuery {
        dataset: second_corpus.to_string_lossy().into_owned(),
        checkpoint,
    })
    .unwrap_err();
    assert!(matches!(
        error,
        transformer_rs::core::entity::TransformerError::Checkpoint(_)
    ));
}
