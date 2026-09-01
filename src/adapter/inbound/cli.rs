use crate::adapter::outbound::{BinaryCheckpointStore, BpeTokenizer, TextDatasetReader};
use crate::core::entity::{
    GenerationConfig, ModelConfig, SamplingConfig, Text, Tokenizer, TransformerError,
};
use crate::core::use_case::{
    EvaluateTextQuery, EvaluateTextUseCase, GenerateTextQuery, GenerateTextUseCase,
    TrainTextCommand, TrainTextUseCase,
};
use bpe::{BpeTrainer, BpeTrainerConfig};
use clap::{Parser, Subcommand};

const DEFAULT_VOCAB: &str = "data/vocab.json";
const DEFAULT_MERGES: &str = "data/merges.txt";

#[derive(Parser)]
#[command(
    name = "transformer-rs",
    about = "A small decoder-only Transformer implemented in Rust"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug, PartialEq)]
pub enum Command {
    #[command(name = "train-bpe")]
    TrainBpe {
        #[arg(long)]
        dataset: String,
        #[arg(long, default_value = "data")]
        output_dir: String,
        #[arg(long, default_value_t = 1000)]
        vocab_size: usize,
        #[arg(long, default_value_t = 2)]
        min_frequency: u64,
        #[arg(long)]
        force: bool,
    },
    #[command(name = "train-text")]
    TrainText {
        #[arg(long)]
        dataset: String,
        #[arg(long)]
        checkpoint: String,
        #[arg(long, default_value = DEFAULT_VOCAB)]
        vocab: String,
        #[arg(long, default_value = DEFAULT_MERGES)]
        merges: String,
        #[arg(long, default_value_t = 10)]
        epochs: usize,
        #[arg(long, default_value_t = 0.001)]
        learning_rate: f32,
        #[arg(long, default_value_t = 128)]
        max_context: usize,
        #[arg(long, default_value_t = 64)]
        d_model: usize,
        #[arg(long, default_value_t = 4)]
        heads: usize,
        #[arg(long, default_value_t = 2)]
        layers: usize,
        #[arg(long, default_value_t = 128)]
        d_ff: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    Generate {
        #[arg(long)]
        checkpoint: String,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value = DEFAULT_VOCAB)]
        vocab: String,
        #[arg(long, default_value = DEFAULT_MERGES)]
        merges: String,
        #[arg(long, default_value_t = 32)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        #[arg(long)]
        top_k: Option<usize>,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    Evaluate {
        #[arg(long)]
        dataset: String,
        #[arg(long)]
        checkpoint: String,
        #[arg(long, default_value = DEFAULT_VOCAB)]
        vocab: String,
        #[arg(long, default_value = DEFAULT_MERGES)]
        merges: String,
    },
}

pub fn run() -> Result<(), TransformerError> {
    execute(Cli::parse())
}

pub fn execute(cli: Cli) -> Result<(), TransformerError> {
    let checkpoints = BinaryCheckpointStore;
    let dataset_reader = TextDatasetReader;
    match cli.command {
        Command::TrainBpe {
            dataset,
            output_dir,
            vocab_size,
            min_frequency,
            force,
        } => {
            let trainer = BpeTrainer::new(BpeTrainerConfig {
                vocab_size,
                min_frequency,
                show_progress: true,
            })
            .map_err(bpe_error)?;
            let (vocab, merges) = trainer
                .train_file(dataset, output_dir, force)
                .map_err(bpe_error)?;
            println!("vocab={}\nmerges={}", vocab.display(), merges.display());
        }
        Command::TrainText {
            dataset,
            checkpoint,
            vocab,
            merges,
            epochs,
            learning_rate,
            max_context,
            d_model,
            heads,
            layers,
            d_ff,
            seed,
        } => {
            let tokenizer = load_tokenizer(&vocab, &merges)?;
            let metrics = TrainTextUseCase {
                dataset: &dataset_reader,
                checkpoints: &checkpoints,
                tokenizer: &tokenizer,
            }
            .execute(TrainTextCommand {
                dataset,
                checkpoint,
                model_config: ModelConfig {
                    vocab_size: tokenizer.config().vocab_size,
                    max_seq_len: max_context,
                    d_model,
                    num_heads: heads,
                    num_layers: layers,
                    d_ff,
                },
                epochs,
                learning_rate,
                seed,
            })?;
            print_metrics(metrics);
        }
        Command::Generate {
            checkpoint,
            prompt,
            vocab,
            merges,
            max_tokens,
            temperature,
            top_k,
            seed,
        } => {
            let tokenizer = load_tokenizer(&vocab, &merges)?;
            let text = GenerateTextUseCase {
                checkpoints: &checkpoints,
                tokenizer: &tokenizer,
            }
            .execute(GenerateTextQuery {
                checkpoint,
                prompt: Text(prompt),
                config: GenerationConfig {
                    max_tokens,
                    sampling: SamplingConfig {
                        temperature,
                        top_k,
                        seed,
                    },
                },
            })?;
            println!("{}", text.0);
        }
        Command::Evaluate {
            dataset,
            checkpoint,
            vocab,
            merges,
        } => {
            let tokenizer = load_tokenizer(&vocab, &merges)?;
            print_metrics(
                EvaluateTextUseCase {
                    dataset: &dataset_reader,
                    checkpoints: &checkpoints,
                    tokenizer: &tokenizer,
                }
                .execute(EvaluateTextQuery {
                    dataset,
                    checkpoint,
                })?,
            );
        }
    }
    Ok(())
}

fn load_tokenizer(vocab: &str, merges: &str) -> Result<BpeTokenizer, TransformerError> {
    BpeTokenizer::from_files(vocab, merges)
}
fn bpe_error(error: bpe::BpeError) -> TransformerError {
    TransformerError::InvalidData(error.to_string())
}
fn print_metrics(m: crate::core::entity::Metrics) {
    println!(
        "average_loss={:.6}\nperplexity={:.6}\nnext_token_accuracy={:.4}\ntokens={}",
        m.average_loss, m.perplexity, m.next_token_accuracy, m.token_count
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case("train-text", &["--dataset", "train.txt", "--checkpoint", "model.trrs"])]
    #[case("generate", &["--checkpoint", "model.trrs", "--prompt", "hello"])]
    #[case("evaluate", &["--dataset", "train.txt", "--checkpoint", "model.trrs"])]
    fn model_commands_use_default_bpe_paths(#[case] command: &str, #[case] args: &[&str]) {
        let cli = Cli::try_parse_from(
            std::iter::once("transformer-rs")
                .chain(std::iter::once(command))
                .chain(args.iter().copied()),
        )
        .unwrap();
        match cli.command {
            Command::TrainText { vocab, merges, .. }
            | Command::Generate { vocab, merges, .. }
            | Command::Evaluate { vocab, merges, .. } => {
                assert_eq!(vocab, DEFAULT_VOCAB);
                assert_eq!(merges, DEFAULT_MERGES);
            }
            Command::TrainBpe { .. } => unreachable!(),
        }
    }

    #[rstest]
    fn train_bpe_parses_custom_values_and_force() {
        let cli = Cli::try_parse_from([
            "transformer-rs",
            "train-bpe",
            "--dataset",
            "input.txt",
            "--output-dir",
            "tokens",
            "--vocab-size",
            "42",
            "--min-frequency",
            "3",
            "--force",
        ])
        .unwrap();
        assert_eq!(
            cli.command,
            Command::TrainBpe {
                dataset: "input.txt".into(),
                output_dir: "tokens".into(),
                vocab_size: 42,
                min_frequency: 3,
                force: true
            }
        );
    }

    #[rstest]
    fn bpe_cli_workflow_runs_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let dataset = dir.path().join("train.txt");
        let output = dir.path().join("tokens");
        let checkpoint = dir.path().join("model.trrs");
        std::fs::write(&dataset, "hello transformer hello transformer\n").unwrap();
        let path = |path: &std::path::Path| path.to_string_lossy().into_owned();

        execute(Cli {
            command: Command::TrainBpe {
                dataset: path(&dataset),
                output_dir: path(&output),
                vocab_size: 32,
                min_frequency: 1,
                force: false,
            },
        })
        .unwrap();
        let vocab = path(&output.join("vocab.json"));
        let merges = path(&output.join("merges.txt"));
        execute(Cli {
            command: Command::TrainText {
                dataset: path(&dataset),
                checkpoint: path(&checkpoint),
                vocab: vocab.clone(),
                merges: merges.clone(),
                epochs: 1,
                learning_rate: 0.01,
                max_context: 8,
                d_model: 4,
                heads: 2,
                layers: 1,
                d_ff: 8,
                seed: 7,
            },
        })
        .unwrap();
        execute(Cli {
            command: Command::Evaluate {
                dataset: path(&dataset),
                checkpoint: path(&checkpoint),
                vocab: vocab.clone(),
                merges: merges.clone(),
            },
        })
        .unwrap();
        execute(Cli {
            command: Command::Generate {
                checkpoint: path(&checkpoint),
                prompt: "hello".into(),
                vocab,
                merges,
                max_tokens: 1,
                temperature: 0.0,
                top_k: None,
                seed: 7,
            },
        })
        .unwrap();
    }
}
