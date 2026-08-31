use crate::adapter::outbound::{BinaryCheckpointStore, ByteLevelTokenizer, TextDatasetReader};
use crate::core::entity::{GenerationConfig, ModelConfig, SamplingConfig, Text, TransformerError};
use crate::core::use_case::{
    EvaluateTextQuery, EvaluateTextUseCase, GenerateTextQuery, GenerateTextUseCase,
    TrainTextCommand, TrainTextUseCase,
};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "transformer-rs",
    about = "A small decoder-only Transformer implemented in Rust"
)]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    #[command(name = "train-text")]
    TrainText {
        #[arg(long)]
        dataset: String,
        #[arg(long)]
        checkpoint: String,
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
    },
}
pub fn run() -> Result<(), TransformerError> {
    let cli = Cli::parse();
    let tokenizer = ByteLevelTokenizer;
    let checkpoints = BinaryCheckpointStore;
    let dataset_reader = TextDatasetReader;
    match cli.command {
        Command::TrainText {
            dataset,
            checkpoint,
            epochs,
            learning_rate,
            max_context,
            d_model,
            heads,
            layers,
            d_ff,
            seed,
        } => {
            let use_case = TrainTextUseCase {
                dataset: &dataset_reader,
                checkpoints: &checkpoints,
                tokenizer: &tokenizer,
            };
            let metrics = use_case.execute(TrainTextCommand {
                dataset,
                checkpoint,
                model_config: ModelConfig {
                    vocab_size: 259,
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
            max_tokens,
            temperature,
            top_k,
            seed,
        } => {
            let use_case = GenerateTextUseCase {
                checkpoints: &checkpoints,
                tokenizer: &tokenizer,
            };
            let text = use_case.execute(GenerateTextQuery {
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
        } => {
            let use_case = EvaluateTextUseCase {
                dataset: &dataset_reader,
                checkpoints: &checkpoints,
                tokenizer: &tokenizer,
            };
            print_metrics(use_case.execute(EvaluateTextQuery {
                dataset,
                checkpoint,
            })?);
        }
    }
    Ok(())
}
fn print_metrics(m: crate::core::entity::Metrics) {
    println!(
        "average_loss={:.6}\nperplexity={:.6}\nnext_token_accuracy={:.4}\ntokens={}",
        m.average_loss, m.perplexity, m.next_token_accuracy, m.token_count
    );
}
