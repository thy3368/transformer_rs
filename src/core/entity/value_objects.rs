use super::TransformerError;
use serde::{Deserialize, Serialize};

pub const BYTE_VOCAB_SIZE: usize = 259;
pub const BOS_TOKEN: u32 = 256;
pub const EOS_TOKEN: u32 = 257;
pub const PAD_TOKEN: u32 = 258;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Text(pub String);
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenIds(pub Vec<u32>);
#[derive(Clone, Debug, PartialEq)]
pub struct HiddenState(pub super::Tensor);
#[derive(Clone, Debug, PartialEq)]
pub struct Query(pub super::Tensor);
#[derive(Clone, Debug, PartialEq)]
pub struct Key(pub super::Tensor);
#[derive(Clone, Debug, PartialEq)]
pub struct Value(pub super::Tensor);
#[derive(Clone, Debug, PartialEq)]
pub struct ContextVector(pub super::Tensor);
#[derive(Clone, Debug, PartialEq)]
pub struct Logits(pub super::Tensor);
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Loss(pub f32);
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointId(pub String);

impl TokenIds {
    pub fn new(ids: Vec<u32>, vocab_size: usize) -> Result<Self, TransformerError> {
        if ids.iter().any(|&id| id as usize >= vocab_size) {
            return Err(TransformerError::InvalidData(
                "token is outside vocabulary".into(),
            ));
        }
        Ok(Self(ids))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub d_ff: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: BYTE_VOCAB_SIZE,
            max_seq_len: 128,
            d_model: 64,
            num_heads: 4,
            num_layers: 2,
            d_ff: 128,
        }
    }
}

impl ModelConfig {
    pub fn validate(&self) -> Result<(), TransformerError> {
        if self.vocab_size == 0
            || self.max_seq_len < 2
            || self.d_model == 0
            || self.num_heads == 0
            || self.num_layers == 0
            || self.d_ff == 0
        {
            return Err(TransformerError::InvalidConfig(
                "all dimensions must be positive and max_seq_len >= 2".into(),
            ));
        }
        if self.d_model % self.num_heads != 0 {
            return Err(TransformerError::InvalidConfig(
                "d_model must be divisible by num_heads".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenizerConfig {
    pub kind: String,
    pub vocab_size: usize,
    pub bos_token: u32,
    pub eos_token: u32,
    pub pad_token: u32,
}
impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            kind: "byte-level-v1".into(),
            vocab_size: BYTE_VOCAB_SIZE,
            bos_token: BOS_TOKEN,
            eos_token: EOS_TOKEN,
            pad_token: PAD_TOKEN,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub seed: u64,
}
impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub sampling: SamplingConfig,
}
impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 32,
            sampling: SamplingConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ParameterSnapshot {
    pub name: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelSnapshot {
    pub model_config: ModelConfig,
    pub tokenizer_config: TokenizerConfig,
    pub parameters: Vec<ParameterSnapshot>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Metrics {
    pub average_loss: f32,
    pub perplexity: f32,
    pub next_token_accuracy: f32,
    pub token_count: usize,
}
