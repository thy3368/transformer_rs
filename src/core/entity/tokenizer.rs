use super::{Text, TokenIds, TokenizerConfig, TransformerError};

pub trait Tokenizer: Send + Sync {
    fn config(&self) -> TokenizerConfig;
    fn encode(&self, text: &Text, add_special_tokens: bool) -> Result<TokenIds, TransformerError>;
    fn decode(&self, ids: &TokenIds) -> Result<Text, TransformerError>;
}
