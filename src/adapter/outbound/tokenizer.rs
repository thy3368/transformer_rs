use crate::core::entity::{
    BOS_TOKEN, EOS_TOKEN, Text, TokenIds, Tokenizer, TokenizerConfig, TransformerError,
};
use std::path::Path;

pub use bpe::BpeTokens;

pub struct ByteLevelTokenizer;
impl Tokenizer for ByteLevelTokenizer {
    fn config(&self) -> TokenizerConfig {
        TokenizerConfig::default()
    }
    fn encode(&self, text: &Text, special: bool) -> Result<TokenIds, TransformerError> {
        let mut ids = Vec::with_capacity(text.0.len() + 2);
        if special {
            ids.push(BOS_TOKEN);
        }
        ids.extend(text.0.as_bytes().iter().map(|&b| b as u32));
        if special {
            ids.push(EOS_TOKEN);
        }
        Ok(TokenIds(ids))
    }
    fn decode(&self, ids: &TokenIds) -> Result<Text, TransformerError> {
        let bytes: Vec<u8> = ids
            .0
            .iter()
            .filter_map(|&id| if id < BOS_TOKEN { Some(id as u8) } else { None })
            .collect();
        Ok(Text(String::from_utf8_lossy(&bytes).into_owned()))
    }
}

pub struct BpeTokenizer {
    tokenizer: bpe::BpeTokenizer,
    config: TokenizerConfig,
}

impl BpeTokenizer {
    pub fn from_files(
        vocab_path: impl AsRef<Path>,
        merges_path: impl AsRef<Path>,
    ) -> Result<Self, TransformerError> {
        let tokenizer =
            bpe::BpeTokenizer::from_files(vocab_path, merges_path).map_err(tokenizer_error)?;
        let bpe_config = tokenizer.config();
        let config = TokenizerConfig {
            kind: bpe_config.kind.clone(),
            vocab_size: bpe_config.vocab_size,
            bos_token: bpe_config.bos_token,
            eos_token: bpe_config.eos_token,
            pad_token: bpe_config.pad_token,
        };
        Ok(Self { tokenizer, config })
    }

    pub fn tokenize(&self, text: &Text) -> Result<BpeTokens, TransformerError> {
        self.tokenizer.tokenize(&text.0).map_err(tokenizer_error)
    }

    pub fn tokens_to_ids(
        &self,
        tokens: &BpeTokens,
        add_special_tokens: bool,
    ) -> Result<TokenIds, TransformerError> {
        let ids = self
            .tokenizer
            .tokens_to_ids(tokens, add_special_tokens)
            .map_err(tokenizer_error)?;
        TokenIds::new(ids, self.config.vocab_size)
    }
}

impl Tokenizer for BpeTokenizer {
    fn config(&self) -> TokenizerConfig {
        self.config.clone()
    }

    fn encode(&self, text: &Text, add_special_tokens: bool) -> Result<TokenIds, TransformerError> {
        let tokens = self.tokenize(text)?;
        self.tokens_to_ids(&tokens, add_special_tokens)
    }

    fn decode(&self, ids: &TokenIds) -> Result<Text, TransformerError> {
        self.tokenizer
            .decode(&ids.0)
            .map(Text)
            .map_err(tokenizer_error)
    }
}

fn tokenizer_error(error: bpe::BpeError) -> TransformerError {
    TransformerError::InvalidData(format!("BPE tokenizer error: {error}"))
}
