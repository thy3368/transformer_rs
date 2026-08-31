use crate::core::entity::{
    BOS_TOKEN, EOS_TOKEN, Text, TokenIds, Tokenizer, TokenizerConfig, TransformerError,
};
use std::path::Path;
use tokenizers::Tokenizer as HuggingFaceTokenizer;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::whitespace::Whitespace;

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BpeTokens(pub Vec<String>);

pub struct BpeTokenizer {
    tokenizer: HuggingFaceTokenizer,
    config: TokenizerConfig,
    unk_token: u32,
}

impl BpeTokenizer {
    pub fn from_files(
        vocab_path: impl AsRef<Path>,
        merges_path: impl AsRef<Path>,
    ) -> Result<Self, TransformerError> {
        let vocab_path = vocab_path.as_ref().to_string_lossy().into_owned();
        let merges_path = merges_path.as_ref().to_string_lossy().into_owned();
        let model = BPE::from_file(&vocab_path, &merges_path)
            .unk_token("[UNK]".into())
            .build()
            .map_err(tokenizer_error)?;
        let mut tokenizer = HuggingFaceTokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));

        let required_id = |token: &str| {
            tokenizer.token_to_id(token).ok_or_else(|| {
                TransformerError::InvalidConfig(format!(
                    "BPE vocabulary is missing required token {token}"
                ))
            })
        };
        let unk_token = required_id("[UNK]")?;
        let bos_token = required_id("[BOS]")?;
        let eos_token = required_id("[EOS]")?;
        let pad_token = required_id("[PAD]")?;
        let config = TokenizerConfig {
            kind: "bpe-v1".into(),
            vocab_size: tokenizer.get_vocab_size(false),
            bos_token,
            eos_token,
            pad_token,
        };
        Ok(Self {
            tokenizer,
            config,
            unk_token,
        })
    }

    pub fn tokenize(&self, text: &Text) -> Result<BpeTokens, TransformerError> {
        let encoding = self
            .tokenizer
            .encode(text.0.as_str(), false)
            .map_err(tokenizer_error)?;
        Ok(BpeTokens(encoding.get_tokens().to_vec()))
    }

    pub fn tokens_to_ids(
        &self,
        tokens: &BpeTokens,
        add_special_tokens: bool,
    ) -> Result<TokenIds, TransformerError> {
        let mut ids = Vec::with_capacity(tokens.0.len() + usize::from(add_special_tokens) * 2);
        if add_special_tokens {
            ids.push(self.config.bos_token);
        }
        ids.extend(
            tokens
                .0
                .iter()
                .map(|token| self.tokenizer.token_to_id(token).unwrap_or(self.unk_token)),
        );
        if add_special_tokens {
            ids.push(self.config.eos_token);
        }
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
            .decode(&ids.0, true)
            .map(Text)
            .map_err(tokenizer_error)
    }
}

fn tokenizer_error(error: tokenizers::Error) -> TransformerError {
    TransformerError::InvalidData(format!("BPE tokenizer error: {error}"))
}
