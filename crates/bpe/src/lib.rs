use sha2::{Digest, Sha256};
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};
use tokenizers::models::TrainerWrapper;
use tokenizers::models::bpe::{BPE, BpeTrainer as HuggingFaceBpeTrainer};
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::{AddedToken, Model, Tokenizer};

pub const SPECIAL_TOKENS: [&str; 4] = ["[UNK]", "[BOS]", "[EOS]", "[PAD]"];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BpeTokens(pub Vec<String>);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BpeConfig {
    pub kind: String,
    pub vocab_size: usize,
    pub unk_token: u32,
    pub bos_token: u32,
    pub eos_token: u32,
    pub pad_token: u32,
    pub fingerprint: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BpeTrainerConfig {
    pub vocab_size: usize,
    pub min_frequency: u64,
    pub show_progress: bool,
}

impl Default for BpeTrainerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 1000,
            min_frequency: 2,
            show_progress: true,
        }
    }
}

#[derive(Debug)]
pub enum BpeError {
    Io(std::io::Error),
    InvalidConfig(String),
    Tokenizer(String),
    OutputExists(PathBuf),
}

impl Display for BpeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::InvalidConfig(message) | Self::Tokenizer(message) => f.write_str(message),
            Self::OutputExists(path) => write!(f, "output file already exists: {}", path.display()),
        }
    }
}

impl std::error::Error for BpeError {}
impl From<std::io::Error> for BpeError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

pub struct BpeTokenizer {
    tokenizer: Tokenizer,
    config: BpeConfig,
}

impl BpeTokenizer {
    pub fn from_files(
        vocab_path: impl AsRef<Path>,
        merges_path: impl AsRef<Path>,
    ) -> Result<Self, BpeError> {
        let vocab_bytes = std::fs::read(vocab_path.as_ref())?;
        let merges_bytes = std::fs::read(merges_path.as_ref())?;
        let vocab = vocab_path.as_ref().to_string_lossy();
        let merges = merges_path.as_ref().to_string_lossy();
        let model = BPE::from_file(vocab.as_ref(), merges.as_ref())
            .unk_token(SPECIAL_TOKENS[0].into())
            .build()
            .map_err(tokenizer_error)?;
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        let required = |token: &str| {
            tokenizer.token_to_id(token).ok_or_else(|| {
                BpeError::InvalidConfig(format!("BPE vocabulary is missing required token {token}"))
            })
        };
        let config = BpeConfig {
            kind: format!("bpe-v1:{}", fingerprint(&vocab_bytes, &merges_bytes)),
            vocab_size: tokenizer.get_vocab_size(false),
            unk_token: required(SPECIAL_TOKENS[0])?,
            bos_token: required(SPECIAL_TOKENS[1])?,
            eos_token: required(SPECIAL_TOKENS[2])?,
            pad_token: required(SPECIAL_TOKENS[3])?,
            fingerprint: fingerprint(&vocab_bytes, &merges_bytes),
        };
        Ok(Self { tokenizer, config })
    }

    pub fn config(&self) -> &BpeConfig {
        &self.config
    }
    pub fn tokenize(&self, text: &str) -> Result<BpeTokens, BpeError> {
        self.tokenizer
            .encode(text, false)
            .map(|e| BpeTokens(e.get_tokens().to_vec()))
            .map_err(tokenizer_error)
    }
    pub fn tokens_to_ids(&self, tokens: &BpeTokens, special: bool) -> Result<Vec<u32>, BpeError> {
        let mut ids = Vec::with_capacity(tokens.0.len() + usize::from(special) * 2);
        if special {
            ids.push(self.config.bos_token);
        }
        ids.extend(tokens.0.iter().map(|token| {
            self.tokenizer
                .token_to_id(token)
                .unwrap_or(self.config.unk_token)
        }));
        if special {
            ids.push(self.config.eos_token);
        }
        Ok(ids)
    }
    pub fn encode(&self, text: &str, special: bool) -> Result<Vec<u32>, BpeError> {
        self.tokens_to_ids(&self.tokenize(text)?, special)
    }
    pub fn decode(&self, ids: &[u32]) -> Result<String, BpeError> {
        let special = [
            self.config.unk_token,
            self.config.bos_token,
            self.config.eos_token,
            self.config.pad_token,
        ];
        let content: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|id| !special.contains(id))
            .collect();
        self.tokenizer
            .decode(&content, true)
            .map_err(tokenizer_error)
    }
}

pub struct BpeTrainer {
    config: BpeTrainerConfig,
}
impl BpeTrainer {
    pub fn new(config: BpeTrainerConfig) -> Result<Self, BpeError> {
        if config.vocab_size < SPECIAL_TOKENS.len() {
            return Err(BpeError::InvalidConfig(
                "vocab_size must be at least 4".into(),
            ));
        }
        if config.min_frequency == 0 {
            return Err(BpeError::InvalidConfig(
                "min_frequency must be positive".into(),
            ));
        }
        Ok(Self { config })
    }
    pub fn train_file(
        &self,
        dataset: impl AsRef<Path>,
        output_dir: impl AsRef<Path>,
        force: bool,
    ) -> Result<(PathBuf, PathBuf), BpeError> {
        let output_dir = output_dir.as_ref();
        let vocab_path = output_dir.join("vocab.json");
        let merges_path = output_dir.join("merges.txt");
        if !force {
            for path in [&vocab_path, &merges_path] {
                if path.exists() {
                    return Err(BpeError::OutputExists(path.clone()));
                }
            }
        }
        std::fs::metadata(dataset.as_ref())?;
        std::fs::create_dir_all(output_dir)?;
        let model = BPE::builder()
            .unk_token(SPECIAL_TOKENS[0].into())
            .build()
            .map_err(tokenizer_error)?;
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        let trainer = HuggingFaceBpeTrainer::builder()
            .vocab_size(self.config.vocab_size)
            .min_frequency(self.config.min_frequency)
            .show_progress(self.config.show_progress)
            .special_tokens(
                SPECIAL_TOKENS
                    .iter()
                    .map(|token| AddedToken::from(*token, true))
                    .collect(),
            )
            .build();
        let mut trainer = TrainerWrapper::from(trainer);
        tokenizer
            .train_from_files(
                &mut trainer,
                vec![dataset.as_ref().to_string_lossy().into_owned()],
            )
            .map_err(tokenizer_error)?;
        tokenizer
            .get_model()
            .save(output_dir, None)
            .map_err(tokenizer_error)?;
        Ok((vocab_path, merges_path))
    }
}

fn fingerprint(vocab: &[u8], merges: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update((vocab.len() as u64).to_le_bytes());
    hash.update(vocab);
    hash.update((merges.len() as u64).to_le_bytes());
    hash.update(merges);
    format!("{:x}", hash.finalize())
}
fn tokenizer_error(error: impl Display) -> BpeError {
    BpeError::Tokenizer(format!("BPE tokenizer error: {error}"))
}
