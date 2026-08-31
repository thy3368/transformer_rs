use crate::core::entity::{
    BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, Text, TokenIds, Tokenizer, TokenizerConfig, TransformerError,
};
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
            .filter_map(|&id| {
                if id < BOS_TOKEN {
                    Some(id as u8)
                } else if id == BOS_TOKEN || id == EOS_TOKEN || id == PAD_TOKEN {
                    None
                } else {
                    None
                }
            })
            .collect();
        Ok(Text(String::from_utf8_lossy(&bytes).into_owned()))
    }
}
