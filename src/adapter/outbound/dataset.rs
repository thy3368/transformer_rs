use crate::core::entity::{Text, TransformerError};
use crate::core::use_case::DatasetReader;
pub struct TextDatasetReader;
impl DatasetReader for TextDatasetReader {
    fn read(&self, path: &str) -> Result<Text, TransformerError> {
        Ok(Text(std::fs::read_to_string(path)?))
    }
}
