use crate::core::entity::{ModelSnapshot, Text, TransformerError};
pub trait DatasetReader {
    fn read(&self, path: &str) -> Result<Text, TransformerError>;
}
pub trait CheckpointStore {
    fn save(&self, path: &str, snapshot: &ModelSnapshot) -> Result<(), TransformerError>;
    fn load(&self, path: &str) -> Result<ModelSnapshot, TransformerError>;
}
