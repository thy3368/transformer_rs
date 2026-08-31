use crate::core::entity::{ModelSnapshot, TransformerError};
use crate::core::use_case::CheckpointStore;
use serde::{Deserialize, Serialize};
const MAGIC: [u8; 4] = *b"TRRS";
const VERSION: u32 = 1;
#[derive(Serialize, Deserialize)]
struct Envelope {
    magic: [u8; 4],
    version: u32,
    snapshot: ModelSnapshot,
}
pub struct BinaryCheckpointStore;
impl CheckpointStore for BinaryCheckpointStore {
    fn save(&self, path: &str, snapshot: &ModelSnapshot) -> Result<(), TransformerError> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        let bytes = bincode::serialize(&Envelope {
            magic: MAGIC,
            version: VERSION,
            snapshot: snapshot.clone(),
        })
        .map_err(|e| TransformerError::Checkpoint(e.to_string()))?;
        std::fs::write(path, bytes)?;
        Ok(())
    }
    fn load(&self, path: &str) -> Result<ModelSnapshot, TransformerError> {
        let bytes = std::fs::read(path)?;
        let e: Envelope = bincode::deserialize(&bytes)
            .map_err(|e| TransformerError::Checkpoint(format!("invalid or truncated file: {e}")))?;
        if e.magic != MAGIC {
            return Err(TransformerError::Checkpoint("wrong magic".into()));
        }
        if e.version != VERSION {
            return Err(TransformerError::Checkpoint(format!(
                "unsupported version {}",
                e.version
            )));
        }
        Ok(e.snapshot)
    }
}
