use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum TransformerError {
    InvalidConfig(String),
    InvalidData(String),
    InvalidTensor(String),
    Checkpoint(String),
    Io(std::io::Error),
}

impl Display for TransformerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(s) => write!(f, "invalid configuration: {s}"),
            Self::InvalidData(s) => write!(f, "invalid data: {s}"),
            Self::InvalidTensor(s) => write!(f, "invalid tensor: {s}"),
            Self::Checkpoint(s) => write!(f, "checkpoint error: {s}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for TransformerError {}
impl From<std::io::Error> for TransformerError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}
