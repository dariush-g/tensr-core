use std::fmt;

pub type Result<T> = std::result::Result<T, TensorError>;

#[derive(Debug, Clone, PartialEq)]
pub enum TensorError {
    ShapeMismatch,
    IndexOutOfBounds,
    NotImplemented,
    DimensionalMismatch,
    InvalidPermutation,
    BroadcastError,
    MixedStorage,
    IncompatibleTypes(&'static str),
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeMismatch => write!(f, "Tensor shapes do not match"),
            TensorError::IndexOutOfBounds => write!(f, "Index is out of bounds"),
            TensorError::NotImplemented => write!(f, "T not implemented"),
            TensorError::InvalidPermutation => write!(f, "Invalid permutation"),
            TensorError::DimensionalMismatch => write!(f, "Dimensions do not match"),
            TensorError::MixedStorage => write!(f, ""),
            TensorError::IncompatibleTypes(..) => write!(f, ""),
            TensorError::BroadcastError => write!(f, ""),
        }
    }
}

impl std::error::Error for TensorError {}
