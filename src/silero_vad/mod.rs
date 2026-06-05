//! Core Silero VAD building blocks shared across the Rust crate.
//!
//! This module groups the bundled ONNX weights (`data`), model-loading helpers
//! (`model`) and post-processing utilities (`utils_vad`) exposed by the crate.

pub mod data;
pub mod model;
pub mod utils_vad;

use thiserror::Error;
use ort::Error as OrtGenericError;
use ort::session::builder::SessionBuilder;

/// Unified error type returned by Silero VAD helpers.
#[derive(Debug, Error)]
pub enum SileroError {
    /// Arbitrary message produced by downstream crates or custom guards.
    #[error("{0}")]
    Message(String),
    /// ONNX Runtime error.
    #[error(transparent)]
    Ort(#[from] OrtGenericError),
}

/// Convenience alias for results returned by public Silero VAD functions.
pub type Result<T> = std::result::Result<T, SileroError>;

impl From<ndarray::ShapeError> for SileroError {
    fn from(value: ndarray::ShapeError) -> Self {
        Self::Message(value.to_string())
    }
}

impl From<ort::Error<SessionBuilder>> for SileroError {
    fn from(err: ort::Error<SessionBuilder>) -> Self {
        SileroError::Ort(err.into())
    }
}
