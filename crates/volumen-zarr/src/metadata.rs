use serde::{Deserialize, Serialize};
use std::path::Path;
use volumen_core::{Result, VolumenError};

/// Deserialized `.zarray` metadata from OME-Zarr.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZarrMetadata {
    pub zarr_format: u32,
    pub shape: Vec<u64>,
    pub chunks: Vec<u64>,
    pub dtype: String,
    pub order: Option<String>,
    pub compressor: Option<serde_json::Value>,
}

impl ZarrMetadata {
    /// Reads and parses `.zarray` from a directory.
    pub fn load_from_dir<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let meta_file = dir.as_ref().join(".zarray");
        if !meta_file.exists() {
            return Err(VolumenError::Zarr(format!(
                ".zarray file not found in {}",
                dir.as_ref().display()
            )));
        }
        let data = std::fs::read_to_string(&meta_file)?;
        let meta: Self = serde_json::from_str(&data)?;
        Ok(meta)
    }

    #[inline]
    pub fn dimensions(&self) -> (u64, u64, u64) {
        if self.shape.len() >= 3 {
            (self.shape[2], self.shape[1], self.shape[0]) // x, y, z in spatial order
        } else {
            (0, 0, 0)
        }
    }

    #[inline]
    pub fn chunk_sizes(&self) -> (u64, u64, u64) {
        if self.chunks.len() >= 3 {
            (self.chunks[2], self.chunks[1], self.chunks[0])
        } else {
            (128, 128, 128)
        }
    }
}
