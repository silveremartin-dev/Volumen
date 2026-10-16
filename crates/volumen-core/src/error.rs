use thiserror::Error;

/// Result alias for all Volumen operations.
pub type Result<T> = std::result::Result<T, VolumenError>;

/// Master error enumeration for Volumen.
#[derive(Error, Debug)]
pub enum VolumenError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Zarr format error: {0}")]
    Zarr(String),

    #[error("Chunk not found at coordinate ({x}, {y}, {z})")]
    ChunkNotFound { x: u64, y: u64, z: u64 },

    #[error("Geometry error: {0}")]
    Geometry(String),

    #[error("Mesh parsing error in OBJ file: {0}")]
    MeshParse(String),

    #[error("Out of bounds voxel query: ({x}, {y}, {z}) outside ({dim_x}, {dim_y}, {dim_z})")]
    OutOfBounds {
        x: f64,
        y: f64,
        z: f64,
        dim_x: u64,
        dim_y: u64,
        dim_z: u64,
    },

    #[error("Cache capacity exceeded: {0}")]
    CacheFull(String),

    #[error("TIFF image error: {0}")]
    Tiff(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}
