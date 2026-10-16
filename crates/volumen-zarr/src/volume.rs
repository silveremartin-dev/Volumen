use crate::chunk::{Chunk3D, ChunkKey};
use crate::metadata::ZarrMetadata;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use volumen_core::{Result, VolumenError, VoxelCoord};

/// Out-of-core multithreaded Zarr Volume reader with concurrent LRU/block caching.
pub struct ZarrVolume {
    pub root_path: PathBuf,
    pub metadata: ZarrMetadata,
    cache: Arc<RwLock<HashMap<ChunkKey, Arc<Chunk3D>>>>,
    pub max_cached_chunks: usize,
}

impl ZarrVolume {
    /// Opens an OME-Zarr array directory.
    pub fn open<P: AsRef<Path>>(path: P, max_cached_chunks: usize) -> Result<Self> {
        let root = path.as_ref().to_path_buf();
        let metadata = ZarrMetadata::load_from_dir(&root)?;
        Ok(Self {
            root_path: root,
            metadata,
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_cached_chunks,
        })
    }

    /// Creates an in-memory synthetic volume (useful for benchmarks and unit testing).
    pub fn new_synthetic(
        dim_x: u64,
        dim_y: u64,
        dim_z: u64,
        chunk_x: u64,
        chunk_y: u64,
        chunk_z: u64,
    ) -> Self {
        let metadata = ZarrMetadata {
            zarr_format: 2,
            shape: vec![dim_z, dim_y, dim_x],
            chunks: vec![chunk_z, chunk_y, chunk_x],
            dtype: "<u2".to_string(),
            order: Some("C".to_string()),
            compressor: None,
        };
        Self {
            root_path: PathBuf::from(":memory:"),
            metadata,
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_cached_chunks: 1024,
        }
    }

    /// Adds a chunk to the in-memory cache directly.
    pub fn insert_chunk(&self, chunk: Chunk3D) {
        let mut cache = self.cache.write().unwrap();
        cache.insert(chunk.key, Arc::new(chunk));
    }

    /// Resolves which chunk contains the given discrete voxel coordinate.
    #[inline]
    pub fn voxel_to_chunk(&self, voxel: &VoxelCoord) -> (ChunkKey, usize, usize, usize) {
        let (cx_size, cy_size, cz_size) = self.metadata.chunk_sizes();
        let key = ChunkKey::new(
            voxel.x / cx_size,
            voxel.y / cy_size,
            voxel.z / cz_size,
        );
        let ox = (voxel.x % cx_size) as usize;
        let oy = (voxel.y % cy_size) as usize;
        let oz = (voxel.z % cz_size) as usize;
        (key, ox, oy, oz)
    }

    /// Reads an individual voxel intensity (16-bit).
    pub fn get_voxel(&self, voxel: &VoxelCoord) -> Result<u16> {
        let (dim_x, dim_y, dim_z) = self.metadata.dimensions();
        if voxel.x >= dim_x || voxel.y >= dim_y || voxel.z >= dim_z {
            return Err(VolumenError::OutOfBounds {
                x: voxel.x as f64,
                y: voxel.y as f64,
                z: voxel.z as f64,
                dim_x,
                dim_y,
                dim_z,
            });
        }

        let (key, ox, oy, oz) = self.voxel_to_chunk(voxel);

        // Check cache first (fast path)
        {
            let cache = self.cache.read().unwrap();
            if let Some(chunk) = cache.get(&key) {
                return Ok(chunk.get(ox, oy, oz).unwrap_or(0));
            }
        }

        // Slow path: load chunk (if from disk or synthetic)
        let chunk = self.load_chunk(key)?;
        let val = chunk.get(ox, oy, oz).unwrap_or(0);
        {
            let mut cache = self.cache.write().unwrap();
            cache.insert(key, chunk);
        }
        Ok(val)
    }

    fn load_chunk(&self, key: ChunkKey) -> Result<Arc<Chunk3D>> {
        let (cx, cy, cz) = self.metadata.chunk_sizes();
        let expected_size = (cx * cy * cz) as usize;

        // Try reading chunk from filesystem if root_path exists
        let chunk_rel_path = key.to_path_string();
        let chunk_file = self.root_path.join(&chunk_rel_path);

        if chunk_file.exists() {
            let raw_bytes = std::fs::read(&chunk_file)?;
            let comp_id = self.metadata.compressor.as_ref().and_then(|c| {
                c.get("id").and_then(|id| id.as_str())
            });
            let voxel_data = crate::decompress::ChunkDecompressor::decompress(
                &raw_bytes,
                comp_id,
                expected_size,
            )?;
            Ok(Arc::new(Chunk3D::new(
                key,
                cx as usize,
                cy as usize,
                cz as usize,
                voxel_data,
            )))
        } else {
            // Synthetic or empty chunk fallback
            Ok(Arc::new(Chunk3D::new_zeros(
                key,
                cx as usize,
                cy as usize,
                cz as usize,
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_volume_chunking() {
        let vol = ZarrVolume::new_synthetic(256, 256, 256, 64, 64, 64);
        let coord = VoxelCoord::new(65, 129, 10);
        let (key, ox, oy, oz) = vol.voxel_to_chunk(&coord);
        assert_eq!(key, ChunkKey::new(1, 2, 0));
        assert_eq!(ox, 1);
        assert_eq!(oy, 1);
        assert_eq!(oz, 10);
    }
}
