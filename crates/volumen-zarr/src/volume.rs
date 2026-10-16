use crate::chunk::{Chunk3D, ChunkKey};
use crate::metadata::ZarrMetadata;
use crate::remote::RemoteZarrFetcher;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use volumen_core::{Result, VolumenError, VoxelCoord};

struct CacheItem {
    chunk: Arc<Chunk3D>,
    last_access: u64,
}

/// Out-of-core multithreaded Zarr Volume reader with concurrent LRU/block caching
/// and support for direct local disk access and live S3 / HTTP streaming.
pub struct ZarrVolume {
    pub location: String,
    pub is_remote: bool,
    pub metadata: ZarrMetadata,
    cache: Arc<RwLock<HashMap<ChunkKey, CacheItem>>>,
    pub max_cached_chunks: usize,
    access_counter: Arc<AtomicU64>,
}

impl ZarrVolume {
    /// Opens an OME-Zarr array from a local directory or remote S3 / HTTP URL.
    pub fn open_auto(path_or_url: &str, max_cached_chunks: usize) -> Result<Self> {
        let trimmed = path_or_url.trim();
        if trimmed.starts_with("s3://") || trimmed.starts_with("http://") || trimmed.starts_with("https://") {
            Self::open_remote(trimmed, max_cached_chunks)
        } else {
            Self::open(trimmed, max_cached_chunks)
        }
    }

    /// Opens an OME-Zarr array from a local directory.
    pub fn open<P: AsRef<Path>>(path: P, max_cached_chunks: usize) -> Result<Self> {
        let root = path.as_ref().to_path_buf();
        let metadata = ZarrMetadata::load_from_dir(&root)?;
        Ok(Self {
            location: root.to_string_lossy().to_string(),
            is_remote: false,
            metadata,
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_cached_chunks,
            access_counter: Arc::new(AtomicU64::new(1)),
        })
    }

    /// Opens an OME-Zarr array streaming directly over S3 or HTTP/HTTPS.
    pub fn open_remote(url: &str, max_cached_chunks: usize) -> Result<Self> {
        let resolved_url = RemoteZarrFetcher::resolve_url(url);
        let metadata = RemoteZarrFetcher::fetch_metadata(&resolved_url)?;
        tracing::info!(
            "Opened remote OME-Zarr volume at {} (shape: {:?}, chunks: {:?})",
            resolved_url,
            metadata.shape,
            metadata.chunks
        );
        Ok(Self {
            location: resolved_url,
            is_remote: true,
            metadata,
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_cached_chunks,
            access_counter: Arc::new(AtomicU64::new(1)),
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
            location: ":memory:".to_string(),
            is_remote: false,
            metadata,
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_cached_chunks: 1024,
            access_counter: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Adds a chunk to the in-memory cache directly.
    pub fn insert_chunk(&self, chunk: Chunk3D) {
        let mut cache = self.cache.write().unwrap();
        let tick = self.access_counter.fetch_add(1, Ordering::Relaxed);
        cache.insert(chunk.key, CacheItem {
            chunk: Arc::new(chunk),
            last_access: tick,
        });
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
            if let Some(item) = cache.get(&key) {
                return Ok(item.chunk.get(ox, oy, oz).unwrap_or(0));
            }
        }

        // Slow path: load chunk (from disk, remote S3/HTTP, or synthetic)
        let chunk = self.load_chunk(key)?;
        let val = chunk.get(ox, oy, oz).unwrap_or(0);
        
        {
            let mut cache = self.cache.write().unwrap();
            let tick = self.access_counter.fetch_add(1, Ordering::Relaxed);
            
            // LRU eviction if cache exceeds capacity
            if cache.len() >= self.max_cached_chunks && self.max_cached_chunks > 0 {
                // Remove oldest 25% of entries
                let mut entries: Vec<(ChunkKey, u64)> = cache
                    .iter()
                    .map(|(k, v)| (*k, v.last_access))
                    .collect();
                entries.sort_unstable_by_key(|&(_, t)| t);
                let remove_count = (cache.len() / 4).max(1);
                for (old_key, _) in entries.iter().take(remove_count) {
                    cache.remove(old_key);
                }
            }

            cache.insert(key, CacheItem {
                chunk,
                last_access: tick,
            });
        }

        Ok(val)
    }

    /// Number of chunks currently present in the LRU cache.
    pub fn cached_chunks_count(&self) -> usize {
        self.cache.read().unwrap().len()
    }

    fn load_chunk(&self, key: ChunkKey) -> Result<Arc<Chunk3D>> {
        let (cx, cy, cz) = self.metadata.chunk_sizes();
        let expected_size = (cx * cy * cz) as usize;
        let comp_id = self.metadata.compressor.as_ref().and_then(|c| {
            c.get("id").and_then(|id| id.as_str())
        });

        if self.is_remote {
            let chunk_rel_path = key.to_path_string();
            let chunk_url = format!("{}/{}", self.location, chunk_rel_path);

            if let Some(raw_bytes) = RemoteZarrFetcher::fetch_bytes(&chunk_url)? {
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
                // Fallback to dot notation or empty chunk
                let alt_path = format!("{}.{}.{}", key.cz, key.cy, key.cx);
                let alt_url = format!("{}/{}", self.location, alt_path);
                if let Some(raw_bytes) = RemoteZarrFetcher::fetch_bytes(&alt_url)? {
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
                    Ok(Arc::new(Chunk3D::new_zeros(
                        key,
                        cx as usize,
                        cy as usize,
                        cz as usize,
                    )))
                }
            }
        } else {
            let chunk_rel_path = key.to_path_string();
            let root_path = PathBuf::from(&self.location);
            let chunk_file = root_path.join(&chunk_rel_path);

            if chunk_file.exists() {
                let raw_bytes = std::fs::read(&chunk_file)?;
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
                let alt_file = root_path.join(format!("{}.{}.{}", key.cz, key.cy, key.cx));
                if alt_file.exists() {
                    let raw_bytes = std::fs::read(&alt_file)?;
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
                    Ok(Arc::new(Chunk3D::new_zeros(
                        key,
                        cx as usize,
                        cy as usize,
                        cz as usize,
                    )))
                }
            }
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

    #[test]
    fn test_lru_cache_eviction() {
        // Cache max 4 chunks
        let vol = ZarrVolume::new_synthetic(512, 512, 512, 64, 64, 64);
        // Reduce max_cached_chunks to 4
        let vol = ZarrVolume {
            max_cached_chunks: 4,
            ..vol
        };

        // Access 8 different chunks
        for i in 0..8 {
            let coord = VoxelCoord::new(i * 64, 0, 0);
            let _ = vol.get_voxel(&coord);
        }

        // Cache must not exceed max_cached_chunks
        assert!(vol.cached_chunks_count() <= 4);
    }
}
