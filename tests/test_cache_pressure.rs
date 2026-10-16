use rayon::prelude::*;
use volumen_core::VoxelCoord;
use volumen_zarr::{Chunk3D, ChunkKey, ZarrVolume};

#[test]
fn test_lru_cache_bounded_memory_pressure() {
    // 1. Create a synthetic volume with 16 chunks (4x4x1)
    let chunk_size = 64u64;
    let vol = ZarrVolume::new_synthetic(256, 256, 64, chunk_size, chunk_size, chunk_size);
    
    // Explicitly constrain cache capacity to only 4 chunks (simulating --cache-gb 0.01)
    let constrained_vol = ZarrVolume {
        max_cached_chunks: 4,
        ..vol
    };

    // Populate each of the 16 chunks with distinct gradient data
    for cz in 0..1 {
        for cy in 0..4 {
            for cx in 0..4 {
                let key = ChunkKey::new(cx, cy, cz);
                let val_base = ((cx * 10 + cy) as u16) * 100;
                let mut data = vec![0u16; (chunk_size * chunk_size * chunk_size) as usize];
                data.fill(val_base);
                constrained_vol.insert_chunk(Chunk3D::new(
                    key,
                    chunk_size as usize,
                    chunk_size as usize,
                    chunk_size as usize,
                    data,
                ));
            }
        }
    }

    // 2. Query voxels across all 16 chunks concurrently from 32 Rayon threads
    let query_points: Vec<VoxelCoord> = (0..10_000)
        .map(|i| {
            let cx = (i % 4) * chunk_size + 10;
            let cy = ((i / 4) % 4) * chunk_size + 10;
            VoxelCoord::new(cx, cy, 10)
        })
        .collect();

    query_points.par_iter().for_each(|coord| {
        let val = constrained_vol.get_voxel(coord).expect("Voxel read should succeed");
        let expected_cx = coord.x / chunk_size;
        let expected_cy = coord.y / chunk_size;
        let expected_val = ((expected_cx * 10 + expected_cy) as u16) * 100;
        assert_eq!(val, expected_val, "Voxel intensity must match chunk data regardless of cache eviction");
    });

    // 3. Invariant check: Cache size must never exceed max_cached_chunks
    let active_chunks = constrained_vol.cached_chunks_count();
    assert!(
        active_chunks <= 4,
        "LRU cache exceeded bounded memory limit: {} > 4",
        active_chunks
    );
}
