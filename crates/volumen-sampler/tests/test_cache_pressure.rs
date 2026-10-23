use rayon::prelude::*;
use tempfile::tempdir;
use volumen_core::VoxelCoord;
use volumen_zarr::writer::ZarrWriter;
use volumen_zarr::ZarrVolume;

#[test]
fn test_lru_cache_bounded_memory_pressure() {
    let tmp = tempdir().unwrap();
    let chunk_size = 64u64;

    // 1. Create a zarr array on disk with 16 chunks (4x4x1)
    let writer = ZarrWriter::create(
        tmp.path(),
        [64, 256, 256],
        [chunk_size, chunk_size, chunk_size],
    ).unwrap();

    for cy in 0..4 {
        for cx in 0..4 {
            let val_base = ((cx * 10 + cy) as u16) * 100;
            let data = vec![val_base; (chunk_size * chunk_size * chunk_size) as usize];
            writer.write_chunk(0, cy, cx, &data).unwrap();
        }
    }

    // 2. Open volume with cache bounded to 4 chunks
    let vol = ZarrVolume::open(tmp.path(), 4).expect("Failed to open zarr volume");

    // 3. Query voxels across all 16 chunks concurrently from Rayon threads
    let query_points: Vec<VoxelCoord> = (0..10_000)
        .map(|i| {
            let cx = (i % 4) * chunk_size + 10;
            let cy = ((i / 4) % 4) * chunk_size + 10;
            VoxelCoord::new(cx, cy, 10)
        })
        .collect();

    query_points.par_iter().for_each(|coord| {
        let val = vol.get_voxel(coord).expect("Voxel read should succeed");
        let expected_cx = coord.x / chunk_size;
        let expected_cy = coord.y / chunk_size;
        let expected_val = ((expected_cx * 10 + expected_cy) as u16) * 100;
        assert_eq!(val, expected_val, "Voxel intensity must match chunk data regardless of cache eviction");
    });

    // 4. Invariant check: Cache size must never exceed max_cached_chunks
    let active_chunks = vol.cached_chunks_count();
    assert!(
        active_chunks <= 4,
        "LRU cache exceeded bounded memory limit: {} > 4",
        active_chunks
    );
}
