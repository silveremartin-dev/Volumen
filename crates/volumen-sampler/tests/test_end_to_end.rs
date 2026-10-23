use std::fs::File;
use std::io::Write;
use tempfile::tempdir;
use volumen_geom::{Face, NormalEstimator, SurfaceMesh};
use volumen_sampler::{SurfaceVolumeConfig, SurfaceVolumeGenerator};
use volumen_zarr::chunk::{Chunk3D, ChunkKey};
use volumen_zarr::ZarrVolume;

#[test]
fn test_end_to_end_surface_volume_pipeline() {
    // 1. Create a synthetic test volume: 128x128x128 with 64x64x64 chunks
    let vol = ZarrVolume::new_synthetic(128, 128, 128, 64, 64, 64);

    // Populate a chunk with a test intensity pattern
    let mut chunk = Chunk3D::new_zeros(ChunkKey::new(0, 0, 0), 64, 64, 64);
    for z in 0..64 {
        for y in 0..64 {
            for x in 0..64 {
                chunk.set(x, y, z, ((x + y + z) * 100) as u16);
            }
        }
    }
    vol.insert_chunk(chunk);

    // 2. Create a planar test mesh
    let mut mesh = SurfaceMesh::new();
    mesh.vertices.push(volumen_core::Point3D::new(10.0, 10.0, 10.0));
    mesh.vertices.push(volumen_core::Point3D::new(20.0, 10.0, 10.0));
    mesh.vertices.push(volumen_core::Point3D::new(10.0, 20.0, 10.0));
    mesh.faces.push(Face {
        v: [0, 1, 2],
        uv: None,
    });

    NormalEstimator::compute_vertex_normals(&mut mesh);
    assert_eq!(mesh.normals.len(), 3);

    // 3. Render 8 layers to a temporary directory
    let dir = tempdir().expect("Failed to create tempdir");
    let config = SurfaceVolumeConfig {
        num_layers: 8,
        step_size: 0.5,
        width: 32,
        height: 32,
        affine: None,
        flip_normals: false,
        auto_center: false,
    };


    SurfaceVolumeGenerator::render_mesh_to_dir(&vol, &mesh, &config, dir.path())
        .expect("Surface volume generation should succeed");

    // Verify generated TIFF slices
    for k in 0..8 {
        let slice_file = dir.path().join(format!("{:02}.tif", k));
        assert!(slice_file.exists(), "Slice {:02}.tif must exist", k);
        assert!(
            slice_file.metadata().unwrap().len() > 0,
            "Slice file must not be empty"
        );
    }
}
