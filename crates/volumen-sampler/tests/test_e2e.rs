use tempfile::tempdir;
use volumen_core::{Point3D, Vec3D};
use volumen_geom::{Face, SurfaceMesh, UV};
use volumen_sampler::{SurfaceVolumeConfig, SurfaceVolumeGenerator};
use volumen_zarr::{Chunk3D, ChunkKey, ZarrVolume};

#[test]
fn test_end_to_end_tiff_and_zarr_rendering() {
    let tmp = tempdir().unwrap();
    let tif_out = tmp.path().join("layers_tif");
    let zarr_out = tmp.path().join("layers_zarr");

    // 1. Setup synthetic volume
    let vol = ZarrVolume::new_synthetic(128, 128, 128, 64, 64, 64);
    let mut chunk_data = vec![0u16; 64 * 64 * 64];
    chunk_data.fill(42000);
    vol.insert_chunk(Chunk3D::new(ChunkKey::new(0, 0, 0), 64, 64, 64, chunk_data));

    // 2. Setup simple unit quad mesh (2 triangles)
    let mesh = SurfaceMesh {
        vertices: vec![
            Point3D::new(10.0, 10.0, 32.0),
            Point3D::new(50.0, 10.0, 32.0),
            Point3D::new(50.0, 50.0, 32.0),
            Point3D::new(10.0, 50.0, 32.0),
        ],
        uvs: vec![
            UV::new(0.0, 0.0),
            UV::new(1.0, 0.0),
            UV::new(1.0, 1.0),
            UV::new(0.0, 1.0),
        ],
        normals: vec![
            Vec3D::new(0.0, 0.0, 1.0),
            Vec3D::new(0.0, 0.0, 1.0),
            Vec3D::new(0.0, 0.0, 1.0),
            Vec3D::new(0.0, 0.0, 1.0),
        ],
        faces: vec![
            Face {
                v: [0, 1, 2],
                uv: Some([0, 1, 2]),
            },
            Face {
                v: [0, 2, 3],
                uv: Some([0, 2, 3]),
            },
        ],
    };

    let config = SurfaceVolumeConfig {
        num_layers: 16,
        step_size: 1.0,
        width: 64,
        height: 64,
        affine: None,
        flip_normals: false,
        auto_center: false,
    };

    // 3. Render directly to TIFF slices
    SurfaceVolumeGenerator::render_mesh_to_dir(&vol, &mesh, &config, &tif_out)
        .expect("TIFF rendering should succeed");

    for layer in 0..16 {
        let tiff_file = tif_out.join(format!("{:02}.tif", layer));
        assert!(tiff_file.exists(), "Layer TIFF {:02}.tif must exist", layer);
        assert!(
            tiff_file.metadata().unwrap().len() > 0,
            "TIFF must not be empty"
        );
    }

    // 4. Render directly to chunked OME-Zarr array
    SurfaceVolumeGenerator::render_mesh_to_zarr(&vol, &mesh, &config, &zarr_out)
        .expect("Chunked Zarr rendering should succeed");

    let meta_file = zarr_out.join(".zarray");
    assert!(
        meta_file.exists(),
        ".zarray metadata file must exist in chunked Zarr output"
    );
    let meta_content = std::fs::read_to_string(&meta_file).unwrap();
    assert!(
        meta_content.contains("\"zarr_format\": 2"),
        "Valid Zarr v2 format metadata required"
    );
}

#[test]
fn test_in_memory_slices_and_auto_center() {
    let vol = ZarrVolume::new_synthetic(128, 128, 128, 64, 64, 64);
    let mut chunk_data = vec![0u16; 64 * 64 * 64];
    chunk_data.fill(30000);
    vol.insert_chunk(Chunk3D::new(ChunkKey::new(0, 0, 0), 64, 64, 64, chunk_data));

    let mesh = SurfaceMesh {
        vertices: vec![
            Point3D::new(10.0, 10.0, 32.0),
            Point3D::new(50.0, 10.0, 32.0),
            Point3D::new(50.0, 50.0, 32.0),
            Point3D::new(10.0, 50.0, 32.0),
        ],
        uvs: vec![
            UV::new(0.0, 0.0),
            UV::new(1.0, 0.0),
            UV::new(1.0, 1.0),
            UV::new(0.0, 1.0),
        ],
        normals: vec![
            Vec3D::new(0.0, 0.0, 1.0),
            Vec3D::new(0.0, 0.0, 1.0),
            Vec3D::new(0.0, 0.0, 1.0),
            Vec3D::new(0.0, 0.0, 1.0),
        ],
        faces: vec![
            Face {
                v: [0, 1, 2],
                uv: Some([0, 1, 2]),
            },
            Face {
                v: [0, 2, 3],
                uv: Some([0, 2, 3]),
            },
        ],
    };

    let config = SurfaceVolumeConfig {
        num_layers: 8,
        step_size: 1.0,
        width: 32,
        height: 32,
        affine: None,
        flip_normals: false,
        auto_center: true,
    };

    let slices = SurfaceVolumeGenerator::render_mesh_to_slices(&vol, &mesh, &config)
        .expect("In-memory slice rendering with auto-centering must succeed");

    assert_eq!(slices.len(), 8);
    assert_eq!(slices[0].len(), 32 * 32);
    // Voxel values inside the synthetic chunk should match
    assert!(slices[4][16 * 32 + 16] > 0);
}
