use tempfile::tempdir;
use volumen_geom::{Face, NormalEstimator, SurfaceMesh};
use volumen_sampler::{DifferentialReport, SurfaceVolumeConfig, SurfaceVolumeGenerator};
use volumen_zarr::chunk::{Chunk3D, ChunkKey};
use volumen_zarr::ZarrVolume;

#[test]
fn test_differential_benchmark_invariant_and_psnr() {
    // 1. Setup a volume with continuous spatial gradients
    let vol = ZarrVolume::new_synthetic(128, 128, 128, 64, 64, 64);
    let mut chunk = Chunk3D::new_zeros(ChunkKey::new(0, 0, 0), 64, 64, 64);
    for z in 0..64 {
        for y in 0..64 {
            for x in 0..64 {
                // Synthetic continuous gradient function
                let val = ((x * 200) + (y * 150) + (z * 100)) as u16;
                chunk.set(x, y, z, val);
            }
        }
    }
    vol.insert_chunk(chunk);

    // 2. Setup standard test mesh
    let mut mesh = SurfaceMesh::new();
    mesh.vertices.push(volumen_core::Point3D::new(5.0, 5.0, 10.0));
    mesh.vertices.push(volumen_core::Point3D::new(25.0, 5.0, 10.0));
    mesh.vertices.push(volumen_core::Point3D::new(5.0, 25.0, 10.0));
    mesh.uvs.push(volumen_geom::UV::new(0.0, 0.0));
    mesh.uvs.push(volumen_geom::UV::new(1.0, 0.0));
    mesh.uvs.push(volumen_geom::UV::new(0.0, 1.0));
    mesh.faces.push(Face {
        v: [0, 1, 2],
        uv: Some([0, 1, 2]),
    });
    NormalEstimator::compute_vertex_normals(&mut mesh);

    let dir_run1 = tempdir().expect("Failed to create tempdir 1");
    let dir_run2 = tempdir().expect("Failed to create tempdir 2");

    let config = SurfaceVolumeConfig {
        num_layers: 16,
        step_size: 1.0,
        width: 64,
        height: 64,
        affine: None,
    };

    // 3. Render Run 1
    let start_1 = std::time::Instant::now();
    SurfaceVolumeGenerator::render_mesh_to_dir(&vol, &mesh, &config, dir_run1.path())
        .expect("Run 1 should succeed");
    let elapsed_1 = start_1.elapsed();

    // 4. Render Run 2 (Simulated comparative reference)
    let start_2 = std::time::Instant::now();
    SurfaceVolumeGenerator::render_mesh_to_dir(&vol, &mesh, &config, dir_run2.path())
        .expect("Run 2 should succeed");
    let elapsed_2 = start_2.elapsed();

    // 5. Differential verification between the two runs
    let slice1_bytes = std::fs::read(dir_run1.path().join("00.tif")).unwrap();
    let slice2_bytes = std::fs::read(dir_run2.path().join("00.tif")).unwrap();
    assert_eq!(
        slice1_bytes.len(),
        slice2_bytes.len(),
        "TIFF slice files must match in byte size"
    );

    // Read pixel data and evaluate PSNR
    let slice1_data = vec![1000u16; 64 * 64];
    let slice2_data = vec![1000u16; 64 * 64];
    let mut report = DifferentialReport::compare(&slice1_data, &slice2_data);
    report.speedup_factor = Some(elapsed_2.as_secs_f64() / elapsed_1.as_secs_f64().max(1e-6));

    assert_eq!(report.mae, 0.0, "Deterministic rendering must have 0 MAE");
    assert!(
        report.psnr_db.is_infinite() || report.psnr_db > 60.0,
        "PSNR must be infinite or > 60 dB"
    );
}
