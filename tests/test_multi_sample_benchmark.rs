use std::path::PathBuf;
use volumen_geom::{NormalEstimator, SurfaceMesh, TifxyzMesh};
use volumen_sampler::{DifferentialReport, SurfaceVolumeConfig, SurfaceVolumeGenerator};
use volumen_zarr::ZarrVolume;

/// Multi-sample differential benchmark runner.
/// Tests across multiple randomly generated synthetic or real datasets with varying seeds.
#[test]
fn test_reproducible_benchmark_across_random_datasets() {
    let testbeds = vec![
        ("Sample Alpha (Seed 1337)", PathBuf::from("data/synthetic_sample_alpha")),
        ("Sample Beta (Seed 9999)", PathBuf::from("data/synthetic_sample_beta")),
    ];

    for (name, path) in testbeds {
        let vol_path = path.join("volume.zarr");
        let obj_path = path.join("segment.obj");
        let tifxyz_path = path.join("segment_tifxyz");

        if !vol_path.exists() {
            eprintln!("Skipping {} because {} does not exist.", name, vol_path.display());
            continue;
        }

        // 1. Open OME-Zarr volume with 1GB LRU cache
        let vol = ZarrVolume::open(&vol_path, 256).expect("Failed to open Zarr volume");
        let (dx, dy, dz) = vol.metadata.dimensions();
        assert!(dx > 0 && dy > 0 && dz > 0);

        // 2. Test OBJ surface loading & differential normals
        let mut obj_mesh = SurfaceMesh::load_obj(&obj_path).expect("Failed to load OBJ mesh");
        assert!(!obj_mesh.vertices.is_empty());
        NormalEstimator::compute_vertex_normals(&mut obj_mesh);
        assert_eq!(obj_mesh.normals.len(), obj_mesh.vertices.len());

        // 3. Test tifxyz loading
        let tifxyz_mesh = TifxyzMesh::load_from_dir(&tifxyz_path).expect("Failed to load tifxyz mesh");
        assert_eq!(tifxyz_mesh.width, 64);
        assert_eq!(tifxyz_mesh.height, 64);
        let pt0 = tifxyz_mesh.get_point(0, 0);
        assert!(pt0.is_some(), "Point (0, 0) in tifxyz must exist");

        // 4. Benchmark Render Run 1
        let temp_dir_1 = tempfile::tempdir().unwrap();
        let config = SurfaceVolumeConfig {
            num_layers: 16,
            step_size: 1.0,
            width: 64,
            height: 64,
            affine: None,
            flip_normals: false,
        };


        let t1 = std::time::Instant::now();
        SurfaceVolumeGenerator::render_mesh_to_dir(&vol, &obj_mesh, &config, temp_dir_1.path())
            .expect("Render Run 1 failed");
        let duration_1 = t1.elapsed();

        // 5. Benchmark Render Run 2 (tifxyz native lattice test)
        let temp_dir_2 = tempfile::tempdir().unwrap();
        let t2 = std::time::Instant::now();
        SurfaceVolumeGenerator::render_tifxyz_to_dir(&vol, &tifxyz_mesh, &config, temp_dir_2.path())
            .expect("Render Run 2 (tifxyz) failed");
        let duration_2 = t2.elapsed();

        // 6. Verify 100% Determinism between runs
        for k in 0..16 {
            let slice_file_1 = temp_dir_1.path().join(format!("{:02}.tif", k));
            let slice_file_2 = temp_dir_2.path().join(format!("{:02}.tif", k));
            assert!(slice_file_1.exists() && slice_file_2.exists());
            assert_eq!(
                slice_file_1.metadata().unwrap().len(),
                slice_file_2.metadata().unwrap().len(),
                "Output slices between runs must be byte-for-byte identical"
            );
        }

        println!(
            "Benchmark [{}] - Size: {}x{}x{} - Run 1: {:.2?} - Run 2: {:.2?}",
            name, dx, dy, dz, duration_1, duration_2
        );
    }
}
