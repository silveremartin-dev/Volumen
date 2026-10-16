use tempfile::tempdir;
use volumen_geom::{SpiralParameters, SpiralUnroller};
use volumen_sampler::{SpiralUnrollConfig, SurfaceVolumeGenerator};
use volumen_zarr::chunk::{Chunk3D, ChunkKey};
use volumen_zarr::ZarrVolume;

#[test]
fn test_spiral_differential_geometry_and_normals() {
    let a = 20.0;
    let b = 3.0;
    let params = SpiralParameters::new(50.0, 50.0, a, b);

    // Test across several turns
    for turns in [0.25, 0.5, 1.0, 2.5, 5.0] {
        let theta = turns * 2.0 * std::f64::consts::PI;
        let pt = params.point_at(theta, 10.0);
        let normal = params.normal_at(theta);

        // 1. Normal magnitude must be 1.0 +- 1e-6
        let norm = (normal.x * normal.x + normal.y * normal.y + normal.z * normal.z).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "Spiral normal must be strictly unit length, got {}",
            norm
        );

        // 2. Normal must have z = 0 for standard cylinder spiral
        assert_eq!(normal.z, 0.0);

        // 3. Normal must be orthogonal to analytical tangent vector dP/dtheta
        let r = params.radius_at(theta);
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let tx = b * cos_t - r * sin_t;
        let ty = b * sin_t + r * cos_t;
        let dot = normal.x * tx + normal.y * ty;
        assert!(
            dot.abs() < 1e-6,
            "Normal must be orthogonal to tangent: dot product is {}",
            dot
        );

        // 4. Point must project back to the exact winding
        let (winding, residual) = params.project_point(&pt);
        assert!(
            (winding - turns).abs() < 1e-5,
            "Point must project back to turn {}",
            turns
        );
        assert!(
            residual.abs() < 1e-5,
            "Residual must be negligible on-manifold: {}",
            residual
        );
    }
}

#[test]
fn test_spiral_arc_length_unrolling_monotonic() {
    let b = 1.5;
    let mut prev_l = 0.0;

    for turn in 1..=10 {
        let theta = (turn as f64) * 2.0 * std::f64::consts::PI;
        let l = SpiralUnroller::compute_arc_length(b, theta);
        assert!(l > prev_l, "Arc length must be strictly monotonically increasing");
        prev_l = l;
    }
}

#[test]
fn test_spiral_surface_volume_render_end_to_end() {
    let dir = tempdir().expect("Failed to create tempdir");
    let vol = ZarrVolume::new_synthetic(128, 128, 64, 64, 64, 64);

    let center_x = 64.0;
    let center_y = 64.0;
    let a = 15.0;
    let b = 2.0;
    let params = SpiralParameters::new(center_x, center_y, a, b);

    // Populate chunks with a synthetic intensity ridge along the spiral manifold
    for cz in 0..1 {
        for cy in 0..2 {
            for cx in 0..2 {
                let mut chunk = Chunk3D::new_zeros(ChunkKey::new(cx, cy, cz), 64, 64, 64);
                let origin_x = cx * 64;
                let origin_y = cy * 64;
                let origin_z = cz * 64;

                for z in 0..64 {
                    for y in 0..64 {
                        for x in 0..64 {
                            let world_x = (origin_x + x) as f64;
                            let world_y = (origin_y + y) as f64;
                            let world_z = (origin_z + z) as f64;

                            let p = volumen_core::Point3D::new(world_x, world_y, world_z);
                            let (_, residual) = params.project_point(&p);

                            // Gaussian ridge around residual = 0 (spiral surface)
                            let sigma = 1.5;
                            let intensity = (50000.0 * (-0.5 * (residual / sigma).powi(2)).exp()) as u16;
                            chunk.set(x, y, z, intensity);
                        }
                    }
                }
                vol.insert_chunk(chunk);
            }
        }
    }

    let config = SpiralUnrollConfig {
        theta_min: 0.5 * std::f64::consts::PI,
        theta_max: 2.0 * std::f64::consts::PI,
        z_min: 10.0,
        z_max: 30.0,
        width: 128,
        height: 64,
        num_layers: 9,
        step_size: 1.0,
    };

    SurfaceVolumeGenerator::render_spiral_to_dir(&vol, &params, &config, dir.path())
        .expect("Spiral render must succeed");

    // Verify all 9 layer files were generated
    for layer in 0..9 {
        let slice_path = dir.path().join(format!("{:02}.tif", layer));
        assert!(slice_path.exists(), "Layer slice {:02}.tif must exist", layer);
    }
}

#[test]
fn test_deformed_spiral_surface_volume_render() {
    use volumen_core::Vec3D;
    use volumen_geom::{DeformationField, DeformedSpiralManifold};

    let dir = tempdir().expect("Failed to create tempdir");
    let vol = ZarrVolume::new_synthetic(128, 128, 64, 64, 64, 64);

    let params = SpiralParameters::new(64.0, 64.0, 15.0, 2.0);

    // Create a 2x2 displacement field simulating scroll ovalization/flattening
    let disp = vec![
        Vec3D::new(2.0, -1.0, 0.0),
        Vec3D::new(-2.0, 1.0, 0.0),
        Vec3D::new(2.0, -1.0, 0.0),
        Vec3D::new(-2.0, 1.0, 0.0),
    ];
    let field = DeformationField::new(
        0.5 * std::f64::consts::PI,
        2.0 * std::f64::consts::PI,
        10.0,
        30.0,
        2,
        2,
        disp,
    )
    .unwrap();

    let manifold = DeformedSpiralManifold::new(params, Some(field));

    let config = SpiralUnrollConfig {
        theta_min: 0.5 * std::f64::consts::PI,
        theta_max: 2.0 * std::f64::consts::PI,
        z_min: 10.0,
        z_max: 30.0,
        width: 64,
        height: 32,
        num_layers: 5,
        step_size: 1.0,
    };

    SurfaceVolumeGenerator::render_deformed_spiral_to_dir(&vol, &manifold, &config, dir.path())
        .expect("Deformed spiral render must succeed");

    for layer in 0..5 {
        let slice_path = dir.path().join(format!("{:02}.tif", layer));
        assert!(slice_path.exists(), "Deformed layer slice {:02}.tif must exist", layer);
    }
}

