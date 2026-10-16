use crate::interpolator::TrilinearSampler;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::fs::create_dir_all;
use std::path::Path;
use volumen_core::{Result, Vec3D};
use volumen_geom::{AffineTransform, SurfaceMesh, TifxyzMesh, UVSpatialIndex};
use volumen_zarr::ZarrVolume;

/// Configuration for 2.5D surface volume extraction.
#[derive(Debug, Clone)]
pub struct SurfaceVolumeConfig {
    pub num_layers: usize,
    pub step_size: f64,
    pub width: usize,
    pub height: usize,
    pub affine: Option<AffineTransform>,
}

impl Default for SurfaceVolumeConfig {
    fn default() -> Self {
        Self {
            num_layers: 64,
            step_size: 1.0,
            width: 1024,
            height: 1024,
            affine: None,
        }
    }
}

/// High-performance Surface Volume generator with exact barycentric ray-casting.
pub struct SurfaceVolumeGenerator;

impl SurfaceVolumeGenerator {
    /// Renders a surface volume using true barycentric ray-casting across a Wavefront OBJ mesh.
    pub fn render_mesh_to_dir<P: AsRef<Path>>(
        volume: &ZarrVolume,
        mesh: &SurfaceMesh,
        config: &SurfaceVolumeConfig,
        output_dir: P,
    ) -> Result<()> {
        let out_path = output_dir.as_ref();
        create_dir_all(out_path)?;

        tracing::info!("Building accelerated UV spatial index...");
        let uv_index = UVSpatialIndex::build(mesh, 128);

        let num_layers = config.num_layers;
        let half_layers = (num_layers as f64 - 1.0) / 2.0;

        tracing::info!(
            "Rendering {} layers (step size = {}) to {}",
            num_layers,
            config.step_size,
            out_path.display()
        );

        let progress = ProgressBar::new(num_layers as u64);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} layers ({eta})")
                .unwrap(),
        );

        let affine = config.affine.unwrap_or_else(AffineTransform::identity);

        for k in 0..num_layers {
            let offset = (k as f64 - half_layers) * config.step_size;
            let mut slice_data = vec![0u16; config.width * config.height];

            slice_data
                .par_chunks_mut(config.width)
                .enumerate()
                .for_each(|(y, row)| {
                    let v_coord = y as f64 / (config.height as f64 - 1.0).max(1.0);
                    for (x, pixel) in row.iter_mut().enumerate() {
                        let u_coord = x as f64 / (config.width as f64 - 1.0).max(1.0);

                        // True barycentric UV-to-3D query
                        if let Some((base_pt, normal)) = uv_index.sample_surface(mesh, u_coord, v_coord) {
                            // Apply affine transform if present
                            let pt_transformed = affine.transform_point(&base_pt);
                            let normal_transformed = affine.transform_normal(&normal);

                            // Extrude along normal
                            let ray_pt = pt_transformed + (normal_transformed * offset);
                            let intensity = TrilinearSampler::sample(volume, &ray_pt);
                            *pixel = intensity.clamp(0.0, 65535.0) as u16;
                        } else {
                            *pixel = 0;
                        }
                    }
                });

            let layer_filename = format!("{:02}.tif", k);
            let layer_path = out_path.join(layer_filename);
            save_layer_tiff(&layer_path, config.width as u32, config.height as u32, &slice_data)?;

            progress.inc(1);
        }

        progress.finish_with_message("Mesh surface volume render complete.");
        Ok(())
    }

    /// Renders a surface volume directly from a VC3D `tifxyz` regular lattice.
    pub fn render_tifxyz_to_dir<P: AsRef<Path>>(
        volume: &ZarrVolume,
        tifxyz: &TifxyzMesh,
        config: &SurfaceVolumeConfig,
        output_dir: P,
    ) -> Result<()> {
        let out_path = output_dir.as_ref();
        create_dir_all(out_path)?;

        let num_layers = config.num_layers;
        let half_layers = (num_layers as f64 - 1.0) / 2.0;

        let affine = config.affine.unwrap_or_else(AffineTransform::identity);

        for k in 0..num_layers {
            let offset = (k as f64 - half_layers) * config.step_size;
            let mut slice_data = vec![0u16; tifxyz.width * tifxyz.height];

            slice_data
                .par_chunks_mut(tifxyz.width)
                .enumerate()
                .for_each(|(y, row)| {
                    for (x, pixel) in row.iter_mut().enumerate() {
                        if let Some(base_pt) = tifxyz.get_point(x, y) {
                            // Estimate normal from neighboring grid cells
                            let normal = estimate_grid_normal(tifxyz, x, y);
                            let pt_t = affine.transform_point(&base_pt);
                            let n_t = affine.transform_normal(&normal);

                            let ray_pt = pt_t + (n_t * offset);
                            let intensity = TrilinearSampler::sample(volume, &ray_pt);
                            *pixel = intensity.clamp(0.0, 65535.0) as u16;
                        } else {
                            *pixel = 0;
                        }
                    }
                });

            let layer_filename = format!("{:02}.tif", k);
            let layer_path = out_path.join(layer_filename);
            save_layer_tiff(&layer_path, tifxyz.width as u32, tifxyz.height as u32, &slice_data)?;
        }

        Ok(())
    }
}

fn estimate_grid_normal(mesh: &TifxyzMesh, x: usize, y: usize) -> Vec3D {
    let p_c = mesh.get_point(x, y).unwrap_or(volumen_core::Point3D::new(0.0, 0.0, 0.0));
    let p_r = mesh.get_point(x + 1, y).unwrap_or(p_c);
    let p_d = mesh.get_point(x, y + 1).unwrap_or(p_c);

    let du = p_r - p_c;
    let dv = p_d - p_c;
    du.cross(&dv).normalize().unwrap_or(Vec3D::new(0.0, 0.0, 1.0))
}

fn save_layer_tiff(path: &Path, width: u32, height: u32, data: &[u16]) -> Result<()> {
    use std::fs::File;
    use tiff::encoder::*;

    let file = File::create(path)?;
    let mut encoder = TiffEncoder::new(file)
        .map_err(|e| volumen_core::VolumenError::Tiff(e.to_string()))?;

    encoder
        .write_image::<colortype::Gray16>(width, height, data)
        .map_err(|e| volumen_core::VolumenError::Tiff(e.to_string()))?;

    Ok(())
}
