use crate::interpolator::TrilinearSampler;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::fs::create_dir_all;
use std::path::Path;
use volumen_core::{Result, Vec3D};
use volumen_geom::SurfaceMesh;
use volumen_zarr::ZarrVolume;

/// Configuration for 2.5D surface volume extraction.
#[derive(Debug, Clone)]
pub struct SurfaceVolumeConfig {
    pub num_layers: usize,
    pub step_size: f64,
    pub width: usize,
    pub height: usize,
}

impl Default for SurfaceVolumeConfig {
    fn default() -> Self {
        Self {
            num_layers: 64,
            step_size: 1.0,
            width: 1024,
            height: 1024,
        }
    }
}

/// Generates multi-layer 2.5D surface volume stacks from a volume and mesh.
pub struct SurfaceVolumeGenerator;

impl SurfaceVolumeGenerator {
    /// Renders a surface volume using parallel ray-casting across the mesh surface.
    pub fn render_to_dir<P: AsRef<Path>>(
        volume: &ZarrVolume,
        mesh: &SurfaceMesh,
        config: &SurfaceVolumeConfig,
        output_dir: P,
    ) -> Result<()> {
        let out_path = output_dir.as_ref();
        create_dir_all(out_path)?;

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

        // For each layer k, evaluate the slice in parallel
        for k in 0..num_layers {
            let offset = (k as f64 - half_layers) * config.step_size;

            // Generate an image slice buffer (16-bit grayscale)
            let mut slice_data = vec![0u16; config.width * config.height];

            // Sample points in parallel across the grid
            slice_data
                .par_chunks_mut(config.width)
                .enumerate()
                .for_each(|(y, row)| {
                    let v_coord = y as f64 / config.height as f64;
                    for (x, pixel) in row.iter_mut().enumerate() {
                        let u_coord = x as f64 / config.width as f64;

                        // Simple planar / vertex proxy mapping
                        if !mesh.vertices.is_empty() {
                            let base_pt = mesh.vertices[0];
                            let normal = mesh
                                .normals
                                .first()
                                .copied()
                                .unwrap_or(Vec3D::new(0.0, 0.0, 1.0));

                            let ray_pt = base_pt
                                + Vec3D::new(
                                    (u_coord - 0.5) * config.width as f64,
                                    (v_coord - 0.5) * config.height as f64,
                                    0.0,
                                )
                                + (normal * offset);

                            let intensity = TrilinearSampler::sample(volume, &ray_pt);
                            *pixel = intensity.clamp(0.0, 65535.0) as u16;
                        }
                    }
                });

            // Write layer TIFF
            let layer_filename = format!("{:02}.tif", k);
            let layer_path = out_path.join(layer_filename);
            save_layer_tiff(&layer_path, config.width as u32, config.height as u32, &slice_data)?;

            progress.inc(1);
        }

        progress.finish_with_message("Surface volume render complete.");
        Ok(())
    }
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
