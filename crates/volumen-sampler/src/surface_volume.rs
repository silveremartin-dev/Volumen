use crate::interpolator::TrilinearSampler;
use rayon::prelude::*;
use std::fs::create_dir_all;
use std::path::Path;
use volumen_core::{Result, Vec3D};
use volumen_geom::{AffineTransform, SurfaceMesh, TifxyzMesh, UVSpatialIndex};
use volumen_zarr::{ZarrVolume, ZarrWriter};

/// Configuration for 2.5D surface volume extraction.
#[derive(Debug, Clone)]
pub struct SurfaceVolumeConfig {
    pub num_layers: usize,
    pub step_size: f64,
    pub width: usize,
    pub height: usize,
    pub affine: Option<AffineTransform>,
    pub flip_normals: bool,
    pub auto_center: bool,
}

impl Default for SurfaceVolumeConfig {
    fn default() -> Self {
        Self {
            num_layers: 64,
            step_size: 1.0,
            width: 1024,
            height: 1024,
            affine: None,
            flip_normals: false,
            auto_center: false,
        }
    }
}

/// Configuration for continuous Archimedean spiral unrolling.
#[derive(Debug, Clone)]
pub struct SpiralUnrollConfig {
    pub theta_min: f64,
    pub theta_max: f64,
    pub z_min: f64,
    pub z_max: f64,
    pub width: usize,
    pub height: usize,
    pub num_layers: usize,
    pub step_size: f64,
}

impl Default for SpiralUnrollConfig {
    fn default() -> Self {
        Self {
            theta_min: 0.0,
            theta_max: 4.0 * std::f64::consts::PI, // Default 2 turns
            z_min: 0.0,
            z_max: 100.0,
            width: 1024,
            height: 512,
            num_layers: 32,
            step_size: 1.0,
        }
    }
}

/// High-performance Surface Volume generator with exact barycentric ray-casting.
pub struct SurfaceVolumeGenerator;

impl SurfaceVolumeGenerator {
    /// Renders a surface volume using true barycentric ray-casting across a Wavefront OBJ mesh.
    /// Renders a surface volume using true barycentric ray-casting directly into in-memory layer buffers.
    pub fn render_mesh_to_slices(
        volume: &ZarrVolume,
        mesh: &SurfaceMesh,
        config: &SurfaceVolumeConfig,
    ) -> Result<Vec<Vec<u16>>> {
        let uv_index = UVSpatialIndex::build(mesh, 128);
        let num_layers = config.num_layers;
        let half_layers = (num_layers as f64 - 1.0) / 2.0;
        let affine = config.affine.unwrap_or_else(AffineTransform::identity);

        let mut slices = Vec::with_capacity(num_layers);

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

                        if let Some((base_pt, normal)) =
                            uv_index.sample_surface(mesh, u_coord, v_coord)
                        {
                            let pt_transformed = affine.transform_point(&base_pt);
                            let normal_transformed = affine.transform_normal(&normal);
                            let normal_final = if config.flip_normals {
                                normal_transformed * -1.0
                            } else {
                                normal_transformed
                            };

                            let pt_base = if config.auto_center {
                                let shift = compute_sheet_centering_offset(
                                    volume,
                                    &pt_transformed,
                                    &normal_final,
                                    6.0,
                                );
                                pt_transformed + (normal_final * shift)
                            } else {
                                pt_transformed
                            };

                            let ray_pt = pt_base + (normal_final * offset);
                            let intensity = TrilinearSampler::sample(volume, &ray_pt);
                            *pixel = intensity.round().clamp(0.0, 65535.0) as u16;
                        } else {
                            *pixel = 0;
                        }
                    }
                });

            slices.push(slice_data);
        }

        Ok(slices)
    }

    /// Renders a surface volume using true barycentric ray-casting across a Wavefront OBJ mesh to TIFF files.
    pub fn render_mesh_to_dir<P: AsRef<Path>>(
        volume: &ZarrVolume,
        mesh: &SurfaceMesh,
        config: &SurfaceVolumeConfig,
        output_dir: P,
    ) -> Result<()> {
        let out_path = output_dir.as_ref();
        create_dir_all(out_path)?;

        let slices = Self::render_mesh_to_slices(volume, mesh, config)?;
        for (k, slice_data) in slices.iter().enumerate() {
            let layer_filename = format!("{:02}.tif", k);
            let layer_path = out_path.join(layer_filename);
            save_layer_tiff(
                &layer_path,
                config.width as u32,
                config.height as u32,
                slice_data,
            )?;
        }

        Ok(())
    }

    /// Renders a surface volume using true barycentric ray-casting directly into a chunked OME-Zarr array.
    pub fn render_mesh_to_zarr<P: AsRef<Path>>(
        volume: &ZarrVolume,
        mesh: &SurfaceMesh,
        config: &SurfaceVolumeConfig,
        zarr_output_dir: P,
    ) -> Result<()> {
        let slices = Self::render_mesh_to_slices(volume, mesh, config)?;
        let writer = ZarrWriter::create(
            zarr_output_dir,
            [
                config.num_layers as u64,
                config.height as u64,
                config.width as u64,
            ],
            [config.num_layers as u64, 128, 128],
        )?;
        writer.write_slices(&slices, config.width, config.height)?;
        Ok(())
    }

    /// Renders a surface volume directly from a VC3D `tifxyz` regular lattice.
    /// Renders a surface volume directly from a VC3D `tifxyz` regular lattice into in-memory layer buffers.
    pub fn render_tifxyz_to_slices(
        volume: &ZarrVolume,
        tifxyz: &TifxyzMesh,
        config: &SurfaceVolumeConfig,
    ) -> Result<Vec<Vec<u16>>> {
        let num_layers = config.num_layers;
        let half_layers = (num_layers as f64 - 1.0) / 2.0;
        let affine = config.affine.unwrap_or_else(AffineTransform::identity);

        let mut slices = Vec::with_capacity(num_layers);

        for k in 0..num_layers {
            let offset = (k as f64 - half_layers) * config.step_size;
            let mut slice_data = vec![0u16; tifxyz.width * tifxyz.height];

            slice_data
                .par_chunks_mut(tifxyz.width)
                .enumerate()
                .for_each(|(y, row)| {
                    for (x, pixel) in row.iter_mut().enumerate() {
                        if let Some(base_pt) = tifxyz.get_point(x, y) {
                            let normal = estimate_grid_normal(tifxyz, x, y);
                            let pt_t = affine.transform_point(&base_pt);
                            let n_t = affine.transform_normal(&normal);
                            let n_final = if config.flip_normals { n_t * -1.0 } else { n_t };

                            let pt_base = if config.auto_center {
                                let shift =
                                    compute_sheet_centering_offset(volume, &pt_t, &n_final, 6.0);
                                pt_t + (n_final * shift)
                            } else {
                                pt_t
                            };

                            let ray_pt = pt_base + (n_final * offset);
                            let intensity = TrilinearSampler::sample(volume, &ray_pt);
                            *pixel = intensity.round().clamp(0.0, 65535.0) as u16;
                        } else {
                            *pixel = 0;
                        }
                    }
                });

            slices.push(slice_data);
        }

        Ok(slices)
    }

    /// Renders a surface volume directly from a VC3D `tifxyz` regular lattice to TIFF files.
    pub fn render_tifxyz_to_dir<P: AsRef<Path>>(
        volume: &ZarrVolume,
        tifxyz: &TifxyzMesh,
        config: &SurfaceVolumeConfig,
        output_dir: P,
    ) -> Result<()> {
        let out_path = output_dir.as_ref();
        create_dir_all(out_path)?;

        let slices = Self::render_tifxyz_to_slices(volume, tifxyz, config)?;
        for (k, slice_data) in slices.iter().enumerate() {
            let layer_filename = format!("{:02}.tif", k);
            let layer_path = out_path.join(layer_filename);
            save_layer_tiff(
                &layer_path,
                tifxyz.width as u32,
                tifxyz.height as u32,
                slice_data,
            )?;
        }

        Ok(())
    }

    /// Renders an unrolled surface volume directly from an analytical 3D Archimedean spiral manifold.
    pub fn render_spiral_to_dir<P: AsRef<Path>>(
        volume: &ZarrVolume,
        params: &volumen_geom::SpiralParameters,
        config: &SpiralUnrollConfig,
        output_dir: P,
    ) -> Result<()> {
        let manifold = volumen_geom::DeformedSpiralManifold::new(*params, None);
        Self::render_deformed_spiral_to_dir(volume, &manifold, config, output_dir)
    }

    /// Renders an unrolled surface volume directly from an analytical or non-rigidly deformed
    /// Archimedean spiral manifold into in-memory layer buffers.
    pub fn render_deformed_spiral_to_slices(
        volume: &ZarrVolume,
        manifold: &volumen_geom::DeformedSpiralManifold,
        config: &SpiralUnrollConfig,
    ) -> Result<Vec<Vec<u16>>> {
        let num_layers = config.num_layers;
        let half_layers = (num_layers as f64 - 1.0) / 2.0;

        let d_theta = if config.width > 1 {
            (config.theta_max - config.theta_min) / (config.width as f64 - 1.0)
        } else {
            0.0
        };

        let d_z = if config.height > 1 {
            (config.z_max - config.z_min) / (config.height as f64 - 1.0)
        } else {
            0.0
        };

        // Precompute grid points and normals across (width, height)
        let total_pixels = config.width * config.height;
        let grid_pts_normals: Vec<(volumen_core::Point3D, volumen_core::Vec3D)> = (0..total_pixels)
            .into_par_iter()
            .map(|idx| {
                let x = idx % config.width;
                let y = idx / config.width;
                let theta = config.theta_min + (x as f64) * d_theta;
                let z = config.z_min + (y as f64) * d_z;
                let base_pt = manifold.point_at(theta, z);
                let normal = manifold.normal_at(theta, z);
                (base_pt, normal)
            })
            .collect();

        let mut slices = Vec::with_capacity(num_layers);

        for k in 0..num_layers {
            let offset = (k as f64 - half_layers) * config.step_size;
            let mut slice_data = vec![0u16; config.width * config.height];

            slice_data
                .par_chunks_mut(config.width)
                .enumerate()
                .for_each(|(y, row)| {
                    let y_offset = y * config.width;
                    for (x, pixel) in row.iter_mut().enumerate() {
                        let (base_pt, normal) = grid_pts_normals[y_offset + x];
                        let ray_pt = base_pt + (normal * offset);
                        let intensity = TrilinearSampler::sample(volume, &ray_pt);
                        *pixel = intensity.round().clamp(0.0, 65535.0) as u16;
                    }
                });

            slices.push(slice_data);
        }

        Ok(slices)
    }

    /// Renders an unrolled surface volume directly from an analytical or non-rigidly deformed
    /// Archimedean spiral manifold to TIFF files. This accounts for real carbonized scroll compression and local warping.
    pub fn render_deformed_spiral_to_dir<P: AsRef<Path>>(
        volume: &ZarrVolume,
        manifold: &volumen_geom::DeformedSpiralManifold,
        config: &SpiralUnrollConfig,
        output_dir: P,
    ) -> Result<()> {
        let out_path = output_dir.as_ref();
        create_dir_all(out_path)?;

        tracing::info!(
            "Unrolling spiral manifold: theta=[{:.2}, {:.2}], z=[{:.2}, {:.2}], {} layers to {} (non-rigid deformation: {})",
            config.theta_min,
            config.theta_max,
            config.z_min,
            config.z_max,
            config.num_layers,
            out_path.display(),
            manifold.deformation.is_some()
        );

        let slices = Self::render_deformed_spiral_to_slices(volume, manifold, config)?;
        for (k, slice_data) in slices.iter().enumerate() {
            let layer_filename = format!("{:02}.tif", k);
            let layer_path = out_path.join(layer_filename);
            save_layer_tiff(
                &layer_path,
                config.width as u32,
                config.height as u32,
                slice_data,
            )?;
        }

        Ok(())
    }

    /// Renders an unrolled spiral surface volume directly into a chunked OME-Zarr array.
    pub fn render_deformed_spiral_to_zarr<P: AsRef<Path>>(
        volume: &ZarrVolume,
        manifold: &volumen_geom::DeformedSpiralManifold,
        config: &SpiralUnrollConfig,
        zarr_output_dir: P,
    ) -> Result<()> {
        let slices = Self::render_deformed_spiral_to_slices(volume, manifold, config)?;
        let writer = ZarrWriter::create(
            zarr_output_dir,
            [
                config.num_layers as u64,
                config.height as u64,
                config.width as u64,
            ],
            [config.num_layers as u64, 128, 128],
        )?;
        writer.write_slices(&slices, config.width, config.height)?;
        Ok(())
    }
}

fn estimate_grid_normal(mesh: &TifxyzMesh, x: usize, y: usize) -> Vec3D {
    let p_c = mesh
        .get_point(x, y)
        .unwrap_or(volumen_core::Point3D::new(0.0, 0.0, 0.0));
    let p_r = mesh.get_point(x + 1, y).unwrap_or(p_c);
    let p_d = mesh.get_point(x, y + 1).unwrap_or(p_c);

    let du = p_r - p_c;
    let dv = p_d - p_c;
    du.cross(&dv)
        .normalize()
        .unwrap_or(Vec3D::new(0.0, 0.0, 1.0))
}

/// Probes intensity along normal in range [-probe_range, +probe_range] voxels
/// and calculates the barycentric center of mass of the papyrus sheet density.
pub fn compute_sheet_centering_offset(
    volume: &ZarrVolume,
    base_pt: &volumen_core::Point3D,
    normal: &Vec3D,
    probe_range: f64,
) -> f64 {
    let steps = 16;
    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;

    for i in 0..=steps {
        let t = -probe_range + (2.0 * probe_range * (i as f64) / (steps as f64));
        let pt = *base_pt + (*normal * t);
        let val = TrilinearSampler::sample(volume, &pt);
        // Papyrus density contrast over background air
        if val > 1000.0 {
            let weight = (val - 1000.0).powi(2);
            weighted_sum += t * weight;
            total_weight += weight;
        }
    }

    if total_weight > 1e-6 {
        (weighted_sum / total_weight).clamp(-probe_range * 0.75, probe_range * 0.75)
    } else {
        0.0
    }
}

fn save_layer_tiff(path: &Path, width: u32, height: u32, data: &[u16]) -> Result<()> {
    use std::fs::File;
    use tiff::encoder::*;

    let file = File::create(path)?;
    let mut encoder =
        TiffEncoder::new(file).map_err(|e| volumen_core::VolumenError::Tiff(e.to_string()))?;

    encoder
        .write_image::<colortype::Gray16>(width, height, data)
        .map_err(|e| volumen_core::VolumenError::Tiff(e.to_string()))?;

    Ok(())
}
