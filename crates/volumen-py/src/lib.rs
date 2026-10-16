use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;
use volumen_geom::{NormalEstimator, SpiralParameters, SurfaceMesh};
use volumen_sampler::{SpiralUnrollConfig, SurfaceVolumeConfig, SurfaceVolumeGenerator};
use volumen_zarr::ZarrVolume;

/// Renders a 2.5D surface volume from an OME-Zarr array and an OBJ or tifxyz mesh.
/// Outputs either multi-layer 16-bit TIFF images, a native chunked OME-Zarr array, or both.
#[pyfunction]
#[pyo3(signature = (
    volume,
    mesh,
    output = None,
    zarr_output = None,
    num_layers = 64,
    step_size = 1.0,
    flip_normals = false,
    width = 1024,
    height = 1024,
    max_cache_gb = 8
))]
fn render_surface(
    py: Python<'_>,
    volume: String,
    mesh: String,
    output: Option<String>,
    zarr_output: Option<String>,
    num_layers: usize,
    step_size: f64,
    flip_normals: bool,
    width: usize,
    height: usize,
    max_cache_gb: usize,
) -> PyResult<()> {
    py.allow_threads(|| {
        let max_chunks = (max_cache_gb * 1024 * 1024 * 1024) / (128 * 128 * 128 * 2);
        let vol = ZarrVolume::open_auto(&volume, max_chunks)
            .map_err(|e| PyIOError::new_err(format!("Failed to open Zarr volume {}: {}", volume, e)))?;

        let config = SurfaceVolumeConfig {
            num_layers,
            step_size,
            width,
            height,
            affine: None,
            flip_normals,
        };

        let mesh_path = Path::new(&mesh);
        if mesh_path.is_dir() {
            let tifxyz = volumen_geom::TifxyzMesh::load_from_dir(mesh_path)
                .map_err(|e| PyValueError::new_err(format!("Failed to load tifxyz mesh: {}", e)))?;
            if let Some(ref out_dir) = output {
                SurfaceVolumeGenerator::render_tifxyz_to_dir(&vol, &tifxyz, &config, out_dir)
                    .map_err(|e| PyIOError::new_err(format!("Render error: {}", e)))?;
            }
        } else {
            let mut surface_mesh = SurfaceMesh::load_obj(mesh_path)
                .map_err(|e| PyValueError::new_err(format!("Failed to parse OBJ mesh {}: {}", mesh, e)))?;
            if surface_mesh.normals.is_empty() {
                NormalEstimator::compute_vertex_normals(&mut surface_mesh);
            }

            if let Some(ref out_dir) = output {
                SurfaceVolumeGenerator::render_mesh_to_dir(&vol, &surface_mesh, &config, out_dir)
                    .map_err(|e| PyIOError::new_err(format!("TIFF render error: {}", e)))?;
            }

            if let Some(ref zarr_dir) = zarr_output {
                SurfaceVolumeGenerator::render_mesh_to_zarr(&vol, &surface_mesh, &config, zarr_dir)
                    .map_err(|e| PyIOError::new_err(format!("Zarr render error: {}", e)))?;
            }
        }

        Ok(())
    })
}

/// Unrolls a continuous 3D Archimedean spiral manifold directly into a 2.5D surface volume,
/// with optional non-rigid deformation field support.
#[pyfunction]
#[pyo3(signature = (
    volume,
    output = None,
    zarr_output = None,
    center_x = 0.0,
    center_y = 0.0,
    a = 10.0,
    b = 2.0,
    turns_start = 0.0,
    turns_end = 2.0,
    z_min = 0.0,
    z_max = 100.0,
    num_layers = 32,
    step_size = 1.0,
    width = 1024,
    height = 512,
    deformation_field = None,
    max_cache_gb = 8
))]
fn unroll_spiral(
    py: Python<'_>,
    volume: String,
    output: Option<String>,
    zarr_output: Option<String>,
    center_x: f64,
    center_y: f64,
    a: f64,
    b: f64,
    turns_start: f64,
    turns_end: f64,
    z_min: f64,
    z_max: f64,
    num_layers: usize,
    step_size: f64,
    width: usize,
    height: usize,
    deformation_field: Option<String>,
    max_cache_gb: usize,
) -> PyResult<()> {
    py.allow_threads(|| {
        let max_chunks = (max_cache_gb * 1024 * 1024 * 1024) / (128 * 128 * 128 * 2);
        let vol = ZarrVolume::open_auto(&volume, max_chunks)
            .map_err(|e| PyIOError::new_err(format!("Failed to open Zarr volume {}: {}", volume, e)))?;

        let theta_min = turns_start * 2.0 * std::f64::consts::PI;
        let theta_max = turns_end * 2.0 * std::f64::consts::PI;

        let params = SpiralParameters::new(center_x, center_y, a, b);
        let deformation = if let Some(ref def_path) = deformation_field {
            Some(
                volumen_geom::DeformationField::load_auto(def_path, theta_min, theta_max, z_min, z_max)
                    .map_err(|e| PyValueError::new_err(format!("Failed to load deformation field: {}", e)))?,
            )
        } else {
            None
        };

        let manifold = volumen_geom::DeformedSpiralManifold::new(params, deformation);

        let config = SpiralUnrollConfig {
            theta_min,
            theta_max,
            z_min,
            z_max,
            width,
            height,
            num_layers,
            step_size,
        };

        if let Some(ref out_dir) = output {
            SurfaceVolumeGenerator::render_deformed_spiral_to_dir(&vol, &manifold, &config, out_dir)
                .map_err(|e| PyIOError::new_err(format!("TIFF spiral unroll error: {}", e)))?;
        }

        if let Some(ref zarr_dir) = zarr_output {
            SurfaceVolumeGenerator::render_deformed_spiral_to_zarr(&vol, &manifold, &config, zarr_dir)
                .map_err(|e| PyIOError::new_err(format!("Zarr spiral unroll error: {}", e)))?;
        }

        Ok(())
    })
}

/// Inspects OME-Zarr metadata and dimensions.
#[pyfunction]
fn inspect_volume(py: Python<'_>, volume: String) -> PyResult<PyObject> {
    let vol = ZarrVolume::open_auto(&volume, 16)
        .map_err(|e| PyIOError::new_err(format!("Failed to open volume: {}", e)))?;

    let meta = &vol.metadata;
    let (dx, dy, dz) = meta.dimensions();
    let (cx, cy, cz) = meta.chunk_sizes();

    let dict = PyDict::new(py);
    dict.set_item("zarr_format", meta.zarr_format)?;
    dict.set_item("dimensions", (dx, dy, dz))?;
    dict.set_item("chunks", (cx, cy, cz))?;
    dict.set_item("dtype", &meta.dtype)?;
    dict.set_item("is_remote", vol.is_remote)?;

    Ok(dict.into())
}

/// Volumen virtual unwrapping engine Python extension module.
#[pymodule]
fn volumen(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(render_surface, m)?)?;
    m.add_function(wrap_pyfunction!(unroll_spiral, m)?)?;
    m.add_function(wrap_pyfunction!(inspect_volume, m)?)?;
    Ok(())
}
