use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use volumen_geom::{NormalEstimator, SurfaceMesh, TifxyzMesh};
use volumen_sampler::{SurfaceVolumeConfig, SurfaceVolumeGenerator};
use volumen_zarr::ZarrVolume;

/// Drop-in replacement for Volume Cartographer's legacy `vc_render` / `vc_render_tifxyz`.
/// Emulates identical CLI syntax for seamless Khartes and villa/render_ink.py integration.
#[derive(Parser, Debug)]
#[command(name = "vc_render")]
#[command(about = "High-performance drop-in replacement for legacy Volume Cartographer vc_render")]
struct VcRenderArgs {
    /// Path to the segmented mesh Wavefront OBJ file or tifxyz directory
    #[arg(short = 's', long = "segmentation", alias = "mesh")]
    segmentation: PathBuf,

    /// Path to the volume (or volume name within a volpkg, or s3:// URL)
    #[arg(short = 'v', long = "volume")]
    volume: Option<String>,

    /// Optional path to Volume Cartographer .volpkg directory
    #[arg(long = "volpkg")]
    volpkg: Option<PathBuf>,

    /// Output directory where 2.5D TIFF layer slices will be saved
    #[arg(short = 'o', long = "output", alias = "tif-output")]
    output: PathBuf,

    /// Number of depth layers to extract along the surface normal
    #[arg(short = 'n', long = "num-slices", default_value_t = 64)]
    num_slices: usize,

    /// Normal sampling step size in voxels
    #[arg(long = "step", alias = "slice-step", default_value_t = 1.0)]
    step: f64,

    /// Scale multiplier for output rendering
    #[arg(long = "scale", default_value_t = 1.0)]
    scale: f64,

    /// Scale factor applied to segmentation mesh coordinates
    #[arg(long = "scale-segmentation", default_value_t = 1.0)]
    scale_segmentation: f64,

    /// OME-Zarr resolution pyramid group index
    #[arg(short = 'g', long = "group-idx", default_value_t = 0)]
    group_idx: usize,

    /// Reverse slice ordering along normal
    #[arg(long = "flip-normals", default_value_t = false)]
    flip_normals: bool,

    /// Automatically center the surface extraction on the papyrus wall
    #[arg(long = "auto-center", default_value_t = false)]
    auto_center: bool,

    /// Output image width in pixels (for OBJ meshes; tifxyz preserves native grid width)
    #[arg(long = "width", default_value_t = 1024)]
    width: usize,

    /// Output image height in pixels (for OBJ meshes; tifxyz preserves native grid height)
    #[arg(long = "height", default_value_t = 1024)]
    height: usize,

    /// Maximum chunk cache memory in GB
    #[arg(long = "cache-gb", default_value_t = 8)]
    max_cache_gb: usize,
}

fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = VcRenderArgs::parse();

    // Resolve volume path: support --volpkg <dir> with --volume <name>
    let volume_target = match (&args.volpkg, &args.volume) {
        (Some(pkg), Some(vol_name)) => {
            let candidate = pkg.join("volumes").join(vol_name);
            if candidate.exists() {
                candidate.to_string_lossy().to_string()
            } else {
                vol_name.clone()
            }
        }
        (Some(pkg), None) => {
            // Find first volume directory in <volpkg>/volumes/
            let vol_dir = pkg.join("volumes");
            if let Ok(entries) = std::fs::read_dir(&vol_dir) {
                let first = entries.filter_map(|e| e.ok()).next();
                if let Some(entry) = first {
                    entry.path().to_string_lossy().to_string()
                } else {
                    pkg.to_string_lossy().to_string()
                }
            } else {
                pkg.to_string_lossy().to_string()
            }
        }
        (None, Some(vol)) => vol.clone(),
        (None, None) => {
            anyhow::bail!("Error: either --volume or --volpkg must be provided to vc_render");
        }
    };

    println!(
        "[vc_render shim] Launching high-speed Volumen engine for: {}",
        volume_target
    );
    let max_chunks = (args.max_cache_gb * 1024 * 1024 * 1024) / (128 * 128 * 128 * 2);
    let vol = ZarrVolume::open_auto(&volume_target, max_chunks)?;

    let config = SurfaceVolumeConfig {
        num_layers: args.num_slices,
        step_size: args.step,
        width: args.width,
        height: args.height,
        affine: None,
        flip_normals: args.flip_normals,
        auto_center: args.auto_center,
    };

    if args.segmentation.is_dir() {
        println!(
            "[vc_render shim] Loading VC3D tifxyz lattice: {}",
            args.segmentation.display()
        );
        let tifxyz = TifxyzMesh::load_from_dir(&args.segmentation)?;
        SurfaceVolumeGenerator::render_tifxyz_to_dir(&vol, &tifxyz, &config, &args.output)?;
    } else {
        println!(
            "[vc_render shim] Loading Wavefront OBJ mesh: {}",
            args.segmentation.display()
        );
        let mut surface_mesh = SurfaceMesh::load_obj(&args.segmentation)?;
        if surface_mesh.normals.is_empty() {
            NormalEstimator::compute_vertex_normals(&mut surface_mesh);
        }
        SurfaceVolumeGenerator::render_mesh_to_dir(&vol, &surface_mesh, &config, &args.output)?;
    }

    println!(
        "[vc_render shim] Complete. Layers successfully written to {}",
        args.output.display()
    );
    Ok(())
}
