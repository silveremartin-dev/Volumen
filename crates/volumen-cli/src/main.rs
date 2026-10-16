use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use volumen_geom::{NormalEstimator, SpiralParameters, SurfaceMesh};
use volumen_sampler::{SpiralUnrollConfig, SurfaceVolumeConfig, SurfaceVolumeGenerator};
use volumen_zarr::ZarrVolume;

#[derive(Parser)]
#[command(name = "volumen")]
#[command(about = "High-performance virtual unwrapping engine for the Vesuvius Challenge", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Render a 2.5D surface volume from an OME-Zarr volume and an OBJ or tifxyz mesh (drop-in replacement for vc_render_tifxyz)
    Render {
        /// Path to the OME-Zarr scroll volume root directory
        #[arg(short, long, alias = "v")]
        volume: PathBuf,

        /// Path to the segmented mesh Wavefront OBJ file or tifxyz directory
        #[arg(short, long, alias = "segmentation", alias = "s")]
        mesh: PathBuf,

        /// Output directory where 2.5D TIFF layer slices will be saved
        #[arg(short, long, alias = "tif-output")]
        output: PathBuf,

        /// Number of depth layers to extract along the surface normal
        #[arg(short, long, alias = "num-slices", alias = "n", default_value_t = 64)]
        layers: usize,

        /// Normal sampling step size in voxels
        #[arg(short, long, alias = "slice-step", default_value_t = 1.0)]
        step: f64,

        /// Scale multiplier for output rendering
        #[arg(long, default_value_t = 1.0)]
        scale: f64,

        /// Scale factor applied to segmentation mesh coordinates
        #[arg(long, default_value_t = 1.0)]
        scale_segmentation: f64,

        /// OME-Zarr resolution pyramid group index
        #[arg(short = 'g', long, default_value_t = 0)]
        group_idx: usize,

        /// Reverse slice ordering along normal
        #[arg(long, default_value_t = false)]
        flip_normals: bool,

        /// Output image width in pixels (for OBJ meshes; tifxyz preserves native grid width)
        #[arg(long, default_value_t = 1024)]
        width: usize,

        /// Output image height in pixels (for OBJ meshes; tifxyz preserves native grid height)
        #[arg(long, default_value_t = 1024)]
        height: usize,

        /// Maximum chunk cache memory in GB
        #[arg(long, alias = "cache-gb", default_value_t = 8)]
        max_cache_gb: usize,
    },


    /// Inspect OME-Zarr metadata, chunks, and dimensions
    Inspect {
        /// Path to the OME-Zarr scroll volume root directory
        #[arg(short, long)]
        volume: PathBuf,
    },

    /// Verify Wavefront OBJ mesh integrity, normal vectors, and bounding box
    Verify {
        /// Path to the Wavefront OBJ file
        #[arg(short, long)]
        mesh: PathBuf,
    },

    /// Unroll continuous 3D Archimedean spiral manifold directly into 2.5D surface volume
    UnrollSpiral {
        /// Path to the OME-Zarr scroll volume root directory
        #[arg(short, long)]
        volume: PathBuf,

        /// Output directory where unrolled TIFF layer slices will be saved
        #[arg(short, long)]
        output: PathBuf,

        /// Center X coordinate of scroll spiral core
        #[arg(long, default_value_t = 0.0)]
        center_x: f64,

        /// Center Y coordinate of scroll spiral core
        #[arg(long, default_value_t = 0.0)]
        center_y: f64,

        /// Base radius at theta = 0
        #[arg(short, long, default_value_t = 10.0)]
        a: f64,

        /// Radial expansion pitch per radian: r(theta) = a + b * theta
        #[arg(short, long, default_value_t = 2.0)]
        b: f64,

        /// Starting angle in turns (e.g. 0.0)
        #[arg(long, default_value_t = 0.0)]
        turns_start: f64,

        /// Ending angle in turns (e.g. 5.0)
        #[arg(long, default_value_t = 2.0)]
        turns_end: f64,

        /// Starting Z coordinate in voxels
        #[arg(long, default_value_t = 0.0)]
        z_min: f64,

        /// Ending Z coordinate in voxels
        #[arg(long, default_value_t = 100.0)]
        z_max: f64,

        /// Number of depth layers across papyrus sheet
        #[arg(short, long, default_value_t = 32)]
        layers: usize,

        /// Normal sampling step size in voxels
        #[arg(short, long, default_value_t = 1.0)]
        step: f64,

        /// Output image width (angular / arc-length resolution)
        #[arg(long, default_value_t = 1024)]
        width: usize,

        /// Output image height (axial Z resolution)
        #[arg(long, default_value_t = 512)]
        height: usize,

        /// Maximum chunk cache memory in GB
        #[arg(long, default_value_t = 8)]
        max_cache_gb: usize,

        /// Optional path to non-rigid deformation field JSON (produced by fit_spiral / neural network)
        #[arg(long, alias = "flow-field")]
        deformation_field: Option<PathBuf>,
    },

}

fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Render {
            volume,
            mesh,
            output,
            layers,
            step,
            scale: _scale,
            scale_segmentation: _scale_segmentation,
            group_idx: _group_idx,
            flip_normals,
            width,
            height,
            max_cache_gb,
        } => {
            println!("Opening OME-Zarr volume: {}", volume.display());
            let max_chunks = (max_cache_gb * 1024 * 1024 * 1024) / (128 * 128 * 128 * 2);
            let vol = ZarrVolume::open(&volume, max_chunks)?;

            let config = SurfaceVolumeConfig {
                num_layers: layers,
                step_size: step,
                width,
                height,
                affine: None,
                flip_normals,
            };


            if mesh.is_dir() {
                println!("Loading VC3D tifxyz lattice: {}", mesh.display());
                let tifxyz = volumen_geom::TifxyzMesh::load_from_dir(&mesh)?;
                SurfaceVolumeGenerator::render_tifxyz_to_dir(&vol, &tifxyz, &config, &output)?;
            } else {
                println!("Loading Wavefront OBJ mesh: {}", mesh.display());
                let mut surface_mesh = SurfaceMesh::load_obj(&mesh)?;
                if surface_mesh.normals.is_empty() {
                    println!("Computing smooth differential vertex normals...");
                    NormalEstimator::compute_vertex_normals(&mut surface_mesh);
                }
                SurfaceVolumeGenerator::render_mesh_to_dir(&vol, &surface_mesh, &config, &output)?;
            }
            println!("Surface volume successfully rendered to {}", output.display());
        }


        Commands::Inspect { volume } => {
            println!("Inspecting OME-Zarr volume: {}", volume.display());
            let meta = volumen_zarr::ZarrMetadata::load_from_dir(&volume)?;
            let (dx, dy, dz) = meta.dimensions();
            let (cx, cy, cz) = meta.chunk_sizes();
            println!("- Zarr Format: {}", meta.zarr_format);
            println!("- Dimensions (X, Y, Z): ({}, {}, {}) voxels", dx, dy, dz);
            println!("- Chunk Dimensions (CX, CY, CZ): ({}, {}, {}) voxels", cx, cy, cz);
            println!("- Data Type: {}", meta.dtype);
        }

        Commands::Verify { mesh } => {
            println!("Verifying Wavefront OBJ mesh: {}", mesh.display());
            let surface_mesh = SurfaceMesh::load_obj(&mesh)?;
            println!("- Vertices: {}", surface_mesh.vertices.len());
            println!("- Faces: {}", surface_mesh.faces.len());
            println!("- UV Coordinates: {}", surface_mesh.uvs.len());
            println!("- Normals: {}", surface_mesh.normals.len());

            if let Some(bbox) = surface_mesh.bounding_box() {
                let (w, h, d) = bbox.dimensions();
                println!(
                    "- Bounding Box: [{:.2}, {:.2}, {:.2}] to [{:.2}, {:.2}, {:.2}]",
                    bbox.min.x, bbox.min.y, bbox.min.z, bbox.max.x, bbox.max.y, bbox.max.z
                );
                println!("- Dimensions (W, H, D): {:.2} x {:.2} x {:.2}", w, h, d);
            }

            if surface_mesh.normals.is_empty() {
                println!("- Warning: No normal vectors defined in file. Volumen can synthesize them on-the-fly.");
            }
        }

        Commands::UnrollSpiral {
            volume,
            output,
            center_x,
            center_y,
            a,
            b,
            turns_start,
            turns_end,
            z_min,
            z_max,
            layers,
            step,
            width,
            height,
            max_cache_gb,
            deformation_field,
        } => {
            println!("Opening OME-Zarr volume: {}", volume.display());
            let max_chunks = (max_cache_gb * 1024 * 1024 * 1024) / (128 * 128 * 128 * 2);
            let vol = ZarrVolume::open(&volume, max_chunks)?;

            let params = SpiralParameters::new(center_x, center_y, a, b);
            let deformation = if let Some(ref def_path) = deformation_field {
                println!("Loading non-rigid deformation field: {}", def_path.display());
                Some(volumen_geom::DeformationField::load_json(def_path)?)
            } else {
                None
            };
            let manifold = volumen_geom::DeformedSpiralManifold::new(params, deformation);

            let theta_min = turns_start * 2.0 * std::f64::consts::PI;
            let theta_max = turns_end * 2.0 * std::f64::consts::PI;

            let config = SpiralUnrollConfig {
                theta_min,
                theta_max,
                z_min,
                z_max,
                width,
                height,
                num_layers: layers,
                step_size: step,
            };

            println!(
                "Unrolling spiral manifold [turns: {:.2} -> {:.2}, z: {:.1} -> {:.1}] to {}...",
                turns_start, turns_end, z_min, z_max, output.display()
            );

            SurfaceVolumeGenerator::render_deformed_spiral_to_dir(&vol, &manifold, &config, &output)?;
            println!("Spiral surface volume successfully unrolled to {}", output.display());
        }

    }

    Ok(())
}

