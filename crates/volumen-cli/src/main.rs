use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use volumen_geom::{NormalEstimator, SurfaceMesh};
use volumen_sampler::{SurfaceVolumeConfig, SurfaceVolumeGenerator};
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
    /// Render a 2.5D surface volume from an OME-Zarr volume and an OBJ mesh
    Render {
        /// Path to the OME-Zarr scroll volume root directory
        #[arg(short, long)]
        volume: PathBuf,

        /// Path to the segmented mesh Wavefront OBJ file
        #[arg(short, long)]
        mesh: PathBuf,

        /// Output directory where 2.5D TIFF layer slices will be saved
        #[arg(short, long)]
        output: PathBuf,

        /// Number of depth layers to extract along the surface normal
        #[arg(short, long, default_value_t = 64)]
        layers: usize,

        /// Normal sampling step size in voxels
        #[arg(short, long, default_value_t = 1.0)]
        step: f64,

        /// Output image width in pixels
        #[arg(long, default_value_t = 1024)]
        width: usize,

        /// Output image height in pixels
        #[arg(long, default_value_t = 1024)]
        height: usize,

        /// Maximum chunk cache memory in GB
        #[arg(long, default_value_t = 8)]
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
            width,
            height,
            max_cache_gb,
        } => {
            println!("Opening OME-Zarr volume: {}", volume.display());
            let max_chunks = (max_cache_gb * 1024 * 1024 * 1024) / (128 * 128 * 128 * 2);
            let vol = ZarrVolume::open(&volume, max_chunks)?;

            println!("Loading Wavefront OBJ mesh: {}", mesh.display());
            let mut surface_mesh = SurfaceMesh::load_obj(&mesh)?;

            if surface_mesh.normals.is_empty() {
                println!("Computing smooth differential vertex normals...");
                NormalEstimator::compute_vertex_normals(&mut surface_mesh);
            }

            let config = SurfaceVolumeConfig {
                num_layers: layers,
                step_size: step,
                width,
                height,
                affine: None,
            };

            if mesh.is_dir() {
                println!("Loading VC3D tifxyz lattice: {}", mesh.display());
                let tifxyz = volumen_geom::TifxyzMesh::load_from_dir(&mesh)?;
                SurfaceVolumeGenerator::render_tifxyz_to_dir(&vol, &tifxyz, &config, &output)?;
            } else {
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
    }

    Ok(())
}
