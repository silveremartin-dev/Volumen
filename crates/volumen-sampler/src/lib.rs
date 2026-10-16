//! High-throughput parallel trilinear interpolation and 2.5D surface volume generator.

pub mod interpolator;
pub mod surface_volume;

pub use interpolator::TrilinearSampler;
pub use surface_volume::{SurfaceVolumeConfig, SurfaceVolumeGenerator};
