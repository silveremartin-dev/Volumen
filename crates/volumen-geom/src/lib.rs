pub mod affine;
pub mod barycentric;
pub mod mesh;
pub mod normal;
pub mod spiral;
pub mod tifxyz;

pub use affine::AffineTransform;
pub use barycentric::{compute_barycentric_2d, UVSpatialIndex};
pub use mesh::{Face, SurfaceMesh, Vertex, UV};
pub use normal::NormalEstimator;
pub use spiral::{DeformationField, DeformedSpiralManifold, SpiralParameters, SpiralUnroller};
pub use tifxyz::TifxyzMesh;
