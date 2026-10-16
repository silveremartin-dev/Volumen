//! Mesh geometry, differential surface normals, OBJ parser, and topological validation.

pub mod mesh;
pub mod normal;

pub use mesh::{Face, SurfaceMesh, Vertex, UV};
pub use normal::NormalEstimator;
