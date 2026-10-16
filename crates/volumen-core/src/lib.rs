//! Core types, 3D vector primitives, bounding boxes, and error types for Volumen.

pub mod bbox;
pub mod coords;
pub mod error;

pub use bbox::BoundingBox3D;
pub use coords::{Point3D, Vec3D, VoxelCoord};
pub use error::{Result, VolumenError};
