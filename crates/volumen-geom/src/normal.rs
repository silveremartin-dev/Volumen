use crate::mesh::SurfaceMesh;
use volumen_core::Vec3D;

/// Computes vertex normals through area-weighted averaging of adjacent face normals.
pub struct NormalEstimator;

impl NormalEstimator {
    /// Recomputes smooth vertex normals for a given SurfaceMesh.
    pub fn compute_vertex_normals(mesh: &mut SurfaceMesh) {
        let mut vertex_normals = vec![Vec3D::new(0.0, 0.0, 0.0); mesh.vertices.len()];

        for face in &mesh.faces {
            let i0 = face.v[0];
            let i1 = face.v[1];
            let i2 = face.v[2];

            if i0 >= mesh.vertices.len() || i1 >= mesh.vertices.len() || i2 >= mesh.vertices.len() {
                continue;
            }

            let p0 = mesh.vertices[i0];
            let p1 = mesh.vertices[i1];
            let p2 = mesh.vertices[i2];

            let edge1 = p1 - p0;
            let edge2 = p2 - p0;
            let face_normal = edge1.cross(&edge2); // Magnitude proportional to face area

            vertex_normals[i0] = Vec3D::new(
                vertex_normals[i0].x + face_normal.x,
                vertex_normals[i0].y + face_normal.y,
                vertex_normals[i0].z + face_normal.z,
            );
            vertex_normals[i1] = Vec3D::new(
                vertex_normals[i1].x + face_normal.x,
                vertex_normals[i1].y + face_normal.y,
                vertex_normals[i1].z + face_normal.z,
            );
            vertex_normals[i2] = Vec3D::new(
                vertex_normals[i2].x + face_normal.x,
                vertex_normals[i2].y + face_normal.y,
                vertex_normals[i2].z + face_normal.z,
            );
        }

        // Normalize each accumulated normal vector
        mesh.normals = vertex_normals
            .into_iter()
            .map(|n| n.normalize().unwrap_or(Vec3D::new(0.0, 0.0, 1.0)))
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{Face, SurfaceMesh};
    use volumen_core::Point3D;

    #[test]
    fn test_planar_triangle_normal() {
        let mut mesh = SurfaceMesh::new();
        // Triangle on the XY plane: normal points along +Z
        mesh.vertices.push(Point3D::new(0.0, 0.0, 0.0));
        mesh.vertices.push(Point3D::new(1.0, 0.0, 0.0));
        mesh.vertices.push(Point3D::new(0.0, 1.0, 0.0));
        mesh.faces.push(Face {
            v: [0, 1, 2],
            uv: None,
        });

        NormalEstimator::compute_vertex_normals(&mut mesh);
        assert_eq!(mesh.normals.len(), 3);
        for n in &mesh.normals {
            assert!((n.x).abs() < 1e-9);
            assert!((n.y).abs() < 1e-9);
            assert!((n.z - 1.0).abs() < 1e-9);
        }
    }
}
