use crate::mesh::{SurfaceMesh, UV};
use volumen_core::{Point3D, Vec3D};

/// Accelerated 2D spatial grid for fast (u, v) -> triangle candidate lookup.
pub struct UVSpatialIndex {
    grid_res: usize,
    /// For each grid cell (cu, cv), list of face indices that overlap the cell.
    cells: Vec<Vec<usize>>,
}

impl UVSpatialIndex {
    /// Builds a spatial hash index over the UV parameterization of a SurfaceMesh.
    pub fn build(mesh: &SurfaceMesh, grid_res: usize) -> Self {
        let mut cells = vec![Vec::new(); grid_res * grid_res];

        for (face_idx, face) in mesh.faces.iter().enumerate() {
            if let Some(uv_indices) = face.uv {
                let uv0 = mesh.uvs[uv_indices[0]];
                let uv1 = mesh.uvs[uv_indices[1]];
                let uv2 = mesh.uvs[uv_indices[2]];

                let min_u = uv0.u.min(uv1.u).min(uv2.u).clamp(0.0, 1.0);
                let max_u = uv0.u.max(uv1.u).max(uv2.u).clamp(0.0, 1.0);
                let min_v = uv0.v.min(uv1.v).min(uv2.v).clamp(0.0, 1.0);
                let max_v = uv0.v.max(uv1.v).max(uv2.v).clamp(0.0, 1.0);

                let cu_min = ((min_u * (grid_res as f64 - 1e-5)).floor() as usize).min(grid_res - 1);
                let cu_max = ((max_u * (grid_res as f64 - 1e-5)).floor() as usize).min(grid_res - 1);
                let cv_min = ((min_v * (grid_res as f64 - 1e-5)).floor() as usize).min(grid_res - 1);
                let cv_max = ((max_v * (grid_res as f64 - 1e-5)).floor() as usize).min(grid_res - 1);

                for cv in cv_min..=cv_max {
                    for cu in cu_min..=cu_max {
                        cells[cv * grid_res + cu].push(face_idx);
                    }
                }
            }
        }

        Self { grid_res, cells }
    }

    /// Finds the 3D surface point and interpolated unit normal for any continuous (u, v) coordinate.
    pub fn sample_surface(
        &self,
        mesh: &SurfaceMesh,
        u: f64,
        v: f64,
    ) -> Option<(Point3D, Vec3D)> {
        if u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0 {
            return None;
        }

        let cu = ((u * (self.grid_res as f64 - 1e-5)).floor() as usize).min(self.grid_res - 1);
        let cv = ((v * (self.grid_res as f64 - 1e-5)).floor() as usize).min(self.grid_res - 1);

        let candidate_faces = &self.cells[cv * self.grid_res + cu];

        for &face_idx in candidate_faces {
            let face = &mesh.faces[face_idx];
            if let Some(uv_indices) = face.uv {
                let uv0 = mesh.uvs[uv_indices[0]];
                let uv1 = mesh.uvs[uv_indices[1]];
                let uv2 = mesh.uvs[uv_indices[2]];

                if let Some((w0, w1, w2)) = compute_barycentric_2d(u, v, &uv0, &uv1, &uv2) {
                    // Interpolate 3D position
                    let p0 = mesh.vertices[face.v[0]];
                    let p1 = mesh.vertices[face.v[1]];
                    let p2 = mesh.vertices[face.v[2]];

                    let pos = Point3D::new(
                        w0 * p0.x + w1 * p1.x + w2 * p2.x,
                        w0 * p0.y + w1 * p1.y + w2 * p2.y,
                        w0 * p0.z + w1 * p1.z + w2 * p2.z,
                    );

                    // Interpolate normal vector
                    let n0 = mesh.normals.get(face.v[0]).copied().unwrap_or(Vec3D::new(0.0, 0.0, 1.0));
                    let n1 = mesh.normals.get(face.v[1]).copied().unwrap_or(Vec3D::new(0.0, 0.0, 1.0));
                    let n2 = mesh.normals.get(face.v[2]).copied().unwrap_or(Vec3D::new(0.0, 0.0, 1.0));

                    let normal = Vec3D::new(
                        w0 * n0.x + w1 * n1.x + w2 * n2.x,
                        w0 * n0.y + w1 * n1.y + w2 * n2.y,
                        w0 * n0.z + w1 * n2.z + w2 * n2.z,
                    ).normalize().unwrap_or(Vec3D::new(0.0, 0.0, 1.0));

                    return Some((pos, normal));
                }
            }
        }

        None
    }
}

/// Evaluates 2D barycentric weights (w0, w1, w2) for point (u, v) in triangle (a, b, c).
/// Returns Some((w0, w1, w2)) if (u, v) is inside triangle, else None.
#[inline]
pub fn compute_barycentric_2d(
    u: f64,
    v: f64,
    a: &UV,
    b: &UV,
    c: &UV,
) -> Option<(f64, f64, f64)> {
    let det = (b.v - c.v) * (a.u - c.u) + (c.u - b.u) * (a.v - c.v);
    if det.abs() < 1e-12 {
        return None;
    }

    let w0 = ((b.v - c.v) * (u - c.u) + (c.u - b.u) * (v - c.v)) / det;
    let w1 = ((c.v - a.v) * (u - c.u) + (a.u - c.u) * (v - c.v)) / det;
    let w2 = 1.0 - w0 - w1;

    let eps = -1e-5;
    if w0 >= eps && w1 >= eps && w2 >= eps {
        Some((w0.max(0.0), w1.max(0.0), w2.max(0.0)))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_barycentric_weights_inside() {
        let a = UV::new(0.0, 0.0);
        let b = UV::new(1.0, 0.0);
        let c = UV::new(0.0, 1.0);

        // Center of triangle: (1/3, 1/3)
        let weights = compute_barycentric_2d(1.0 / 3.0, 1.0 / 3.0, &a, &b, &c);
        assert!(weights.is_some());
        let (w0, w1, w2) = weights.unwrap();
        assert!((w0 - 1.0 / 3.0).abs() < 1e-6);
        assert!((w1 - 1.0 / 3.0).abs() < 1e-6);
        assert!((w2 - 1.0 / 3.0).abs() < 1e-6);
    }
}
