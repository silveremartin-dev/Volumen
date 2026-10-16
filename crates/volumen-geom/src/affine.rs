use glam::DMat4;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::Path;
use volumen_core::{Point3D, Result, Vec3D, VolumenError};

/// 4x4 Affine transformation for aligning segments with global tomographic coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AffineTransform {
    pub matrix: [[f64; 4]; 4],
}

impl AffineTransform {
    pub fn identity() -> Self {
        Self {
            matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Loads affine transform from official VC3D `transform.json`.
    pub fn load_from_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let json: serde_json::Value = serde_json::from_reader(file)?;

        if let Some(mat_val) = json.get("transformation_matrix") {
            if let Some(rows) = mat_val.as_array() {
                if rows.len() >= 3 {
                    let mut mat = [[0.0; 4]; 4];
                    for r in 0..rows.len().min(4) {
                        if let Some(cols) = rows[r].as_array() {
                            for c in 0..cols.len().min(4) {
                                mat[r][c] = cols[c].as_f64().unwrap_or(0.0);
                            }
                        }
                    }
                    if rows.len() == 3 {
                        mat[3] = [0.0, 0.0, 0.0, 1.0];
                    }
                    return Ok(Self { matrix: mat });
                }
            }
        }

        Err(VolumenError::Geometry(
            "Invalid transformation_matrix in transform.json".to_string(),
        ))
    }

    /// Transforms a 3D point: P' = T * P
    #[inline]
    pub fn transform_point(&self, p: &Point3D) -> Point3D {
        let m = &self.matrix;
        let x = m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3];
        let y = m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3];
        let z = m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3];
        Point3D::new(x, y, z)
    }

    /// Transforms a normal vector using the inverse transpose: N' = (M^-1)^T * N
    #[inline]
    pub fn transform_normal(&self, n: &Vec3D) -> Vec3D {
        let dmat = DMat4::from_cols_array_2d(&self.matrix);
        let inv_t = dmat.inverse().transpose();

        let nx = inv_t.x_axis.x * n.x + inv_t.y_axis.x * n.y + inv_t.z_axis.x * n.z;
        let ny = inv_t.x_axis.y * n.x + inv_t.y_axis.y * n.y + inv_t.z_axis.y * n.z;
        let nz = inv_t.x_axis.z * n.x + inv_t.y_axis.z * n.y + inv_t.z_axis.z * n.z;

        Vec3D::new(nx, ny, nz).normalize().unwrap_or(*n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affine_identity_translation() {
        let mut aff = AffineTransform::identity();
        aff.matrix[0][3] = 100.0; // translate X +100
        let p = Point3D::new(5.0, 10.0, 15.0);
        let pt = aff.transform_point(&p);
        assert_eq!(pt, Point3D::new(105.0, 10.0, 15.0));

        let n = Vec3D::new(0.0, 0.0, 1.0);
        let nt = aff.transform_normal(&n);
        assert_eq!(nt, Vec3D::new(0.0, 0.0, 1.0));
    }
}
