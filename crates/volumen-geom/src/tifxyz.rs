use std::fs::File;
use std::path::Path;
use tiff::decoder::{Decoder, DecodingResult};
use volumen_core::{Point3D, Result, VolumenError};

/// Regular 2D QuadSurface mesh representation as used by Volume Cartographer 3D (VC3D)
/// in `ScrollPrize/villa`.
///
/// Encoded as three 32-bit floating-point TIFF files (`x.tif`, `y.tif`, `z.tif`)
/// where each pixel `(u, v)` represents the 3D position `(x, y, z)` in voxel space.
/// Cells with `x = -1.0` or non-finite values represent missing/culled surface regions.
#[derive(Debug, Clone)]
pub struct TifxyzMesh {
    pub width: usize,
    pub height: usize,
    pub xs: Vec<f32>,
    pub ys: Vec<f32>,
    pub zs: Vec<f32>,
}

impl TifxyzMesh {
    /// Loads a `tifxyz` directory containing `x.tif`, `y.tif`, and `z.tif`.
    pub fn load_from_dir<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir = dir.as_ref();
        let x_path = dir.join("x.tif");
        let y_path = dir.join("y.tif");
        let z_path = dir.join("z.tif");

        if !x_path.exists() || !y_path.exists() || !z_path.exists() {
            return Err(VolumenError::MeshParse(format!(
                "tifxyz directory must contain x.tif, y.tif, and z.tif: {}",
                dir.display()
            )));
        }

        let (w_x, h_x, xs) = read_tiff_f32(&x_path)?;
        let (w_y, h_y, ys) = read_tiff_f32(&y_path)?;
        let (w_z, h_z, zs) = read_tiff_f32(&z_path)?;

        if w_x != w_y || w_x != w_z || h_x != h_y || h_x != h_z {
            return Err(VolumenError::MeshParse(format!(
                "tifxyz coordinate dimensions mismatch: x=({}x{}), y=({}x{}), z=({}x{})",
                w_x, h_x, w_y, h_y, w_z, h_z
            )));
        }

        Ok(Self {
            width: w_x,
            height: h_x,
            xs,
            ys,
            zs,
        })
    }

    /// Queries the 3D position at grid coordinate (u, v).
    /// Returns `None` if out of bounds or masked out (e.g. x == -1.0 or non-finite).
    #[inline]
    pub fn get_point(&self, u: usize, v: usize) -> Option<Point3D> {
        if u < self.width && v < self.height {
            let idx = v * self.width + u;
            let x = self.xs[idx];
            let y = self.ys[idx];
            let z = self.zs[idx];

            if x >= 0.0 && x.is_finite() && y.is_finite() && z.is_finite() {
                Some(Point3D::new(x as f64, y as f64, z as f64))
            } else {
                None
            }
        } else {
            None
        }
    }
}

fn read_tiff_f32(path: &Path) -> Result<(usize, usize, Vec<f32>)> {
    let file = File::open(path)?;
    let mut decoder = Decoder::new(file).map_err(|e| {
        VolumenError::Tiff(format!("Failed to open TIFF {}: {}", path.display(), e))
    })?;

    let (w, h) = decoder
        .dimensions()
        .map_err(|e| VolumenError::Tiff(e.to_string()))?;

    let result = decoder.read_image().map_err(|e| {
        VolumenError::Tiff(format!("Failed to decode TIFF {}: {}", path.display(), e))
    })?;

    match result {
        DecodingResult::F32(data) => Ok((w as usize, h as usize, data)),
        _ => Err(VolumenError::Tiff(format!(
            "Expected 32-bit floating point TIFF in {}",
            path.display()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tifxyz_point_lookup() {
        let mesh = TifxyzMesh {
            width: 2,
            height: 2,
            xs: vec![10.0, 20.0, -1.0, 40.0],
            ys: vec![11.0, 21.0, 0.0, 41.0],
            zs: vec![12.0, 22.0, 0.0, 42.0],
        };

        assert_eq!(mesh.get_point(0, 0), Some(Point3D::new(10.0, 11.0, 12.0)));
        assert_eq!(mesh.get_point(1, 0), Some(Point3D::new(20.0, 21.0, 22.0)));
        assert_eq!(mesh.get_point(0, 1), None); // Masked out (-1.0)
        assert_eq!(mesh.get_point(1, 1), Some(Point3D::new(40.0, 41.0, 42.0)));
    }
}
