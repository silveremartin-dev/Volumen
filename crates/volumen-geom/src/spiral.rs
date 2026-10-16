use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use volumen_core::{Point3D, Result, Vec3D, VolumenError};

/// Parameters defining a generalized 3D Archimedean spiral:
/// r(theta, z) = a(z) + b(z) * theta
#[derive(Debug, Clone, Copy)]
pub struct SpiralParameters {
    pub center_x: f64,
    pub center_y: f64,
    pub a: f64,          // Base core radius
    pub b: f64,          // Radial spacing between successive windings (thickness of papyrus wrap)
    pub pitch_per_turn: f64, // 2 * pi * b
}

impl SpiralParameters {
    pub fn new(center_x: f64, center_y: f64, a: f64, b: f64) -> Self {
        Self {
            center_x,
            center_y,
            a,
            b,
            pitch_per_turn: 2.0 * std::f64::consts::PI * b,
        }
    }

    /// Evaluates the continuous radial distance for any cumulative winding angle theta.
    #[inline]
    pub fn radius_at(&self, theta: f64) -> f64 {
        self.a + self.b * theta
    }

    /// Projects an arbitrary 3D voxel coordinate onto the spiral manifold:
    /// Returns (winding_number, residual_distance).
    #[inline]
    pub fn project_point(&self, p: &Point3D) -> (f64, f64) {
        let dx = p.x - self.center_x;
        let dy = p.y - self.center_y;
        let r = (dx * dx + dy * dy).sqrt();

        // Solve for winding k such that r = a + b * (theta + 2 * pi * k)
        let total_theta = ((r - self.a) / self.b).max(0.0);
        let winding_k = total_theta / (2.0 * std::f64::consts::PI);
        let residual = r - self.radius_at(total_theta);

        (winding_k, residual)
    }

    /// Evaluates the 3D position on the spiral manifold at cumulative angle theta and height z.
    #[inline]
    pub fn point_at(&self, theta: f64, z: f64) -> Point3D {
        let r = self.radius_at(theta);
        Point3D::new(
            self.center_x + r * theta.cos(),
            self.center_y + r * theta.sin(),
            z,
        )
    }

    /// Evaluates the outward unit normal vector on the spiral surface at cumulative angle theta.
    /// Orthogonal to tangent vector t = dP/dtheta in the xy-plane and to z-axis.
    #[inline]
    pub fn normal_at(&self, theta: f64) -> volumen_core::Vec3D {
        let r = self.radius_at(theta);
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let tx = self.b * cos_t - r * sin_t;
        let ty = self.b * sin_t + r * cos_t;
        let norm = (tx * tx + ty * ty).sqrt();
        if norm > 1e-12 {
            volumen_core::Vec3D::new(ty / norm, -tx / norm, 0.0)
        } else {
            volumen_core::Vec3D::new(cos_t, sin_t, 0.0)
        }
    }
}

/// Continuous Spiral Unrolling Engine.
/// Resolves topological bifurcations and maps spiral 3D space into a flattened isometric 2D domain.
pub struct SpiralUnroller;

impl SpiralUnroller {
    /// Computes the arc length of the spiral from theta = 0 to theta = theta_max.
    /// Analytical formula: L = (b/2) * [theta * sqrt(1 + theta^2) + ln(theta + sqrt(1 + theta^2))]
    pub fn compute_arc_length(b: f64, theta: f64) -> f64 {
        if theta <= 0.0 {
            return 0.0;
        }
        let t2 = theta * theta;
        let sqrt_term = (1.0 + t2).sqrt();
        0.5 * b * (theta * sqrt_term + (theta + sqrt_term).ln())
    }

    /// Generates flattened (u, v) isometric coordinates for points along a winding.
    pub fn unroll_point(params: &SpiralParameters, p: &Point3D) -> (f64, f64) {
        let (winding, _) = params.project_point(p);
        let total_theta = winding * 2.0 * std::f64::consts::PI;
        let u_arc_length = Self::compute_arc_length(params.b, total_theta);
        let v_height = p.z; // Axial coordinate along scroll cylinder
        (u_arc_length, v_height)
    }
}

/// 2D grid of 3D displacement vectors (dx, dy, dz) indexed by (theta, z).
/// Typically produced by deep learning deformation estimators (e.g. `fit_spiral` in `ScrollPrize/villa`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeformationField {
    pub theta_min: f64,
    pub theta_max: f64,
    pub z_min: f64,
    pub z_max: f64,
    pub width: usize,   // Number of theta steps
    pub height: usize,  // Number of z steps
    pub displacements: Vec<Vec3D>, // Row-major: displacements[y * width + x]
}

impl DeformationField {
    pub fn new(
        theta_min: f64,
        theta_max: f64,
        z_min: f64,
        z_max: f64,
        width: usize,
        height: usize,
        displacements: Vec<Vec3D>,
    ) -> Result<Self> {
        if displacements.len() != width * height {
            return Err(VolumenError::InvalidParameter(format!(
                "Deformation field size mismatch: expected {} vectors, got {}",
                width * height,
                displacements.len()
            )));
        }

        Ok(Self {
            theta_min,
            theta_max,
            z_min,
            z_max,
            width,
            height,
            displacements,
        })
    }

    /// Load deformation field from JSON file
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let reader = BufReader::new(file);
        let field: Self = serde_json::from_reader(reader)
            .map_err(|e| VolumenError::MeshParse(format!("Failed to parse deformation JSON: {}", e)))?;
        Ok(field)
    }

    /// Bilinear interpolation of displacement at continuous (theta, z)
    #[inline]
    pub fn sample(&self, theta: f64, z: f64) -> Vec3D {
        if self.width <= 1 || self.height <= 1 {
            return self.displacements.first().copied().unwrap_or(Vec3D::new(0.0, 0.0, 0.0));
        }

        let u = ((theta - self.theta_min) / (self.theta_max - self.theta_min).max(1e-12))
            .clamp(0.0, 1.0) * (self.width - 1) as f64;
        let v = ((z - self.z_min) / (self.z_max - self.z_min).max(1e-12))
            .clamp(0.0, 1.0) * (self.height - 1) as f64;

        let x0 = (u.floor() as usize).min(self.width - 2);
        let y0 = (v.floor() as usize).min(self.height - 2);
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let fx = u - x0 as f64;
        let fy = v - y0 as f64;

        let d00 = self.displacements[y0 * self.width + x0];
        let d10 = self.displacements[y0 * self.width + x1];
        let d01 = self.displacements[y1 * self.width + x0];
        let d11 = self.displacements[y1 * self.width + x1];

        let d0 = d00 * (1.0 - fx) + d10 * fx;
        let d1 = d01 * (1.0 - fx) + d11 * fx;
        d0 * (1.0 - fy) + d1 * fy
    }
}

/// Continuous spiral manifold with optional non-rigid deformation field.
/// Bypasses the rigid Archimedean assumption by accounting for physical papyrus compression,
/// flattening, and local warping.
#[derive(Debug, Clone)]
pub struct DeformedSpiralManifold {
    pub base_params: SpiralParameters,
    pub deformation: Option<DeformationField>,
}

impl DeformedSpiralManifold {
    pub fn new(base_params: SpiralParameters, deformation: Option<DeformationField>) -> Self {
        Self {
            base_params,
            deformation,
        }
    }

    /// Evaluates the true 3D point on the manifold at (theta, z), including non-rigid deformation.
    #[inline]
    pub fn point_at(&self, theta: f64, z: f64) -> Point3D {
        let base = self.base_params.point_at(theta, z);
        if let Some(ref flow) = self.deformation {
            base + flow.sample(theta, z)
        } else {
            base
        }
    }

    /// Evaluates the smooth surface normal at (theta, z) by differential tangents.
    #[inline]
    pub fn normal_at(&self, theta: f64, z: f64) -> Vec3D {
        if self.deformation.is_none() {
            return self.base_params.normal_at(theta);
        }

        let eps_t = 1e-3;
        let eps_z = 1.0;

        let p_t_plus = self.point_at(theta + eps_t, z);
        let p_t_minus = self.point_at(theta - eps_t, z);
        let t_theta = Vec3D::new(
            (p_t_plus.x - p_t_minus.x) / (2.0 * eps_t),
            (p_t_plus.y - p_t_minus.y) / (2.0 * eps_t),
            (p_t_plus.z - p_t_minus.z) / (2.0 * eps_t),
        );

        let p_z_plus = self.point_at(theta, z + eps_z);
        let p_z_minus = self.point_at(theta, z - eps_z);
        let t_z = Vec3D::new(
            (p_z_plus.x - p_z_minus.x) / (2.0 * eps_z),
            (p_z_plus.y - p_z_minus.y) / (2.0 * eps_z),
            (p_z_plus.z - p_z_minus.z) / (2.0 * eps_z),
        );

        let cross = t_theta.cross(&t_z);
        if let Some(n) = cross.normalize() {
            // Ensure normal points outward (away from core)
            let canonical_n = self.base_params.normal_at(theta);
            if n.dot(&canonical_n) < 0.0 {
                n * -1.0
            } else {
                n
            }
        } else {
            self.base_params.normal_at(theta)
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spiral_projection_exact() {
        let params = SpiralParameters::new(100.0, 100.0, 10.0, 2.0);
        let theta = 4.0 * std::f64::consts::PI; // Exactly 2 complete turns
        let r = params.radius_at(theta);
        let p = Point3D::new(100.0 + r, 100.0, 50.0);

        let (winding, residual) = params.project_point(&p);
        assert!((winding - 2.0).abs() < 1e-4);
        assert!(residual.abs() < 1e-6);

        let (u, v) = SpiralUnroller::unroll_point(&params, &p);
        assert!(u > 0.0);
        assert_eq!(v, 50.0);
    }

    #[test]
    fn test_deformed_spiral_manifold() {
        let params = SpiralParameters::new(0.0, 0.0, 50.0, 2.0);
        // Create a deformation field that expands radius by 5.0 along x
        let disp = vec![
            Vec3D::new(5.0, 0.0, 0.0),
            Vec3D::new(5.0, 0.0, 0.0),
            Vec3D::new(5.0, 0.0, 0.0),
            Vec3D::new(5.0, 0.0, 0.0),
        ];
        let field = DeformationField::new(0.0, 10.0, 0.0, 100.0, 2, 2, disp).unwrap();
        let manifold = DeformedSpiralManifold::new(params, Some(field));

        let p_canonical = params.point_at(0.0, 50.0);
        let p_deformed = manifold.point_at(0.0, 50.0);
        assert_eq!(p_deformed.x, p_canonical.x + 5.0);
        assert_eq!(p_deformed.y, p_canonical.y);
        assert_eq!(p_deformed.z, 50.0);

        let normal = manifold.normal_at(0.0, 50.0);
        assert!(normal.length() > 0.999 && normal.length() < 1.001);
    }
}

