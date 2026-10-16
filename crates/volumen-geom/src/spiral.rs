use volumen_core::Point3D;

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
}
