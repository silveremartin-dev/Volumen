//! Differential benchmarking metrics (MAE, MSE, PSNR, SSIM) for validating
//! Volumen against Volume Cartographer / vc_render ground truth.

use serde::{Deserialize, Serialize};

/// Statistical report comparing two rendered 2.5D surface volumes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialReport {
    pub total_voxels: usize,
    pub mae: f64,
    pub mse: f64,
    pub psnr_db: f64,
    pub max_absolute_diff: u16,
    pub speedup_factor: Option<f64>,
}

impl DifferentialReport {
    /// Compares two equal-sized 16-bit surface volume buffers.
    pub fn compare(target: &[u16], reference: &[u16]) -> Self {
        assert_eq!(
            target.len(),
            reference.len(),
            "Volume slice buffers must have identical lengths"
        );

        let total = target.len();
        if total == 0 {
            return Self {
                total_voxels: 0,
                mae: 0.0,
                mse: 0.0,
                psnr_db: f64::INFINITY,
                max_absolute_diff: 0,
                speedup_factor: None,
            };
        }

        let mut sum_abs_err: f64 = 0.0;
        let mut sum_sq_err: f64 = 0.0;
        let mut max_diff: u16 = 0;

        for (&a, &b) in target.iter().zip(reference.iter()) {
            let diff = (a as i64 - b as i64).unsigned_abs() as u16;
            max_diff = max_diff.max(diff);
            sum_abs_err += diff as f64;
            sum_sq_err += (diff as f64).powi(2);
        }

        let mae = sum_abs_err / total as f64;
        let mse = sum_sq_err / total as f64;

        let psnr_db = if mse < 1e-12 {
            f64::INFINITY
        } else {
            20.0 * (65535.0f64 / mse.sqrt()).log10()
        };

        Self {
            total_voxels: total,
            mae,
            mse,
            psnr_db,
            max_absolute_diff: max_diff,
            speedup_factor: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_volumes_perfect_psnr() {
        let v1 = vec![1000u16, 2000, 3000, 4000];
        let v2 = v1.clone();
        let report = DifferentialReport::compare(&v1, &v2);
        assert_eq!(report.mae, 0.0);
        assert_eq!(report.mse, 0.0);
        assert!(report.psnr_db.is_infinite());
    }

    #[test]
    fn test_noisy_volumes_finite_psnr() {
        let v1 = vec![1000u16, 2000];
        let v2 = vec![1005u16, 1995];
        let report = DifferentialReport::compare(&v1, &v2);
        assert_eq!(report.mae, 5.0);
        assert_eq!(report.mse, 25.0);
        assert!(report.psnr_db > 80.0); // Very high fidelity
    }
}
