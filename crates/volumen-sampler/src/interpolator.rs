use volumen_core::{Point3D, VoxelCoord};
use volumen_zarr::ZarrVolume;

/// Fast trilinear interpolation engine over a 3D Zarr volume.
pub struct TrilinearSampler;

impl TrilinearSampler {
    /// Samples a continuous 3D coordinate using trilinear interpolation.
    ///
    /// Computes the weighted average of the 8 surrounding lattice voxels:
    /// Returns 0.0 if the point is strictly outside the volume boundary.
    #[inline]
    pub fn sample(volume: &ZarrVolume, p: &Point3D) -> f64 {
        let (dim_x, dim_y, dim_z) = volume.metadata.dimensions();

        let x0 = p.x.floor() as i64;
        let y0 = p.y.floor() as i64;
        let z0 = p.z.floor() as i64;

        if x0 < 0 || y0 < 0 || z0 < 0 {
            return 0.0;
        }

        let ux0 = x0 as u64;
        let uy0 = y0 as u64;
        let uz0 = z0 as u64;

        if ux0 + 1 >= dim_x || uy0 + 1 >= dim_y || uz0 + 1 >= dim_z {
            return 0.0;
        }

        let dx = p.x - p.x.floor();
        let dy = p.y - p.y.floor();
        let dz = p.z - p.z.floor();

        // Query the 8 corners of the unit voxel cube
        let v000 = volume.get_voxel(&VoxelCoord::new(ux0, uy0, uz0)).unwrap_or(0) as f64;
        let v100 = volume.get_voxel(&VoxelCoord::new(ux0 + 1, uy0, uz0)).unwrap_or(0) as f64;
        let v010 = volume.get_voxel(&VoxelCoord::new(ux0, uy0 + 1, uz0)).unwrap_or(0) as f64;
        let v110 = volume.get_voxel(&VoxelCoord::new(ux0 + 1, uy0 + 1, uz0)).unwrap_or(0) as f64;

        let v001 = volume.get_voxel(&VoxelCoord::new(ux0, uy0, uz0 + 1)).unwrap_or(0) as f64;
        let v101 = volume.get_voxel(&VoxelCoord::new(ux0 + 1, uy0, uz0 + 1)).unwrap_or(0) as f64;
        let v011 = volume.get_voxel(&VoxelCoord::new(ux0, uy0 + 1, uz0 + 1)).unwrap_or(0) as f64;
        let v111 = volume.get_voxel(&VoxelCoord::new(ux0 + 1, uy0 + 1, uz0 + 1)).unwrap_or(0) as f64;

        // Trilinear blending
        let c00 = v000 * (1.0 - dx) + v100 * dx;
        let c01 = v001 * (1.0 - dx) + v101 * dx;
        let c10 = v010 * (1.0 - dx) + v110 * dx;
        let c11 = v011 * (1.0 - dx) + v111 * dx;

        let c0 = c00 * (1.0 - dy) + c10 * dy;
        let c1 = c01 * (1.0 - dy) + c11 * dy;

        c0 * (1.0 - dz) + c1 * dz
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use volumen_zarr::chunk::{Chunk3D, ChunkKey};

    #[test]
    fn test_trilinear_interpolation_exact_linear() {
        let vol = ZarrVolume::new_synthetic(100, 100, 100, 32, 32, 32);
        let mut chunk = Chunk3D::new_zeros(ChunkKey::new(0, 0, 0), 32, 32, 32);

        // Set corner intensities: v000 = 100, v100 = 200
        chunk.set(0, 0, 0, 100);
        chunk.set(1, 0, 0, 200);
        vol.insert_chunk(chunk);

        // Point exactly in the middle along X: (0.5, 0.0, 0.0) -> expected 150.0
        let val = TrilinearSampler::sample(&vol, &Point3D::new(0.5, 0.0, 0.0));
        assert!((val - 150.0).abs() < 1e-6);
    }
}
