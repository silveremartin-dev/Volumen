use serde::{Deserialize, Serialize};

/// Unique identifier for a 3D chunk block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkKey {
    pub cx: u64,
    pub cy: u64,
    pub cz: u64,
}

impl ChunkKey {
    #[inline]
    pub const fn new(cx: u64, cy: u64, cz: u64) -> Self {
        Self { cx, cy, cz }
    }

    /// Formats chunk filename following Zarr 2.0 convention (z/y/x or z.y.x).
    pub fn to_path_string(&self) -> String {
        format!("{}/{}/{}", self.cz, self.cy, self.cx)
    }
}

/// A 3D chunk buffer in memory storing 16-bit unsigned voxel data.
#[derive(Debug, Clone)]
pub struct Chunk3D {
    pub key: ChunkKey,
    pub dim_x: usize,
    pub dim_y: usize,
    pub dim_z: usize,
    pub data: Vec<u16>,
}

impl Chunk3D {
    pub fn new(key: ChunkKey, dim_x: usize, dim_y: usize, dim_z: usize, data: Vec<u16>) -> Self {
        assert_eq!(data.len(), dim_x * dim_y * dim_z);
        Self {
            key,
            dim_x,
            dim_y,
            dim_z,
            data,
        }
    }

    pub fn new_zeros(key: ChunkKey, dim_x: usize, dim_y: usize, dim_z: usize) -> Self {
        let size = dim_x * dim_y * dim_z;
        Self {
            key,
            dim_x,
            dim_y,
            dim_z,
            data: vec![0u16; size],
        }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> Option<u16> {
        if x < self.dim_x && y < self.dim_y && z < self.dim_z {
            let idx = z * (self.dim_x * self.dim_y) + y * self.dim_x + x;
            Some(self.data[idx])
        } else {
            None
        }
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, val: u16) {
        if x < self.dim_x && y < self.dim_y && z < self.dim_z {
            let idx = z * (self.dim_x * self.dim_y) + y * self.dim_x + x;
            self.data[idx] = val;
        }
    }
}
