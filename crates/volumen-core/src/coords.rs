use glam::DVec3;
use serde::{Deserialize, Serialize};

/// Discrete voxel lattice coordinate (integral voxel index in volume).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VoxelCoord {
    pub x: u64,
    pub y: u64,
    pub z: u64,
}

impl VoxelCoord {
    #[inline]
    pub const fn new(x: u64, y: u64, z: u64) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn to_point3d(&self) -> Point3D {
        Point3D::new(self.x as f64, self.y as f64, self.z as f64)
    }
}

/// Continuous spatial point in 3D (double precision).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3D {
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn to_dvec3(&self) -> DVec3 {
        DVec3::new(self.x, self.y, self.z)
    }

    #[inline]
    pub fn from_dvec3(v: DVec3) -> Self {
        Self::new(v.x, v.y, v.z)
    }

    #[inline]
    pub fn distance_to(&self, other: &Point3D) -> f64 {
        self.to_dvec3().distance(other.to_dvec3())
    }

    #[inline]
    pub fn floor_voxel(&self) -> (i64, i64, i64) {
        (
            self.x.floor() as i64,
            self.y.floor() as i64,
            self.z.floor() as i64,
        )
    }
}

/// 3D Vector for directions, normals, and offsets.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vec3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3D {
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn to_dvec3(&self) -> DVec3 {
        DVec3::new(self.x, self.y, self.z)
    }

    #[inline]
    pub fn from_dvec3(v: DVec3) -> Self {
        Self::new(v.x, v.y, v.z)
    }

    #[inline]
    pub fn length(&self) -> f64 {
        self.to_dvec3().length()
    }

    #[inline]
    pub fn normalize(&self) -> Option<Self> {
        let v = self.to_dvec3();
        let len = v.length();
        if len > 1e-12 {
            let n = v / len;
            Some(Self::new(n.x, n.y, n.z))
        } else {
            None
        }
    }

    #[inline]
    pub fn dot(&self, other: &Vec3D) -> f64 {
        self.to_dvec3().dot(other.to_dvec3())
    }

    #[inline]
    pub fn cross(&self, other: &Vec3D) -> Self {
        Self::from_dvec3(self.to_dvec3().cross(other.to_dvec3()))
    }
}

impl std::ops::Add<Vec3D> for Point3D {
    type Output = Point3D;

    #[inline]
    fn add(self, rhs: Vec3D) -> Point3D {
        Point3D::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl std::ops::Sub<Point3D> for Point3D {
    type Output = Vec3D;

    #[inline]
    fn sub(self, rhs: Point3D) -> Vec3D {
        Vec3D::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl std::ops::Mul<f64> for Vec3D {
    type Output = Vec3D;

    #[inline]
    fn mul(self, scalar: f64) -> Vec3D {
        Vec3D::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }
}

impl std::ops::Add<Vec3D> for Vec3D {
    type Output = Vec3D;

    #[inline]
    fn add(self, rhs: Vec3D) -> Vec3D {
        Vec3D::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_normalization() {
        let v = Vec3D::new(3.0, 0.0, 4.0);
        assert_eq!(v.length(), 5.0);
        let n = v.normalize().expect("Normalization should succeed");
        assert!((n.length() - 1.0).abs() < 1e-9);
        assert!((n.x - 0.6).abs() < 1e-9);
        assert!((n.z - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_point_vector_addition() {
        let p = Point3D::new(10.0, 20.0, 30.0);
        let v = Vec3D::new(1.0, -2.0, 3.0);
        let p2 = p + v;
        assert_eq!(p2, Point3D::new(11.0, 18.0, 33.0));
    }
}
