use crate::coords::Point3D;
use serde::{Deserialize, Serialize};

/// Axis-aligned 3D bounding box.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox3D {
    pub min: Point3D,
    pub max: Point3D,
}

impl BoundingBox3D {
    #[inline]
    pub const fn new(min: Point3D, max: Point3D) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn contains(&self, p: &Point3D) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }

    #[inline]
    pub fn dimensions(&self) -> (f64, f64, f64) {
        (
            (self.max.x - self.min.x).max(0.0),
            (self.max.y - self.min.y).max(0.0),
            (self.max.z - self.min.z).max(0.0),
        )
    }

    #[inline]
    pub fn expand_to_include(&mut self, p: &Point3D) {
        self.min.x = self.min.x.min(p.x);
        self.min.y = self.min.y.min(p.y);
        self.min.z = self.min.z.min(p.z);
        self.max.x = self.max.x.max(p.x);
        self.max.y = self.max.y.max(p.y);
        self.max.z = self.max.z.max(p.z);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounding_box_containment() {
        let bbox = BoundingBox3D::new(Point3D::new(0.0, 0.0, 0.0), Point3D::new(10.0, 10.0, 10.0));
        assert!(bbox.contains(&Point3D::new(5.0, 5.0, 5.0)));
        assert!(!bbox.contains(&Point3D::new(15.0, 5.0, 5.0)));
    }
}
