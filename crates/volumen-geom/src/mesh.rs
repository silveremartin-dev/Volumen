use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use volumen_core::{BoundingBox3D, Point3D, Result, Vec3D, VolumenError};

/// 2D UV Texture Coordinate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UV {
    pub u: f64,
    pub v: f64,
}

impl UV {
    #[inline]
    pub const fn new(u: f64, v: f64) -> Self {
        Self { u, v }
    }
}

/// A 3D mesh vertex with optional UV and normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    pub pos: Point3D,
    pub uv: Option<UV>,
    pub normal: Option<Vec3D>,
}

/// A triangular face referencing 3 vertex indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Face {
    pub v: [usize; 3],
    pub uv: Option<[usize; 3]>,
}

/// Surface Mesh representing an unrolled or 3D segmented papyrus sheet.
#[derive(Debug, Clone, Default)]
pub struct SurfaceMesh {
    pub vertices: Vec<Point3D>,
    pub uvs: Vec<UV>,
    pub normals: Vec<Vec3D>,
    pub faces: Vec<Face>,
}

impl SurfaceMesh {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            uvs: Vec::new(),
            normals: Vec::new(),
            faces: Vec::new(),
        }
    }

    /// Loads a Wavefront `.obj` file.
    pub fn load_obj<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut mesh = Self::new();

        for (line_idx, line) in reader.lines().enumerate() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let mut parts = trimmed.split_whitespace();
            let prefix = parts.next().unwrap_or("");

            match prefix {
                "v" => {
                    let x: f64 = parts
                        .next()
                        .ok_or_else(|| {
                            VolumenError::MeshParse(format!("Line {}: missing x", line_idx))
                        })?
                        .parse()
                        .map_err(|e| {
                            VolumenError::MeshParse(format!("Line {}: {}", line_idx, e))
                        })?;
                    let y: f64 = parts
                        .next()
                        .ok_or_else(|| {
                            VolumenError::MeshParse(format!("Line {}: missing y", line_idx))
                        })?
                        .parse()
                        .map_err(|e| {
                            VolumenError::MeshParse(format!("Line {}: {}", line_idx, e))
                        })?;
                    let z: f64 = parts
                        .next()
                        .ok_or_else(|| {
                            VolumenError::MeshParse(format!("Line {}: missing z", line_idx))
                        })?
                        .parse()
                        .map_err(|e| {
                            VolumenError::MeshParse(format!("Line {}: {}", line_idx, e))
                        })?;
                    mesh.vertices.push(Point3D::new(x, y, z));
                }
                "vt" => {
                    let u: f64 = parts.next().unwrap_or("0").parse().unwrap_or(0.0);
                    let v: f64 = parts.next().unwrap_or("0").parse().unwrap_or(0.0);
                    mesh.uvs.push(UV::new(u, v));
                }
                "vn" => {
                    let nx: f64 = parts.next().unwrap_or("0").parse().unwrap_or(0.0);
                    let ny: f64 = parts.next().unwrap_or("0").parse().unwrap_or(0.0);
                    let nz: f64 = parts.next().unwrap_or("1").parse().unwrap_or(1.0);
                    mesh.normals.push(Vec3D::new(nx, ny, nz));
                }
                "f" => {
                    let mut face_v = Vec::new();
                    let mut face_uv = Vec::new();

                    for vertex_str in parts {
                        let subparts: Vec<&str> = vertex_str.split('/').collect();
                        if let Some(v_str) = subparts.first() {
                            if let Ok(v_idx) = v_str.parse::<usize>() {
                                if v_idx > 0 {
                                    face_v.push(v_idx - 1); // 1-indexed to 0-indexed
                                }
                            }
                        }
                        if subparts.len() > 1 && !subparts[1].is_empty() {
                            if let Ok(uv_idx) = subparts[1].parse::<usize>() {
                                if uv_idx > 0 {
                                    face_uv.push(uv_idx - 1);
                                }
                            }
                        }
                    }

                    if face_v.len() >= 3 {
                        // Triangulate if quad/polygon
                        for i in 1..face_v.len() - 1 {
                            let uv_opt = if face_uv.len() >= face_v.len() {
                                Some([face_uv[0], face_uv[i], face_uv[i + 1]])
                            } else {
                                None
                            };
                            mesh.faces.push(Face {
                                v: [face_v[0], face_v[i], face_v[i + 1]],
                                uv: uv_opt,
                            });
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(mesh)
    }

    /// Computes the bounding box of the entire mesh.
    pub fn bounding_box(&self) -> Option<BoundingBox3D> {
        let first = self.vertices.first()?;
        let mut bbox = BoundingBox3D::new(*first, *first);
        for v in &self.vertices[1..] {
            bbox.expand_to_include(v);
        }
        Some(bbox)
    }
}
