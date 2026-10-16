use std::path::Path;
use volumen_geom::{NormalEstimator, SurfaceMesh};

#[test]
fn test_real_scroll1_segment_mesh_loading() {
    let obj_path = Path::new("data/real_scroll1_20230503225234/20230503225234.obj");
    if !obj_path.exists() {
        eprintln!("Real segment OBJ file not downloaded; skipping test.");
        return;
    }

    let mut mesh = SurfaceMesh::load_obj(obj_path).expect("Must parse real Vesuvius OBJ file");
    
    // Check real segment topology
    assert_eq!(mesh.vertices.len(), 100, "Segment has exactly 100 vertices");
    assert_eq!(mesh.faces.len(), 157, "Segment has 157 triangular faces");
    assert_eq!(mesh.uvs.len(), 100, "Segment has 100 UV texture coordinates");

    // Compute differential normals
    NormalEstimator::compute_vertex_normals(&mut mesh);
    assert_eq!(mesh.normals.len(), 100, "Each vertex must have a smooth normal");

    // Check that all normals are unit length
    for n in &mesh.normals {
        let len = (n.x * n.x + n.y * n.y + n.z * n.z).sqrt();
        assert!((len - 1.0).abs() < 1e-5, "Normal must be normalized: {}", len);
    }

    // Check physical bounding box in Scroll 1 voxel space
    let bbox = mesh.bounding_box().expect("Bounding box must exist");
    println!("Real Scroll 1 Segment BBox: min={:?}, max={:?}", bbox.min, bbox.max);
    assert!(bbox.min.x >= 2600.0 && bbox.max.x <= 2800.0, "X must be within Scroll 1 volume");
    assert!(bbox.min.y >= 2600.0 && bbox.max.y <= 2800.0, "Y must be within Scroll 1 volume");
}
