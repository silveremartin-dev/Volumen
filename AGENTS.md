# AGENTS.md — Developer & AI Agent Collaboration Guide

## 1. System Overview & Purpose

`Volumen` is a high-performance, modular virtual unwrapping engine written in modern Rust for the **Vesuvius Challenge** (the computational recovery of carbonized Herculaneum papyri).

Its primary mission is to replace legacy, fragile, monolithic C++ tooling (such as *Volume Cartographer*'s `vc_render`) and ad-hoc Python glue scripts with an ultra-fast, memory-safe, out-of-core pipeline capable of:
1. **Streaming multiresolution volumetric data** (OME-Zarr / multiscale chunked arrays) without VRAM/RAM exhaustion.
2. **Executing parallel 3D trilinear ray-casting & normal sampling** across 3D meshes to produce 2.5D surface volumes (30 to 64 depth layers) in seconds instead of hours.
3. **Resolving topological ambiguities & de-bifurcation** where carbonized, pyrolyzed papyrus spires merge without radiologic density contrast.

---

## 2. Core Architectural Principles for AI Agents & Developers

When contributing to or extending this codebase, all AI agents and contributors must strictly enforce the following principles:

### 2.1 Compatibility with Vesuvius Data Standards (Zero Disruption)
- **Never invent proprietary binary container formats** for upstream/downstream communication.
- **Strictly adhere to community interchange formats**:
  - Volumetric inputs: **OME-Zarr** (`.zarr` directory tree with JSON metadata and chunked compressed/uncompressed arrays) and standard TIFF stacks.
  - Surface meshes: **Wavefront OBJ** (`.obj`) with 3D vertex coordinates $(x,y,z)$ and normalized $(u,v)$ parameterization coordinates.
  - Surface volumes: Multi-layer **TIFF 16-bit** stacks or direct Zarr outputs consumable by downstream PyTorch ink detection models.

### 2.2 Memory Safety & Out-of-Core Processing
- A complete Herculaneum scroll scan reaches **5 to 10 Terabytes** at $2.4\ \mu\text{m} \sim 7.91\ \mu\text{m}$.
- **Never attempt to load an entire volume into system RAM**.
- All volume operations must run through chunked LRU caches and asynchronous block streaming (`volumen-zarr`).
- Memory usage must remain bounded, deterministic, and configurable via CLI flags (e.g., `--max-cache-gb`).

### 2.3 Mathematical Rigor & Differential Geometry
- Normal estimation along papyrus sheets must adhere strictly to differential surface properties:
  $$\mathbf{n}(u, v) = \frac{\frac{\partial \mathbf{S}}{\partial u} \times \frac{\partial \mathbf{S}}{\partial v}}{\left\| \frac{\partial \mathbf{S}}{\partial u} \times \frac{\partial \mathbf{S}}{\partial v} \right\|}$$
- Ray sampling along the normal $\mathbf{n}$ must employ trilinear interpolation over voxel coordinates $(x, y, z)$ without edge artifacts:
  $$\mathbf{P}(k) = \mathbf{V}_0 + \left(k - \frac{N - 1}{2}\right) \cdot \delta \cdot \mathbf{n}, \quad k \in [0, N-1]$$
- Avoid naive thresholding on voxel intensities: pyrolyzed papyrus exhibits near-identical electron density across compressed spires.

### 2.4 High Concurrency & SIMD Vectorization
- Exploit data-level parallelism using `rayon` across mesh triangles and independent ray queries.
- Ensure linear algebra primitives benefit from SIMD vectorization via `glam` and `nalgebra`.
- Keep the rendering engine headless, modular, and callable both as a standalone CLI (`volumen-cli`) and as an embeddable Rust crate.

---

## 3. Subsystem Breakdown & Crate Topology

```
┌────────────────────────────────────────────────────────────────────────┐
│                      Volumen Workspace Architecture                     │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────┐    ┌────────────────────┐    ┌─────────────────────┐
│ 1. volumen-core  │ ──▶│ 2. volumen-zarr    │ ──▶│ 3. volumen-geom     │
│ - Voxel & Point3 │    │ - OME-Zarr Parser  │    │ - Wavefront OBJ     │
│ - Bounding Boxes │    │ - Chunk LRU Cache  │    │ - Normal Estimation │
│ - Coordinate Sys │    │ - Out-of-core IO   │    │ - Curvature/Topology│
└──────────────────┘    └────────────────────┘    └─────────────────────┘
                                                              │
                                                              ▼
┌─────────────────────────────────────────┐    ┌────────────────────────┐
│ 5. volumen-cli                          │ ◀──│ 4. volumen-sampler     │
│ - `volumen render` (replaces vc_render) │    │ - Trilinear Sampler    │
│ - `volumen verify` (mesh sanity checks) │    │ - Parallel Ray-caster  │
│ - `volumen inspect` (Zarr & metadata)   │    │ - 16-bit TIFF Exporter │
└─────────────────────────────────────────┘    └────────────────────────┘
```

### 3.1 `volumen-core`
- **Responsibility**: Foundational types, vector math, bounding boxes, spatial coordinates, and error primitives.
- **Key Types**:
  - `Point3D`, `Vec3D`, `VoxelCoord`
  - `BoundingBox3D`
  - `VolumenError`, `Result<T>`

### 3.2 `volumen-zarr`
- **Responsibility**: Out-of-core OME-Zarr parsing, multiscale chunk fetching, and thread-safe block caching.
- **Key Types**:
  - `ZarrVolume`: Abstraction over multiscale `.zarr` stores.
  - `ChunkKey`, `ChunkGrid`: Coordinate translation between world voxel space and storage chunks.
  - `BlockCache`: Thread-safe LRU cache backed by memory bounds.

### 3.3 `volumen-geom`
- **Responsibility**: Surface geometry, OBJ parsing, mesh integrity, and sheet topology.
- **Key Types**:
  - `SurfaceMesh`: Representation of vertex coordinates, UV texture mappings, and face indices.
  - `NormalEstimator`: Computes smooth vertex and face normals, accounting for winding order.
  - `TopologicalValidator`: Detects self-intersections, inverted faces, and discontinuous jumps.

### 3.4 `volumen-sampler`
- **Responsibility**: The high-performance computational core that generates 2.5D surface volumes.
- **Key Types**:
  - `TrilinearInterpolator`: High-speed trilinear voxel sampling with boundary clamping.
  - `SurfaceVolumeGenerator`: Parallel ray caster generating $N$-layer voxel stacks along the mesh normal field.
  - `TiffExporter`: Writes multi-page or stacked 16-bit grayscale TIFF images conforming to Volume Cartographer specs.

### 3.5 `volumen-cli`
- **Responsibility**: User-facing command-line interface designed as a drop-in high-speed replacement for legacy tooling.
- **Commands**:
  - `volumen render --volume <scroll.zarr> --mesh <segment.obj> --output <dir> --layers 64 --step 1.0`
  - `volumen inspect --volume <scroll.zarr>`
  - `volumen verify --mesh <segment.obj>`

---

## 4. Testing & Verification Protocols

Every component must include exhaustive tests covering both synthetic geometries and edge cases:

1. **Unit Tests**:
   - Every crate must provide in-line unit tests (`cargo test --all`).
2. **Mathematical Invariant Tests**:
   - Trilinear interpolation must evaluate to exact values on linear gradient synthetic volumes.
   - Normal vectors must remain unit length ($\|\mathbf{n}\| = 1.0 \pm 10^{-6}$) and strictly orthogonal to planar meshes.
3. **Property-Based Testing**:
   - Use `proptest` for bounding box intersections and coordinate space conversions.
4. **Integration Benchmarks**:
   - Integration tests in `/tests` simulating end-to-end rendering on synthetic volumetric cylinders and spirals.

```bash
# Run all workspace tests
cargo test --workspace --verbose

# Run with release optimizations
cargo test --workspace --release
```

---

## 5. Coding Standards & Conventions
- **Rust Edition**: 2021+.
- **Zero Panic in Libraries**: Functions in `volumen-core`, `volumen-zarr`, `volumen-geom`, and `volumen-sampler` must never call `unwrap()` or `panic!()` on user/external data. Always return `Result<T, VolumenError>`.
- **Clippy & Formatting**:
  ```bash
  cargo fmt --all -- --check
  cargo clippy --workspace --all-targets -- -D warnings
  ```
- **Documentation**: All public traits, structs, and methods must have docstrings in English detailing parameter meanings, units (voxels vs micrometers), and failure conditions.
