# Volumen System Architecture & Dataflow

## 1. Overview & Problem Statement

The computational virtual unwrapping pipeline developed for the Herculaneum papyri is tasked with processing three-dimensional X-ray computed tomography data of unprecedented scale. A single scroll scan produces up to 10 Terabytes of volumetric data represented in regular voxel lattices at 2.4 to 7.91 micrometers per voxel.

Historically, the pipeline relied on disparate tools connected by shell scripts:
1. `Volume Cartographer` (C++17) handled mesh manipulation and volume texturing.
2. Custom Python scripts handled Zarr slicing, instance clustering, and ML inference.
3. Intermediate data was repeatedly serialized as uncompressed TIFF slices or massive OBJ meshes.

This architecture causes:
- **I/O Choke Points**: Disk transfer saturation caused by millions of individual files.
- **Memory Spikes**: Lack of bounded out-of-core caches leading to worker crashes.
- **Fragile Integration**: Inconsistent coordinate transformations between voxel indices, physical metric space, and UV texture domains.

`Volumen` replaces this disparate toolchain with a cohesive, high-performance Rust ecosystem designed around strict memory bounds, unified coordinate spaces, and parallel SIMD operations.

---

## 2. Coordinate Systems & Spaces

Volumen defines four strictly typed, non-ambiguous spatial domains:

```
[Voxel Coordinate Space]       (Integer grid: ix, iy, iz in [0, Dimension))
          ▲
          │ Scaling by resolution (e.g. 7.91 µm / voxel)
          ▼
[Physical World Space]         (Continuous metric coordinates in mm / µm)
          ▲
          │ Mesh barycentric embedding
          ▼
[Parametric Sheet Space]       (Continuous 2D UV coordinates in [0.0, 1.0])
          ▲
          │ Normal extrusion: S(u,v) + t * n(u,v)
          ▼
[Surface Volume Stack Space]   (2.5D Layer grid: [u_pixel, v_pixel, layer_k])
```

1. **Voxel Space ($\mathbb{Z}^3$)**: The discrete lattice of the tomographic scan.
2. **World Space ($\mathbb{R}^3$)**: The metric physical frame of the synchrotron apparatus.
3. **Parametric UV Space ($\mathbb{R}^2$)**: The two-dimensional flattened domain of the papyrus sheet produced via As-Rigid-As-Possible (ARAP) or Scalable Locally Injective Mappings (SLIM).
4. **Surface Volume Space ($\mathbb{Z}^2 \times \mathbb{Z}$)**: The discrete multi-layer texturing domain, where each pixel $(u, v)$ has $K$ depth layers along the normal vector field.

---

## 3. High-Throughput Processing Pipeline

```
                     ┌─────────────────────────────┐
                     │ OME-Zarr Multiscale Volume  │
                     └──────────────┬──────────────┘
                                    │
                        Chunk requests on-demand
                                    │
                                    ▼
┌──────────────────┐    ┌─────────────────────────────┐
│  Segmented Mesh  │───▶│ volumen-zarr Block Cache    │
│  (.obj with UVs) │    │ (Concurrent LRU in RAM)     │
└────────┬─────────┘    └──────────────┬──────────────┘
         │                             │
         │ Vertex & UV Barycentrics    │ Voxel neighborhoods
         ▼                             ▼
┌─────────────────────────────────────────────────────┐
│ volumen-sampler (Rayon Thread Pool)                 │
│                                                     │
│ For each (u, v) pixel:                              │
│   1. Find 3D base point S(u, v) via barycentric xfm │
│   2. Query smooth unit normal vector n(u, v)        │
│   3. Compute ray: P(k) = S(u,v) + offset(k) * n     │
│   4. Perform SIMD trilinear interpolation           │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
          ┌───────────────────────────────────┐
          │ High-speed TIFF Stack / Zarr 2.5D │
          │ (Ready for PyTorch Ink Detection) │
          └───────────────────────────────────┘
```

---

## 4. Key Design Decisions

1. **Zero-Copy / Bounded Memory**:
   - The memory budget is determined exclusively by `--max-cache-gb`.
   - Volumes are accessed through an out-of-core block abstraction, guaranteeing that 10 TB volumes can be processed on commodity 16 GB workstations.
2. **Thread Concurrency via Work-Stealing**:
   - Surface patches are broken into tile blocks and scheduled across `Rayon` worker threads.
   - Cache contention is minimized through spatial locality partitioning.
3. **Strict Mathematical Guarantees**:
   - Degenerate triangles and self-intersecting folds are caught in preprocessing by `volumen-geom`.
   - Normal vectors are normalized and orientation-checked to prevent inverted surface layers.
