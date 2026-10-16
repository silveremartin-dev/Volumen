# OME-Zarr Out-of-Core Volumetric Streaming

## 1. Motivation

A full micro-CT acquisition of a Herculaneum scroll consists of tens of thousands of $8\text{K} \times 8\text{K}$ 16-bit projection slices, resulting in datasets ranging from 2 TB to 10 TB. Loading such datasets into system memory is impossible on standard workstations.

The Vesuvius Challenge ecosystem adopted the **OME-Zarr** format (NGFF - Next Generation File Formats specification), which structures volumetric datasets as:
- Multi-resolution hierarchical pyramids (levels `0` through `4+`).
- Uniform 3D chunking (typically $128 \times 128 \times 128$ or $256 \times 256 \times 256$ voxels).
- Chunk-level compression (Blosc, Zstandard, or uncompressed raw binary).

`volumen-zarr` provides an out-of-core, asynchronous, thread-safe reader with an integrated LRU cache specifically optimized for non-axis-aligned ray-casting across tilted papyrus meshes.

---

## 2. Chunk Addressing & Index Translation

Given a continuous 3D coordinate point $\mathbf{p} = (x, y, z) \in \mathbb{R}^3$, the corresponding integer voxel index is:
$$i_x = \lfloor x \rfloor, \quad i_y = \lfloor y \rfloor, \quad i_z = \lfloor z \rfloor$$

For a volume partitioned into chunks of dimensions $(C_x, C_y, C_z)$:
- **Chunk Coordinate**:
  $$\mathbf{c} = \left( \lfloor i_x / C_x \rfloor, \lfloor i_y / C_y \rfloor, \lfloor i_z / C_z \rfloor \right)$$
- **Local Voxel Offset within Chunk**:
  $$\mathbf{o} = \left( i_x \pmod{C_x}, i_y \pmod{C_y}, i_z \pmod{C_z} \right)$$
- **Flat In-Chunk Memory Index**:
  $$\text{index} = o_z \cdot (C_x \cdot C_y) + o_y \cdot C_x + o_x$$

---

## 3. LRU Cache & Spatial Locality

Because a papyrus surface mesh weaves continuously through 3D space, ray sampling along adjacent surface pixels exhibits high spatial locality: consecutive rays frequently access voxels within the same or neighboring chunks.

```
                  ┌──────────────────────────────┐
                  │ Ray Sampling Trajectory      │
                  │   R(t) = P0 + t * n          │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                     Voxel query (x, y, z)
                                 │
                                 ▼
                    ┌─────────────────────────┐
             ───────┤ In LRU Cache in Memory? ├───────
            │ Yes   └─────────────────────────┘   No  │
            ▼                                         ▼
   Direct Voxel Read                     Fetch chunk from NVMe/SSD
   (Zero I/O overhead)                   Decompress chunk
                                         Store in Concurrent LRU
                                         Evict oldest chunk if limit exceeded
```

### Cache Configuration
The `volumen-zarr` block cache is configured with a strict memory cap:
- Default: 8 GB.
- Multi-threaded: Thread-safe reads using read-write locks (`RwLock`) or lock-free concurrent hash maps.
- Proactive eviction prevents out-of-memory panics.
