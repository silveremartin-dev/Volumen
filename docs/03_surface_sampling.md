# Surface Volume Normal Sampling & Ray-Casting

## 1. Concept: What is a Surface Volume?

In carbonized papyrus analysis, ink (carbon-based or heavy-metal trace pigment) does not rest on a perfectly flat plane; it penetrates microscopically into the upper cellular fibers of the papyrus sheet. Furthermore, the extracted 3D mesh is subject to slight segmentation misalignments (typically $\pm 2$ to $\pm 5$ voxels).

To overcome this, the Vesuvius Challenge does not sample a single 2D layer. Instead, it extracts a **Surface Volume**:
- A 2.5D slab of voxels sampled along the normal vector $\mathbf{n}$ at every pixel of the flattened mesh UV space.
- Typically contains **32 or 64 depth layers**.
- Downstream machine learning models (3D ResNets, Vision Transformers) ingest this multi-layer volume to detect ink signals across depth slices.

---

## 2. Geometry of Normal Ray-Casting

```
            +32 voxels ───────── Layer +31 (Top air boundary)
                 │
                 │   ▲ Normal vector n(u, v)
                 │   │
              0  *───┴── Papyrus Mesh Surface Point S(u, v)
                 │
                 │
            -32 voxels ───────── Layer 0 (Bottom papyrus/fiber boundary)
```

For each flattened UV pixel $(u, v)$ in the output image:
1. Compute the corresponding 3D world position $\mathbf{S}(u, v)$ via barycentric interpolation over the mesh triangle containing $(u, v)$.
2. Compute the interpolated unit surface normal $\mathbf{n}(u, v)$.
3. For each layer $k \in \{0, \dots, N-1\}$:
   $$\mathbf{P}_k(u, v) = \mathbf{S}(u, v) + \left(k - \frac{N - 1}{2}\right) \cdot \delta \cdot \mathbf{n}(u, v)$$
   where $\delta$ is the inter-layer step size (typically $1.0$ voxel).
4. Evaluate voxel intensity at $\mathbf{P}_k(u, v)$ using trilinear interpolation:
   $$I_k(u, v) = \text{TrilinearSample}(\text{Volume}, \mathbf{P}_k(u, v))$$

---

## 3. High-Speed Trilinear Interpolation

Given a floating-point coordinate $\mathbf{P} = (x, y, z)$ within the voxel grid, trilinear interpolation computes the weighted average of the 8 surrounding lattice vertices:

```
        v011 ──────────── v111
        /│                /│
       / │               / │
    v001 ──────────── v101 │
     │   │             │   │
     │  v010 ──────────│─ v110
     │  /              │  /
     │ /               │ /
    v000 ──────────── v100
```

Let:
$$x_0 = \lfloor x \rfloor, \quad y_0 = \lfloor y \rfloor, \quad z_0 = \lfloor z \rfloor$$
$$d_x = x - x_0, \quad d_y = y - y_0, \quad d_z = z - z_0$$

Then:
$$\begin{aligned}
c_{00} &= v_{000}(1 - d_x) + v_{100}d_x \\
c_{01} &= v_{001}(1 - d_x) + v_{101}d_x \\
c_{10} &= v_{010}(1 - d_x) + v_{110}d_x \\
c_{11} &= v_{011}(1 - d_x) + v_{111}d_x \\
c_0    &= c_{00}(1 - d_y) + c_{10}d_y \\
c_1    &= c_{01}(1 - d_y) + c_{11}d_y \\
I      &= c_0(1 - d_z) + c_1 d_z
\end{aligned}$$

`volumen-sampler` implements this interpolation using SIMD arithmetic, executing all 8 vertex lookups and linear blending operations with high throughput.
