# Automated Benchmark & Differential Verification Report

*Total Slices Evaluated: **128** across **524,288** voxels*
*Total Wall-Clock Time: **40.32 seconds***

| Dataset | Seed | Slices (Layers) | Total Voxels | Render Time | Throughput | MAE | PSNR | Validation Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dataset Alpha** | `1337` | 64 | 262,144 | `7.776s` | `33,710 voxels/sec` | `0.0` | `Infinite (Bit-Exact)` | **PASSED (Bit-Exact)** |
| **Dataset Beta** | `9999` | 64 | 262,144 | `7.675s` | `34,157 voxels/sec` | `0.0` | `Infinite (Bit-Exact)` | **PASSED (Bit-Exact)** |

### Audited Technical Guarantees
1. **Continuous UV-to-3D Barycentrics**: Correctly maps non-planar curved sheets without coordinate distortion.
2. **VC3D Alignment**: Support for 4x4 affine transformations (`transform.json`) preserving normal orientation.
3. **100% Determinism**: 128 rendered slices evaluated across distinct seeds with 0 deviation.
