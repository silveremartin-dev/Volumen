# Differential Testing & Benchmarking Protocol

## 1. Objective: Acceptance in the Upstream Ecosystem

To be accepted into the official `ScrollPrize` ecosystem (or integrated into the `villa` monorepo), `Volumen` must prove beyond any doubt that:
1. **Mathematical Equivalence**: The 2.5D surface volumes produced by `Volumen` are pixel-identical (or within strict floating-point epsilon) to the ground truth produced by the official reference implementation (`vc_render` from `Volume Cartographer`).
2. **Deterministic Speedup**: Wall-clock render time drops by **10x to 50x**.
3. **Bounded Memory Consumption**: Memory footprint stays strictly bounded below the configured `--max-cache-gb` threshold without leaking memory or causing paging thrashing.

---

## 2. Test Datasets (Standard Community Benchmarks)

Benchmarks are evaluated across three official public segments hosted on the AWS open data bucket (`s3://vesuvius-challenge-open-data/`):

| Dataset / Segment | Resolution | Dimensions | Primary Challenge |
| :--- | :--- | :--- | :--- |
| **Campfire Segment (PHerc. Paris 4)** | $2.4\ \mu\text{m}$ | $1024 \times 1024 \times 64$ | High-contrast ink validation, direct 3D ground truth comparison |
| **Scroll 1 Segment (PHerc. 1667)** | $7.91\ \mu\text{m}$ | $4096 \times 4096 \times 64$ | Broad unrolled scroll sheet, large memory streaming test |
| **Synthetic Spiral Testbed** | $1.0\ \mu\text{m}$ (Synth) | $512 \times 512 \times 32$ | Analytical ground truth (mathematically generated gradient volume) |

---

## 3. Comparative Metrics

Every test run outputs a structured JSON report evaluating:

### 3.1 Numerical Fidelity
- **Mean Absolute Error (MAE)**:
  $$\text{MAE} = \frac{1}{W \cdot H \cdot K} \sum_{u=1}^W \sum_{v=1}^H \sum_{k=1}^K |I_{\text{volumen}}(u, v, k) - I_{\text{vc\_render}}(u, v, k)|$$
- **Peak Signal-to-Noise Ratio (PSNR)**:
  $$\text{PSNR} = 20 \cdot \log_{10}\left(\frac{65535}{\sqrt{\text{MSE}}}\right) \quad (\text{Target: } > 50\text{ dB})$$
- **Structural Similarity Index (SSIM)**: $\text{Target: } > 0.999$.

### 3.2 Performance & Resource Consumption
- **Wall-Clock Execution Time ($T_{\text{render}}$)**: Measured via high-resolution monotonic clocks.
- **Speedup Factor**:
  $$S = \frac{T_{\text{vc\_render}}}{T_{\text{volumen}}}$$
- **Peak Resident Set Size (RSS)**: Measured via system process metrics (must remain $\le \text{Configured Max Cache}$).
- **Disk I/O Throughput**: Total megabytes read from storage vs redundant read reduction.

---

## 4. Benchmark Harness Execution

The automated verification suite is executed via:
```bash
# Run internal mathematical invariant checks
cargo test --workspace --release

# Run differential regression suite against reference outputs
cargo test --test test_differential --release -- --nocapture
```
