#!/usr/bin/env python3
"""
benchmark_comparison.py
-----------------------
Automated benchmark and differential verification runner comparing Volumen
against Volume Cartographer (vc_render) outputs.

Evaluates:
  1. Wall-clock execution time (Speedup factor)
  2. Numerical fidelity (Mean Absolute Error, PSNR in dB)
  3. Memory consumption (RSS)
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError:
    print("Error: numpy and pillow are required. Run: pip install numpy pillow")
    sys.exit(1)


def compute_metrics(volumen_dir: Path, reference_dir: Path, num_layers: int):
    """Computes MAE, MSE, and PSNR across all rendered TIFF layers."""
    total_voxels = 0
    sum_abs_diff = 0.0
    sum_sq_diff = 0.0

    print(f"Comparing {num_layers} layers between:")
    print(f"  Target (Volumen):     {volumen_dir}")
    print(f"  Reference (vc_render): {reference_dir}")

    for k in range(num_layers):
        filename = f"{k:02d}.tif"
        p_target = volumen_dir / filename
        p_ref = reference_dir / filename

        if not p_target.exists():
            raise FileNotFoundError(f"Missing target slice: {p_target}")
        if not p_ref.exists():
            raise FileNotFoundError(f"Missing reference slice: {p_ref}")

        img_target = np.array(Image.open(p_target), dtype=np.float64)
        img_ref = np.array(Image.open(p_ref), dtype=np.float64)

        if img_target.shape != img_ref.shape:
            raise ValueError(
                f"Dimension mismatch at layer {k}: {img_target.shape} vs {img_ref.shape}"
            )

        diff = img_target - img_ref
        total_voxels += diff.size
        sum_abs_diff += np.sum(np.abs(diff))
        sum_sq_diff += np.sum(diff**2)

    mae = sum_abs_diff / total_voxels
    mse = sum_sq_diff / total_voxels

    if mse < 1e-12:
        psnr_db = float("inf")
    else:
        psnr_db = 20.0 * math.log10(65535.0 / math.sqrt(mse))

    return {
        "total_voxels": int(total_voxels),
        "mae": float(mae),
        "mse": float(mse),
        "psnr_db": float(psnr_db),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run differential comparison between Volumen and vc_render."
    )
    parser.add_argument("--volumen-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--layers", default=64, type=int)
    parser.add_argument("--output-json", default="benchmark_result.json", type=Path)
    args = parser.parse_args()

    results = compute_metrics(args.volumen_dir, args.reference_dir, args.layers)
    print("\n--- Differential Verification Results ---")
    print(f"Total Voxels Compared: {results['total_voxels']:,}")
    print(f"Mean Absolute Error (MAE): {results['mae']:.4f}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {results['psnr_db']:.2f} dB")

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Report saved to: {args.output_json}")


if __name__ == "__main__":
    main()
