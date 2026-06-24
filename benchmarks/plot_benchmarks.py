"""
Generate line plots from LU benchmark results.

For each (gpu, dtype) combination, produces one SVG plot with a subplots grid
(one subplot per batch size). Only includes data where batch >= 4 and N >= 256.
X-axis: matrix size (log scale)
Y-axis: time in microseconds (log scale)
Lines: custom (solid), magma (dashed), cusolver (dot-dashed)
"""

import os
import re
import glob
import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BENCHMARK_DIR, "plots")

# Files to include (post June 6, excluding cuda132 variants)
INCLUDE_PREFIXES = ["a100", "h100", "gb200", "l40s", "rtx5090"]

MIN_BATCH = 4
MIN_SIZE = 256


def parse_benchmark_file(filepath):
    """Parse a benchmark .txt file and return structured data.

    Returns:
        (has_magma, data_by_batch) where data_by_batch is a dict:
        {batch_size: [(matrix_size, magma_time, cusolver_time, custom_time), ...]}
        Only includes entries with batch >= MIN_BATCH and matrix_size >= MIN_SIZE.
        If has_magma is False, magma_time will be None for all entries.
    """
    data_by_batch = defaultdict(list)

    with open(filepath, "r") as f:
        content = f.read()

    # Detect if file has 3 columns (magma | cusolver | custom) or 2 (cusolver | custom)
    has_magma = "magma" in content

    if has_magma:
        pattern = re.compile(
            r"\(\s*(\d+(?:,\s*\d+)*)\s*\)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)"
        )
    else:
        pattern = re.compile(
            r"\(\s*(\d+(?:,\s*\d+)*)\s*\)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)"
        )

    for match in pattern.finditer(content):
        dims = [int(x.strip()) for x in match.group(1).split(",")]

        if has_magma:
            magma = float(match.group(2))
            cusolver = float(match.group(3))
            custom = float(match.group(4))
        else:
            magma = None
            cusolver = float(match.group(2))
            custom = float(match.group(3))

        if len(dims) == 2:
            batch_size = 1
            matrix_size = dims[0]
        elif len(dims) == 3:
            batch_size = dims[0]
            matrix_size = dims[1]
        else:
            continue

        # Filter: only batch >= 4 and size >= 256
        if batch_size < MIN_BATCH or matrix_size < MIN_SIZE:
            continue

        data_by_batch[batch_size].append((matrix_size, magma, cusolver, custom))

    # Sort each batch's data by matrix_size
    for batch in data_by_batch:
        data_by_batch[batch].sort(key=lambda x: x[0])

    return has_magma, data_by_batch


def extract_gpu_dtype(filename):
    """Extract GPU name and dtype from filename like 'h100_float32.txt'."""
    name = os.path.splitext(filename)[0]
    for dtype in ["complex128", "complex64", "float64", "float32"]:
        if name.endswith(dtype):
            gpu = name[: -(len(dtype) + 1)]  # strip _dtype
            return gpu, dtype
    return name, "unknown"


def plot_benchmark(gpu, dtype, has_magma, data_by_batch, output_dir):
    """Generate one plot per (gpu, dtype) with subplots grid for batch sizes."""
    os.makedirs(output_dir, exist_ok=True)

    batch_sizes = sorted(data_by_batch.keys())
    n_batches = len(batch_sizes)

    if n_batches == 0:
        return

    # Determine grid layout
    ncols = min(3, n_batches)
    nrows = math.ceil(n_batches / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                             squeeze=False, sharex=True, sharey=True)

    for idx, batch_size in enumerate(batch_sizes):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]

        rows = data_by_batch[batch_size]
        matrix_sizes = [r[0] for r in rows]
        cusolver_times = [r[2] for r in rows]
        custom_times = [r[3] for r in rows]

        ax.plot(matrix_sizes, custom_times, "-o", label="custom",
                linewidth=2, markersize=4)
        if has_magma:
            magma_times = [r[1] for r in rows]
            ax.plot(matrix_sizes, magma_times, "--s", label="magma",
                    linewidth=2, markersize=4)
        ax.plot(matrix_sizes, cusolver_times, "-.^", label="cusolver",
                linewidth=2, markersize=4)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"batch = {batch_size}", fontsize=11)
        ax.grid(True, alpha=0.3, which="both")

        ax.set_xticks(matrix_sizes)
        ax.set_xticklabels([str(s) for s in matrix_sizes], rotation=45,
                           ha="right", fontsize=8)

        if col == 0:
            ax.set_ylabel("Time (μs, log scale)", fontsize=10)
        if row == nrows - 1:
            ax.set_xlabel("Matrix size N×N (log scale)", fontsize=10)

        # Only show legend in first subplot
        if idx == 0:
            ax.legend(fontsize=9)

    # Hide unused subplots
    for idx in range(n_batches, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"LU Factorization — {gpu.upper()} / {dtype}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    fname = f"{gpu}_{dtype}.svg"
    outpath = os.path.join(output_dir, fname)
    fig.savefig(outpath, format="svg")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(BENCHMARK_DIR, "*.txt")))
    files = [
        f
        for f in files
        if any(os.path.basename(f).startswith(p) for p in INCLUDE_PREFIXES)
        and "cuda132" not in os.path.basename(f)
    ]

    print(f"Processing {len(files)} benchmark files...\n")

    for filepath in files:
        filename = os.path.basename(filepath)
        gpu, dtype = extract_gpu_dtype(filename)
        print(f"[{gpu} / {dtype}]")

        has_magma, data_by_batch = parse_benchmark_file(filepath)
        if not data_by_batch:
            print("  No data found, skipping.")
            continue

        plot_benchmark(gpu, dtype, has_magma, data_by_batch, OUTPUT_DIR)
        print()

    print(f"Done. Plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
