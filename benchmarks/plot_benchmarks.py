"""
Generate comparison plots: custom (new cusolver) vs cusolver (old) vs magma.

For each (gpu, dtype) combination, produces one SVG with a subplot grid
(one subplot per batch size). Only includes data where batch >= 4 and N >= 256.

Naming convention:
  - "custom"   = cusolver column from GPU_new_dtype.txt  (solid line)
  - "cusolver" = cusolver column from GPU_old_dtype.txt  (dashed line)
  - "magma"    = average of magma columns from both files (dot-dashed line)

Each cusolver/magma point is annotated with slowdown vs custom (e.g. ×2.3).
"""

import os
import re
import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BENCHMARK_DIR, "plots")

MIN_BATCH = 4
MIN_SIZE = 256


def parse_file(filepath):
    """Parse a benchmark .txt file with 2-column format (magma | cusolver).

    Detects the time unit from "Times are in ..." line and normalizes to ms.

    Returns:
        dict: {(batch, size): (magma_time_ms, cusolver_time_ms)}
    """
    data = {}

    with open(filepath, "r") as f:
        content = f.read()

    # Detect time unit and compute conversion factor to milliseconds
    scale = 1.0  # default: assume ms
    unit_match = re.search(r"Times are in (\w+)", content)
    if unit_match:
        unit = unit_match.group(1).lower()
        if unit == "microseconds":
            scale = 1e-3
        elif unit == "milliseconds":
            scale = 1.0
        elif unit == "seconds":
            scale = 1e3

    pattern = re.compile(
        r"\(\s*(\d+(?:,\s*\d+)*)\s*\)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)"
    )

    for match in pattern.finditer(content):
        dims = [int(x.strip()) for x in match.group(1).split(",")]
        magma = float(match.group(2)) * scale
        cusolver = float(match.group(3)) * scale

        if len(dims) == 2:
            batch_size = 1
            matrix_size = dims[0]
        elif len(dims) == 3:
            batch_size = dims[0]
            matrix_size = dims[1]
        else:
            continue

        if batch_size < MIN_BATCH or matrix_size < MIN_SIZE:
            continue

        data[(batch_size, matrix_size)] = (magma, cusolver)

    return data


def discover_pairs():
    """Auto-discover all (gpu, dtype) pairs that have both _new_ and _old_ files.

    Scans the benchmark directory for files matching GPU_new_DTYPE.txt and
    finds their GPU_old_DTYPE.txt counterparts.

    Returns:
        list of (gpu, dtype, new_path, old_path)
    """
    pairs = []
    seen = set()

    pattern = re.compile(r"^(.+)_new_(.+)\.txt$")

    for fname in sorted(os.listdir(BENCHMARK_DIR)):
        match = pattern.match(fname)
        if not match:
            continue

        gpu = match.group(1)
        dtype = match.group(2)

        if (gpu, dtype) in seen:
            continue

        new_path = os.path.join(BENCHMARK_DIR, fname)
        old_path = os.path.join(BENCHMARK_DIR, f"{gpu}_old_{dtype}.txt")

        if os.path.exists(old_path):
            pairs.append((gpu, dtype, new_path, old_path))
            seen.add((gpu, dtype))

    return pairs


def build_combined_data(new_data, old_data):
    """Combine new and old benchmark data.

    Returns:
        dict: {batch: [(size, custom_time, cusolver_time, magma_time), ...]}
        where:
          custom_time  = cusolver from new file
          cusolver_time = cusolver from old file
          magma_time   = average of magma from both files
    """
    combined = defaultdict(list)

    # Use keys present in both files
    common_keys = set(new_data.keys()) & set(old_data.keys())

    for batch, size in sorted(common_keys):
        magma_new, cusolver_new = new_data[(batch, size)]
        magma_old, cusolver_old = old_data[(batch, size)]

        custom_time = cusolver_new       # new cusolver = "custom"
        cusolver_time = cusolver_old     # old cusolver = "cusolver"
        magma_time = (magma_new + magma_old) / 2.0  # average magma

        combined[batch].append((size, custom_time, cusolver_time, magma_time))

    # Sort by matrix size within each batch
    for batch in combined:
        combined[batch].sort(key=lambda x: x[0])

    return combined


def plot_comparison(gpu, dtype, data_by_batch, output_dir):
    """Generate one SVG plot per (gpu, dtype) with subplots for each batch size.

    Each subplot has a line plot on top and a small table below showing the
    slowdown of cusolver and magma relative to custom.
    """
    import matplotlib.gridspec as gridspec

    os.makedirs(output_dir, exist_ok=True)

    batch_sizes = sorted(data_by_batch.keys())
    n_batches = len(batch_sizes)

    if n_batches == 0:
        return

    ncols = min(3, n_batches)
    nrows = math.ceil(n_batches / ncols)

    # Each batch gets a plot area (height 4) + table area (height 1)
    fig = plt.figure(figsize=(7 * ncols, 6 * nrows))
    outer_gs = gridspec.GridSpec(
        nrows, ncols, figure=fig,
        hspace=0.45, wspace=0.3,
    )

    color_custom = "#2196F3"
    color_cusolver = "#FF8C00"   # light orange
    color_magma = "#66BB6A"      # light green

    for idx, batch_size in enumerate(batch_sizes):
        row_idx = idx // ncols
        col_idx = idx % ncols

        # Split each cell into plot (top, 75%) and table (bottom, 25%)
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer_gs[row_idx, col_idx],
            height_ratios=[4, 1], hspace=0.05,
        )

        ax = fig.add_subplot(inner_gs[0])
        ax_table = fig.add_subplot(inner_gs[1])

        rows = data_by_batch[batch_size]
        sizes = [r[0] for r in rows]
        custom_times = [r[1] for r in rows]
        cusolver_times = [r[2] for r in rows]
        magma_times = [r[3] for r in rows]

        # --- Line plot ---
        ax.plot(sizes, custom_times, "-o", label="custom",
                linewidth=2, markersize=5, color=color_custom)
        ax.plot(sizes, cusolver_times, "--s", label="cusolver",
                linewidth=2, markersize=5, color=color_cusolver)
        ax.plot(sizes, magma_times, "-.^", label="magma",
                linewidth=2, markersize=5, color=color_magma)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"batch = {batch_size}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")

        ax.set_xticks(sizes)
        ax.set_xticklabels([])  # hide x labels on plot (table has them)

        if col_idx == 0:
            ax.set_ylabel("Time (ms, log scale)", fontsize=10)

        if idx == 0:
            ax.legend(fontsize=9, loc="upper left")

        # --- Slowdown table ---
        ax_table.axis("off")

        # Build table data
        col_labels = [str(s) for s in sizes]
        cs_slowdowns = [
            f"×{cusolver_times[i] / custom_times[i]:.1f}"
            for i in range(len(sizes))
        ]
        mg_slowdowns = [
            f"×{magma_times[i] / custom_times[i]:.1f}"
            for i in range(len(sizes))
        ]

        table_data = [cs_slowdowns, mg_slowdowns]
        row_labels = ["cusolver", "magma"]

        table = ax_table.table(
            cellText=table_data,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc="center",
            rowLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.2)

        # Style the table
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#CCCCCC")
            cell.set_linewidth(0.5)
            if r == 0:
                # Column headers (sizes)
                cell.set_facecolor("#E0E0E0")
                cell.set_text_props(fontweight="bold", fontsize=7)
            elif c == -1:
                # Row labels
                if r == 1:
                    cell.set_facecolor("#FFE0B2")  # cusolver row
                else:
                    cell.set_facecolor("#C8E6C9")  # magma row
                cell.set_text_props(fontweight="bold", fontsize=7)
            else:
                # Data cells
                if r == 1:
                    cell.set_facecolor("#FFF3E0")  # light orange tint
                else:
                    cell.set_facecolor("#E8F5E9")  # light green tint

    fig.suptitle(
        f"LU Factorization — {gpu.upper()} / {dtype}\n"
        f"custom = new cuSOLVER batched, cusolver = old cuSOLVER, magma = MAGMA (avg)",
        fontsize=13, fontweight="bold",
    )

    fname = f"{gpu}_{dtype}.svg"
    outpath = os.path.join(output_dir, fname)
    fig.savefig(outpath, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pairs = discover_pairs()
    print(f"Found {len(pairs)} (GPU, dtype) pairs to plot.\n")

    for gpu, dtype, new_path, old_path in pairs:
        print(f"[{gpu.upper()} / {dtype}]")

        new_data = parse_file(new_path)
        old_data = parse_file(old_path)

        if not new_data or not old_data:
            print("  Insufficient data, skipping.")
            continue

        data_by_batch = build_combined_data(new_data, old_data)
        plot_comparison(gpu, dtype, data_by_batch, OUTPUT_DIR)
        print()

    print(f"Done. Plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
