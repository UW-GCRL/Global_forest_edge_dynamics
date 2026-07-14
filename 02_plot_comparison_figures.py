"""
Step 2 - Visualization #1: per-sample forest-change comparison figure.

For every sample folder that has area_5yr.tif and edge_5yr.tif, builds a 3x5
panel (rows = forest extent / forest-edge length / high-res image; columns =
the five target years) and saves it as forest_change_comparison.png inside the
sample folder.

Run AFTER 01_crop_highres_images.py so the embedded high-res images are already
cropped to their red border.

Usage:
    python 02_plot_comparison_figures.py
"""

import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image

import config


def extract_coordinates_from_meta(subfolder):
    """Pull the 0.01deg pixel bounds out of meta.txt for the figure title."""
    import re

    meta_path = os.path.join(subfolder, "meta.txt")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if "0.01deg pixel bounds" in line:
                    m = re.search(
                        r"\(L,B,R,T\):\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)",
                        line,
                    )
                    if m:
                        left, bottom, right, top = m.groups()
                        return f"({left} deg, {bottom} deg) to ({right} deg, {top} deg)"
    except Exception as e:
        print(f"Warning: could not read meta.txt in {subfolder}: {e}")
    return None


def read_tif_bands(tif_path):
    """Read all bands of a multi-band GeoTIFF as (bands, H, W)."""
    with rasterio.open(tif_path) as src:
        return src.read()


def create_comparison_figure(subfolder):
    """Build and save the 3x5 comparison figure for one sample folder."""
    years = config.TARGET_YEARS
    year_indices = {y: i for i, y in enumerate(years)}

    area_tif = os.path.join(subfolder, config.AREA_TIF)
    edge_tif = os.path.join(subfolder, config.EDGE_TIF)
    if not os.path.exists(area_tif) or not os.path.exists(edge_tif):
        print(f"Skipping {subfolder}: missing TIF files")
        return None

    area_data = read_tif_bands(area_tif)   # (5, H, W)
    edge_data = read_tif_bands(edge_tif)   # (5, H, W)

    coords = extract_coordinates_from_meta(subfolder)
    title = f"Validation Sample: {coords}" if coords else \
        f"Validation Sample: {os.path.basename(subfolder)}"

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(4, 5, figure=fig, height_ratios=[1, 1, 0.08, 1],
                           hspace=0.25, wspace=0.15)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    axes = [[fig.add_subplot(gs[row, col]) for col in range(5)] for row in (0, 1, 3)]

    im = None
    for col, year in enumerate(years):
        band_idx = year_indices[year]

        # Row 1: forest extent (green = forest, white = non-forest)
        area_band = area_data[band_idx]
        forest_mask = np.zeros((*area_band.shape, 3))
        forest_mask[area_band > 0] = [0, 0.5, 0]
        forest_mask[area_band == 0] = [1, 1, 1]
        axes[0][col].imshow(forest_mask, aspect="auto")
        axes[0][col].set_title(f"Forest Extent\n{year}", fontsize=10)
        axes[0][col].axis("off")

        # Row 2: forest-edge length (heatmap)
        edge_band = edge_data[band_idx]
        im = axes[1][col].imshow(edge_band, cmap="YlOrRd", vmin=0, aspect="auto")
        axes[1][col].set_title(f"Forest Edge\n{year}", fontsize=10)
        axes[1][col].axis("off")

        # Row 3: matching high-res image
        img_path, img_year = config.find_matching_image(subfolder, year)
        if img_path:
            axes[2][col].imshow(np.array(Image.open(img_path)), aspect="auto")
            axes[2][col].set_title(f"High-Res Image\n{img_year}", fontsize=10)
        else:
            axes[2][col].text(0.5, 0.5, "No Image", ha="center", va="center",
                              transform=axes[2][col].transAxes, fontsize=12, color="red")
            axes[2][col].set_title(f"High-Res Image\n{year} (N/A)", fontsize=10)
        axes[2][col].axis("off")

    axes[0][0].set_ylabel("Forest Extent", fontsize=11, fontweight="bold", labelpad=10)
    axes[1][0].set_ylabel("Forest Edge Length", fontsize=11, fontweight="bold", labelpad=10)
    axes[2][0].set_ylabel("High-Resolution Image", fontsize=11, fontweight="bold", labelpad=10)

    if im is not None:
        cbar_ax = fig.add_axes([0.375, 0.39, 0.25, 0.015])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Edge Length (m)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    output_path = os.path.join(subfolder, config.COMPARISON_PNG)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Created: {output_path}")
    return output_path


def main():
    root = config.SAMPLES_ROOT
    print("=" * 80)
    print("Forest Change Comparison Figure Generator")
    print("=" * 80)
    print(f"Root folder: {root}")
    print(f"Target years: {config.TARGET_YEARS}  (image tolerance +/- {config.YEAR_TOLERANCE})")

    if not os.path.exists(root):
        print(f"Error: folder does not exist: {root}")
        return

    outputs = []
    for cur, _, files in os.walk(root):
        if config.AREA_TIF in files and config.EDGE_TIF in files:
            print(f"\nProcessing: {cur}")
            out = create_comparison_figure(cur)
            if out:
                outputs.append(out)

    print("\n" + "=" * 80)
    print(f"Complete! Generated {len(outputs)} comparison figures")
    print("=" * 80)


if __name__ == "__main__":
    main()
