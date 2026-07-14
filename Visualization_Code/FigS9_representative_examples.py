"""
Fig. S9 - Representative examples of visual validation.

A 2x2 composite. Each panel shows, top to bottom: forest-extent maps
(green = forest, white = non-forest), forest-edge maps colored by edge length
(m), and very-high-resolution satellite imagery, across the five target years.

Panels:
  a. Sample 50 - Brazil (Cerrado), 52.72W, 17.87S   (afforestation)
  b. Sample 63 - Laos (SE Asia), 102.56E, 18.07N     (fragmented, roads)
  c. Sample 16 - Maine, USA, 69.76W, 45.01N          (disturbance + recovery)
  d. Sample 53 - South Africa, 30.47E, 26.11S        (plantation blocks)

Source: reorganized from "Validation_comparison_visualizaiton.ipynb", extended
to composite four selected samples into one supplementary figure.
"""

import os
import re

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image

# ---------------------------------------------------------------------------
_OUT = (
    r"G:\Hangkai\Global Forest Edge Other Data"
    r"\validation_samples_0p01deg_per_sample_folder\samples\output"
)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

YEARS = [2000, 2005, 2010, 2015, 2020]
YEAR_TOLERANCE = 2
YEAR_PATTERN = r"(199[0-9]|20[0-1][0-9]|202[0-5])"

# (panel_letter, title, sample_folder)
PANELS = [
    ("a", "Sample 50 - Brazil (Cerrado), 52.72W, 17.87S", os.path.join(_OUT, "sample_00695")),
    ("b", "Sample 63 - Laos (SE Asia), 102.56E, 18.07N",  os.path.join(_OUT, "sample_01095_A")),
    ("c", "Sample 16 - Maine, USA, 69.76W, 45.01N",       os.path.join(_OUT, "sample_00203")),
    ("d", "Sample 53 - South Africa, 30.47E, 26.11S",     os.path.join(_OUT, "sample_01010_A")),
]
# ---------------------------------------------------------------------------


def find_matching_image(folder, target_year, tolerance=YEAR_TOLERANCE):
    """First non-backup PNG whose embedded year is within tolerance."""
    if not os.path.isdir(folder):
        return None, None
    for f in os.listdir(folder):
        if f.endswith("_original.png") or not f.lower().endswith(".png"):
            continue
        m = re.search(YEAR_PATTERN, f)
        if m and abs(int(m.group(1)) - target_year) <= tolerance:
            return os.path.join(folder, f), int(m.group(1))
    return None, None


def read_bands(tif_path):
    with rasterio.open(tif_path) as src:
        return src.read()


def render_panel(fig, outer_cell, folder, letter, title):
    """Render one 3-row x 5-year panel inside `outer_cell` of the figure grid."""
    area = read_bands(os.path.join(folder, "area_5yr.tif"))
    edge = read_bands(os.path.join(folder, "edge_5yr.tif"))

    inner = gridspec.GridSpecFromSubplotSpec(
        3, 5, subplot_spec=outer_cell, hspace=0.08, wspace=0.05
    )

    im_edge = None
    for col, year in enumerate(YEARS):
        # Row 0: forest extent
        band = area[col]
        rgb = np.zeros((*band.shape, 3))
        rgb[band > 0] = [0, 0.5, 0]
        rgb[band == 0] = [1, 1, 1]
        ax = fig.add_subplot(inner[0, col])
        ax.imshow(rgb, aspect="auto"); ax.axis("off")
        ax.set_title(str(year), fontsize=7)

        # Row 1: forest edge
        ax = fig.add_subplot(inner[1, col])
        im_edge = ax.imshow(edge[col], cmap="YlOrRd", vmin=0, aspect="auto")
        ax.axis("off")

        # Row 2: high-res image
        ax = fig.add_subplot(inner[2, col])
        img_path, img_year = find_matching_image(folder, year)
        if img_path:
            ax.imshow(np.array(Image.open(img_path)), aspect="auto")
            ax.set_title(str(img_year), fontsize=7, y=-0.22)
        else:
            ax.text(0.5, 0.5, "No Image", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="red")
        ax.axis("off")

    # Panel title spanning the inner grid
    top_ax = fig.add_subplot(outer_cell)
    top_ax.axis("off")
    top_ax.set_title(f"{letter}. {title}", loc="left", fontsize=11,
                     fontweight="bold", pad=14)
    return im_edge


def main():
    fig = plt.figure(figsize=(20, 13))
    outer = gridspec.GridSpec(2, 2, figure=fig, hspace=0.18, wspace=0.10)

    im_edge = None
    for (letter, title, folder), cell in zip(PANELS, outer):
        if not os.path.isdir(folder):
            print(f"[WARN] missing folder for panel {letter}: {folder}")
            continue
        im_edge = render_panel(fig, cell, folder, letter, title)

    # Shared edge-length colorbar along the bottom
    if im_edge is not None:
        cbar_ax = fig.add_axes([0.35, 0.055, 0.30, 0.012])
        cbar = fig.colorbar(im_edge, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Edge length (m)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    for ext in ("png", "pdf"):
        out = os.path.join(OUT_DIR, f"FigS9_representative_examples.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
