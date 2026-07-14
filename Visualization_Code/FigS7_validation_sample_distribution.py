"""
Fig. S7 - Distribution of validation samples.

Robinson world map with a red point for each of the 100 randomly selected
validation samples. Reads the geolocation summary CSV.

Source: reorganized from "Geolocation of Validation Samples.ipynb" (cell 1).
"""

import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
GEO_SUMMARY_CSV = (
    r"G:\Hangkai\Global_Forest_edge_mapping_data\Validation_Samples"
    r"\validation_samples_geolocation_summary_hangkai.csv"
)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
# ---------------------------------------------------------------------------


def main():
    df = pd.read_csv(GEO_SUMMARY_CSV)
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df = df.dropna(subset=["longitude", "latitude"])
    df = df[df["longitude"].between(-180, 180) & df["latitude"].between(-90, 90)]
    print(f"Plotting {len(df)} validation samples")

    fig = plt.figure(figsize=(12, 7))
    ax = plt.axes(projection=ccrs.Robinson())

    ax.add_feature(cfeature.OCEAN, facecolor="#D4E7F5", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#F5E6D3", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="#404040", zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="gray", alpha=0.3, zorder=2)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.3,
                      linestyle="--", x_inline=False, y_inline=False)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {"size": 11, "color": "black"}
    gl.ylabel_style = {"size": 11, "color": "black"}

    ax.scatter(df["longitude"].values, df["latitude"].values,
               transform=ccrs.PlateCarree(), c="red", s=28, alpha=0.85,
               edgecolor="#8B0000", linewidth=0.4, zorder=10)

    ax.set_global()
    plt.tight_layout()

    for ext in ("png", "pdf"):
        out = os.path.join(OUT_DIR, f"FigS7_validation_sample_distribution.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
