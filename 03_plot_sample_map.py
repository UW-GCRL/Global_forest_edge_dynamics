"""
Step 3 - Visualization #2: global map of validation-sample locations.

Reads the geolocation summary CSV (config.GEO_SUMMARY_CSV) and scatters every
sample on a Robinson world map.

Usage:
    python 03_plot_sample_map.py               # show interactively
    python 03_plot_sample_map.py --save out.png
"""

import argparse

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd

import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", metavar="PATH", default=None,
                        help="save the figure to PATH instead of showing it")
    args = parser.parse_args()

    df = pd.read_csv(config.GEO_SUMMARY_CSV)
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df = df.dropna(subset=["longitude", "latitude"])
    df = df[df["longitude"].between(-180, 180) & df["latitude"].between(-90, 90)]
    print(f"Found {len(df)} sample locations in CSV")

    fig = plt.figure(figsize=(24, 14))
    ax = plt.axes(projection=ccrs.Robinson())

    ax.add_feature(cfeature.OCEAN, facecolor="#D4E7F5", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#F5E6D3", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor="#404040", zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="gray", alpha=0.3, zorder=2)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.3,
                      linestyle="--", x_inline=False, y_inline=False)
    gl.top_labels = gl.bottom_labels = gl.left_labels = gl.right_labels = True
    gl.xlabel_style = {"size": 25, "color": "black"}
    gl.ylabel_style = {"size": 25, "color": "black"}

    ax.scatter(df["longitude"].values, df["latitude"].values,
               transform=ccrs.PlateCarree(), c="red", s=80, alpha=0.75,
               edgecolor="#8B0000", linewidth=0.8, zorder=10)

    ax.set_global()
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"Saved: {args.save}")
    else:
        plt.show()

    print("\nSummary Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Longitude range: {df['longitude'].min():.2f} to {df['longitude'].max():.2f}")
    print(f"Latitude range:  {df['latitude'].min():.2f} to {df['latitude'].max():.2f}")


if __name__ == "__main__":
    main()
