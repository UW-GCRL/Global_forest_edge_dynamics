"""
Optional companion to Visualization #2 - tabulate the sample geography.

Walks the sample folders under config.SAMPLES_ROOT, reads center_lon/center_lat
from each meta.txt, and prints counts by continent, climate zone, hemisphere,
and a continent x climate-zone cross-tabulation.

This is analysis (printed tables), not a figure; it lives here because it comes
from the same notebook as the sample map.

Usage:
    python 04_sample_distribution_stats.py
"""

import os
import re

import config


def classify_continent(lon, lat):
    if lat < -60:
        return "Antarctica"
    if -180 <= lon < -30:
        return "North America" if lat >= 15 else "South America"
    if -30 <= lon < 40 and lat >= 35:
        return "Europe"
    if -20 <= lon < 55 and -35 < lat < 38:
        return "Africa"
    if 25 <= lon < 180 and lat >= 8:
        if lon >= 110 and lat < 25 and 110 <= lon <= 180 and -50 <= lat < 25:
            return "Oceania"
        return "Asia"
    if 95 <= lon <= 180 and -50 <= lat < 25:
        return "Oceania"
    if lon >= 40 and lat < 8:
        return "Asia"
    return "Unknown"


def classify_climate_zone(lat):
    abs_lat = abs(lat)
    if abs_lat <= 23.5:
        return "Tropical (0-23.5 deg)"
    if abs_lat <= 35:
        return "Subtropical (23.5-35 deg)"
    if abs_lat <= 66.5:
        return "Temperate (35-66.5 deg)"
    return "Polar/Cold (>66.5 deg)"


def classify_hemisphere(lat):
    return "Northern Hemisphere" if lat >= 0 else "Southern Hemisphere"


def read_sample_coords(root):
    """Return lists of (lon, lat) parsed from every meta.txt under `root`."""
    lons, lats = [], []
    for subfolder in os.listdir(root):
        meta = os.path.join(root, subfolder, "meta.txt")
        if not os.path.isfile(meta):
            continue
        with open(meta, "r", encoding="utf-8") as f:
            content = f.read()
        m = re.search(r"center_lon,\s*center_lat:\s*([-\d.]+),\s*([-\d.]+)", content)
        if m:
            lons.append(float(m.group(1)))
            lats.append(float(m.group(2)))
    return lons, lats


def print_counts(title, values, order=None):
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    keys = order if order else sorted(counts, key=lambda k: -counts[k])
    n = len(values)
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)
    for k in keys:
        if k in counts:
            print(f"{k:<28} {counts[k]:<8} {counts[k] / n * 100:>6.2f}%")
    return counts


def main():
    root = config.SAMPLES_ROOT
    print(f"Reading sample data from: {root}")
    lons, lats = read_sample_coords(root)
    print(f"Successfully read {len(lons)} samples")
    if not lons:
        print("Error: no sample data found. Check config.SAMPLES_ROOT.")
        return

    continents = [classify_continent(lon, lat) for lon, lat in zip(lons, lats)]
    climate_zones = [classify_climate_zone(lat) for lat in lats]
    hemispheres = [classify_hemisphere(lat) for lat in lats]

    zone_order = [
        "Tropical (0-23.5 deg)",
        "Subtropical (23.5-35 deg)",
        "Temperate (35-66.5 deg)",
        "Polar/Cold (>66.5 deg)",
    ]

    print("\n" + "=" * 60)
    print(f"GEOGRAPHIC AND CLIMATE DISTRIBUTION  (total samples: {len(lons)})")
    print("=" * 60)

    continent_counts = print_counts("CONTINENTAL DISTRIBUTION", continents)
    print_counts("CLIMATE ZONE DISTRIBUTION", climate_zones, order=zone_order)
    print_counts("HEMISPHERE DISTRIBUTION", hemispheres)

    # Continent x climate-zone cross-tabulation
    cross = {}
    for cont, zone in zip(continents, climate_zones):
        cross.setdefault(cont, {})[zone] = cross.setdefault(cont, {}).get(zone, 0) + 1

    print("\n" + "-" * 100)
    print("CONTINENT x CLIMATE ZONE CROSS-TABULATION")
    print("-" * 100)
    header = f"{'Continent':<18}" + "".join(f"{z:<26}" for z in zone_order) + "Total"
    print(header)
    for cont in sorted(continent_counts, key=lambda k: -continent_counts[k]):
        row = f"{cont:<18}"
        total = 0
        for zone in zone_order:
            c = cross.get(cont, {}).get(zone, 0)
            total += c
            row += f"{(c if c else '-'):<26}"
        print(row + str(total))


if __name__ == "__main__":
    main()
