"""
Step 1 - Crop the downloaded high-resolution images to their red border.

For every sample folder under config.SAMPLES_ROOT this script:
  1. Deletes old backups (*_original.png)
  2. Finds all valid year-named PNGs (config.TARGET_YEARS +/- tolerance)
  3. Backs each one up as *_original.png
  4. Crops it to the red bounding box, discarding the excess outside

Run this BEFORE 02_plot_comparison_figures.py, because the comparison figure
embeds the cropped PNGs.

Usage:
    python 01_crop_highres_images.py          # prompts before modifying files
    python 01_crop_highres_images.py --yes     # skip the confirmation prompt
"""

import argparse
import os
import shutil

import cv2
import numpy as np

import config


def crop_with_border(image_path):
    """Crop `image_path` to its red border in place, backing it up first."""
    backup_path = image_path.replace(".png", "_original.png")
    if os.path.exists(backup_path):
        print("    Warning: backup already exists, skipping backup creation")
    else:
        shutil.copy2(image_path, backup_path)

    img = cv2.imread(image_path)
    if img is None:
        print("    Error: cannot read image")
        return False

    original_size = f"{img.shape[1]}x{img.shape[0]}"

    # Detect red (two hue ranges wrap around 0 in HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2),
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"    Warning: no red border detected (size: {original_size})")
        return False

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Keep the border itself, trim only the excess outside it
    margin = 1
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img.shape[1] - x, w + 2 * margin)
    h = min(img.shape[0] - y, h + 2 * margin)

    cv2.imwrite(image_path, img[y:y + h, x:x + w])
    print(f"    Cropped: {original_size} -> {w}x{h}")
    return True


def cleanup_backup_files(folder_path):
    """Remove all *_original.png backups so cropping starts from clean files."""
    print("=" * 80)
    print("STEP 1: CLEANING UP BACKUP FILES")
    print("=" * 80)

    deleted = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith("_original.png"):
                os.remove(os.path.join(root, file))
                print(f"Deleting: {file}")
                deleted += 1
    print(f"\nDeleted {deleted} backup files\n")
    return deleted


def crop_all_images(folder_path):
    """Backup + crop every valid year-named PNG under `folder_path`."""
    print("=" * 80)
    print("STEP 2: CROPPING ALL IMAGES")
    print("=" * 80)
    print(f"Target years: {config.TARGET_YEARS} (+/- {config.YEAR_TOLERANCE})\n")

    stats = {"samples": 0, "processed": 0, "images": 0, "cropped": 0, "failed": 0}

    for subfolder in sorted(os.listdir(folder_path)):
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        stats["samples"] += 1

        valid = []
        for year in config.TARGET_YEARS:
            path, found = config.find_matching_image(subfolder_path, year)
            if path is not None:
                valid.append((path, found))

        if not valid:
            continue
        stats["processed"] += 1
        print(f"Sample: {subfolder} ({len(valid)} images)")
        for path, found in valid:
            stats["images"] += 1
            print(f"  Processing: {os.path.basename(path)} (year: {found})")
            if crop_with_border(path):
                stats["cropped"] += 1
            else:
                stats["failed"] += 1
        print()

    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-y", "--yes", action="store_true",
                        help="skip the confirmation prompt")
    args = parser.parse_args()

    folder = config.SAMPLES_ROOT
    print(f"Working folder: {folder}")
    if not os.path.exists(folder):
        print(f"Error: folder does not exist: {folder}")
        return

    if not args.yes:
        if input("\nThis modifies PNGs in place. Proceed? (yes/no): ").strip().lower() not in ("yes", "y"):
            print("Cancelled.")
            return

    deleted = cleanup_backup_files(folder)
    stats = crop_all_images(folder)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Backups deleted:      {deleted}")
    print(f"  Samples scanned:      {stats['samples']}")
    print(f"  Samples with images:  {stats['processed']}")
    print(f"  Images found:         {stats['images']}")
    print(f"  Successfully cropped: {stats['cropped']}")
    print(f"  Failed:               {stats['failed']}")
    if stats["images"]:
        print(f"  Success rate:         {stats['cropped'] / stats['images'] * 100:.1f}%")


if __name__ == "__main__":
    main()
