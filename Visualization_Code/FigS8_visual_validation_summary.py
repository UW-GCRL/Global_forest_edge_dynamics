"""
Fig. S8 - Summary of visual validation results for forest-edge predictions.

Panel a: stacked bar of edge-consistency assessments across the 100 samples for
         each mapped year (Consistent / Undeterminable / Inconsistent).
Panel b: stacked bar of the dynamic-agreement assessment.

Reads the merged two-reviewer validation table (Y = consistent, U =
undeterminable, N = inconsistent).

Source: built from forest_edge_validation_merged.csv.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
MERGED_CSV = (
    r"G:\Hangkai\Global_Forest_edge_mapping_data\Validation_Samples"
    r"\forest_edge_validation_merged.csv"
)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

YEARS = [2000, 2005, 2010, 2015, 2020]
YEAR_COLS = [f"Year_{y}_Edge_Consistency" for y in YEARS]

# Y = Consistent, U = Undeterminable, N = Inconsistent
CATEGORIES = ["Y", "U", "N"]
COLORS = {"Y": "#2e9e3f", "U": "#f5a623", "N": "#e02b1d"}
LABELS = {
    "Y": "Consistent (Y)",
    "U": "Undeterminable (U)",
    "N": "Inconsistent (N)",
}
# ---------------------------------------------------------------------------


def counts(series):
    """Return {Y, U, N} counts for a column of assessments."""
    vc = series.astype(str).str.strip().str.upper().value_counts()
    return {c: int(vc.get(c, 0)) for c in CATEGORIES}


def stacked_bar(ax, x_positions, per_x_counts, width=0.6):
    """Draw stacked Y/U/N bars; per_x_counts is a list of {Y,U,N} dicts."""
    bottoms = [0] * len(x_positions)
    for cat in CATEGORIES:
        heights = [c[cat] for c in per_x_counts]
        ax.bar(x_positions, heights, width, bottom=bottoms,
               color=COLORS[cat], label=LABELS[cat], edgecolor="white", linewidth=0.4)
        bottoms = [b + h for b, h in zip(bottoms, heights)]


def main():
    df = pd.read_csv(MERGED_CSV, comment="#")

    year_counts = [counts(df[col]) for col in YEAR_COLS]
    dyn_counts = counts(df["Dynamic_Agreement"])

    fig = plt.figure(figsize=(11, 5.5))
    gs = GridSpec(1, 2, width_ratios=[5, 1], wspace=0.25, figure=fig)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharey=ax_a)

    # Panel a
    xa = list(range(len(YEARS)))
    stacked_bar(ax_a, xa, year_counts)
    ax_a.set_xticks(xa)
    ax_a.set_xticklabels(YEARS)
    ax_a.set_xlabel("Year")
    ax_a.set_ylabel("Number of samples")
    ax_a.set_ylim(0, 100)
    ax_a.set_title("a. Edge consistency by year", loc="left", fontweight="bold")
    ax_a.legend(loc="lower right", frameon=True, fontsize=9)

    # Panel b
    stacked_bar(ax_b, [0], [dyn_counts], width=0.5)
    ax_b.set_xticks([0])
    ax_b.set_xticklabels(["Dynamic\nAgreement"])
    ax_b.set_title("b. Dynamics", loc="left", fontweight="bold")
    plt.setp(ax_b.get_yticklabels(), visible=False)

    for ax in (ax_a, ax_b):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(OUT_DIR, f"FigS8_visual_validation_summary.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)

    # console summary
    print("\nEdge consistency counts (Y/U/N):")
    for y, c in zip(YEARS, year_counts):
        print(f"  {y}: {c}")
    print(f"Dynamic agreement: {dyn_counts}")


if __name__ == "__main__":
    main()
