"""
Compare global hydrate carbon stock estimates (Figure 5c).

This script reads the three scenarios from this study and a curated list
of published global hydrate carbon stock estimates, and draws a horizontal
comparison plot on a logarithmic x-axis.

The output is a single PNG file. No PDF is produced.

Output
------
``fig5c_carbon_stock_comparison.png``
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

FIG_WIDTH_IN = 23.0 / 2.54
FIG_HEIGHT_IN = 15.0 / 2.54

FS_LABEL = 25
FS_TICK = 21
FS_VALUE = 21

COLOR_OUR = "#C44E52"
COLOR_LIT = "#55A868"

X_LIMITS = (200, 90000)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
OUR_STUDY = [
    ("This study (low)", 2410),
    ("This study (medium)", 5469),
    ("This study (high)", 17136),
]

# (year, label, low, high, single_value_or_None)
LITERATURE = [
    (1994, "Gornitz & Fung (1994)", 14200, 14200, 14200),
    (1988, "Kvenvolden (1988)", 11000, 11000, 11000),
    (2004, "Milkov (2004)", 500, 2500, None),
    (2004, "Buffett & Archer (2004)", 3000, 3000, 3000),
    (2005, "Klauda & Sandler (2005)", 55800, 55800, 55800),
    (2009, "Archer et al. (2009)", 1600, 2000, None),
    (2011, "Boswell & Collett (2011)", 1500, 1500, 1500),
    (2011, "Dickens (2011)", 170, 12700, None),
    (2015, "Kretschmer et al. (2015)", 1146, 1146, 1146),
    (2017, "Ruppel & Kessler (2017)", 1800, 1800, 1800),
    (2023, "Wang et al. (2023)", 1100, 12400, None),
]

# Authors whose single-value estimate is a lower bound
GE_AUTHORS = {
    "Kretschmer et al. (2015)",
    "Ruppel & Kessler (2017)",
}


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_comparison(out_path):
    """Draw and save the carbon-stock comparison figure.

    Parameters
    ----------
    out_path : str
        Output PNG path.
    """
    labels = [s[0] for s in OUR_STUDY] + [
        x[1] for x in LITERATURE[::-1]
    ]
    y_pos = np.arange(len(labels))
    n_our = len(OUR_STUDY)

    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=300
    )

    # -----------------------------------------------------------------
    # This study
    # -----------------------------------------------------------------
    for i, (_, value) in enumerate(OUR_STUDY):
        ax.barh(
            y_pos[i],
            value,
            height=0.6,
            color=COLOR_OUR,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.text(
            value * 1.08,
            y_pos[i],
            f"{value:,}",
            va="center",
            fontsize=FS_VALUE,
            fontweight="bold",
        )

    # -----------------------------------------------------------------
    # Literature
    # -----------------------------------------------------------------
    lit_reversed = LITERATURE[::-1]

    for i, (_, author, low, high, single) in enumerate(lit_reversed):
        y = y_pos[n_our + i]

        if single is not None:
            ax.plot(
                single,
                y,
                marker="D",
                markersize=10,
                color=COLOR_LIT,
                markeredgecolor="black",
                markeredgewidth=0.8,
                zorder=5,
            )

            prefix = "≥" if author in GE_AUTHORS else ""

            if "Klauda" in author:
                ax.text(
                    single * 0.90,
                    y,
                    f"{prefix}{single:,}",
                    va="center",
                    ha="right",
                    fontsize=FS_VALUE,
                )
            else:
                ax.text(
                    single * 1.08,
                    y,
                    f"{prefix}{single:,}",
                    va="center",
                    fontsize=FS_VALUE,
                )
        else:
            ax.plot(
                [low, high],
                [y, y],
                color=COLOR_LIT,
                linewidth=3.2,
                solid_capstyle="round",
                zorder=5,
            )
            ax.plot(
                [low, high],
                [y, y],
                marker="|",
                markersize=11,
                color="black",
                markeredgewidth=1.5,
                zorder=6,
                linestyle="none",
            )
            ax.text(
                high * 1.08,
                y,
                f"{low:,}–{high:,}",
                va="center",
                fontsize=FS_VALUE,
            )

    # -----------------------------------------------------------------
    # Axes and grid
    # -----------------------------------------------------------------
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=FS_TICK)
    ax.set_xscale("log")
    ax.set_xlim(*X_LIMITS)
    ax.set_xlabel(
        "Global hydrate carbon stock (Gt C)", fontsize=FS_LABEL
    )
    ax.tick_params(axis="x", labelsize=FS_TICK)

    ax.grid(axis="x", alpha=0.25, linewidth=0.5, which="major")
    ax.grid(axis="x", alpha=0.12, linewidth=0.4, which="minor")
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    ax.axhline(
        y=n_our - 0.5,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------
def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare global hydrate carbon stock estimates."
    )
    parser.add_argument(
        "--out",
        default="fig5c_carbon_stock_comparison.png",
        help="Output PNG path.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the comparison plotting workflow."""
    args = parse_args(argv)
    plot_comparison(args.out)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)