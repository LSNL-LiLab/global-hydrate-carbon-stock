"""
Compare true-positive and false-negative samples for selected features.

This script reads the true-positive and false-negative tables exported by
the spatial cross-validation step, standardizes each feature using the
combined TP+FN distribution, and draws side-by-side boxplots.

Inputs
------
- ``Classifier_models/true_positives_cv.csv``
- ``Classifier_models/false_negatives_cv.csv``

Output
------
``fig4d_true_positive_vs_false_negative.png``
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

COLOR_TP = "#4C72B0"
COLOR_FN = "#C44E52"

FS_TICK = 21
FS_LABEL = 21
FS_LEGEND = 21

FIG_WIDTH_IN = 19.0 / 2.54
FIG_HEIGHT_IN = 12.0 / 2.54

COMPARE_FEATURES = [
    ("crustal_age", "Crustal\nage"),
    ("sediment_thickness", "Sediment\nthickness"),
    ("dist_to_subduction", "Dist. to\nsubduction"),
    ("geothermal_gradient", "Geothermal\ngradient"),
]


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------
def load_cv_tables(models_dir):
    """Load the true-positive and false-negative CV tables."""
    tp_path = os.path.join(models_dir, "true_positives_cv.csv")
    fn_path = os.path.join(models_dir, "false_negatives_cv.csv")

    for path in (tp_path, fn_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input not found: {path}")

    df_tp = pd.read_csv(tp_path)
    df_fn = pd.read_csv(fn_path)
    print(f"TP: {len(df_tp)}, FN: {len(df_fn)}")
    return df_tp, df_fn


def standardize_against_pooled(df_tp, df_fn, compare_features):
    """Standardize each feature using the pooled TP+FN distribution.

    Returns
    -------
    tuple
        ``(data_tp_z, data_fn_z, labels)``.
    """
    data_tp_z, data_fn_z, labels = [], [], []

    for feat, label in compare_features:
        if feat not in df_tp.columns or feat not in df_fn.columns:
            continue

        tp_v = df_tp[feat].dropna().values
        fn_v = df_fn[feat].dropna().values
        if len(tp_v) == 0 or len(fn_v) == 0:
            continue

        pooled = np.concatenate([tp_v, fn_v])
        mean, std = pooled.mean(), pooled.std()
        if std > 0:
            data_tp_z.append((tp_v - mean) / std)
            data_fn_z.append((fn_v - mean) / std)
            labels.append(label)

    return data_tp_z, data_fn_z, labels


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def draw_boxplots(ax, data_tp_z, data_fn_z, labels, n_tp, n_fn):
    """Draw paired boxplots with a styled legend."""
    positions = np.arange(len(labels))
    width = 0.35

    common_box = {
        "patch_artist": True,
        "showfliers": False,
        "medianprops": {"color": "black", "linewidth": 1.5},
        "whiskerprops": {"color": "black", "linewidth": 0.8},
        "capprops": {"color": "black", "linewidth": 0.8},
    }

    bp_tp = ax.boxplot(
        data_tp_z,
        positions=positions - width / 2,
        widths=width,
        boxprops={
            "facecolor": COLOR_TP,
            "alpha": 0.85,
            "edgecolor": "black",
            "linewidth": 0.8,
        },
        **common_box,
    )
    bp_fn = ax.boxplot(
        data_fn_z,
        positions=positions + width / 2,
        widths=width,
        boxprops={
            "facecolor": COLOR_FN,
            "alpha": 0.85,
            "edgecolor": "black",
            "linewidth": 0.8,
        },
        **common_box,
    )

    ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=FS_TICK, linespacing=1)
    ax.set_ylabel("Z-score (standardized value)", fontsize=FS_LABEL)
    ax.set_ylim(-2.2, 3.8)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.grid(axis="y", alpha=0.2)

    # -----------------------------------------------------------------
    # Legend (all tunable parameters kept with short English notes)
    # -----------------------------------------------------------------
    legend = ax.legend(
        [bp_tp["boxes"][0], bp_fn["boxes"][0]],
        [f"True positives (n={n_tp})", f"False negatives (n={n_fn})"],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.96),
        fontsize=FS_LEGEND,
        labelspacing=0.15,
        handletextpad=0.3,
        handlelength=1.5,
        handleheight=1.0,
        frameon=True,
        fancybox=True,
        edgecolor="black",
        facecolor="white",
        framealpha=0.3,
        borderpad=0,
        borderaxespad=0,
        ncol=1,
        columnspacing=1.0,
        markerscale=1.0,
        title=None,
        title_fontsize=None,
    )
    legend.get_frame().set_boxstyle(
        "round,pad=0.3,rounding_size=0.3"
    )
    legend.get_frame().set_linewidth(1.6)


def make_figure(df_tp, df_fn, out_path):
    """Build and save the TP-vs-FN boxplot figure."""
    data_tp_z, data_fn_z, labels = standardize_against_pooled(
        df_tp, df_fn, COMPARE_FEATURES
    )
    if not labels:
        raise ValueError(
            "No overlapping features with valid data; nothing to plot."
        )

    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=300
    )
    draw_boxplots(ax, data_tp_z, data_fn_z, labels, len(df_tp), len(df_fn))

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
        description=(
            "Compare true-positive and false-negative samples across "
            "selected features."
        )
    )
    parser.add_argument(
        "--models-dir",
        default="Classifier_models",
        help=(
            "Directory containing true_positives_cv.csv and "
            "false_negatives_cv.csv."
        ),
    )
    parser.add_argument(
        "--out",
        default="fig4d_true_positive_vs_false_negative.png",
        help="Output PNG path.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the comparison plotting workflow."""
    args = parse_args(argv)

    print("Loading CV tables...")
    df_tp, df_fn = load_cv_tables(args.models_dir)

    print("\nBuilding figure...")
    make_figure(df_tp, df_fn, args.out)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)