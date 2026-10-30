"""
Plot the SHAP beeswarm summary for the BSR-occurrence classifier.

This script reads the exported SHAP values and feature matrix, computes
the mean absolute SHAP value per feature, and draws a custom beeswarm
plot for the top features. Each point is coloured by the normalized
feature value, and a horizontal colour bar with a rounded frame is
embedded in the axes.

Inputs
------
- ``Classifier_models/shap_values.npy``
- ``Classifier_models/shap_features.csv``

Output
------
``fig4c_shap_beeswarm.png``
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LinearSegmentedColormap


# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

FS_TICK = 23
FS_LABEL = 25
FS_CBAR = 23
FS_CBARTK = 21

FIG_WIDTH_IN = 23.0 / 2.54
FIG_HEIGHT_IN = 12.0 / 2.54

# Blue -> red without the darkest ends
COLOR_STOPS = [
    "#3B4CC0",
    "#7B9FF9",
    "#B8D0F0",
    "#F2F2F2",
    "#F5C0A8",
    "#EE8468",
    "#B40426",
]
CMAP_SHAP = LinearSegmentedColormap.from_list(
    "blue_red", COLOR_STOPS, N=256
)

FEATURE_NAME_MAP = {
    "water_depth": "Water depth",
    "crustal_age": "Crustal age",
    "sediment_thickness": "Sed. thickness",
    "dist_to_subduction": "Dist. subduction",
    "tectonic_environment": "Tectonic env.",
    "bottom_salinity": "Bot. salinity",
    "bottom_temp": "Bot. temperature",
    "dist_to_ridge": "Dist. ridge",
    "BGHS_thickness": "BGHS thickness",
    "geothermal_gradient": "Geoth. gradient",
    "slope": "Slope",
}


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------
def load_shap_inputs(models_dir):
    """Load SHAP values and the corresponding feature matrix.

    Returns
    -------
    tuple
        ``(shap_values, features)``.
    """
    shap_path = os.path.join(models_dir, "shap_values.npy")
    feat_path = os.path.join(models_dir, "shap_features.csv")

    for path in (shap_path, feat_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input not found: {path}")

    shap_values = np.load(shap_path)
    features = pd.read_csv(feat_path)
    print(f"SHAP matrix: {shap_values.shape}")
    return shap_values, features


# ---------------------------------------------------------------------
# Beeswarm plot
# ---------------------------------------------------------------------
def plot_shap_beeswarm(
    ax,
    shap_values,
    features,
    feature_name_map,
    cmap,
    max_display=11,
    fontsize_tick=FS_TICK,
    fontsize_label=FS_LABEL,
    fontsize_cbar=FS_CBAR,
    fontsize_cbartk=FS_CBARTK,
):
    """Draw a SHAP beeswarm plot on ``ax``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    shap_values : np.ndarray
        2-D array of SHAP values, shape ``(n_samples, n_features)``.
    features : pd.DataFrame
        Feature matrix aligned with ``shap_values``.
    feature_name_map : dict
        Mapping from raw column name to display label.
    cmap : matplotlib.colors.Colormap
        Colormap for feature values.
    max_display : int
        Maximum number of features to show.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:max_display]
    rng = np.random.RandomState(42)

    for row_idx, feat_idx in enumerate(order):
        y_pos = len(order) - row_idx - 1
        sv = shap_values[:, feat_idx]
        fv = features.iloc[:, feat_idx].values.astype(float)

        fv_min, fv_max = np.nanpercentile(fv, [5, 95])
        if fv_max > fv_min:
            fv_norm = np.clip((fv - fv_min) / (fv_max - fv_min), 0, 1)
        else:
            fv_norm = np.full_like(fv, 0.5)

        mask = ~np.isnan(sv) & ~np.isnan(fv_norm)
        sv_c, fv_c = sv[mask], fv_norm[mask]
        if len(sv_c) == 0:
            continue

        jitter = rng.uniform(-0.4, 0.4, size=len(sv_c))
        alpha = min(0.7, max(0.2, 500 / len(sv_c)))
        ax.scatter(
            sv_c,
            y_pos + jitter,
            c=fv_c,
            cmap=cmap,
            s=8,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )

    labels = [
        feature_name_map.get(features.columns[i], features.columns[i])
        for i in order
    ][::-1]
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=fontsize_tick)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel(
        "SHAP value (impact on BSR probability)", fontsize=fontsize_label
    )
    ax.tick_params(axis="x", labelsize=fontsize_tick)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="x", alpha=0.2)

    # -----------------------------------------------------------------
    # Horizontal colour bar with a rounded frame
    # -----------------------------------------------------------------
    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])

    cax_pos = [0.08, 0.24, 0.25, 0.06]  # [x, y, width, height]
    cax = ax.inset_axes(cax_pos)
    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Feature value", fontsize=fontsize_cbar, labelpad=6)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.ax.tick_params(labelsize=fontsize_cbartk)

    x0, y0, w, h = cax_pos
    pad_x = 0.05
    pad_y = 0.115
    rect = mpatches.FancyBboxPatch(
        (x0 - pad_x, y0 - pad_y - 0.09),
        w + 2 * pad_x,
        h + 2 * pad_y,
        boxstyle="round,pad=0.005,rounding_size=0.02",
        linewidth=1.6,
        edgecolor="black",
        facecolor="white",
        alpha=0.3,
        transform=ax.transAxes,
        zorder=0,
    )
    ax.add_patch(rect)


def make_beeswarm_figure(shap_values, features, out_path):
    """Create and save the beeswarm figure."""
    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=300
    )

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:11]

    plot_shap_beeswarm(
        ax,
        shap_values[:, top_idx],
        features.iloc[:, top_idx],
        FEATURE_NAME_MAP,
        CMAP_SHAP,
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
        description="Plot the SHAP beeswarm summary for BSR probability."
    )
    parser.add_argument(
        "--models-dir",
        default="Classifier_models",
        help="Directory containing shap_values.npy and shap_features.csv.",
    )
    parser.add_argument(
        "--out",
        default="fig4c_shap_beeswarm.png",
        help="Output PNG path.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the plotting workflow."""
    args = parse_args(argv)

    print("Loading SHAP inputs...")
    shap_values, features = load_shap_inputs(args.models_dir)

    print("\nBuilding beeswarm figure...")
    make_beeswarm_figure(shap_values, features, args.out)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)