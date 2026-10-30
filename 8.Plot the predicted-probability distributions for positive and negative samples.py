"""
Plot the predicted-probability distributions for positive and negative samples.

This script reads the constrained BSR occurrence probability map and the
BSR machine-learning dataset, extracts the predicted probability at each
sample location, and draws kernel-density curves for positive (label=1)
and negative (label=0) samples.

Output
------
``fig4b_positive_negative_probability.png``
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

FS_LABEL = 23
FS_TICK = 22
FS_LEGEND = 21

FIG_WIDTH_IN = 18.9 / 2.54
FIG_HEIGHT_IN = 12.7 / 2.54


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------
def read_probability_raster(path):
    """Read the probability GeoTIFF as a 2-D xarray DataArray.

    Returns
    -------
    xr.DataArray
        Probability field with dims ``("y", "x")``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Probability raster not found: {path}")

    prob = xr.open_dataarray(path, engine="rasterio")
    if prob.ndim == 3:
        prob = prob.isel(band=0)
    if "lat" in prob.dims:
        prob = prob.rename({"lat": "y", "lon": "x"})
    return prob


def read_ml_dataset(path):
    """Read the BSR machine-learning dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"ML dataset not found: {path}")
    df = pd.read_csv(path)
    print(f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def extract_probabilities(df, prob_da, nodata):
    """Sample the probability raster at each row's (X, Y) location.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``X`` and ``Y``.
    prob_da : xr.DataArray
        Probability field with dims ``("y", "x")``.
    nodata : float
        No-data value in the raster.

    Returns
    -------
    np.ndarray
        1-D array of sampled probabilities; NaN where sampling fails.
    """
    probs = []
    for _, row in df.iterrows():
        try:
            p = float(
                prob_da.sel(
                    x=row["X"], y=row["Y"], method="nearest"
                ).values
            )
            if np.isclose(p, nodata):
                p = np.nan
        except Exception:  # noqa: BLE001
            p = np.nan
        probs.append(p)
    return np.array(probs)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_probability_distribution(pos_probs, neg_probs, out_path):
    """Draw KDE curves for positive and negative samples.

    Parameters
    ----------
    pos_probs : np.ndarray
        Predicted probabilities for positive samples.
    neg_probs : np.ndarray
        Predicted probabilities for negative samples.
    out_path : str
        Output image path.
    """
    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=300
    )

    x_range = np.linspace(0, 1, 500)

    kde_neg = gaussian_kde(neg_probs)
    y_neg = kde_neg(x_range)
    ax.fill_between(
        x_range, y_neg, alpha=0.35, color="#2166ac",
        label=f"Negative samples (n={len(neg_probs)})",
    )
    ax.plot(x_range, y_neg, color="#2166ac", linewidth=1.5)

    kde_pos = gaussian_kde(pos_probs)
    y_pos = kde_pos(x_range)
    ax.fill_between(
        x_range, y_pos, alpha=0.35, color="#b2182b",
        label=f"Positive samples (n={len(pos_probs)})",
    )
    ax.plot(x_range, y_pos, color="#b2182b", linewidth=1.5)

    ax.axvline(
        0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.8,
        label="Threshold = 0.5",
    )

    ax.set_xlabel("Predicted probability", fontsize=FS_LABEL)
    ax.set_ylabel("Probability density", fontsize=FS_LABEL)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.legend(loc="upper center", fontsize=FS_LEGEND)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------
def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot predicted-probability distributions for positive and "
            "negative BSR samples."
        )
    )
    parser.add_argument(
        "--prob-tif",
        default="BSR_occurrence_probability_constrained.tif",
        help="Constrained probability GeoTIFF.",
    )
    parser.add_argument(
        "--ml-dataset",
        default="BSR_ML_dataset.csv",
        help="BSR machine-learning dataset.",
    )
    parser.add_argument(
        "--out",
        default="fig4b_positive_negative_probability.png",
        help="Output PNG path.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the probability-distribution plotting workflow."""
    args = parse_args(argv)

    # -----------------------------------------------------------------
    # 1. Read inputs
    # -----------------------------------------------------------------
    print("Reading probability raster...")
    prob = read_probability_raster(args.prob_tif)
    nodata = prob.rio.nodata if prob.rio.nodata is not None else -9999.0

    print("Reading ML dataset...")
    df_ml = read_ml_dataset(args.ml_dataset)

    # -----------------------------------------------------------------
    # 2. Extract probabilities at sample locations
    # -----------------------------------------------------------------
    print("Extracting predicted probabilities...")
    df_ml["pred_prob"] = extract_probabilities(df_ml, prob, nodata)

    pos_probs = df_ml.loc[df_ml["label"] == 1, "pred_prob"].dropna().values
    neg_probs = df_ml.loc[df_ml["label"] == 0, "pred_prob"].dropna().values

    print(
        f"Positive samples: n={len(pos_probs)}, "
        f"mean={pos_probs.mean():.4f}"
    )
    print(
        f"Negative samples: n={len(neg_probs)}, "
        f"mean={neg_probs.mean():.4f}"
    )

    # -----------------------------------------------------------------
    # 3. Plot and save
    # -----------------------------------------------------------------
    plot_probability_distribution(pos_probs, neg_probs, args.out)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)