"""
Train and evaluate an XGBoost regressor for BSR residual prediction.

This script uses only positive BSR observations. The target is the BSR
residual, defined as::

    residual = BSR_depth - BGHS_thickness

Spatial cross-validation (5-degree GroupKFold blocks) is used to avoid
spatial autocorrelation inflating model performance. The script prints
per-fold metrics, cross-fold summary statistics, feature importance,
and saves a three-panel performance figure.

Inputs
------
``BSR_ML_dataset.csv``

Output
------
``BSR_depth_model_performance.png``

Reproducibility note
--------------------
XGBoost can show small run-to-run variability due to floating-point
non-determinism, thread scheduling, and differences in library versions
or hardware. The random seed is fixed (``RANDOM_STATE = 42``), so results
should be close across runs, but exact metric values may differ slightly
from those reported in the paper. The overall trends and conclusions are
stable, and the methodology and evaluation protocol are fully
reproducible.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
FEATURE_COLS = [
    "crustal_age",
    "dist_to_ridge",
    "dist_to_subduction",
    "sediment_thickness",
    "BGHS_thickness",
    "bottom_salinity",
    "bottom_temp",
    "geothermal_gradient",
    "slope",
    "water_depth",
    "tectonic_environment",
]

BLOCK_SIZE = 5.0
N_SPLITS = 5
RANDOM_STATE = 42
RESIDUAL_CLIP = 1000.0     # discard |residual| above this (metres)


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------
def load_positive_samples(csv_path, residual_clip=RESIDUAL_CLIP):
    """Load positive BSR samples and compute the BSR residual.

    Parameters
    ----------
    csv_path : str
        Input CSV produced by ``build_bsr_dataset.py``.
    residual_clip : float
        Absolute residual threshold above which samples are discarded.

    Returns
    -------
    pd.DataFrame
        Filtered positive samples with a ``BSR_residual`` column.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Raw rows: {len(df)}")

    df_pos = df[df["label"] == 1].copy()
    print(f"Positive rows: {len(df_pos)}")

    df_pos["BSR_residual"] = df_pos["BSR"] - df_pos["BGHS_thickness"]

    valid = df_pos.dropna(subset=["BSR_residual"])
    valid = valid[np.abs(valid["BSR_residual"]) < residual_clip].copy()

    print(f"Valid training samples: {len(valid)}")
    print(
        f"BSR depth range: {valid['BSR'].min():.1f} - "
        f"{valid['BSR'].max():.1f} m"
    )
    print(
        f"Residual range: {valid['BSR_residual'].min():.1f} - "
        f"{valid['BSR_residual'].max():.1f} m"
    )
    print(f"Mean residual: {valid['BSR_residual'].mean():.1f} m")

    return valid


def add_spatial_blocks(df, block_size=BLOCK_SIZE):
    """Assign a spatial block ID to each row and return the group array."""
    df["spatial_block"] = (
        np.floor(df["X"] / block_size).astype(int).astype(str)
        + "_"
        + np.floor(df["Y"] / block_size).astype(int).astype(str)
    )
    n_blocks = df["spatial_block"].nunique()
    print(
        f"\nSpatial blocks: {block_size} x {block_size} degrees, "
        f"{n_blocks} unique blocks"
    )
    return df["spatial_block"].values


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
def build_xgb_regressor():
    """Return the XGBoost regressor used for BSR residual prediction."""
    return xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbosity=0,
    )


# ---------------------------------------------------------------------
# Spatial cross-validation
# ---------------------------------------------------------------------
def run_regression_cv(X, y, groups, n_splits=N_SPLITS):
    """Run GroupKFold spatial CV for the XGBoost regressor.

    Returns
    -------
    tuple
        ``(fold_results, importances, y_true, y_pred)``.
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []
    all_importances = []
    all_y_true, all_y_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model = build_xgb_regressor()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        mae = float(mean_absolute_error(y_te, y_pred))
        r2 = float(r2_score(y_te, y_pred))

        fold_results.append({
            "fold": fold + 1,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        })
        all_importances.append(model.feature_importances_)
        all_y_true.extend(y_te.values)
        all_y_pred.extend(y_pred)

        print(
            f"  Fold {fold + 1}: train={len(train_idx)}, "
            f"test={len(test_idx)} | "
            f"RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}"
        )

    return (
        fold_results,
        np.array(all_importances),
        np.array(all_y_true),
        np.array(all_y_pred),
    )


def summarize_cv(fold_results):
    """Print fold-level table and mean +/- std summary."""
    df_f = pd.DataFrame(fold_results)
    print("\n" + "=" * 60)
    print("  XGBoost regression summary")
    print("=" * 60)
    print(df_f.to_string(index=False))

    print(
        f"\n  Mean RMSE: {df_f['rmse'].mean():.2f} "
        f"+/- {df_f['rmse'].std():.2f} m"
    )
    print(
        f"  Mean MAE:  {df_f['mae'].mean():.2f} "
        f"+/- {df_f['mae'].std():.2f} m"
    )
    print(
        f"  Mean R2:   {df_f['r2'].mean():.4f} "
        f"+/- {df_f['r2'].std():.4f}"
    )
    return df_f


def build_importance_table(importances, feature_cols):
    """Return a sorted DataFrame of mean and std feature importance."""
    imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance_mean": importances.mean(axis=0),
        "Importance_std": importances.std(axis=0),
    })
    imp_df["rank"] = (
        imp_df["Importance_mean"].rank(ascending=False).astype(int)
    )
    return imp_df.sort_values("Importance_mean", ascending=False)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_performance(y_true, y_pred, imp_df, out_path):
    """Draw and save the three-panel performance figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Predicted vs observed
    ax = axes[0]
    ax.scatter(y_true, y_pred, s=10, alpha=0.5, c="steelblue")
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", lw=1.5, label="1:1 line")
    ax.set_xlabel("Observed BSR residual (m)")
    ax.set_ylabel("Predicted BSR residual (m)")
    ax.set_title(
        f"XGBoost: RMSE={np.sqrt(mean_squared_error(y_true, y_pred)):.1f} m"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Residual density (placeholder for consistent 3-panel layout)
    ax = axes[1]
    residuals = y_pred - y_true
    ax.hist(residuals, bins=50, color="coral", alpha=0.75)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Prediction residual (m)")
    ax.set_ylabel("Count")
    ax.set_title("Prediction residual distribution")
    ax.grid(alpha=0.3)

    # 3. Feature importance
    ax = axes[2]
    imp_sorted = imp_df.sort_values("Importance_mean")
    ax.barh(
        imp_sorted["Feature"],
        imp_sorted["Importance_mean"],
        color="steelblue",
        alpha=0.7,
    )
    ax.set_xlabel("Feature importance")
    ax.set_title("XGBoost feature importance")
    ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------
def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate XGBoost for BSR residual prediction."
    )
    parser.add_argument(
        "--csv",
        default="BSR_ML_dataset.csv",
        help="Input CSV produced by build_bsr_dataset.py.",
    )
    parser.add_argument(
        "--out",
        default="BSR_depth_model_performance.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--block-size",
        type=float,
        default=BLOCK_SIZE,
        help="Spatial block size in degrees for GroupKFold.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=N_SPLITS,
        help="Number of spatial CV folds.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the full regression workflow."""
    args = parse_args(argv)

    # -----------------------------------------------------------------
    # 1. Load and prepare data
    # -----------------------------------------------------------------
    valid = load_positive_samples(args.csv)
    groups = add_spatial_blocks(valid, args.block_size)

    X = valid[FEATURE_COLS]
    y = valid["BSR_residual"]

    # -----------------------------------------------------------------
    # 2. Spatial CV
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  XGBoost regression spatial CV")
    print("=" * 60)

    fold_results, importances, y_true, y_pred = run_regression_cv(
        X, y, groups, args.n_splits
    )

    # -----------------------------------------------------------------
    # 3. Summaries
    # -----------------------------------------------------------------
    summarize_cv(fold_results)

    print("\n" + "=" * 60)
    print("  Feature importance (mean across folds)")
    print("=" * 60)

    imp_df = build_importance_table(importances, FEATURE_COLS)
    print(imp_df.to_string(index=False))

    # -----------------------------------------------------------------
    # 4. Plot
    # -----------------------------------------------------------------
    plot_performance(y_true, y_pred, imp_df, args.out)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)