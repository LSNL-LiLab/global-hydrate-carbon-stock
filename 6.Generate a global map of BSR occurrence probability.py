"""
Generate a global map of BSR occurrence probability.

This script trains a final XGBoost classifier on the full BSR dataset,
aligns the 11 feature rasters to a common 0.25-degree grid, predicts
the probability of BSR occurrence at every ocean cell, and writes the
result as a GeoTIFF.

Reproducibility note
--------------------
XGBoost can show small run-to-run variability due to floating-point
non-determinism, thread scheduling, and differences in library versions
or hardware. The random seed is fixed (``RANDOM_STATE = 42``), so
results should be close across runs, but exact probability values may
differ slightly from those reported in the paper. The overall spatial
pattern and threshold-based statistics are stable.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import rasterio
import xgboost as xgb
from rasterio.warp import reproject, Resampling

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
FEATURE_MAP = {
    "crustal_age_0.25deg.tif": "crustal_age",
    "dist_to_ridge_0.25deg.tif": "dist_to_ridge",
    "dist_to_subduction_0.25deg.tif": "dist_to_subduction",
    "globa_sediment_thickness.tif": "sediment_thickness",
    "global_BGHS_thickness_0.25deg.tif": "BGHS_thickness",
    "global_bottom_salinity.tif": "bottom_salinity",
    "global_bottom_temp.tif": "bottom_temp",
    "global_geothermal_gradient.tif": "geothermal_gradient",
    "global_slope.tif": "slope",
    "global_water_depth.tif": "water_depth",
    "tectonic_environment_0.25deg.tif": "tectonic_environment",
}
FEATURE_COLS = list(FEATURE_MAP.values())

REFERENCE_TIF = "global_BGHS_thickness_0.25deg.tif"

SCALE_POS_WEIGHT = 5.0
RANDOM_STATE = 42
OUT_NODATA = -9999.0

THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
def build_final_xgb(scale_pos_weight=SCALE_POS_WEIGHT):
    """Return the final XGBoost classifier used for global prediction."""
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        use_label_encoder=False,
    )


def train_model(csv_path, feature_cols):
    """Train the final XGBoost model on the full dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    X_train = df[feature_cols]
    y_train = df["label"]

    print(f"Training XGBoost (scale_pos_weight={SCALE_POS_WEIGHT})...")
    model = build_final_xgb()
    model.fit(X_train, y_train)
    print("Training finished.")
    return model


# ---------------------------------------------------------------------
# Raster alignment
# ---------------------------------------------------------------------
def read_reference_grid(tif_path):
    """Read reference grid metadata (shape, transform, CRS, profile)."""
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"Reference raster not found: {tif_path}")

    with rasterio.open(tif_path) as src:
        return {
            "profile": src.profile.copy(),
            "transform": src.transform,
            "crs": src.crs,
            "shape": src.shape,
            "nodata": src.nodata,
        }


def align_raster(tif_path, ref, is_categorical=False):
    """Align one raster to the reference grid.

    Parameters
    ----------
    tif_path : str
        Path to the input raster.
    ref : dict
        Reference grid metadata from ``read_reference_grid``.
    is_categorical : bool
        If True, use nearest-neighbour resampling.

    Returns
    -------
    np.ndarray
        2-D array aligned to the reference grid; NaN for invalid cells.
    """
    with rasterio.open(tif_path) as src:
        src_nodata = src.nodata
        resampling = (
            Resampling.nearest if is_categorical else Resampling.bilinear
        )

        same_grid = (
            src.shape == ref["shape"]
            and src.transform == ref["transform"]
            and src.crs == ref["crs"]
        )
        if same_grid:
            data = src.read(1).astype(np.float32)
        else:
            data = np.empty(ref["shape"], dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref["transform"],
                dst_crs=ref["crs"],
                resampling=resampling,
            )

    if src_nodata is not None:
        data[data == src_nodata] = np.nan
    data[~np.isfinite(data)] = np.nan
    data[np.abs(data) > 1e30] = np.nan
    return data


def load_all_features(feat_dir, ref, feature_map):
    """Load and align all feature rasters to the reference grid."""
    print("\nAligning feature rasters...")
    feat_data = {}
    for fname, col in feature_map.items():
        tif_path = os.path.join(feat_dir, fname)
        if not os.path.exists(tif_path):
            print(f"  WARNING: missing raster: {fname}")
            feat_data[col] = np.full(ref["shape"], np.nan, dtype=np.float32)
            continue

        is_categorical = col == "tectonic_environment"
        data = align_raster(tif_path, ref, is_categorical)
        feat_data[col] = data

        n_valid = int((~np.isnan(data)).sum())
        print(f"  OK  {col:25s} valid cells: {n_valid}")

    return feat_data


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------
def predict_probability(model, feat_data, feature_cols):
    """Predict BSR occurrence probability at every valid ocean cell.

    Returns
    -------
    tuple
        ``(rows, cols, proba)``.
    """
    bghs = feat_data["BGHS_thickness"]
    ocean_mask = ~np.isnan(bghs)
    n_ocean = int(ocean_mask.sum())
    print(f"\nOcean cells: {n_ocean}")

    rows, cols = np.where(ocean_mask)
    X_pred = np.column_stack(
        [feat_data[col][rows, cols] for col in feature_cols]
    )
    print(f"Prediction matrix: {X_pred.shape}")

    print("\nPredicting...")
    proba = model.predict_proba(X_pred)[:, 1]
    print(
        f"Prediction finished. Probability range: "
        f"{proba.min():.4f} - {proba.max():.4f}"
    )
    print(f"Mean probability: {proba.mean():.4f}")

    return rows, cols, proba


def write_probability_tif(out_path, ref, rows, cols, proba, nodata=OUT_NODATA):
    """Write the probability array to a GeoTIFF on the reference grid."""
    out_data = np.full(ref["shape"], nodata, dtype=np.float32)
    out_data[rows, cols] = proba

    profile = ref["profile"].copy()
    profile.update(dtype="float32", count=1, nodata=nodata)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_data, 1)

    print(f"\nSaved: {out_path}")


def print_threshold_stats(proba, thresholds=THRESHOLDS):
    """Print the number of cells above each probability threshold."""
    print("\n========== Prediction statistics ==========")
    for t in thresholds:
        n = int((proba >= t).sum())
        pct = n / len(proba) * 100
        print(f"  P >= {t:.1f}: {n:8d} cells ({pct:.2f}%)")


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------
def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a global BSR occurrence probability map."
    )
    parser.add_argument(
        "--feat-dir",
        default="Global feature dataset",
        help="Directory containing the feature GeoTIFFs.",
    )
    parser.add_argument(
        "--csv",
        default="BSR_ML_dataset.csv",
        help="Training CSV produced by build_bsr_dataset.py.",
    )
    parser.add_argument(
        "--out-tif",
        default="BSR_occurrence_probability.tif",
        help="Output probability GeoTIFF.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the full training-and-prediction workflow."""
    args = parse_args(argv)

    # -----------------------------------------------------------------
    # 1. Train final model
    # -----------------------------------------------------------------
    model = train_model(args.csv, FEATURE_COLS)

    # -----------------------------------------------------------------
    # 2. Reference grid
    # -----------------------------------------------------------------
    ref_path = os.path.join(args.feat_dir, REFERENCE_TIF)
    ref = read_reference_grid(ref_path)
    print(f"\nReference grid: {ref['shape'][0]} rows x {ref['shape'][1]} cols")
    print(f"CRS: {ref['crs']}")

    # -----------------------------------------------------------------
    # 3. Align features
    # -----------------------------------------------------------------
    feat_data = load_all_features(args.feat_dir, ref, FEATURE_MAP)

    # -----------------------------------------------------------------
    # 4. Predict
    # -----------------------------------------------------------------
    rows, cols, proba = predict_probability(model, feat_data, FEATURE_COLS)

    # -----------------------------------------------------------------
    # 5. Write output
    # -----------------------------------------------------------------
    write_probability_tif(args.out_tif, ref, rows, cols, proba)

    # -----------------------------------------------------------------
    # 6. Threshold statistics
    # -----------------------------------------------------------------
    print_threshold_stats(proba)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)