"""
Predict global BSR burial depth from the trained XGBoost residual model.

This script trains an XGBoost regressor on positive BSR observations to
predict the BSR residual (observed BSR depth minus theoretical BGHS
thickness). It then:

1. Aligns the 11 feature rasters to the reference grid of the
   constrained BSR-occurrence probability map.
2. Predicts the residual at every ocean cell.
3. Converts the residual to absolute BSR depth by adding the BGHS
   thickness back.
4. Keeps only cells with occurrence probability above a user-defined
   threshold.
5. Writes the result as ``BSR_depth_prediction.tif``.

Reproducibility note
--------------------
XGBoost can show small run-to-run variability due to floating-point
non-determinism, thread scheduling, and differences in library versions
or hardware. The random seed is fixed (``RANDOM_STATE = 42``), so results
should be close across runs, but exact depth values may differ slightly
from those reported in the paper. The overall spatial pattern and
summary statistics are stable.
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

RANDOM_STATE = 42
RESIDUAL_CLIP = 1000.0      # metres; discard larger absolute residuals
PROBA_THRESHOLD = 0.5
OUT_NODATA = -9999.0
PERCENTILES = (10, 25, 50, 75, 90)


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def load_training_data(csv_path, residual_clip=RESIDUAL_CLIP):
    """Load positive samples and compute the BSR residual target."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df_pos = df[df["label"] == 1].copy()
    df_pos["BSR_residual"] = df_pos["BSR"] - df_pos["BGHS_thickness"]

    valid = df_pos.dropna(subset=["BSR_residual"])
    valid = valid[np.abs(valid["BSR_residual"]) < residual_clip].copy()

    print(f"Training samples: {len(valid)}")
    return valid


def train_residual_model(valid):
    """Fit the XGBoost regressor on positive BSR residuals."""
    print("Training XGBoost regression model...")
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(valid[FEATURE_COLS], valid["BSR_residual"])
    print("Training finished.")
    return model


# ---------------------------------------------------------------------
# Raster alignment
# ---------------------------------------------------------------------
def read_reference_grid(tif_path):
    """Read the reference grid and the probability field."""
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"Reference raster not found: {tif_path}")

    with rasterio.open(tif_path) as src:
        proba = src.read(1).astype(np.float32)
        return {
            "profile": src.profile.copy(),
            "transform": src.transform,
            "crs": src.crs,
            "shape": src.shape,
            "nodata": src.nodata,
            "proba": proba,
        }


def align_raster(tif_path, ref, is_categorical=False):
    """Align a raster to the reference grid; nodata -> NaN."""
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
    """Align all feature rasters to the reference grid."""
    print("\nAligning feature rasters...")
    feat_data = {}
    for fname, col in feature_map.items():
        tif_path = os.path.join(feat_dir, fname)
        if not os.path.exists(tif_path):
            print(f"  WARNING: missing raster: {fname}")
            feat_data[col] = np.full(ref["shape"], np.nan, dtype=np.float32)
            continue

        is_categorical = col == "tectonic_environment"
        feat_data[col] = align_raster(tif_path, ref, is_categorical)
        print(f"  OK  {col}")

    return feat_data


# ---------------------------------------------------------------------
# Prediction and output
# ---------------------------------------------------------------------
def predict_residuals(model, feat_data, feature_cols):
    """Predict BSR residual at every valid ocean cell.

    Returns
    -------
    tuple
        ``(rows, cols, residual_pred, bsr_depth)``.
    """
    bghs = feat_data["BGHS_thickness"]
    ocean_mask = ~np.isnan(bghs)
    rows, cols = np.where(ocean_mask)

    X_pred = np.column_stack(
        [feat_data[col][rows, cols] for col in feature_cols]
    )
    print(f"\nPrediction matrix: {X_pred.shape}")

    print("\nPredicting BSR residual...")
    residual = model.predict(X_pred)
    print(
        f"Residual range: {residual.min():.1f} - {residual.max():.1f} m"
    )

    bghs_flat = bghs[rows, cols]
    bsr_depth = np.maximum(residual + bghs_flat, 0.0)

    print(
        f"BSR depth range: {bsr_depth.min():.1f} - "
        f"{bsr_depth.max():.1f} m"
    )
    print(f"BSR depth mean: {bsr_depth.mean():.1f} m")

    return rows, cols, residual, bsr_depth


def apply_probability_mask(rows, cols, bsr_depth, proba, threshold):
    """Keep BSR depth only where occurrence probability >= threshold."""
    high_prob_mask = proba[rows, cols] >= threshold
    print(
        f"\nHigh-probability cells (>= {threshold}): "
        f"{int(high_prob_mask.sum())}"
    )
    return high_prob_mask


def write_depth_tif(out_path, ref, rows, cols, bsr_depth, keep_mask):
    """Write the masked BSR depth array to a GeoTIFF."""
    out_data = np.full(ref["shape"], OUT_NODATA, dtype=np.float32)
    out_data[rows[keep_mask], cols[keep_mask]] = bsr_depth[keep_mask]

    profile = ref["profile"].copy()
    profile.update(dtype="float32", count=1, nodata=OUT_NODATA)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_data, 1)

    print(f"\nSaved: {out_path}")
    return out_data


def report_depth_statistics(out_data, nodata=OUT_NODATA):
    """Print summary statistics of the final BSR depth map."""
    valid_depth = out_data[out_data != nodata]
    if valid_depth.size == 0:
        print("\nNo valid depth cells; statistics skipped.")
        return

    print("\n========== BSR depth statistics ==========")
    print(f"Valid cells: {len(valid_depth)}")
    print(
        f"Depth range: {valid_depth.min():.1f} - "
        f"{valid_depth.max():.1f} m"
    )
    print(f"Mean depth:  {valid_depth.mean():.1f} m")
    print(f"Median depth: {np.median(valid_depth):.1f} m")
    for q in PERCENTILES:
        print(
            f"  {q}th percentile: {np.percentile(valid_depth, q):.1f} m"
        )


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------
def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict global BSR depth from the XGBoost residual model."
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
        "--prob-tif",
        default="BSR_occurrence_probability_constrained.tif",
        help="Constrained occurrence probability GeoTIFF (reference grid).",
    )
    parser.add_argument(
        "--out-tif",
        default="BSR_depth_prediction.tif",
        help="Output BSR depth GeoTIFF.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=PROBA_THRESHOLD,
        help="Occurrence-probability threshold for keeping a cell.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the full BSR depth prediction workflow."""
    args = parse_args(argv)

    # -----------------------------------------------------------------
    # 1. Train residual model
    # -----------------------------------------------------------------
    valid = load_training_data(args.csv)
    model = train_residual_model(valid)

    # -----------------------------------------------------------------
    # 2. Reference grid
    # -----------------------------------------------------------------
    ref = read_reference_grid(args.prob_tif)
    proba = ref["proba"]
    ocean_mask = (proba != ref["nodata"]) & ~np.isnan(proba)
    print(
        f"\nReference grid: {ref['shape']}, "
        f"ocean cells: {int(ocean_mask.sum())}"
    )

    # -----------------------------------------------------------------
    # 3. Align features
    # -----------------------------------------------------------------
    feat_data = load_all_features(args.feat_dir, ref, FEATURE_MAP)

    # -----------------------------------------------------------------
    # 4. Predict residuals and BSR depth
    # -----------------------------------------------------------------
    rows, cols, residual, bsr_depth = predict_residuals(
        model, feat_data, FEATURE_COLS
    )

    # -----------------------------------------------------------------
    # 5. Probability constraint
    # -----------------------------------------------------------------
    keep_mask = apply_probability_mask(
        rows, cols, bsr_depth, proba, args.threshold
    )

    # -----------------------------------------------------------------
    # 6. Write output
    # -----------------------------------------------------------------
    out_data = write_depth_tif(
        args.out_tif, ref, rows, cols, bsr_depth, keep_mask
    )
    print(f"Final BSR depth cells: {int(keep_mask.sum())}")

    # -----------------------------------------------------------------
    # 7. Statistics
    # -----------------------------------------------------------------
    report_depth_statistics(out_data)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)