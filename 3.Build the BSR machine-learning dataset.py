"""
Build the BSR machine-learning dataset.

This script merges positive BSR observations and negative (pseudo-absence)
points, then samples 11 environmental rasters at each point. The result is
a single CSV that can be fed into the classification and regression models.

Expected inputs
---------------
- ``Global BSR actual observation dataset.txt``
  Tab-separated file with columns ``X``, ``Y``, ``BSR``.
- ``BSR_negative_points.txt``
  Tab-separated file with columns ``X``, ``Y``.
- ``Global feature dataset/``
  Directory containing the 11 GeoTIFFs listed in ``FEATURE_MAP``.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import rasterio


# ---------------------------------------------------------------------
# Raster file name -> output column name
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
    "Ocean_stratigraphic_age.tif": "ocean_stratigraphic_age",
    "tectonic_environment_0.25deg.tif": "tectonic_environment",
}


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------
def read_points(pos_file, neg_file):
    """Read positive and negative point tables and merge them.

    Parameters
    ----------
    pos_file : str
        Tab-separated file with ``X``, ``Y``, ``BSR`` columns.
    neg_file : str
        Tab-separated file with ``X``, ``Y`` columns.

    Returns
    -------
    pd.DataFrame
        Merged table with columns ``X``, ``Y``, ``BSR``, ``label``.
    """
    for path in (pos_file, neg_file):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Point file not found: {path}")

    pos = pd.read_csv(pos_file, sep="\t")
    neg = pd.read_csv(neg_file, sep="\t")
    print(f"Positives: {pos.shape[0]} rows; negatives: {neg.shape[0]} rows")

    pos["label"] = 1
    neg["label"] = 0
    neg["BSR"] = -1

    df = pd.concat([pos, neg], ignore_index=True)
    df = df[["X", "Y", "BSR", "label"]]
    print(f"Merged: {df.shape[0]} rows")
    return df


def sample_raster(tif_path, coords):
    """Sample the first band of a raster at the given coordinates.

    Parameters
    ----------
    tif_path : str
        Path to a GeoTIFF.
    coords : list of tuple
        ``(x, y)`` pairs in the raster's CRS.

    Returns
    -------
    np.ndarray
        1-D array of sampled values, shape ``(len(coords),)``.
    """
    with rasterio.open(tif_path) as src:
        nodata = src.nodata
        if src.count > 1:
            name = os.path.basename(tif_path)
            print(f"  [{name}] has {src.count} bands; using band 1")
        vals = np.array([v[0] for v in src.sample(coords)], dtype=float)

    if nodata is not None:
        vals[vals == nodata] = np.nan
    vals[~np.isfinite(vals)] = np.nan
    vals[np.abs(vals) > 1e30] = np.nan
    return vals


def attach_features(df, feat_dir, feature_map):
    """Sample all features and attach them as new columns.

    Parameters
    ----------
    df : pd.DataFrame
        Table with ``X`` and ``Y`` columns.
    feat_dir : str
        Directory containing the GeoTIFFs.
    feature_map : dict
        Mapping from raster file name to output column name.

    Returns
    -------
    tuple
        ``(df_with_features, missing_info)``.
    """
    coords = list(zip(df["X"].values, df["Y"].values))
    missing_info = {}

    print("\nSampling rasters...")
    for fname, col_name in feature_map.items():
        tif_path = os.path.join(feat_dir, fname)
        if not os.path.exists(tif_path):
            print(f"  WARNING: missing raster: {tif_path}")
            df[col_name] = np.nan
            missing_info[col_name] = "file not found"
            continue

        vals = sample_raster(tif_path, coords)
        df[col_name] = vals

        n_nan = int(np.isnan(vals).sum())
        missing_info[col_name] = f"{n_nan} / {len(vals)} missing"
        print(f"  OK  {col_name:28s} <- {fname}  | missing {n_nan}")

    return df, missing_info


def report_missing(df, missing_info):
    """Print NaN counts for each feature, split by label."""
    print("\n========== Missing-value summary ==========")
    print(f"Positives (label=1): {(df['label'] == 1).sum()} rows")
    print(f"Negatives (label=0): {(df['label'] == 0).sum()} rows")

    for col, info in missing_info.items():
        print(f"  {col:28s}: {info}")

    print("\nNaN counts by label (feature columns):")
    feature_cols = [c for c in df.columns if c not in ("X", "Y", "BSR", "label")]
    nan_by_label = df.groupby("label")[feature_cols].apply(
        lambda g: g.isna().sum()
    )
    print(nan_by_label.T)


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------
def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge BSR points and sample environmental rasters."
    )
    parser.add_argument(
        "--feat-dir",
        default="Global feature dataset",
        help="Directory containing the feature GeoTIFFs.",
    )
    parser.add_argument(
        "--pos-file",
        default="Global BSR actual observation dataset.txt",
        help="Tab-separated file with positive BSR points.",
    )
    parser.add_argument(
        "--neg-file",
        default="BSR_negative_points.txt",
        help="Tab-separated file with negative points.",
    )
    parser.add_argument(
        "--out-csv",
        default="BSR_ML_dataset.csv",
        help="Output CSV path.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the dataset-building workflow."""
    args = parse_args(argv)

    # -----------------------------------------------------------------
    # 1. Merge point tables
    # -----------------------------------------------------------------
    df = read_points(args.pos_file, args.neg_file)

    # -----------------------------------------------------------------
    # 2. Sample features
    # -----------------------------------------------------------------
    df, missing_info = attach_features(df, args.feat_dir, FEATURE_MAP)

    # -----------------------------------------------------------------
    # 3. Write output
    # -----------------------------------------------------------------
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved: {args.out_csv}")
    print(f"Rows: {len(df)}, columns: {df.shape[1]}")
    print(f"Columns: {list(df.columns)}")

    # -----------------------------------------------------------------
    # 4. Report missing values
    # -----------------------------------------------------------------
    report_missing(df, missing_info)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)