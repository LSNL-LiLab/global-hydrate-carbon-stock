"""
Apply a shallow-water physical constraint to the BSR occurrence probability map.

Any ocean cell with water depth below ``WATER_DEPTH_MIN`` is physically
unable to host a BSR, so its predicted probability is capped at
``PROBA_CAP``. The script reads the unconstrained probability raster and
the water-depth raster, aligns them, applies the constraint, and writes
the constrained probability map:

``BSR_occurrence_probability_constrained.tif``

It also reports threshold statistics before/after the constraint and
verifies that known BSR points were not accidentally suppressed.

Reproducibility note
--------------------
The constraint is deterministic given the input rasters. The only
source of run-to-run variation is the earlier XGBoost prediction
step; this script itself does not retrain or resample randomly.

Author: <your name>
License: MIT (or your preferred license)
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
WATER_DEPTH_MIN = 400.0     # metres; shallower than this, BSR is impossible
PROBA_CAP = 0.0             # probability assigned in shallow water
THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9)


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------
def read_raster(path):
    """Read the first band of a raster and return data and metadata."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raster not found: {path}")

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        meta = {
            "transform": src.transform,
            "crs": src.crs,
            "shape": src.shape,
            "profile": src.profile.copy(),
            "nodata": src.nodata,
        }
    return data, meta


def align_to_reference(src_path, ref_meta, resampling=Resampling.bilinear):
    """Reproject a raster to the reference grid and convert nodata to NaN."""
    with rasterio.open(src_path) as src:
        dst = np.empty(ref_meta["shape"], dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_meta["transform"],
            dst_crs=ref_meta["crs"],
            resampling=resampling,
        )
        src_nodata = src.nodata

    if src_nodata is not None:
        dst[dst == src_nodata] = np.nan
    dst[~np.isfinite(dst)] = np.nan
    return dst


def write_raster(path, data, ref_meta, nodata):
    """Write a 2-D array as a GeoTIFF using the reference profile."""
    profile = ref_meta["profile"].copy()
    profile.update(dtype="float32", count=1, nodata=nodata)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)


# ---------------------------------------------------------------------
# Constraint logic
# ---------------------------------------------------------------------
def apply_shallow_water_constraint(proba, wd_abs, ocean_mask, depth_min, cap):
    """Cap probability to ``cap`` wherever depth < ``depth_min``.

    Returns
    -------
    tuple
        ``(constrained, shallow_mask)``.
    """
    shallow_mask = (wd_abs < depth_min) & ocean_mask

    constrained = proba.copy()
    constrained[shallow_mask] = cap

    return constrained, shallow_mask


def report_thresholds(proba_before, proba_after, ocean_mask, thresholds):
    """Print the number of cells above each threshold, before/after."""
    print("\n" + "=" * 70)
    print("  Threshold statistics before/after constraint")
    print("=" * 70)
    print(f"{'Threshold':<12}{'Before':<15}{'After':<15}{'Change':<15}")
    print("-" * 55)
    for t in thresholds:
        n_before = int(((proba_before >= t) & ocean_mask).sum())
        n_after = int(((proba_after >= t) & ocean_mask).sum())
        print(f">={t:<10.1f}{n_before:<15d}{n_after:<15d}"
              f"{n_after - n_before:+d}")


def verify_constraint(proba_before, proba_after, shallow_mask, depth_min, cap):
    """Sanity-check that shallow cells are capped and deep cells untouched."""
    print("\n" + "=" * 70)
    print("  Constraint verification")
    print("=" * 70)

    shallow_after = proba_after[shallow_mask]
    if shallow_after.size == 0:
        print("No shallow-water cells found; nothing to verify.")
        return

    print(f"Shallow cells max after constraint:  {shallow_after.max():.4f}")
    print(f"Shallow cells mean after constraint: {shallow_after.mean():.4f}")

    if shallow_after.max() <= cap + 1e-6:
        print(f"OK: shallow-water probabilities capped at {cap}.")
    else:
        print(f"WARNING: some shallow cells exceed {cap}.")

    deep_mask = ~shallow_mask
    deep_before = proba_before[deep_mask]
    deep_after = proba_after[deep_mask]
    if np.allclose(deep_before, deep_after, equal_nan=True):
        print("OK: deep-water predictions unchanged.")
    else:
        print("WARNING: deep-water predictions were modified.")


def verify_known_bsr_points(bsr_path, proba_before, proba_after, ref_meta,
                            prob_nodata):
    """Check that known BSR points were not suppressed by the constraint."""
    if not os.path.exists(bsr_path):
        print(f"\nKnown-BSR check skipped: file not found: {bsr_path}")
        return

    print("\n" + "=" * 70)
    print("  Known BSR point verification")
    print("=" * 70)

    bsr_df = pd.read_csv(bsr_path, sep="\t")

    def lookup(lon, lat, arr):
        try:
            row, col = rasterio.transform.rowcol(
                ref_meta["transform"], lon, lat
            )
            if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
                v = arr[row, col]
                if v != prob_nodata and not np.isnan(v):
                    return float(v)
        except Exception:  # noqa: BLE001
            pass
        return np.nan

    bsr_df["proba_before"] = bsr_df.apply(
        lambda r: lookup(r["X"], r["Y"], proba_before), axis=1
    )
    bsr_df["proba_after"] = bsr_df.apply(
        lambda r: lookup(r["X"], r["Y"], proba_after), axis=1
    )

    valid = bsr_df.dropna(subset=["proba_before", "proba_after"])
    print(f"Valid known BSR points: {len(valid)}")
    print(f"Mean probability before: {valid['proba_before'].mean():.4f}")
    print(f"Mean probability after:  {valid['proba_after'].mean():.4f}")

    affected = valid[valid["proba_before"] != valid["proba_after"]]
    print(f"Known BSR points affected: {len(affected)}")

    if len(affected) > 0:
        print("\nAffected points (likely shallow-water data issues):")
        print(
            affected[["X", "Y", "BSR", "proba_before", "proba_after"]]
            .head(10)
            .to_string(index=False)
        )
    else:
        print("OK: no known BSR point was affected by the constraint.")


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------
def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply shallow-water constraint to BSR probability map."
    )
    parser.add_argument(
        "--feat-dir",
        default="Global feature dataset",
        help="Directory containing the water-depth raster.",
    )
    parser.add_argument(
        "--prob-tif",
        default="BSR_occurrence_probability.tif",
        help="Unconstrained probability GeoTIFF.",
    )
    parser.add_argument(
        "--wd-tif",
        default=None,
        help=(
            "Water-depth GeoTIFF. Defaults to "
            "<feat-dir>/global_water_depth.tif."
        ),
    )
    parser.add_argument(
        "--out-constrained",
        default="BSR_occurrence_probability_constrained.tif",
        help="Output constrained probability GeoTIFF.",
    )
    parser.add_argument(
        "--bsr-points",
        default="Global BSR actual observation dataset.txt",
        help="Known BSR points for verification (tab-separated).",
    )
    parser.add_argument(
        "--depth-min",
        type=float,
        default=WATER_DEPTH_MIN,
        help="Minimum water depth for BSR occurrence (metres).",
    )
    parser.add_argument(
        "--cap",
        type=float,
        default=PROBA_CAP,
        help="Probability assigned in shallow water.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the constraint workflow."""
    args = parse_args(argv)
    wd_tif = args.wd_tif or os.path.join(
        args.feat_dir, "global_water_depth.tif"
    )

    # -----------------------------------------------------------------
    # 1. Read unconstrained probability
    # -----------------------------------------------------------------
    print("=" * 70)
    print("  Reading unconstrained probability")
    print("=" * 70)

    proba, ref_meta = read_raster(args.prob_tif)
    prob_nodata = ref_meta["nodata"]
    print(
        f"Shape: {ref_meta['shape']}, CRS: {ref_meta['crs']}, "
        f"nodata: {prob_nodata}"
    )

    ocean_mask = (proba != prob_nodata) & ~np.isnan(proba)
    print(f"Ocean cells: {int(ocean_mask.sum()):,}")

    # -----------------------------------------------------------------
    # 2. Align water depth
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Aligning water-depth raster")
    print("=" * 70)

    wd_aligned = align_to_reference(wd_tif, ref_meta)
    wd_abs = np.abs(wd_aligned)
    print(
        f"Depth range: {np.nanmin(wd_abs):.0f} - "
        f"{np.nanmax(wd_abs):.0f} m"
    )

    # -----------------------------------------------------------------
    # 3. Apply constraint
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Applying shallow-water constraint")
    print("=" * 70)

    proba_constrained, shallow_mask = apply_shallow_water_constraint(
        proba, wd_abs, ocean_mask, args.depth_min, args.cap
    )

    n_shallow = int(shallow_mask.sum())
    n_high = int((shallow_mask & (proba >= 0.5)).sum())
    n_changed = int((proba != proba_constrained).sum())

    print(f"Depth threshold: <{args.depth_min:.0f} m")
    print(f"Shallow cells: {n_shallow:,}")
    print(f"Shallow cells with P>=0.5 before: {n_high:,}")
    if n_shallow > 0:
        print(
            f"Shallow cells mean probability before: "
            f"{proba[shallow_mask].mean():.4f}"
        )
    print(f"Cells modified: {n_changed:,}")
    if n_changed > 0:
        diff_values = proba[shallow_mask] - args.cap
        print(f"Max correction: {diff_values.max():.4f}")
        print(f"Mean correction (modified cells): {diff_values.mean():.4f}")

    # -----------------------------------------------------------------
    # 4. Statistics
    # -----------------------------------------------------------------
    report_thresholds(proba, proba_constrained, ocean_mask, THRESHOLDS)

    mean_before = proba[ocean_mask].mean()
    mean_after = proba_constrained[ocean_mask].mean()
    print(
        f"\nGlobal mean probability: {mean_before:.4f} -> "
        f"{mean_after:.4f} ({mean_after - mean_before:+.4f})"
    )

    # -----------------------------------------------------------------
    # 5. Write output
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Writing output")
    print("=" * 70)

    write_raster(
        args.out_constrained, proba_constrained, ref_meta, prob_nodata
    )
    print(f"Saved constrained probability: {args.out_constrained}")

    # -----------------------------------------------------------------
    # 6. Verification
    # -----------------------------------------------------------------
    verify_constraint(
        proba, proba_constrained, shallow_mask, args.depth_min, args.cap
    )
    verify_known_bsr_points(
        args.bsr_points, proba, proba_constrained, ref_meta, prob_nodata
    )

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)