"""
Estimate global hydrate carbon stock from the predicted BSR depth map.

This script reads the predicted BSR burial depth raster, computes the
spherical area of each grid cell, and applies the volumetric method to
estimate global hydrate carbon stock under three parameter scenarios:

- Low:    porosity 50%, saturation 1%,  gas factor 160
- Medium: porosity 55%, saturation 2%,  gas factor 165
- High:   porosity 65%, saturation 5%,  gas factor 175

Only the total carbon stock (and derived CH4 mass, CH4 volume, CO2-eq)
is reported to the terminal. No figure, CSV, or GeoTIFF is written.

Reproducibility note
--------------------
The calculation is deterministic given the input raster. No random
seed, resampling, or model training is involved.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd  # noqa: F401  (kept for parity with upstream pipeline)
import rasterio

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
SCENARIOS = {
    "Low":    {"phi": 0.50, "Sh": 0.01, "PF": 160},
    "Medium": {"phi": 0.55, "Sh": 0.02, "PF": 165},
    "High":   {"phi": 0.65, "Sh": 0.05, "PF": 175},
}

C_DENSITY = 0.000537        # t C per m3 CH4 at STP
CH4_C_FRACTION = 0.75       # 12 / 16
CH4_DENSITY_T_PER_M3 = 0.0007167   # t CH4 per m3 at STP
GWP_100 = 28
EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------
def read_depth_raster(path):
    """Read the BSR depth raster and return data plus metadata."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Depth raster not found: {path}")

    with rasterio.open(path) as src:
        depth = src.read(1).astype(np.float64)
        meta = {
            "transform": src.transform,
            "crs": src.crs,
            "shape": src.shape,
            "nodata": src.nodata,
        }
    return depth, meta


# ---------------------------------------------------------------------
# Cell area
# ---------------------------------------------------------------------
def compute_cell_area(meta, radius_km=EARTH_RADIUS_KM):
    """Return a 2-D array of cell areas in m2 on a regular lat/lon grid."""
    transform = meta["transform"]
    nrows, ncols = meta["shape"]

    dlon = abs(transform.a) * np.pi / 180.0
    dlat = abs(transform.e) * np.pi / 180.0

    row_areas = np.zeros(nrows)
    for i in range(nrows):
        lat_top = transform.f + i * transform.e
        lat_bot = transform.f + (i + 1) * transform.e
        row_areas[i] = radius_km ** 2 * dlon * abs(
            np.sin(np.radians(lat_top)) - np.sin(np.radians(lat_bot))
        )

    return np.tile(row_areas[:, np.newaxis], (1, ncols)) * 1e6  # m2


# ---------------------------------------------------------------------
# Carbon stock
# ---------------------------------------------------------------------
def estimate_carbon_stock(depth, area_2d, valid_mask, phi, sh, pf):
    """Compute carbon stock per scenario.

    Parameters
    ----------
    depth : np.ndarray
        BSR depth (m).
    area_2d : np.ndarray
        Cell area (m2).
    valid_mask : np.ndarray
        Boolean mask of valid cells.
    phi : float
        Sediment porosity.
    sh : float
        Hydrate saturation.
    pf : float
        Gas production factor.

    Returns
    -------
    dict
        Aggregated stock metrics.
    """
    thickness = np.where(valid_mask, depth, 0.0)

    carbon_per_cell = area_2d * thickness * phi * sh * pf * C_DENSITY
    carbon_per_cell[~valid_mask] = 0.0

    total_carbon = carbon_per_cell.sum()
    total_ch4 = total_carbon / CH4_C_FRACTION
    total_carbon_gt = total_carbon / 1e9
    total_ch4_gt = total_ch4 / 1e9
    total_ch4_tm3 = total_ch4 / CH4_DENSITY_T_PER_M3 / 1e12
    co2_eq_gt = total_ch4_gt * GWP_100

    return {
        "phi": phi,
        "sh": sh,
        "pf": pf,
        "carbon_gt": total_carbon_gt,
        "ch4_gt": total_ch4_gt,
        "ch4_tm3": total_ch4_tm3,
        "co2_eq_gt": co2_eq_gt,
    }


def report_results(results):
    """Print a compact table of scenario results."""
    print("\n" + "=" * 70)
    print("  Global hydrate carbon stock")
    print("=" * 70)

    for name, r in results.items():
        print(f"\n[{name}]  phi={r['phi']:.0%}, "
              f"Sh={r['sh']:.0%}, PF={r['pf']}")
        print(f"  Carbon stock : {r['carbon_gt']:.1f} Gt C")
        print(f"  CH4 mass     : {r['ch4_gt']:.1f} Gt CH4")
        print(f"  CH4 volume   : {r['ch4_tm3']:.0f} Tm3 (STP)")
        print(f"  CO2-eq       : {r['co2_eq_gt']:.0f} Gt CO2-eq "
              f"(GWP={GWP_100})")


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------
def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Estimate global hydrate carbon stock from BSR depth."
    )
    parser.add_argument(
        "--depth-tif",
        default="BSR_depth_prediction.tif",
        help="Predicted BSR depth GeoTIFF.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the carbon stock estimation workflow."""
    args = parse_args(argv)

    # -----------------------------------------------------------------
    # 1. Read depth raster
    # -----------------------------------------------------------------
    depth, meta = read_depth_raster(args.depth_tif)
    nodata = meta["nodata"]

    valid_mask = (
        (depth != nodata) & ~np.isnan(depth) & (depth > 0)
    )
    print(f"Valid cells: {int(valid_mask.sum())}")

    # -----------------------------------------------------------------
    # 2. Cell area
    # -----------------------------------------------------------------
    area_2d = compute_cell_area(meta)

    # -----------------------------------------------------------------
    # 3. Carbon stock per scenario
    # -----------------------------------------------------------------
    results = {}
    for name, params in SCENARIOS.items():
        results[name] = estimate_carbon_stock(
            depth,
            area_2d,
            valid_mask,
            params["phi"],
            params["Sh"],
            params["PF"],
        )

    # -----------------------------------------------------------------
    # 4. Report
    # -----------------------------------------------------------------
    report_results(results)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)