"""
Compare carbon stock across major BSR regions (Figure 5d).

Reads the predicted BSR depth raster, computes per-cell carbon stock
using the medium scenario parameters, aggregates the stock within
predefined regional bounding boxes, sorts the regions by stock, and
draws a horizontal bar chart.

Output
------
``fig5d_regional_carbon_stock.png``
"""

import argparse
import os
import sys

import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

FIG_WIDTH_IN = 22.0 / 2.54
FIG_HEIGHT_IN = 15.0 / 2.54

FS_LABEL = 25
FS_TICK = 21
FS_VALUE = 21

# Medium scenario (matches Table 2)
C_DENSITY = 0.000537
PHI_MID = 0.55
SH_MID = 0.02
PF_MID = 165

EARTH_RADIUS_KM = 6371.0

# Bounding boxes as (lon_min, lon_max, lat_min, lat_max)
REGIONS = {
    "South China Sea": (105, 125, 5, 25),
    "Gulf of Mexico": (-98, -82, 22, 31),
    "Blake Ridge": (-80, -70, 28, 35),
    "Nankai Trough": (132, 142, 30, 35),
    "Black Sea": (28, 42, 41, 46),
    "Indian Margin": (80, 88, 14, 21),
    "Andaman Sea": (92, 98, 10, 15),
    "Hikurangi": (175, 180, -42, -38),
    "Beaufort Sea": (-150, -130, 70, 72),
    "Storegga": (0, 12, 62, 68),
}

BAR_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#64B5CD",
    "#B07AA1", "#9C755F",
]


# ---------------------------------------------------------------------
# I/O and area
# ---------------------------------------------------------------------
def read_depth_raster(path):
    """Read BSR depth and return data plus metadata."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Depth raster not found: {path}")

    with rasterio.open(path) as src:
        depth = src.read(1).astype(np.float64)
        meta = {
            "transform": src.transform,
            "shape": src.shape,
            "nodata": src.nodata,
        }
    return depth, meta


def compute_cell_area(meta):
    """Return cell areas in m2 for a regular lat/lon grid."""
    transform = meta["transform"]
    nrows, ncols = meta["shape"]

    dlon = abs(transform.a) * np.pi / 180.0
    row_areas = np.zeros(nrows)
    for i in range(nrows):
        lat_top = transform.f + i * transform.e
        lat_bot = transform.f + (i + 1) * transform.e
        row_areas[i] = EARTH_RADIUS_KM ** 2 * dlon * abs(
            np.sin(np.radians(lat_top)) - np.sin(np.radians(lat_bot))
        )
    return np.tile(row_areas[:, np.newaxis], (1, ncols)) * 1e6


def build_coordinate_grids(meta):
    """Return 2-D longitude and latitude arrays at cell centres."""
    transform = meta["transform"]
    nrows, ncols = meta["shape"]

    lons = np.array(
        [transform.c + (i + 0.5) * transform.a for i in range(ncols)]
    )
    lats = np.array(
        [transform.f + (j + 0.5) * transform.e for j in range(nrows)]
    )
    return np.meshgrid(lons, lats)


# ---------------------------------------------------------------------
# Regional aggregation
# ---------------------------------------------------------------------
def aggregate_regions(carbon, lon_grid, lat_grid, valid, regions):
    """Sum carbon stock within each bounding box (Gt C)."""
    region_carbon = {}
    for name, (lon_min, lon_max, lat_min, lat_max) in regions.items():
        mask = (
            valid
            & (lon_grid >= lon_min)
            & (lon_grid <= lon_max)
            & (lat_grid >= lat_min)
            & (lat_grid <= lat_max)
        )
        region_carbon[name] = carbon[mask].sum() / 1e9
        print(f"  {name}: {region_carbon[name]:.2f} Gt C")
    return region_carbon


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_regions(region_carbon, out_path):
    """Draw and save the regional carbon-stock bar chart."""
    ordered = dict(
        sorted(region_carbon.items(), key=lambda x: x[1], reverse=True)
    )
    names = list(ordered.keys())
    values = list(ordered.values())

    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=300
    )

    bars = ax.barh(
        names,
        values,
        color=BAR_COLORS[: len(names)],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
    )

    offset = max(values) * 0.01
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            fontsize=FS_VALUE,
            fontweight="bold",
        )

    ax.set_xlabel("Carbon stock (Gt C)", fontsize=FS_LABEL)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.set_xlim(0, max(values) * 1.15)
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

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
        description="Plot carbon stock across major BSR regions."
    )
    parser.add_argument(
        "--depth-tif",
        default="BSR_depth_prediction.tif",
        help="Predicted BSR depth GeoTIFF.",
    )
    parser.add_argument(
        "--out",
        default="fig5d_regional_carbon_stock.png",
        help="Output PNG path.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the regional carbon-stock plotting workflow."""
    args = parse_args(argv)

    # -----------------------------------------------------------------
    # 1. Read depth raster
    # -----------------------------------------------------------------
    print("Reading depth raster...")
    depth, meta = read_depth_raster(args.depth_tif)
    nodata = meta["nodata"]

    valid = (depth != nodata) & ~np.isnan(depth) & (depth > 0)
    print(f"Valid cells: {int(valid.sum())}")

    # -----------------------------------------------------------------
    # 2. Carbon stock per cell (medium scenario)
    # -----------------------------------------------------------------
    area_2d = compute_cell_area(meta)
    carbon = area_2d * depth * PHI_MID * SH_MID * PF_MID * C_DENSITY
    carbon[~valid] = 0.0

    print(
        f"Total carbon (medium scenario): "
        f"{carbon.sum() / 1e9:.1f} Gt C"
    )

    # -----------------------------------------------------------------
    # 3. Coordinate grids and regional aggregation
    # -----------------------------------------------------------------
    lon_grid, lat_grid = build_coordinate_grids(meta)
    print("\nRegional carbon stock:")
    region_carbon = aggregate_regions(
        carbon, lon_grid, lat_grid, valid, REGIONS
    )

    # -----------------------------------------------------------------
    # 4. Plot
    # -----------------------------------------------------------------
    plot_regions(region_carbon, args.out)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)