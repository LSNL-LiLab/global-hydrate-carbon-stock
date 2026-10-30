"""
Plot carbon stock as a function of BSR burial depth (Figure 5b).

Reads the predicted BSR depth raster, computes per-cell carbon stock
using the medium scenario parameters, bins carbon stock by depth, and
draws a smoothed curve with a peak annotation.

Output
------
``fig5b_carbon_vs_bsr_depth.png``

"""

import os
import sys

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

FIG_WIDTH_IN = 18.0 / 2.54
FIG_HEIGHT_IN = 12.0 / 2.54

FS_LABEL = 21
FS_TICK = 21
FS_VALUE = 21

# Medium scenario (matches Table 2)
C_DENSITY = 0.000537
PHI_MID = 0.55
SH_MID = 0.02
PF_MID = 165

EARTH_RADIUS_KM = 6371.0
DEPTH_BIN_EDGES = np.arange(100, 1250, 25)


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


# ---------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------
def carbon_by_depth(depth_valid, carbon_valid, bin_edges):
    """Sum carbon stock within each depth bin (Gt C)."""
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    carbon_by_bin = np.zeros(len(bin_centers))

    for i in range(len(bin_centers)):
        mask = (depth_valid >= bin_edges[i]) & (depth_valid < bin_edges[i + 1])
        carbon_by_bin[i] = carbon_valid[mask].sum() / 1e9

    nonzero = carbon_by_bin > 0
    return bin_centers[nonzero], carbon_by_bin[nonzero]


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
def plot_curve(bin_centers, carbon_by_bin, out_png):
    """Draw and save the smoothed carbon-depth curve."""
    carbon_smooth = gaussian_filter1d(carbon_by_bin, sigma=1.5)

    x_smooth = np.linspace(
        bin_centers.min(), bin_centers.max(), 500
    )
    spline = make_interp_spline(bin_centers, carbon_smooth, k=3)
    y_smooth = np.maximum(spline(x_smooth), 0)

    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=300
    )

    ax.plot(x_smooth, y_smooth, color="#4C72B0", linewidth=2.2)
    ax.fill_between(x_smooth, y_smooth, alpha=0.25, color="#4C72B0")

    peak_idx = np.argmax(y_smooth)
    peak_depth = x_smooth[peak_idx]
    peak_value = y_smooth[peak_idx]
    ax.annotate(
        f"Peak: {peak_value:.1f} Gt C\nat {peak_depth:.0f} m",
        xy=(peak_depth, peak_value),
        xytext=(peak_depth + 150, peak_value * 0.8),
        fontsize=FS_VALUE,
        color="black",
        arrowprops=dict(arrowstyle="->", color="gray", linewidth=1.0),
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9,
        ),
    )

    ax.set_xlabel("BSR burial depth (m)", fontsize=FS_LABEL)
    ax.set_ylabel("Carbon stock (Gt C)", fontsize=FS_LABEL)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.set_xlim(x_smooth.min(), x_smooth.max())

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    depth_tif = "BSR_depth_prediction.tif"
    out_png = "fig5b_carbon_vs_bsr_depth.png"

    print("Reading depth raster...")
    depth, meta = read_depth_raster(depth_tif)
    nodata = meta["nodata"]

    valid = (
        (depth != nodata) & ~np.isnan(depth) & (depth > 0)
    )
    print(f"Valid cells: {int(valid.sum())}")

    area_2d = compute_cell_area(meta)
    carbon = area_2d * depth * PHI_MID * SH_MID * PF_MID * C_DENSITY
    carbon[~valid] = 0.0

    print(f"Total carbon (medium scenario): {carbon.sum() / 1e9:.1f} Gt C")

    depth_valid = depth[valid]
    carbon_valid = carbon[valid]

    bin_centers, carbon_by_bin = carbon_by_depth(
        depth_valid, carbon_valid, DEPTH_BIN_EDGES
    )
    plot_curve(bin_centers, carbon_by_bin, out_png)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)