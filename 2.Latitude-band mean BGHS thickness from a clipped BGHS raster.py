"""
Latitude-band mean BGHS thickness from a clipped BGHS raster.

Reads a GeoTIFF of BGHS (Base of Gas Hydrate Stability zone) thickness,
aggregates mean thickness per 10-degree latitude band, and writes a
horizontal bar chart.
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

FIG_WIDTH_IN = 18.0 / 2.54      # 18 cm
FIG_HEIGHT_IN = 12.5 / 2.54     # 12.5 cm

FS_LABEL = 26
FS_TICK = 23


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------
def read_raster(path):
    """Read the first band of a GeoTIFF and return data and metadata.

    Parameters
    ----------
    path : str
        Path to the input GeoTIFF.

    Returns
    -------
    tuple
        ``(array, transform, nodata, nrows, ncols)``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input raster not found: {path}")

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float64)
        transform = src.transform
        nodata = src.nodata
        nrows, ncols = data.shape

    return data, transform, nodata, nrows, ncols


def valid_mask(data, nodata):
    """Return a boolean mask of valid (non-nodata, non-NaN) cells."""
    if nodata is not None:
        return (data != nodata) & ~np.isnan(data)
    return ~np.isnan(data)


def band_mean(data, lats, valid, band_edges):
    """Compute the mean of ``data`` within each latitude band.

    Parameters
    ----------
    data : np.ndarray
        2-D array, shape ``(nrows, ncols)``.
    lats : np.ndarray
        1-D array of row-centre latitudes, shape ``(nrows,)``.
    valid : np.ndarray
        Boolean mask of valid cells.
    band_edges : np.ndarray
        Latitude boundaries of the bands.

    Returns
    -------
    list of float
        Mean value per band, in the same order as ``band_edges``.
    """
    band_means = []
    for lat_min, lat_max in zip(band_edges[:-1], band_edges[1:]):
        band_mask = (
            valid
            & (lats[:, np.newaxis] >= lat_min)
            & (lats[:, np.newaxis] < lat_max)
        )
        if band_mask.sum() > 0:
            band_means.append(np.nanmean(data[band_mask]))
        else:
            band_means.append(0.0)
    return band_means


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_band_means(band_centers, band_heights, band_means, out_path):
    """Render and save the latitude-band bar chart.

    Parameters
    ----------
    band_centers : np.ndarray
        Centre latitude of each band.
    band_heights : np.ndarray
        Vertical extent of each band (in degrees).
    band_means : list of float
        Mean BGHS thickness per band.
    out_path : str
        Output image path.
    """
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=300)

    colors = plt.get_cmap("YlGnBu")(
        np.linspace(0.30, 0.85, len(band_means))
    )

    ax.barh(
        band_centers,
        band_means,
        height=band_heights,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
    )

    ax.set_ylim(-90, 90)
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax.set_yticklabels(
        ["90\u00b0S", "60\u00b0S", "30\u00b0S", "0\u00b0",
         "30\u00b0N", "60\u00b0N", "90\u00b0N"],
        fontsize=FS_TICK,
    )

    ax.set_xlabel("Mean BGHS thickness (m)", fontsize=FS_LABEL)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.set_xlim(0, max(band_means) * 1.10)
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------
def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot mean BGHS thickness per 10-degree latitude band."
        )
    )
    parser.add_argument(
        "--tif",
        default=os.path.join(
            "Global feature dataset",
            "global_BGHS_thickness_clipped_800m.tif",
        ),
        help="Input BGHS GeoTIFF.",
    )
    parser.add_argument(
        "--out",
        default="fig3_BGHS_thickness_by_latitude_band.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--band-step",
        type=float,
        default=10.0,
        help="Latitude band width in degrees.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the latitude-band aggregation and plotting workflow."""
    args = parse_args(argv)

    # -----------------------------------------------------------------
    # 1. Read raster
    # -----------------------------------------------------------------
    data, transform, nodata, nrows, _ = read_raster(args.tif)

    lats = np.array(
        [transform.f + (j + 0.5) * transform.e for j in range(nrows)]
    )

    mask = valid_mask(data, nodata)
    data_plot = np.where(mask, data, np.nan)

    # -----------------------------------------------------------------
    # 2. Aggregate per latitude band
    # -----------------------------------------------------------------
    step = args.band_step
    band_edges = np.arange(-90, 90 + step, step)
    band_centers = (band_edges[:-1] + band_edges[1:]) / 2
    band_heights = np.diff(band_edges)

    band_means = band_mean(data_plot, lats, mask, band_edges)

    # -----------------------------------------------------------------
    # 3. Plot and save
    # -----------------------------------------------------------------
    plot_band_means(band_centers, band_heights, band_means, args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)