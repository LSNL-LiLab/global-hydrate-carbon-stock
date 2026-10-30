"""
Global BGHS (Base of Gas Hydrate Stability zone) thickness calculation.

This script computes the theoretical thickness of the methane hydrate
stability zone in marine sediments using a simplified phase-equilibrium
model. It reads gridded oceanographic and geothermal inputs, aligns them
to a common 0.25-degree grid, calculates pressure, equilibrium temperature,
and BGHS thickness, then writes two GeoTIFFs:

1. global_BGHS_thickness_0.25deg.tif
   Raw BGHS thickness, used as a feature for machine learning.

2. global_BGHS_thickness_clipped_800m.tif
   BGHS thickness clipped at a user-defined upper limit, used for mapping.
"""

import argparse
import os
import sys

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401  (required for .rio accessor)


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
RHO_SW = 1030.0          # seawater density (kg/m3)
G = 9.81                 # gravitational acceleration (m/s2)
NODATA_VAL = -9999.0     # nodata value in output GeoTIFFs


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------
def load_and_align(filename, std_lat, std_lon, base_dir):
    """Load a raster and bilinearly resample it to a standard grid.

    Parameters
    ----------
    filename : str
        File name relative to ``base_dir``.
    std_lat : np.ndarray
        Target latitude coordinates (1-D).
    std_lon : np.ndarray
        Target longitude coordinates (1-D).
    base_dir : str
        Directory containing the raster file.

    Returns
    -------
    xr.DataArray
        Raster aligned to ``std_lat`` and ``std_lon``.
    """
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input raster not found: {path}")

    da = xr.open_dataarray(path, engine="rasterio")
    if da.ndim == 3:
        da = da.isel(band=0)
    if "lat" in da.dims:
        da = da.rename({"lat": "y", "lon": "x"})

    needs_resample = (
        da.sizes["y"] != len(std_lat) or da.sizes["x"] != len(std_lon)
    )
    if needs_resample:
        print(f"  Resampling {filename}: {da.sizes} -> "
              f"({len(std_lat)}, {len(std_lon)})")
        da = da.interp(y=std_lat, x=std_lon, method="linear")
    else:
        print(f"  {filename} already aligned")
    return da


def clean_nodata(da):
    """Replace the no-data sentinel and NaNs with NaN."""
    return da.where((da != NODATA_VAL) & (~np.isnan(da)))


def save_geotiff(data_array, out_path, lat, lon, nodata=NODATA_VAL):
    """Write a 2-D array to a GeoTIFF with EPSG:4326 metadata.

    Parameters
    ----------
    data_array : np.ndarray
        2-D array of shape (len(lat), len(lon)).
    out_path : str
        Output file path.
    lat, lon : np.ndarray
        1-D coordinate arrays.
    nodata : float
        No-data value written to the GeoTIFF.
    """
    da = xr.DataArray(
        data_array,
        dims=("y", "x"),
        coords={"y": lat, "x": lon},
        name="BGHS",
    )
    da = da.rio.write_crs("EPSG:4326")
    da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    da.rio.write_nodata(nodata, inplace=True)
    da.rio.to_raster(out_path)


# ---------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------
def equilibrium_temperature(pressure_mpa, salinity_psu):
    """Equilibrium temperature of pure methane hydrate (deg C).

    Simplified Sloan-type fit, corrected for pore-water salinity.

    Parameters
    ----------
    pressure_mpa : np.ndarray
        Seafloor pressure in MPa.
    salinity_psu : np.ndarray
        Bottom-water salinity in PSU.

    Returns
    -------
    np.ndarray
        Equilibrium temperature in deg C.
    """
    p_safe = np.maximum(pressure_mpa, 2.0)          # avoid log of small P
    ln_p = np.log(p_safe)
    t_eq_pure = 6.2 * ln_p + 4.5

    sal_safe = np.nan_to_num(salinity_psu, nan=35.0)
    t_eq = t_eq_pure - 0.075 * sal_safe
    return t_eq


def compute_bghs(t_eq, t_bottom, gradient_km):
    """Compute BGHS thickness in metres below seafloor.

    Parameters
    ----------
    t_eq : np.ndarray
        Equilibrium temperature (deg C).
    t_bottom : np.ndarray
        Bottom-water temperature (deg C).
    gradient_km : np.ndarray
        Geothermal gradient in deg C per km.

    Returns
    -------
    np.ndarray
        BGHS thickness in metres; 0 where the stability condition fails.
    """
    gradient_m = gradient_km / 1000.0               # deg C per m
    delta_t = t_eq - t_bottom

    valid = (delta_t > 0) & (gradient_m > 1e-10)
    bghs = np.where(valid, delta_t / gradient_m, 0.0)
    return bghs


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------
def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute global BGHS thickness from gridded inputs."
    )
    parser.add_argument(
        "--input-dir",
        default="Global feature dataset",
        help="Directory containing input GeoTIFFs.",
    )
    parser.add_argument(
        "--raw-tif",
        default="global_BGHS_thickness_0.25deg.tif",
        help="Output filename for raw BGHS thickness.",
    )
    parser.add_argument(
        "--clip-tif",
        default="global_BGHS_thickness_clipped_800m.tif",
        help="Output filename for clipped BGHS thickness.",
    )
    parser.add_argument(
        "--clip-threshold",
        type=float,
        default=800.0,
        help="Upper clipping limit for mapping (metres).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the full BGHS calculation pipeline."""
    args = parse_args(argv)
    base_dir = args.input_dir
    threshold = args.clip_threshold

    output_raw_tif = os.path.join(base_dir, args.raw_tif)
    output_clip_tif = os.path.join(base_dir, args.clip_tif)

    # -----------------------------------------------------------------
    # 1. Load the standard grid from the temperature raster
    # -----------------------------------------------------------------
    print("Reading standard grid (bottom temperature)...")
    temp = xr.open_dataarray(
        os.path.join(base_dir, "global_bottom_temp.tif"),
        engine="rasterio",
    )
    if temp.ndim == 3:
        temp = temp.isel(band=0)
    if "lat" in temp.dims:
        temp = temp.rename({"lat": "y", "lon": "x"})

    std_lat = temp.coords["y"].values
    std_lon = temp.coords["x"].values
    print(f"Standard grid: {len(std_lat)} rows x {len(std_lon)} cols")

    # -----------------------------------------------------------------
    # 2. Load and align the remaining rasters
    # -----------------------------------------------------------------
    print("\nReading and aligning other rasters...")
    sal = load_and_align(
        "global_bottom_salinity.tif", std_lat, std_lon, base_dir
    )
    grad = load_and_align(
        "global_geothermal_gradient.tif", std_lat, std_lon, base_dir
    )
    depth = load_and_align(
        "global_water_depth.tif", std_lat, std_lon, base_dir
    )

    # -----------------------------------------------------------------
    # 3. Clean no-data and build a strict ocean mask
    # -----------------------------------------------------------------
    temp_clean = clean_nodata(temp)
    sal_clean = clean_nodata(sal)
    grad_clean = clean_nodata(grad)
    depth_clean = clean_nodata(depth)

    ocean_mask = ~np.isnan(temp_clean.values)
    n_ocean = int(np.sum(ocean_mask))
    print(f"Ocean grid cells: {n_ocean:,} / {ocean_mask.size:,}")

    # -----------------------------------------------------------------
    # 4. Seafloor pressure (MPa)
    # -----------------------------------------------------------------
    print("Computing seafloor pressure...")
    pressure_mpa = RHO_SW * G * np.abs(depth_clean.values) / 1e6
    print(
        "Pressure range: {:.2f} - {:.2f} MPa".format(
            np.nanmin(pressure_mpa[ocean_mask]),
            np.nanmax(pressure_mpa[ocean_mask]),
        )
    )

    # -----------------------------------------------------------------
    # 5. Equilibrium temperature (deg C)
    # -----------------------------------------------------------------
    print("Computing equilibrium temperature...")
    t_eq = equilibrium_temperature(pressure_mpa, sal_clean.values)
    print(
        "Equilibrium temperature range: {:.2f} - {:.2f} deg C".format(
            np.nanmin(t_eq[ocean_mask]),
            np.nanmax(t_eq[ocean_mask]),
        )
    )

    # -----------------------------------------------------------------
    # 6. BGHS thickness (mbsf)
    # -----------------------------------------------------------------
    print("Computing BGHS thickness...")
    bghs_raw = compute_bghs(t_eq, temp_clean.values, grad_clean.values)
    bghs_masked = np.where(ocean_mask, bghs_raw, NODATA_VAL)

    # -----------------------------------------------------------------
    # 7. Statistics and clipping
    # -----------------------------------------------------------------
    ocean_thickness = bghs_raw[ocean_mask]
    valid_thickness = ocean_thickness[ocean_thickness > 0]

    print("\n" + "=" * 60)
    print("Global BGHS thickness statistics (ocean only)")
    print("=" * 60)
    print(
        "Cells with stability zone: {:,} ({:.1f}%)".format(
            len(valid_thickness),
            len(valid_thickness) / n_ocean * 100,
        )
    )
    print(
        "Raw thickness range: {:.1f} - {:.1f} m".format(
            np.min(valid_thickness), np.max(valid_thickness)
        )
    )
    print(f"Mean thickness: {np.mean(valid_thickness):.1f} m")
    print(f"Median thickness: {np.median(valid_thickness):.1f} m")
    for q in (10, 25, 50, 75, 90):
        print(
            f"  {q}th percentile: {np.percentile(valid_thickness, q):.1f} m"
        )

    bghs_clipped = np.where(
        ocean_mask & (bghs_raw > threshold), threshold, bghs_masked
    )
    bghs_clipped[~ocean_mask] = NODATA_VAL

    n_clipped = int(np.sum(ocean_mask & (bghs_raw > threshold)))
    print(f"\nClip threshold: {threshold:.0f} m")
    print(
        "Clipped cells: {:,} ({:.2f}%)".format(
            n_clipped, n_clipped / n_ocean * 100
        )
    )

    # -----------------------------------------------------------------
    # 8. Write outputs
    # -----------------------------------------------------------------
    print("\nSaving raw BGHS thickness (for machine learning)...")
    save_geotiff(bghs_masked, output_raw_tif, std_lat, std_lon)
    print(f"  -> {output_raw_tif}")

    print(f"Saving clipped BGHS thickness ({threshold:.0f} m cap)...")
    save_geotiff(bghs_clipped, output_clip_tif, std_lat, std_lon)
    print(f"  -> {output_clip_tif}")

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)