"""
Download Google Maps Satellite imagery for an AOI and save as a georeferenced
GeoTIFF, then build an external overview pyramid (.tif.ovr) for fast display.

Output: GoogleMap-Images.tif (+ GoogleMap-Images.tif.ovr)

Note: this machine has stale system environment variables (CURL_CA_BUNDLE,
PROJ_LIB, GDAL_DATA) left over from an old PostgreSQL/PostGIS install that
point at missing/incompatible files. We override them for this process only
(not system-wide) so rasterio's own bundled PROJ/GDAL data is used instead.
"""

import io
import math
import os
import subprocess
import sys

os.environ["CURL_CA_BUNDLE"] = ""

import numpy as np
import requests
from PIL import Image

_RASTERIO_DIR = os.path.join(os.path.dirname(sys.executable), "..", "Lib", "site-packages", "rasterio")
os.environ["PROJ_LIB"] = os.path.join(_RASTERIO_DIR, "proj_data")
os.environ["GDAL_DATA"] = os.path.join(_RASTERIO_DIR, "gdal_data")

import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, transform as warp_transform

# ---------------------------------------------------------------------------
# AOI (paste the coordinates from GEE Geometry Tools here, lon/lat WGS84)
# ---------------------------------------------------------------------------
AOI_LONLAT = [
    [100.92719193580069, 13.104421800541681],
    [100.92719193580069, 13.101438467119591],
    [100.93089338424124, 13.101438467119591],
    [100.93089338424124, 13.104421800541681],
]

# Output CRS (matches the UTM zone used elsewhere in this project)
OUTPUT_CRS = "EPSG:32647"

# Tile zoom level (19-20 gives ~0.3-0.6 m/px, suitable for building extraction)
ZOOM = 20

TILE_SIZE = 256
GOOGLE_SAT_TILE_URL = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"

OUTPUT_TIF = "GoogleMap-Images.tif"

# gdaladdo.exe is needed to build a true EXTERNAL (.ovr) pyramid - rasterio's
# own Python API can only embed overviews inside the .tif itself. QGIS ships
# its own GDAL build with gdaladdo, so we borrow that.
GDALADDO_EXE = r"C:\Program Files\QGIS 3.28.15\bin\gdaladdo.exe"
GDALADDO_GDAL_DATA = r"C:\Program Files\QGIS 3.28.15\apps\gdal\share\gdal"
GDALADDO_PROJ_LIB = r"C:\Program Files\QGIS 3.28.15\share\proj"


def lonlat_to_tile(lon, lat, zoom):
    lat_rad = math.radians(lat)
    n = 2.0**zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tile_to_lonlat(x, y, zoom):
    n = 2.0**zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lon, lat


def download_tile(x, y, zoom, session):
    url = GOOGLE_SAT_TILE_URL.format(x=x, y=y, z=zoom)
    resp = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def build_external_pyramid(tif_path):
    """Build a .tif.ovr sidecar pyramid using gdaladdo -ro (external mode)."""
    if not os.path.exists(GDALADDO_EXE):
        print(f"WARNING: gdaladdo.exe not found at {GDALADDO_EXE}; skipping external pyramid.")
        return

    env = os.environ.copy()
    env["GDAL_DATA"] = GDALADDO_GDAL_DATA
    env["PROJ_LIB"] = GDALADDO_PROJ_LIB

    subprocess.run(
        [GDALADDO_EXE, "-ro", "-r", "average", tif_path, "2", "4", "8", "16"],
        env=env,
        check=True,
    )


def main():
    lons = [pt[0] for pt in AOI_LONLAT]
    lats = [pt[1] for pt in AOI_LONLAT]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    x_min_f, y_min_f = lonlat_to_tile(min_lon, max_lat, ZOOM)  # top-left
    x_max_f, y_max_f = lonlat_to_tile(max_lon, min_lat, ZOOM)  # bottom-right

    x_start, x_end = math.floor(x_min_f), math.ceil(x_max_f)
    y_start, y_end = math.floor(y_min_f), math.ceil(y_max_f)

    n_cols = x_end - x_start
    n_rows = y_end - y_start
    print(f"AOI covers {n_cols} x {n_rows} tiles at zoom {ZOOM} ({n_cols * n_rows} tiles)")

    mosaic = Image.new("RGB", (n_cols * TILE_SIZE, n_rows * TILE_SIZE))

    with requests.Session() as session:
        for row, ty in enumerate(range(y_start, y_end)):
            for col, tx in enumerate(range(x_start, x_end)):
                tile = download_tile(tx, ty, ZOOM, session)
                mosaic.paste(tile, (col * TILE_SIZE, row * TILE_SIZE))
                print(f"  downloaded tile z={ZOOM} x={tx} y={ty}")

    mosaic_arr = np.array(mosaic)  # (H, W, 3)

    # Georeference the mosaic in Web Mercator (native CRS of the tile grid)
    lon_tl, lat_tl = tile_to_lonlat(x_start, y_start, ZOOM)
    lon_br, lat_br = tile_to_lonlat(x_end, y_end, ZOOM)

    web_merc = CRS.from_epsg(3857)
    wgs84 = CRS.from_epsg(4326)
    xs, ys = warp_transform(wgs84, web_merc, [lon_tl, lon_br], [lat_tl, lat_br])
    x_tl, x_br = xs
    y_tl, y_br = ys

    px_w = (x_br - x_tl) / mosaic_arr.shape[1]
    px_h = (y_tl - y_br) / mosaic_arr.shape[0]
    src_transform = from_origin(x_tl, y_tl, px_w, px_h)

    src_bands = np.moveaxis(mosaic_arr, -1, 0)  # (3, H, W)

    # Reproject straight into the target CRS/output file
    dst_transform, dst_width, dst_height = calculate_default_transform(
        web_merc, OUTPUT_CRS, mosaic_arr.shape[1], mosaic_arr.shape[0],
        left=x_tl, bottom=y_br, right=x_br, top=y_tl,
    )

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 3,
        "width": dst_width,
        "height": dst_height,
        "crs": OUTPUT_CRS,
        "transform": dst_transform,
        "compress": "deflate",
        "photometric": "RGB",
    }

    with rasterio.open(OUTPUT_TIF, "w", **profile) as dst:
        for band_index in range(3):
            reproject(
                source=src_bands[band_index],
                destination=rasterio.band(dst, band_index + 1),
                src_transform=src_transform,
                src_crs=web_merc,
                dst_transform=dst_transform,
                dst_crs=OUTPUT_CRS,
                resampling=Resampling.bilinear,
            )

    print(f"Wrote {OUTPUT_TIF} ({dst_width} x {dst_height}, {OUTPUT_CRS})")

    build_external_pyramid(OUTPUT_TIF)
    print(f"Built external overview pyramid: {OUTPUT_TIF}.ovr")


if __name__ == "__main__":
    main()
