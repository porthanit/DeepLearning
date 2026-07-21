"""
Extract building footprints from a georeferenced satellite mosaic using a
YOLOv8 building-segmentation model (keremberke/yolov8m-building-segmentation,
~55 MB, auto-downloaded from Hugging Face Hub on first run).

Input:  GoogleMap-Images.tif   (from GoogleMap-Download.py)
Output: YOLO-Extraction.tif        binary building mask, same grid as input
        YOLO-Extraction.geojson    building footprint polygons, same CRS
"""

import os
import sys

os.environ["CURL_CA_BUNDLE"] = ""

_RASTERIO_DIR = os.path.join(os.path.dirname(sys.executable), "..", "Lib", "site-packages", "rasterio")
os.environ["PROJ_LIB"] = os.path.join(_RASTERIO_DIR, "proj_data")
os.environ["GDAL_DATA"] = os.path.join(_RASTERIO_DIR, "gdal_data")

import geopandas as gpd
import numpy as np
import rasterio
from huggingface_hub import hf_hub_download
from PIL import Image
from rasterio.features import shapes as raster_shapes
from shapely.geometry import shape
from ultralytics import YOLO

INPUT_TIF = "GoogleMap-Images.tif"
OUTPUT_TIF = "YOLO-Extraction.tif"
OUTPUT_GEOJSON = "YOLO-Extraction.geojson"

MODEL_REPO = "keremberke/yolov8m-building-segmentation"
MODEL_FILENAME = "best.pt"

TILE_SIZE = 640       # matches the model's native training resolution
TILE_OVERLAP = 64      # overlap so buildings that straddle a tile edge still get segmented
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
DEVICE = "cpu"          # no CUDA GPU on this machine


def load_model():
    weight_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    return YOLO(weight_path)


def iter_tiles(width, height, tile_size, overlap):
    step = tile_size - overlap
    for y in range(0, height, step):
        for x in range(0, width, step):
            w = min(tile_size, width - x)
            h = min(tile_size, height - y)
            if w > 0 and h > 0:
                yield x, y, w, h


def main():
    with rasterio.open(INPUT_TIF) as src:
        profile = src.profile
        transform = src.transform
        crs = src.crs
        width, height = src.width, src.height
        # YOLO expects HWC arrays in BGR order (like cv2.imread)
        bgr = np.moveaxis(src.read([3, 2, 1]), 0, -1)

    model = load_model()

    mask = np.zeros((height, width), dtype=np.uint8)

    n_tiles = 0
    n_detections = 0
    for x0, y0, w, h in iter_tiles(width, height, TILE_SIZE, TILE_OVERLAP):
        tile = np.ascontiguousarray(bgr[y0:y0 + h, x0:x0 + w])
        n_tiles += 1

        result = model.predict(
            tile, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, device=DEVICE, verbose=False
        )[0]

        if result.masks is not None:
            for poly_mask in result.masks.data.cpu().numpy():
                resized = Image.fromarray((poly_mask * 255).astype(np.uint8)).resize((w, h))
                binary = np.array(resized) > 127
                mask[y0:y0 + h, x0:x0 + w] |= binary.astype(np.uint8)
                n_detections += 1

        print(f"  tile ({x0},{y0}) {w}x{h}: "
              f"{0 if result.masks is None else len(result.masks.data)} building(s)")

    print(f"Processed {n_tiles} tiles, {n_detections} raw building detections")

    # --- Write raster mask (same grid/CRS as input) ---
    out_profile = profile.copy()
    out_profile.update(count=1, dtype="uint8", compress="deflate")
    with rasterio.open(OUTPUT_TIF, "w", **out_profile) as dst:
        dst.write(mask, 1)
    print(f"Wrote {OUTPUT_TIF}")

    # --- Vectorize mask into building footprint polygons ---
    # overlapping detections between neighbouring tiles were already merged
    # in the raster (bitwise OR above), so this naturally dedupes them.
    polygons = [
        shape(geom)
        for geom, value in raster_shapes(mask, mask=mask > 0, transform=transform)
        if value == 1
    ]

    gdf = gpd.GeoDataFrame({"class": ["building"] * len(polygons)}, geometry=polygons, crs=crs)
    gdf.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
    print(f"Wrote {OUTPUT_GEOJSON} ({len(polygons)} building polygons)")


if __name__ == "__main__":
    main()
