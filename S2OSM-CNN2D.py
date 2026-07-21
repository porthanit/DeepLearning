"""
Train a 2D-CNN land-cover classifier on small image patches centered on the
same ESA WorldCover sample points used by S2OSM-CNN.py (CNN-1D) and the GEE
Random Forest script, for a like-for-like comparison of classifier types.

Unlike CNN-1D (a single pixel's band vector), this feeds a PATCH_SIZE x
PATCH_SIZE x 10-band neighbourhood into a small 2D CNN, so it can use local
spatial context (e.g. a pixel next to a river) in addition to spectral values.

Setup: same as S2OSM-CNN.py -- separate .venv-cnn (see requirements-cnn.txt),
your own EE_PROJECT, and `earthengine authenticate` on first run.
"""

import os
import sys

os.environ["CURL_CA_BUNDLE"] = ""

_RASTERIO_DIR = os.path.join(os.path.dirname(sys.executable), "..", "Lib", "site-packages", "rasterio")
os.environ["PROJ_LIB"] = os.path.join(_RASTERIO_DIR, "proj_data")
os.environ["GDAL_DATA"] = os.path.join(_RASTERIO_DIR, "gdal_data")

import ee
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.warp import transform as warp_transform
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
EE_PROJECT = "thanitnuk"

AOI_RECT = [100.45, 14.25, 100.65, 14.45]  # [minLon, minLat, maxLon, maxLat]

BANDS = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "EVI", "NDWI", "NDBI"]
CLASS_NAMES = {1: "น้ำ", 2: "เมือง", 3: "เกษตร", 4: "ไม้ยืนต้น"}

SAMPLE_SCALE = 10   # resolution used for stratifiedSample() point locations
EXPORT_SCALE = 30   # resolution of the local raster used for patches + export
PATCH_SIZE = 9      # patch neighbourhood fed to the CNN (must be odd)
POINTS_PER_CLASS = 500
RANDOM_SEED = 42
OUTPUT_CRS = "EPSG:32647"

OUTPUT_MODEL = "S2OSM-CNN2D.keras"
OUTPUT_COMPOSITE_TIF = "S2OSM-Composite.tif"
OUTPUT_CLASSIFIED_TIF = "S2OSM-CNN2D-Classified.tif"


def ee_init():
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT)


def build_composite(aoi):
    def mask_s2(img):
        scl = img.select("SCL")
        clear = (
            scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        )
        return img.updateMask(clear).divide(10000).copyProperties(img, ["system:time_start"])

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate("2025-11-01", "2026-02-28")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .map(mask_s2)
        .median()
        .select(["B2", "B3", "B4", "B8", "B11", "B12"])
        .clip(aoi)
    )

    with_indices = (
        s2.addBands(s2.normalizedDifference(["B8", "B4"]).rename("NDVI"))
        .addBands(
            s2.expression(
                "2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)",
                {"NIR": s2.select("B8"), "RED": s2.select("B4"), "BLUE": s2.select("B2")},
            ).rename("EVI")
        )
        .addBands(s2.normalizedDifference(["B3", "B8"]).rename("NDWI"))
        .addBands(s2.normalizedDifference(["B11", "B8"]).rename("NDBI"))
    )
    return with_indices


def build_label_image(aoi):
    world_cover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").clip(aoi)
    return world_cover.remap([80, 50, 40, 10], [1, 2, 3, 4]).rename("label")


def build_samples(aoi, with_indices, label_img):
    samples = (
        with_indices.select(BANDS)
        .addBands(label_img)
        .stratifiedSample(
            numPoints=POINTS_PER_CLASS,
            classBand="label",
            region=aoi,
            scale=SAMPLE_SCALE,
            seed=RANDOM_SEED,
            tileScale=4,
            geometries=True,  # need coordinates to locate each point in the local raster
        )
        .randomColumn("random", RANDOM_SEED)
    )
    return samples


def fetch_dataframe(fc):
    features = fc.getInfo()["features"]
    rows = []
    for f in features:
        row = dict(f["properties"])
        lon, lat = f["geometry"]["coordinates"]
        row["lon"], row["lat"] = lon, lat
        rows.append(row)
    return pd.DataFrame(rows)


def download_image(image, aoi, path, scale):
    url = image.getDownloadURL({"region": aoi, "scale": scale, "crs": OUTPUT_CRS, "format": "GEO_TIFF"})
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)


def build_cnn2d(patch_size, n_bands, n_classes):
    # No BatchNormalization -- see S2OSM-CNN.py for why (collapses val
    # accuracy on this small a dataset).
    model = keras.Sequential(
        [
            keras.Input(shape=(patch_size, patch_size, n_bands)),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def rowcol_for_points(df, crs, transform, height, width):
    xs, ys = warp_transform("EPSG:4326", crs, df["lon"].tolist(), df["lat"].tolist())
    inv = ~transform
    cols, rows = inv * (np.array(xs), np.array(ys))
    cols = np.clip(np.floor(cols).astype(int), 0, width - 1)
    rows = np.clip(np.floor(rows).astype(int), 0, height - 1)
    return rows, cols


def main():
    ee_init()

    aoi = ee.Geometry.Rectangle(AOI_RECT)
    with_indices = build_composite(aoi)
    label_img = build_label_image(aoi)
    samples = build_samples(aoi, with_indices, label_img)

    df = fetch_dataframe(samples)
    print(f"Fetched {len(df)} sample points, per class: {df['label'].value_counts().to_dict()}")

    download_image(with_indices.select(BANDS), aoi, OUTPUT_COMPOSITE_TIF, EXPORT_SCALE)
    print(f"Downloaded {OUTPUT_COMPOSITE_TIF}")

    with rasterio.open(OUTPUT_COMPOSITE_TIF) as src:
        composite = src.read()  # (bands, H, W)
        crs, transform = src.crs, src.transform
        H, W = src.height, src.width

    rows, cols = rowcol_for_points(df, crs, transform, H, W)

    half = PATCH_SIZE // 2
    padded = np.pad(composite, ((0, 0), (half, half), (half, half)), mode="reflect")
    patches = np.stack([padded[:, r:r + PATCH_SIZE, c:c + PATCH_SIZE] for r, c in zip(rows, cols)])
    patches = np.moveaxis(patches, 1, -1)  # (N, patch, patch, bands)

    # keras expects 0-indexed integer classes
    labels0 = df["label"].values.astype(int) - 1

    train_mask = df["random"].values < 0.7
    test_mask = ~train_mask
    print(f"train: {train_mask.sum()}  test: {test_mask.sum()}")

    X_train_raw, X_test_raw = patches[train_mask], patches[test_mask]
    y_train, y_test = labels0[train_mask], labels0[test_mask]

    band_mean = X_train_raw.reshape(-1, len(BANDS)).mean(axis=0)
    band_std = X_train_raw.reshape(-1, len(BANDS)).std(axis=0)
    band_std[band_std == 0] = 1

    X_train = (X_train_raw - band_mean) / band_std
    X_test = (X_test_raw - band_mean) / band_std

    model = build_cnn2d(PATCH_SIZE, len(BANDS), len(CLASS_NAMES))
    model.summary()
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=60,
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", mode="max", patience=15, restore_best_weights=True
            )
        ],
        verbose=2,
    )
    model.save(OUTPUT_MODEL)
    print(f"Saved {OUTPUT_MODEL}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    labels = list(range(len(CLASS_NAMES)))

    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred, labels=labels))
    print("Overall accuracy:", accuracy_score(y_test, y_pred))
    print("Kappa:", cohen_kappa_score(y_test, y_pred))
    print(
        classification_report(
            y_test, y_pred, labels=labels,
            target_names=[CLASS_NAMES[i + 1] for i in labels], zero_division=0,
        )
    )

    # --- Classify the whole AOI, in row-chunks to keep memory bounded ---
    full_padded = np.pad(composite, ((0, 0), (half, half), (half, half)), mode="reflect")
    full_padded = (full_padded - band_mean[:, None, None]) / band_std[:, None, None]

    classified = np.zeros((H, W), dtype=np.uint8)
    CHUNK = 50
    for r0 in range(0, H, CHUNK):
        r1 = min(r0 + CHUNK, H)
        chunk_patches = []
        for r in range(r0, r1):
            for c in range(W):
                chunk_patches.append(full_padded[:, r:r + PATCH_SIZE, c:c + PATCH_SIZE])
        chunk_arr = np.moveaxis(np.stack(chunk_patches), 1, -1)
        pred = np.argmax(model.predict(chunk_arr, batch_size=2048, verbose=0), axis=1) + 1
        classified[r0:r1, :] = pred.reshape(r1 - r0, W)
        print(f"  classified rows {r0}:{r1}")

    with rasterio.open(OUTPUT_COMPOSITE_TIF) as src:
        profile = src.profile
    out_profile = profile.copy()
    out_profile.update(count=1, dtype="uint8", compress="deflate")
    with rasterio.open(OUTPUT_CLASSIFIED_TIF, "w", **out_profile) as dst:
        dst.write(classified, 1)
    print(f"Wrote {OUTPUT_CLASSIFIED_TIF}")


if __name__ == "__main__":
    main()
