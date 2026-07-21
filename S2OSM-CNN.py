"""
Train a 1D-CNN land-cover classifier on the same Sentinel-2 + spectral-index
sampling points used by the Random Forest version (S2OSM-RF), then classify
the whole AOI.

Mirrors the GEE JavaScript workflow 1:1:
  1. AOI + Sentinel-2 median composite + spectral indices (same as RF script)
  2. ESA WorldCover-derived stratified samples (same sampling method, same seed)
  3. 70/30 train/test split (same 'random' column, same 0.7 threshold)
  4. Classifier: CNN-1D instead of ee.Classifier.smileRandomForest
  5. Accuracy / kappa / producer's & consumer's accuracy (same metrics as RF)
  6. Classify the whole AOI -> S2OSM-CNN-Classified.tif

Setup:
  - Needs a Google Cloud project with the Earth Engine API enabled. Fill in
    EE_PROJECT below (see slide "Cloud Project ID").
  - First run will open a browser to sign in via `ee.Authenticate()`.

Note on scale: sample points are still extracted at 10 m (SAMPLE_SCALE) since
stratifiedSample() runs server-side and doesn't need a local download. But
this AOI is ~21 x 22 km, so downloading the full composite at 10 m for local
CNN inference would be ~176 MB (10 bands) -- over Earth Engine's direct
getDownloadURL size limit. EXPORT_SCALE=30 keeps the final classified map
download small enough for a direct download; for a full 10 m map, use
ee.batch.Export.image.toDrive with tiling instead.
"""

import os
import sys

os.environ["CURL_CA_BUNDLE"] = ""

import ee
import numpy as np
import pandas as pd
import rasterio
import requests
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

SAMPLE_SCALE = 10   # resolution used for stratifiedSample() (server-side, no download)
EXPORT_SCALE = 30   # resolution used for the local whole-AOI classified GeoTIFF
POINTS_PER_CLASS = 500
RANDOM_SEED = 42
OUTPUT_CRS = "EPSG:32647"

OUTPUT_MODEL = "S2OSM-CNN.keras"
OUTPUT_COMPOSITE_TIF = "S2OSM-Composite.tif"
OUTPUT_CLASSIFIED_TIF = "S2OSM-CNN-Classified.tif"


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
    return s2, with_indices


def build_samples(aoi, with_indices):
    world_cover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").clip(aoi)
    label = world_cover.remap([80, 50, 40, 10], [1, 2, 3, 4]).rename("label")

    samples = (
        with_indices.select(BANDS)
        .addBands(label)
        .stratifiedSample(
            numPoints=POINTS_PER_CLASS,
            classBand="label",
            region=aoi,
            scale=SAMPLE_SCALE,
            seed=RANDOM_SEED,
            tileScale=4,
            geometries=False,
        )
        .randomColumn("random", RANDOM_SEED)
    )
    return samples


def fetch_dataframe(fc):
    features = fc.getInfo()["features"]
    return pd.DataFrame([f["properties"] for f in features])


def download_composite(with_indices, aoi, path):
    url = with_indices.select(BANDS).getDownloadURL(
        {"region": aoi, "scale": EXPORT_SCALE, "crs": OUTPUT_CRS, "format": "GEO_TIFF"}
    )
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)


def build_cnn(n_bands, n_classes):
    # No BatchNormalization: with only ~1400 training samples and few epochs,
    # its running mean/variance don't converge in time, so the moving-average
    # stats used at inference badly mismatch the per-batch stats used during
    # training -- this alone was enough to collapse val accuracy to ~23%.
    model = keras.Sequential(
        [
            keras.Input(shape=(n_bands, 1)),
            layers.Conv1D(16, 3, padding="same", activation="relu"),
            layers.Conv1D(32, 3, padding="same", activation="relu"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    ee_init()

    aoi = ee.Geometry.Rectangle(AOI_RECT)
    s2, with_indices = build_composite(aoi)
    samples = build_samples(aoi, with_indices)

    df = fetch_dataframe(samples)
    print(f"Fetched {len(df)} sample points, per class: {df['label'].value_counts().to_dict()}")

    train_df = df[df["random"] < 0.7]
    test_df = df[df["random"] >= 0.7]
    print(f"train: {len(train_df)}  test: {len(test_df)}")

    # z-score standardization (fit on train only), rather than min-max --
    # more robust for a mix of reflectance bands and [-1, 1] spectral indices
    band_mean = train_df[BANDS].mean()
    band_std = train_df[BANDS].std().replace(0, 1)

    def scale(x):
        return ((x[BANDS] - band_mean) / band_std).values[..., np.newaxis]

    X_train, X_test = scale(train_df), scale(test_df)
    # keras expects 0-indexed integer classes
    y_train = train_df["label"].values.astype(int) - 1
    y_test = test_df["label"].values.astype(int) - 1

    model = build_cnn(n_bands=len(BANDS), n_classes=len(CLASS_NAMES))
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

    # --- Evaluate on test set (same metrics as the RF script) ---
    y_pred = np.argmax(model.predict(X_test), axis=1)
    labels = list(range(len(CLASS_NAMES)))

    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred, labels=labels))
    print("Overall accuracy:", accuracy_score(y_test, y_pred))
    print("Kappa:", cohen_kappa_score(y_test, y_pred))
    # In this report: recall = producer's accuracy, precision = consumer's accuracy
    print(
        classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=[CLASS_NAMES[i + 1] for i in labels],
            zero_division=0,
        )
    )

    # --- Classify the whole AOI ---
    download_composite(with_indices, aoi, OUTPUT_COMPOSITE_TIF)
    print(f"Downloaded {OUTPUT_COMPOSITE_TIF}")

    with rasterio.open(OUTPUT_COMPOSITE_TIF) as src:
        profile = src.profile
        arr = src.read()  # (bands, H, W)

    h, w = arr.shape[1], arr.shape[2]
    pixels = pd.DataFrame(arr.reshape(len(BANDS), -1).T, columns=BANDS)
    valid = np.isfinite(pixels.values).all(axis=1)

    pixels_scaled = scale(pixels)
    pred = np.zeros(len(pixels), dtype=np.uint8)
    pred[valid] = np.argmax(model.predict(pixels_scaled[valid], batch_size=4096), axis=1) + 1

    classified = pred.reshape(h, w)

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="uint8", compress="deflate")
    with rasterio.open(OUTPUT_CLASSIFIED_TIF, "w", **out_profile) as dst:
        dst.write(classified, 1)
    print(f"Wrote {OUTPUT_CLASSIFIED_TIF}")


if __name__ == "__main__":
    main()
