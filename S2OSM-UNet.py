"""
Train a U-Net semantic-segmentation model for land-cover classification and
compare it against CNN-1D (S2OSM-CNN.py) and CNN-2D (S2OSM-CNN2D.py).

Unlike the point-based CNN-1D/CNN-2D classifiers, U-Net is trained on random
image patches with DENSE per-pixel supervision (every pixel in the patch has
a label, not just the center one) using the full ESA WorldCover label raster
-- this is the training regime U-Net's encoder-decoder + skip-connection
architecture is actually designed for. For a fair final comparison, it's
still evaluated on the SAME held-out sample points used by the other two
scripts (same AOI, same stratifiedSample seed, same 70/30 split).

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

SAMPLE_SCALE = 10
EXPORT_SCALE = 30
POINTS_PER_CLASS = 500
RANDOM_SEED = 42
OUTPUT_CRS = "EPSG:32647"

TRAIN_PATCH = 64        # random-crop patch size used for training
N_TRAIN_PATCHES = 400   # number of random patches sampled across the AOI

OUTPUT_MODEL = "S2OSM-UNet.keras"
OUTPUT_COMPOSITE_TIF = "S2OSM-Composite.tif"
OUTPUT_LABEL_TIF = "S2OSM-Label.tif"
OUTPUT_CLASSIFIED_TIF = "S2OSM-UNet-Classified.tif"


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

    return (
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


def build_label_image(aoi):
    world_cover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").clip(aoi)
    return world_cover.remap([80, 50, 40, 10], [1, 2, 3, 4]).rename("label")


def build_samples(aoi, with_indices, label_img):
    return (
        with_indices.select(BANDS)
        .addBands(label_img)
        .stratifiedSample(
            numPoints=POINTS_PER_CLASS, classBand="label", region=aoi, scale=SAMPLE_SCALE,
            seed=RANDOM_SEED, tileScale=4, geometries=True,
        )
        .randomColumn("random", RANDOM_SEED)
    )


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


def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x


def build_unet(n_bands, n_classes):
    # Fully convolutional: (None, None, n_bands) lets the SAME trained model
    # run on 64x64 training patches AND the full AOI raster at inference.
    # No BatchNormalization -- see S2OSM-CNN.py for why.
    inputs = keras.Input(shape=(None, None, n_bands))
    c1 = conv_block(inputs, 16)
    p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 32)
    p2 = layers.MaxPooling2D()(c2)
    bottleneck = conv_block(p2, 64)
    u2 = layers.Concatenate()([layers.UpSampling2D()(bottleneck), c2])
    d2 = conv_block(u2, 32)
    u1 = layers.Concatenate()([layers.UpSampling2D()(d2), c1])
    d1 = conv_block(u1, 16)
    outputs = layers.Conv2D(n_classes, 1, activation="softmax")(d1)
    model = keras.Model(inputs, outputs)
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
    print(f"Fetched {len(df)} sample points for evaluation, per class: {df['label'].value_counts().to_dict()}")

    download_image(with_indices.select(BANDS), aoi, OUTPUT_COMPOSITE_TIF, EXPORT_SCALE)
    download_image(label_img, aoi, OUTPUT_LABEL_TIF, EXPORT_SCALE)
    print(f"Downloaded {OUTPUT_COMPOSITE_TIF} and {OUTPUT_LABEL_TIF}")

    with rasterio.open(OUTPUT_COMPOSITE_TIF) as src:
        composite = src.read()  # (bands, H, W)
        crs, transform = src.crs, src.transform
        H, W = src.height, src.width
        profile = src.profile

    with rasterio.open(OUTPUT_LABEL_TIF) as src:
        label_raster = src.read(1)

    band_mean = composite.reshape(len(BANDS), -1).mean(axis=1)
    band_std = composite.reshape(len(BANDS), -1).std(axis=1)
    band_std[band_std == 0] = 1
    composite_norm = (composite - band_mean[:, None, None]) / band_std[:, None, None]

    label0 = label_raster.astype(int) - 1  # 0..3 (assumes full WorldCover coverage, see notes in README)

    # --- Dense-supervision training patches: random crops across the AOI ---
    rng = np.random.default_rng(RANDOM_SEED)

    def sample_patches(n):
        rr = rng.integers(0, H - TRAIN_PATCH, size=n)
        cc = rng.integers(0, W - TRAIN_PATCH, size=n)
        X = np.stack([composite_norm[:, r:r + TRAIN_PATCH, c:c + TRAIN_PATCH] for r, c in zip(rr, cc)])
        Y = np.stack([label0[r:r + TRAIN_PATCH, c:c + TRAIN_PATCH] for r, c in zip(rr, cc)])
        return np.moveaxis(X, 1, -1), Y

    X_all, Y_all = sample_patches(N_TRAIN_PATCHES)
    n_val = int(N_TRAIN_PATCHES * 0.2)
    X_train, Y_train = X_all[n_val:], Y_all[n_val:]
    X_val, Y_val = X_all[:n_val], Y_all[:n_val]
    print(f"train patches: {X_train.shape}  val patches: {X_val.shape}")

    model = build_unet(len(BANDS), len(CLASS_NAMES))
    model.summary()
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=60, batch_size=8,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", mode="max", patience=15, restore_best_weights=True
            )
        ],
        verbose=2,
    )
    model.save(OUTPUT_MODEL)
    print(f"Saved {OUTPUT_MODEL}")

    # --- Full-image inference (fully convolutional, single forward pass) ---
    pad_h, pad_w = (-H) % 4, (-W) % 4
    full_in = np.pad(composite_norm, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    full_in = np.moveaxis(full_in, 0, -1)[None, ...]
    full_pred = model.predict(full_in, batch_size=1)[0]
    classified0 = np.argmax(full_pred, axis=-1)[:H, :W]
    classified = (classified0 + 1).astype(np.uint8)

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="uint8", compress="deflate")
    with rasterio.open(OUTPUT_CLASSIFIED_TIF, "w", **out_profile) as dst:
        dst.write(classified, 1)
    print(f"Wrote {OUTPUT_CLASSIFIED_TIF}")

    # --- Evaluate at the same held-out sample points as CNN-1D / CNN-2D ---
    rows, cols = rowcol_for_points(df, crs, transform, H, W)
    raster_label0 = label_raster[rows, cols].astype(int) - 1

    test_mask = df["random"].values >= 0.7
    y_test = raster_label0[test_mask]
    y_pred = classified0[rows[test_mask], cols[test_mask]]
    labels = list(range(len(CLASS_NAMES)))

    print(f"train points: {(~test_mask).sum()}  test points: {test_mask.sum()}")
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


if __name__ == "__main__":
    main()
