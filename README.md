<div align="center">

# DeepLearning

**Deep Learning for Earth Observation** — coursework project from the GISTDA training
*"การเรียนรู้เชิงลึกสำหรับข้อมูลสำรวจโลก"*

![Python](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)
![GDAL](https://img.shields.io/badge/GDAL-rasterio-2b7a78)
![Status](https://img.shields.io/badge/status-in%20progress-yellow)
![License](https://img.shields.io/badge/use-educational-lightgrey)

</div>

---

## Overview

This repo tracks hands-on work from the GISTDA *Deep Learning for Earth Observation*
course: preparing very-high-resolution satellite imagery, training CNN / U-Net models
for geospatial classification, and experimenting with pretrained models (SAM, YOLO,
GeoSAM) for feature extraction.

## Project Structure

```
DeepLearning/
├── GoogleMap-Download.py   # Download + georeference a Google Maps satellite AOI
├── requirements.txt        # Python dependencies
├── .vscode/settings.json   # Interpreter + environment config
└── .gitignore
```

## Getting Started

**Prerequisites:** Python 3.13, a virtual environment, and (for the external overview
pyramid step) a local QGIS install.

```bash
git clone https://github.com/porthanit/DeepLearning.git
cd DeepLearning

python -m venv .venv
.venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

## Usage

### Download a satellite image for an AOI

1. Draw a rectangle in **Google Earth Engine → Geometry Tools** and copy the generated
   `ee.Geometry.Polygon(...)` coordinates.
2. Paste the `[lon, lat]` pairs into `AOI_LONLAT` at the top of
   [`GoogleMap-Download.py`](./GoogleMap-Download.py).
3. Run it:

   ```bash
   python GoogleMap-Download.py
   ```

**Output:**

| File | Description |
|---|---|
| `GoogleMap-Images.tif` | Mosaicked satellite image, reprojected to `EPSG:32647` (UTM 47N) |
| `GoogleMap-Images.tif.ovr` | External overview pyramid (2x / 4x / 8x / 16x) for fast display in QGIS |

Adjust `ZOOM` in the script to trade off resolution vs. download size (zoom 20 ≈
0.15 m/px, suited for building-scale extraction).

## Troubleshooting

This environment has some stale system-wide variables left over from an older
PostgreSQL/PostGIS install (`CURL_CA_BUNDLE`, `PROJ_LIB`, `GDAL_DATA`) that can break
TLS verification and CRS lookups for unrelated geospatial tools. `GoogleMap-Download.py`
overrides these at runtime (scoped to its own process only) rather than requiring a
system-wide fix. If a *different* script in this repo hits a `proj.db` or SSL
certificate error, apply the same pattern.

## Roadmap

- [x] Download & georeference high-resolution AOI imagery
- [ ] Train CNN-1D / CNN-2D / U-Net classifiers
- [ ] Building extraction with GeoSAM / YOLO
- [ ] Gaussian Splatting case study (drone imagery → NeRFStudio → SuperSplat)

## Notes on data sources

Satellite tiles are fetched from Google Maps for **educational, non-commercial
coursework use only**. Review Google's terms of service before any other use.

---

<div align="center">

Thanit Nukoolrat · GISTDA

</div>
