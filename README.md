<div align="center">

# Deep Learning

**การเรียนรู้เชิงลึกสำหรับข้อมูลสำรวจโลก**
*(Deep Learning for Earth Observation)*

Created and modified by Thanit Nukoolrat and Claude

![Python](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)
![GDAL](https://img.shields.io/badge/GDAL-rasterio-2b7a78)
![Status](https://img.shields.io/badge/status-in%20progress-yellow)
![License](https://img.shields.io/badge/use-educational-lightgrey)

</div>

---

## ภาพรวม

การใช้ Deep Learning สำหรับงานสำรวจระยะไกล โดยทำตั้งแต่การเตรียมภาพถ่ายดาวเทียม
ที่มีความละเอียดสูงมาก การเทรนโมเดล CNN / U-Net สำหรับการจำแนกข้อมูลเชิงพื้นที่
ไปจนถึงการทดลองใช้ Pre-Trained Models (SAM, YOLO, GeoSAM) เพื่อสกัดวัตถุออกจากภาพ

## โครงสร้างโปรเจกต์

```
DeepLearning/
├── GoogleMap-Download.py   # ดาวน์โหลด + ใส่พิกัดให้ภาพถ่ายดาวเทียมจาก Google Maps
├── YOLO-Extraction.py      # สกัดอาคารออกจากภาพด้วยโมเดล YOLOv8 segmentation
├── requirements.txt        # รายการไลบรารีที่ต้องติดตั้ง
├── .vscode/settings.json   # ตั้งค่า Interpreter และ environment
└── .gitignore
```

## เริ่มต้นใช้งาน

**สิ่งที่ต้องมี:** Python 3.13, virtual environment และ (สำหรับขั้นตอนสร้าง
external overview pyramid) เครื่องต้องลง QGIS ไว้ด้วย

```bash
git clone https://github.com/porthanit/DeepLearning.git
cd DeepLearning

python -m venv .venv
.venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

## วิธีใช้งาน

### ดาวน์โหลดภาพถ่ายดาวเทียมตามพื้นที่ที่สนใจ (AOI)

1. วาดสี่เหลี่ยมใน **Google Earth Engine → Geometry Tools** แล้วคัดลอกโค้ด
   `ee.Geometry.Polygon(...)` ที่ได้
2. นำพิกัด `[lon, lat]` ไปวางแทนที่ตัวแปร `AOI_LONLAT` ที่ด้านบนของไฟล์
   [`GoogleMap-Download.py`](./GoogleMap-Download.py)
3. รันสคริปต์:

   ```bash
   python GoogleMap-Download.py
   ```

**ผลลัพธ์ที่ได้:**

| ไฟล์ | รายละเอียด |
|---|---|
| `GoogleMap-Images.tif` | ภาพโมเสกที่ต่อแล้ว reproject เป็น `EPSG:32647` (UTM โซน 47N) |
| `GoogleMap-Images.tif.ovr` | External overview pyramid (2x / 4x / 8x / 16x) ช่วยให้เปิดดูใน QGIS ได้เร็วขึ้น |

ปรับค่า `ZOOM` ในสคริปต์เพื่อแลกระหว่างความละเอียดกับขนาดไฟล์ที่ดาวน์โหลด
(zoom 20 ≈ ความละเอียด 0.15 เมตร/พิกเซล เหมาะกับงานสกัดอาคาร)

### สกัดอาคารออกจากภาพด้วย YOLO

รันหลังจากได้ไฟล์ `GoogleMap-Images.tif` แล้ว:

```bash
python YOLO-Extraction.py
```

สคริปต์จะตัดภาพเป็น tile ขนาด 640x640 (มี overlap กันอาคารถูกตัดขอบ) แล้วรัน
โมเดล [`keremberke/yolov8m-building-segmentation`](https://huggingface.co/keremberke/yolov8m-building-segmentation)
(~55 MB ดาวน์โหลดอัตโนมัติจาก Hugging Face Hub ในการรันครั้งแรก) บน CPU

**ผลลัพธ์ที่ได้:**

| ไฟล์ | รายละเอียด |
|---|---|
| `YOLO-Extraction.tif` | Binary mask ของอาคาร กริด/CRS เดียวกับภาพต้นฉบับ |
| `YOLO-Extraction.geojson` | ขอบเขตอาคารเป็น polygon (CRS เดียวกับภาพต้นฉบับ) |

## แก้ปัญหาที่พบบ่อย

เครื่องนี้มีตัวแปร environment ระดับระบบบางตัวที่ตั้งค้างไว้จากการติดตั้ง
PostgreSQL/PostGIS เวอร์ชันเก่า (`CURL_CA_BUNDLE`, `PROJ_LIB`, `GDAL_DATA`)
ซึ่งอาจทำให้การตรวจสอบ TLS หรือการค้นหาระบบพิกัด (CRS) ของเครื่องมือ
geospatial อื่นๆ พังโดยไม่ทราบสาเหตุ ไฟล์ `GoogleMap-Download.py`
แก้ปัญหานี้ด้วยการ override ค่าตัวแปรเหล่านี้ **เฉพาะใน process ของตัวเองเท่านั้น**
โดยไม่ไปยุ่งกับการตั้งค่าระดับระบบ หากสคริปต์ตัวอื่นในรีโปนี้เจอ error
เกี่ยวกับ `proj.db` หรือ SSL certificate ให้ใช้วิธีแก้แบบเดียวกันนี้ได้

## แผนงานต่อไป

- [x] ดาวน์โหลดและใส่พิกัดให้ภาพถ่ายดาวเทียมความละเอียดสูง
- [ ] เทรนโมเดลจำแนกข้อมูลแบบ CNN-1D / CNN-2D / U-Net
- [x] สกัดอาคารด้วย YOLO
- [ ] สกัดอาคารด้วย GeoSAM
- [ ] กรณีศึกษา Gaussian Splatting (ภาพจากโดรน → NeRFStudio → SuperSplat)

## หมายเหตุเกี่ยวกับแหล่งข้อมูล

ภาพถ่ายดาวเทียมถูกดึงมาจาก Google Maps **เพื่อการศึกษาเท่านั้น ไม่ใช่เชิงพาณิชย์**
หากต้องการนำไปใช้ในลักษณะอื่น ควรตรวจสอบเงื่อนไขการให้บริการ (Terms of Service) ของ Google ก่อน

---

<div align="center">

Thanit Nukoolrat · GISTDA

</div>

---

📍 อยากรู้เรื่อง GIS และข้อมูลเชิงพื้นที่เพิ่มเติม? ติดตามที่ [PORTHA Channel](https://youtu.be/8or7MoUCQHY?list=PLh1MlD0Zdj-B_-GZN4WCCY3BKwhWuD5hH)
