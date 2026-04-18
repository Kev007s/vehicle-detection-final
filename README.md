# 🚗 Vehicle Detection using YOLOv8 vs Faster R-CNN

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-red.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Final Project for COMP8044041 - Deep Learning and Its Applications**  
> Bina Nusantara University, Graduate Program


## 🎯 Project Overview

Proyek ini mengimplementasikan dan membandingkan **dua arsitektur deep learning** untuk deteksi kendaraan:

| Model | Tipe | Keunggulan |
|-------|------|------------|
| **YOLOv8** | One-stage detector | Cepat, real-time, akurasi tinggi |
| **Faster R-CNN** | Two-stage detector | Akurat, robust untuk objek kecil |

Model dilatih pada **Kaggle Car Object Detection** dataset (1.001 gambar latih, 175 gambar uji) untuk mendeteksi mobil dalam berbagai kondisi lalu lintas termasuk okklusi, variasi skala, dan perubahan pencahayaan.

---

## 🏆 Key Results

| Metrik | YOLOv8 | Faster R-CNN |
|--------|--------|--------------|
| **mAP@0.5** | **97.1%** | 85.0% |
| **mAP@0.5:0.95** | 68.2% | 59.5% |
| **Precision** | 85.2% | 88.0% |
| **Recall** | 82.1% | 82.0% |
| **F1-Score** | 83.6% | 84.9% |
| **FPS (RTX 4060)** | **122.0** | 20.7 |
| **Inference Time** | **8.2 ms** | 48.2 ms |
| **Model Size** | **21.5 MB** | 158 MB |

### 🎉 Pemenang: **YOLOv8**
- ✅ **5.9x lebih cepat** dari Faster R-CNN
- ✅ **12.1% lebih akurat** (mAP@0.5)
- ✅ **7.3x lebih kecil** ukuran model
- ✅ Direkomendasikan untuk aplikasi real-time (CCTV, kendaraan otonom)

---

## 📊 Dataset

### Car Object Detection (Kaggle)
- **Sumber**: [Kaggle - Car Object Detection](https://www.kaggle.com/datasets/sshikamaru/car-object-detection)
- **Gambar latih**: 1.001 gambar dengan bounding box
- **Gambar uji**: 175 gambar
- **Kelas**: Tunggal (car)
- **Format anotasi**: Pascal VOC (xmin, ymin, xmax, ymax)

### Pembagian Data
| Split      | Jumlah Gambar |
|------------|---------------|
| Train      | 248 (70%)     |
| Validation | 53 (15%)      |
| Test       | 53 (15%)      |

### Augmentasi Data (YOLOv8)
| Augmentasi         | Probabilitas  | Tujuan                         |
|--------------------|---------------|--------------------------------|
| Mosaic             | 1.0           | Multi-skala & simulasi okklusi |
| Horizontal Flip    | 0.5           | Variasi orientasi              |
| HSV (Hue/Sat/Val)  | 0.015/0.7/0.4 | Simulasi pencahayaan           |
| Scale              | 0.5           | Variasi ukuran                 |
| Translate          | 0.1           | Variasi posisi                 |

---

## 🧠 Models

### 1. YOLOv8 (Ultralytics)
- **Arsitektur**: CSPDarknet + C2f + PAN-FPN + Decoupled Head
- **Loss**: CIoU + DFL untuk box regression, BCE untuk klasifikasi
- **Parameter**: 3.2M (versi nano)
- **Pre-trained**: COCO dataset

### 2. Faster R-CNN (TorchVision)
- **Arsitektur**: ResNet50 + FPN + RPN + RoI Align
- **Loss**: Smooth L1 untuk box, Cross-entropy untuk class
- **Parameter**: 41.5M
- **Pre-trained**: COCO dataset

---


---

## ⚙️ Setup & Installation

### Prasyarat
- **Python 3.12** atau lebih tinggi
- **NVIDIA GPU** dengan CUDA 12.1+ (direkomendasikan untuk training)
- **RAM 8GB+**
- **Ruangan disk 10GB+**

### Step 1: Clone Repository
```bash
git clone https://github.com/Kev007s/vehicle-detection-yolo-faster-rcnn.git
cd vehicle-detection-yolo-faster-rcnn
Step 2: Buat Virtual Environment
bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
Step 3: Install Dependencies
bash
pip install --upgrade pip
pip install -r requirements.txt
Step 4: Download Dataset
bash
# Menggunakan Kaggle API (butuh kaggle.json)
kaggle datasets download sshikamaru/car-object-detection -p data/raw/ --unzip

# Atau download manual dari:
# https://www.kaggle.com/datasets/sshikamaru/car-object-detection


🚀 How to Run the Models
1. Data Preparation
Jalankan script ini untuk mempersiapkan dataset ke format YOLO:
python scripts/01_prepare_dataset.py


2. Train YOLOv8
Jalankan script training YOLOv8:
python scripts/02_train_yolov8.py

3. Train Faster R-CNN
Jalankan script training Faster R-CNN:
python scripts/03_train_faster_rcnn_simple.py

4. Evaluate & Compare Models
Jalankan script evaluasi untuk membandingkan kedua model:
python scripts/04_fast_evaluation.py

5. Run Inference Demo (Gambar)
Jalankan deteksi pada gambar test:
python scripts/05_inference_demo.py

6. Video Detection Demo (Untuk Presentasi)
Jalankan deteksi pada video (paling mudah):
python scripts/08_simple_demo.py

🎥 Video Demo
Source: https://www.youtube.com/watch?v=K6xsEng2PhU

Dataset
sshikamaru. (2022). Car Object Detection. Kaggle.

Frameworks
PyTorch: https://pytorch.org

Ultralytics: https://github.com/ultralytics/ultralytics

OpenCV: https://opencv.org

Course: COMP8044041 - Deep Learning and Its Applications

Bina Nusantara University, Graduate Program

📞 Quick Commands Reference
Task	Command
Setup environment	python -m venv venv && venv\Scripts\activate
Install dependencies	pip install -r requirements.txt
Prepare dataset	python scripts/01_prepare_dataset.py
Train YOLOv8	python scripts/02_train_yolov8.py
Train Faster R-CNN	python scripts/03_train_faster_rcnn_simple.py
Evaluate models	python scripts/04_fast_evaluation.py
Demo on images	python scripts/05_inference_demo.py
Demo on video	python scripts/08_simple_demo.py

bash
# 1. Aktifkan environment
venv\Scripts\activate

# 2. Persiapan dataset (WAJIB pertama kali)
python scripts/01_prepare_dataset.py

# 3. Training YOLOv8 (opsional - jika ingin train ulang)
python scripts/02_train_yolov8.py

# 4. Training Faster R-CNN (opsional - jika ingin train ulang)
python scripts/03_train_faster_rcnn_simple.py

# 5. Evaluasi dan bandingkan (WAJIB untuk laporan)
python scripts/04_fast_evaluation.py

# 6. Demo deteksi di gambar
python scripts/05_inference_demo.py

# 7. Demo deteksi di video 
python scripts/08_simple_demo.py
Selesai! Sekarang repository GitHub Anda sudah memiliki README.md yang lengkap dan profesional. 


