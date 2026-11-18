#  Smart Laundry Sorter: Implementasi Hybrid CNN untuk Penyortiran Pakaian Berbasis Morfologi Piksel

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![Microcontroller](https://img.shields.io/badge/Hardware-FPGA-teal.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Kevin Kidney Sumual | 101022300012

Sistem otomatisasi penyortiran pakaian kotor berbasis **Computer Vision** dan **Deep Learning (Convolutional Neural Network)** yang mengklasifikasikan item laundry berdasarkan **bentuk geometri, siluet kontur, dan skala piksel** secara *real-time*.

---

## Daftar Isi
- [Latar Belakang Proyek](#-latar-belakang-proyek)
- [Kelas Klasifikasi Pakaian](#-kelas-klasifikasi-pakaian)
- [Alur Kerja Sistem (End-to-End Pipeline)](#-alur-kerja-sistem-end-to-end-pipeline)
- [Arsitektur Sistem & Hybrid CNN](#-arsitektur-sistem--hybrid-cnn)
- [Struktur Repositori](#-struktur-repositori)
- [Instalasi & Persiapan Lingkungan](#-instalasi--persiapan-lingkungan)
- [Panduan Penggunaan](#-panduan-penggunaan)
- [Strategi Dataset & Augmentasi](#-strategi-dataset--augmentasi)
- [Roadmap Pengembangan](#-roadmap-pengembangan)
- [Lisensi](#-lisensi)

---

## Latar Belakang Proyek

Penyortiran pakaian secara manual di fasilitas laundry komersial maupun industri sering kali memakan waktu lama dan rentan terhadap kesalahan penanganan (seperti pakaian dalam yang terselip, kaus dalam melar akibat tercampur beban berat, atau jaket tebal yang memerlukan siklus pencucian khusus).

Proyek ini menyelesaikan masalah pengenalan objek kain yang bersifat *non-rigid* (mudah terlipat dan berubah bentuk) dengan menggabungkan:
1. **Segmentasi Siluet Biner** untuk mengisolasi bentuk luar pakaian dari latar belakang konveyor.
2. **Letterbox Resizing** untuk mempertahankan rasio aspek asli tanpa distorsi.
3. **Multi-Input Hybrid CNN** yang menggabungkan fitur visual 4-kanal ($\text{RGB} + \text{Mask}$) dengan vektor metadata geometri skala fisik.

---

## Kelas Klasifikasi Pakaian

Model dirancang untuk memisahkan pakaian ke dalam **5 kategori utama** berdasarkan karakteristik morfologi piksel dan penanganan pencucian:

| ID Kelas | Kategori Pakaian | Ciri Geometri & Siluet Piksel | Fitur Struktural Internal | Tindakan Aktuasi / Jalur |
| :---: | :--- | :--- | :--- | :--- |
| **`0`** | **Baju / Atasan** (*Kaus, Kemeja*) | Kontur badan utama dengan tonjolan simetris dua lengan; rasio aspek mendekati $1:1$ hingga $1:1.3$. | Garis kerah (*collar*), lipatan bahu, kancing vertikal, potongan bawah rata. | Keranjang Cuci Reguler |
| **`1`** | **Celana** (*Panjang / Pendek*) | Kontur memanjang (*elongated*) dengan percabangan dua pipa kaki; rasio aspek $1:1.5$ hingga $1:2.5$. | Garis pinggang (*waistband*) tebal, jahitan tengah (*inseam*), saku. | Keranjang Pakaian Bawah |
| **`2`** | **Jaket / Outer** (*Hoodie, Sweater*) | Luas area piksel paling besar dan tebal; siluet lebar dengan lengan penuh dan kerah tinggi/tudung. | Resleting (*zipper*) vertikal, manset karet tebal (*ribbed cuffs*), tali *hoodie*. | Keranjang *Heavy-Duty* |
| **`3`** | **Celana Dalam** (*Briefs, Boxers*) | **Luas area piksel sangat kecil**; bentuk segitiga/trapesium terbalik atau persegi pendek tanpa pipa kaki. | Pita karet pinggang elastis kontras dan lengkungan paha (*leg openings*). | *Mesh Laundry Bag* Khusus |
| **`4`** | **Kaus Dalam** (*Singlet / Tank Top*) | Kontur badan vertikal ramping **tanpa tonjolan lengan** (*sleeveless*). | Potongan leher U/V dalam (*deep scoop neck*) dan dua tali bahu sempit. | Keranjang *Delicates / Gentle* |

---

## Alur Kerja Sistem (End-to-End Pipeline)

1. **Flat-Drop Feeding & IR Trigger**
   Pakaian diletakkan semi-terbentang di atas sabuk konveyor berwarna *matte* netral (kecepatan ideal $0.2 - 0.4\text{ m/s}$). Saat melewati sensor inframerah (*break-beam*), kamera *overhead* di dalam *lightbox* LED ($\text{CRI} > 90$, $5500\text{K}$) mengambil citra secara otomatis.
2. **Segmentasi & Ekstraksi Siluet**
   Algoritma *Background Subtraction* memisahkan objek kain dari permukaan sabuk konveyor untuk menghasilkan citra masker biner hitam-putih.
3. **Ekstraksi Fitur Geometri Skala Fisik**
   Sebelum gambar diubah ukurannya, sistem menghitung parameter fisik aktual dari kontur utama:
   - **Luas Area Piksel ($A$):** Membedakan objek kecil (celana dalam) dengan objek besar (jaket).
   - **Aspect Ratio ($W/H$):** Rasio lebar terhadap tinggi *bounding box* objek.
   - **Solidity ($S = A / A_{\text{convex}}$):** Mengukur tingkat lekukan atau percabangan pipa kaki celana.
4. **Letterbox Resizing**
   Citra *Region of Interest* (ROI) diubah ukurannya menjadi $256 \times 256$ piksel dengan penambahan *padding* hitam pada sisi terpendek agar bentuk asli celana maupun baju tidak terdistorsi.
5. **Inferensi Hybrid CNN & Aktuasi Diverter**
   Model mengeluarkan probabilitas untuk 5 kelas. Jika skor *confidence* $\ge 0.80$, unit komputasi mengirimkan sinyal via UART/GPIO ke mikrokontroler ESP32 untuk menggerakkan lengan pemilah (*diverter*) menuju keranjang yang sesuai.

---

## Arsitektur Sistem & Hybrid CNN

Sistem menggunakan pendekatan **Late Fusion** yang memadukan cabang ekstraksi spasial 4-kanal dengan cabang *Multi-Layer Perceptron* (MLP) untuk skala geometri:

```text
                       [Kamera Overhead + Lightbox]
                                    │
                                    ▼
                     (Frame Citra Asli di Konveyor)
                                    │
                 ┌──────────────────┴──────────────────┐
                 ▼                                     ▼
   [Segmentasi & Letterbox Resize]        [Ekstraksi Fitur Geometri]
    - RGB (3 Kanal) + Masker Biner         - Luas Area Piksel (A)
    - Padding tanpa distorsi               - Aspect Ratio (W/H)
    - Dimensi: 256 x 256 x 4               - Solidity & Perimeter
                 │                                     │
                 ▼                                     ▼
   ┌───────────────────────────┐         ┌───────────────────────────┐
   │    CNN SHAPE BACKBONE     │         │   GEOMETRIC MLP BRANCH    │
   │  - Conv2D + Dilated Conv  │         │  - Dense (16, ReLU)       │
   │  - Spatial Pyramid Pool   │         │  - BatchNormalization     │
   └─────────────┬─────────────┘         └─────────────┬─────────────┘
                 │ (Vektor Fitur Spasial: 256)         │ (Vektor Skala: 16)
                 └──────────────────┬──────────────────┘
                                    ▼
                        [Concatenate Layer (272)]
                                    │
                                    ▼
                      [Dense (128) + Dropout (0.3)]
                                    │
                                    ▼
                         [Softmax Output (5 Kelas)]
         ┌──────────┬──────────┬──────────┬──────────┬──────────┐
         ▼          ▼          ▼          ▼          ▼          ▼
      (Baju)    (Celana)    (Jaket)   (Cel.Dalam) (Kaus Dalam) (Manual)
```

### Komponen Kunci Arsitektur Model:
- **Input Layer 4-Kanal ($256 \times 256 \times 4$):** Menggabungkan citra warna $\text{RGB}$ (untuk detail kerah, resleting, karet pinggang) dengan masker biner (untuk batas tepi luar yang tegas).
- **Dilated Convolutions ($\text{dilation\_rate} = 2$):** Memperluas *receptive field* tanpa menurunkan resolusi spasial secara drastis, membantu jaringan mengenali hubungan dua pipa kaki celana yang terpisah.
- **Spatial Pyramid Pooling (SPP):** Memecah peta fitur akhir ke dalam grid $1 \times 1$, $2 \times 2$, dan $4 \times 4$ secara paralel agar model invarian terhadap variasi ukuran pakaian anak maupun dewasa.
- **Geometric Fusion:** Memastikan objek kecil berbentuk segitiga dengan luas area piksel rendah diklasifikasikan secara akurat sebagai **Celana Dalam** dan tidak tertukar dengan baju terlipat.

---

## Struktur Repositori

```text
smart-laundry-sorter-cnn/
├── configs/
│   └── config.yaml                 # Konfigurasi hyperparameter, threshold, dan port serial
├── dataset/
│   ├── raw/                        # Citra mentah hasil tangkapan kamera konveyor
│   └── processed/                  # Dataset hasil segmentasi masker & metadata CSV
├── firmware/
│   └── esp32_diverter_control/
│       └── esp32_diverter_control.ino  # Kode kontrol sensor IR, motor conveyor, & aktuator
├── models/
│   ├── checkpoints/                # Bobot model terbaik (.h5 / .pt)
│   └── exported/                   # Model terkuantisasi (.onnx / .tflite) untuk Edge AI
├── notebooks/
│   ├── 01_eda_geometry_analysis.ipynb  # Analisis distribusi luas area & aspect ratio
│   └── 02_gradcam_visualization.ipynb  # Validasi area fokus model CNN
├── src/
│   ├── preprocessing/
│   │   ├── segmentation.py         # Background subtraction & ekstraksi masker biner
│   │   └── geometry_extractor.py   # Perhitungan luas piksel, aspect ratio, & solidity
│   ├── model/
│   │   ├── hybrid_cnn.py           # Definisi arsitektur 4-Channel CNN + Geometric MLP
│   │   └── spp_layer.py            # Implementasi Spatial Pyramid Pooling
│   ├── train.py                    # Skrip pelatihan dan evaluasi model
│   └── inference_realtime.py       # Pipeline inferensi kamera + komunikasi serial ESP32
├── requirements.txt
└── README.md
```

---

## Instalasi & Persiapan Lingkungan

1. **Kloning Repositori**
   ```bash
   git clone https://github.com/username/smart-laundry-sorter-cnn.git
   cd smart-laundry-sorter-cnn
   ```

2. **Buat Virtual Environment & Instal Dependensi**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Unggah Firmware ke ESP32**
   Buka berkas `firmware/esp32_diverter_control/esp32_diverter_control.ino` menggunakan Arduino IDE atau PlatformIO, sesuaikan pin GPIO untuk sensor IR dan aktuator servo/pneumatik, lalu unggah ke board ESP32.

---

## Panduan Penggunaan

### 1. Pra-pemrosesan Dataset & Ekstraksi Fitur Geometri
Menghasilkan citra 4-kanal ($\text{RGB} + \text{Mask}$) dan file CSV berisi metadata geometri dari gambar mentah:
```bash
python src/preprocessing/segmentation.py --input_dir dataset/raw --output_dir dataset/processed
```

### 2. Pelatihan Model Hybrid CNN
Melatih model menggunakan konfigurasi yang ditentukan di `configs/config.yaml`:
```bash
python src/train.py --config configs/config.yaml --epochs 50 --batch_size 32
```

### 3. Menjalankan Penyortiran Real-Time
Menjalankan kamera inspeksi, model inferensi, dan pengiriman sinyal aktuasi ke mikrokontroler:
```bash
python src/inference_realtime.py --model models/exported/laundry_hybrid_cnn.onnx --serial_port /dev/ttyUSB0
```

---

## Strategi Dataset & Augmentasi

Untuk mencapai generalisasi tinggi pada kondisi lipatan kain nyata, dataset disusun dengan komposisi berikut:
- **Jumlah Sampel:** Minimal $1.000$ citra per kelas (total $\ge 5.000$ citra) dengan pembagian $70\%$ *Training*, $15\%$ *Validation*, dan $15\%$ *Testing*.
- **Variasi Kondisi Pakaian:** $50\%$ terbentang alami (*flat-lay*), $35\%$ terlipat sebagian (*semi-folded*), dan $15\%$ kusut ringan (*naturally crumpled*).
- **Augmentasi Aman Morfologi:**
  - ✅ *Random Rotation* ($0^\circ - 360^\circ$) dan *Horizontal/Vertical Flip*
  - ✅ *Scale Jitter* ($\pm 15\%$) dengan pembaruan nilai luas piksel proporsional
  - ✅ *Elastic Grid Distortion* halus untuk mensimulasikan kerutan alami kain
  - ❌ **Hindari *Random Cropping*** agar ujung lengan baju atau pipa celana panjang tidak terpotong (yang dapat mengubah label bentuk objek secara keliru).

---

## Roadmap Pengembangan

- [x] Perancangan arsitektur Hybrid 4-Channel CNN + Geometric MLP
- [x] Implementasi modul ekstraksi kontur, *letterbox padding*, dan *solidity*
- [ ] Pengumpulan dataset terstandarisasi di atas *conveyor belt* ($5.000+$ citra)
- [ ] Kuantisasi model ke format INT8/FP16 untuk latensi inferensi $< 35\text{ ms}$ di Raspberry Pi 5
- [ ] Integrasi mekanisme sinkronisasi *rotary encoder* pada konveyor kecepatan variabel
- [ ] Pengembangan *dashboard* monitoring berbasis web untuk statistik *throughput* cucian

---

## 📄 Lisensi

Proyek ini didistribusikan di bawah lisensi **MIT License**. Lihat berkas `LICENSE` untuk informasi lebih lanjut.
