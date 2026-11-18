# Smart Laundry Sorter — Kaggle Fashion Clothing Catalog + Cyclone V

**Kevin Kidney Sumual | 101022300012**

Proyek klasifikasi lima jenis pakaian laundry berdasarkan bentuk/siluet piksel. Dataset sumber yang digunakan adalah [Fashion Clothing Products Catalog — Kaggle](https://www.kaggle.com/datasets/shivamb/fashion-clothing-products-catalog/data), handle `shivamb/fashion-clothing-products-catalog`.

Foto katalog diunduh melalui KaggleHub, subset lima kelas disalin ke `dataset/raw/` agar file sumber dapat dilihat secara lokal, lalu diproses menjadi siluet biner 64×64 dan empat fitur geometri. Model Python memakai PyTorch. RTL yang saat ini tersedia adalah geometry extractor Avalon-ST/Avalon-MM, belum CNN accelerator lengkap.

## Kelas

| ID | Kelas laundry | Kaggle `articleType` yang dipetakan |
|---:|---|---|
| 0 | Baju | Shirts, Tshirts, Tops, Kurtas |
| 1 | Celana | Jeans, Trousers, Track Pants, Shorts, Capris |
| 2 | Jaket / Outer | Jackets, Sweaters, Sweatshirts, Blazers |
| 3 | Celana Dalam | Briefs, Boxers, Trunk |
| 4 | Kaus Dalam | Innerwear Vests, Camisoles |

Mapping resmi proyek ada di `algoritma/class_map.json`. Jika salah satu `articleType` tidak ada di salinan dataset Kaggle Anda, skrip mencatat distribusi kategori dan berhenti jika jumlah kelas kurang dari batas minimum. Tidak ada sampel buatan yang ditambahkan untuk mengisi kelas.

## Struktur repository

Nama folder yang telah ada dipertahankan: `Reference/`, `algoritma/`, `dataset/`, `report/`, dan `verilog/`. Folder `software_niosv/` tetap digunakan untuk contoh firmware.

```text
cnn-kevins/
├── Reference/                         # folder lama, dipertahankan
├── algoritma/
│   ├── class_map.json                 # dataset slug + peta lima kelas
│   ├── preprocessing.py               # segmentasi + siluet 64x64 + geometri
│   ├── prepare_dataset.py              # download Kaggle + salin subset ke dataset/raw
│   ├── model.py                       # Shape-CNN 3 convolutional blocks
│   ├── train.py                       # train/validation/test + metrik
│   ├── predict.py                     # prediksi satu gambar
│   └── test_pipeline.py               # unit test
├── dataset/
│   ├── raw/                           # subset gambar Kaggle (lokal setelah download)
│   │   └── fashion-clothing-products-catalog/
│   │       ├── styles.csv
│   │       ├── manifest.csv
│   │       ├── download_report.json
│   │       └── images/
│   └── processed/                     # NPZ, distribusi, manifest, preview siluet
├── report/
│   ├── DATASET_AND_LABELS.md
│   ├── HARDWARE_STATUS.md
│   ├── TOOLCHAIN_COMPATIBILITY.md
│   └── VALIDATION_STATUS.md
├── verilog/
│   ├── laundry_geom_ip.v              # streaming geometry extractor
│   ├── laundry_geom_hw.tcl            # Platform Designer component descriptor
│   ├── tb_laundry_geom.v              # testbench celana
│   ├── run_modelsim.do                # batch simulation script
│   └── run_modelsim.bat               # Windows launcher
├── software_niosv/src/main.c
├── requirements.txt
└── .gitignore
```

> Folder `dataset/raw/` dan `dataset/processed/` disiapkan di GitHub dengan README/placeholder. File gambar dan hasil eksperimen berukuran besar tidak di-commit secara otomatis; jalankan downloader di komputer lokal untuk mengisi folder tersebut.

## 1. Persiapan software Python

Gunakan Python 3.10+.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Siapkan autentikasi Kaggle/KaggleHub sesuai akun Anda. Dataset harus dapat diakses melalui akun tersebut. Kemudian jalankan dari root repository:

```bash
python -m algoritma.prepare_dataset
```

Perintah itu akan:
1. Mengunduh dataset Kaggle `shivamb/fashion-clothing-products-catalog` ke cache KaggleHub.
2. Mencari CSV katalog yang memiliki kolom `id` dan `articleType`, serta folder gambar.
3. Memilih sampai 1.000 gambar per kelas target (seed 42) dan menyalin gambar asli beserta metadata ke `dataset/raw/fashion-clothing-products-catalog/`.
4. Mengabaikan gambar dengan resolusi sumber di bawah 32×32.
5. Membuat mask siluet 64×64, fitur geometri, manifest, laporan distribusi, dan preview.

Output lokal:
- `dataset/raw/fashion-clothing-products-catalog/styles.csv`
- `dataset/raw/fashion-clothing-products-catalog/manifest.csv`
- `dataset/raw/fashion-clothing-products-catalog/images/*`
- `dataset/processed/laundry64.npz`
- `dataset/processed/laundry64.manifest.csv`
- `dataset/processed/laundry64.distribution.csv`
- `dataset/processed/laundry64.report.json`
- `dataset/processed/mask_preview.png`

Untuk dataset yang sudah diunduh manual, letakkan folder katalog di komputer dan jalankan:

```bash
python -m algoritma.prepare_dataset --dataset-dir "D:/datasets/fashion-clothing-products-catalog"
```

Jika ingin mengganti subset yang sudah ada di `dataset/raw/`:

```bash
python -m algoritma.prepare_dataset --overwrite-raw --max-per-class 1000
```

Periksa `dataset/raw/.../manifest.csv`, distribusi kelas, dan terutama `dataset/processed/mask_preview.png` sebelum training. Foto katalog dapat menampilkan model/mannequin, bayangan, atau latar yang membuat segmentasi tidak sempurna. Dataset katalog juga berbeda domain dengan pakaian di conveyor; foto aktual lightbox/conveyor harus digunakan untuk validasi akhir.

## 2. Training dan inferensi CNN

```bash
python -m algoritma.train --epochs 40 --batch-size 32
python -m algoritma.predict "D:/foto/pakaian.jpg"
```

Model: input mask `1×64×64` → Conv1 `1→8` → Conv2 `8→16` → Conv3 `16→32`, masing-masing Conv2D + ReLU + MaxPool 2×2 → Global Average Pooling (32 fitur) → gabung empat fitur geometri → classifier `36→64→5`.

Empat fitur geometri:
1. normalized foreground area,
2. normalized aspect ratio,
3. bounding-box occupancy (bukan convex-hull solidity),
4. proporsi baris di bagian bawah bounding box yang memiliki tepat dua segmen foreground.

Training memakai split stratified 70%/15%/15%, early stopping berdasarkan validation macro-F1, dan menyimpan accuracy, macro-F1, classification report, confusion matrix serta checkpoint di `report/runs/laundry_shape_cnn/`.

## 3. Validasi Python

```bash
python -m unittest discover -s algoritma -p "test_*.py" -v
```

Fixture dummy hanya memeriksa jalur kode dan bukan hasil training dataset Kaggle asli. Jangan menuliskan akurasi model sebelum training dijalankan pada data sumber yang benar.

## 4. Target hardware dan toolchain

Tool yang diminta sebagai target pengembangan:
- **Intel Quartus Prime Lite Edition v25.1+** dengan **Cyclone V Device Support**.
- **ModelSim-Altera v18 atau lebih baru** untuk simulasi RTL.
- **Ashling RiscFree IDE for Altera** untuk firmware Nios V.

Petunjuk simulasi ModelSim tersedia di `verilog/run_modelsim.do` dan `verilog/run_modelsim.bat`. RTL geometri ditulis dalam Verilog sederhana dan tidak memakai primitive vendor; testbench dapat dikompilasi dengan `vlog -2001`. Jalankan perintah dari Command Prompt:

```bat
cd verilog
run_modelsim.bat
```

Atau dari ModelSim Transcript:

```tcl
do run_modelsim.do
```

**Status kesiapan FPGA:** file `laundry_geom_ip.v` merupakan tahap awal geometry extractor, bukan akselerator CNN. Descriptor `laundry_geom_hw.tcl` menyiapkan antarmuka Avalon-ST dan read-only Avalon-MM untuk Platform Designer. Firmware `software_niosv/src/main.c` hanya membaca telemetry geometri; belum mengklasifikasikan lima kelas.

File project Quartus `.qsf/.qpf` dan pin assignment board-spesifik belum dibuat karena nomor board/part Cyclone V belum ditetapkan. Dukungan Nios V pada kombinasi Quartus Lite/device yang dipakai juga harus dikonfirmasi di Platform Designer lokal; jangan menganggap keberadaan RiscFree otomatis berarti CPU IP tersedia untuk semua device/edition. Detail checklist ada di `report/TOOLCHAIN_COMPATIBILITY.md`.

## 5. Roadmap CNN FPGA

1. Bekukan kontrak fixed-point Python: skala input/weight, lebar accumulator, rounding, saturation, padding, dan pooling.
2. Tambahkan QAT atau kalibrasi kuantisasi dan exporter bobot/bias.
3. Implementasikan Conv1–Conv3, ReLU, pooling, dan GAP dalam RTL.
4. Tambahkan register fitur CNN dan FC/ArgMax di Nios V.
5. Buat testbench bit-exact Python vs ModelSim.
6. Bangun project Quartus sesuai part board, lalu ukur ALM/M10K/DSP, timing, dan latency pada FPGA nyata.

Jangan mengklaim hasil sintesis, speedup, timing closure, atau dukungan perangkat sebelum pengujian dilakukan pada versi software dan board yang benar-benar dipakai.
