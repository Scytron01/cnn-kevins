# Smart Laundry Sorter — Kaggle Fashion Shape-CNN

**Kevin Kidney Sumual | 101022300012**

Proyek klasifikasi pakaian laundry berdasarkan siluet dari dataset Kaggle **Fashion Product Images Small** (`paramaggarwal/fashion-product-images-small`). Foto katalog diproses menjadi mask biner 64×64 dan empat fitur geometri, kemudian digunakan untuk melatih CNN PyTorch lima kelas.

> **Status jujur implementasi:** pipeline Kaggle, preprocessing, training/evaluasi CNN Python, prediksi satu foto, RTL geometry extractor, testbench, dan contoh firmware Nios V untuk membaca metadata geometri disertakan. RTL saat ini **belum merupakan CNN accelerator penuh**. Conv1–Conv3 RTL, kuantisasi fixed-point/INT8, FC/ArgMax di Nios V, dan verifikasi bit-exact adalah pekerjaan lanjutan.

## Kelas

| ID | Kelas | Kaggle `articleType` |
|---:|---|---|
| 0 | Baju | Shirts, Tshirts, Tops, Kurtas |
| 1 | Celana | Jeans, Trousers, Track Pants, Shorts, Capris |
| 2 | Jaket | Jackets, Sweaters, Sweatshirts, Blazers |
| 3 | Celana Dalam | Briefs, Boxers, Trunk |
| 4 | Kaus Dalam | Innerwear Vests, Camisoles |

Mapping ada di `algoritma/class_map.json`. Kategori yang tersedia bergantung pada salinan dataset. Skrip melaporkan distribusi dan menghentikan proses bila suatu kelas tidak memenuhi minimum; jangan membuat label palsu atau menduplikasi sampel untuk menutup kelas kosong.

## Struktur repository

Nama folder lama dipertahankan: `Reference/`, `algoritma/`, `dataset/`, `report/`, dan `verilog/`. Placeholder lama tidak dihapus. Folder `software_niosv/` ditambahkan untuk firmware.

```text
cnn-kevins/
├── Reference/                    # folder lama, dipertahankan
├── algoritma/
│   ├── class_map.json            # articleType -> 5 kelas
│   ├── preprocessing.py          # segmentasi, letterbox, geometri
│   ├── prepare_dataset.py        # Kaggle -> NPZ
│   ├── model.py                  # CNN tiga convolutional blocks
│   ├── train.py                  # training, validation, test
│   ├── predict.py                # inferensi satu foto
│   └── test_pipeline.py          # unit test
├── dataset/                      # dataset lokal (tidak di-commit)
├── report/                       # dokumentasi dan output eksperimen
├── verilog/
│   ├── laundry_geom_ip.v         # Avalon-ST geometry extractor
│   ├── laundry_geom_hw.tcl       # descriptor Platform Designer
│   └── tb_laundry_geom.v         # testbench siluet celana
├── software_niosv/src/main.c     # telemetry geometri Nios V
├── requirements.txt
└── .gitignore
```

## Arsitektur Python

Input mask `1×64×64` → Conv1 `1→8` → Conv2 `8→16` → Conv3 `16→32`, masing-masing Conv2D + ReLU + MaxPool 2×2 → Global Average Pooling (32 fitur) → gabung empat fitur geometri → classifier `36→64→5`.

Empat fitur: normalized area, normalized aspect ratio, bounding-box occupancy, dan proporsi baris bagian bawah bounding box yang memiliki tepat dua segmen foreground. Fitur ketiga adalah **bounding-box occupancy, bukan convex-hull solidity**; definisi tersebut dipilih agar dapat dihitung ulang dari area dan bounding box hardware.

## Instalasi dan dataset Kaggle

Gunakan Python 3.10+.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Siapkan autentikasi Kaggle/KaggleHub pada komputer Anda, lalu jalankan:

```bash
python -m algoritma.prepare_dataset
```

Alternatif unduh manual: arahkan ke folder berisi `styles.csv` dan `images/`.

```bash
python -m algoritma.prepare_dataset --no-download --dataset-dir "D:/dataset/fashion-product-images-small"
```

Parameter lengkap: `python -m algoritma.prepare_dataset --help`. Default minimum resolusi sumber 32×32, mask 64×64, maksimum 2500 sampel/kelas, minimum 20 sampel/kelas. Output lokal: `dataset/processed/laundry64.npz`, `laundry64.report.json`, `laundry64.distribution.csv`, dan `mask_preview.png`.

**Periksa `mask_preview.png` sebelum training.** Dataset katalog bukan citra pakaian di conveyor; model/mannequin, latar putih, bayangan, dan objek yang menyentuh tepi gambar dapat mengganggu segmentasi. Kumpulkan pula foto conveyor/lightbox untuk validasi domain akhir. Dataset mentah, NPZ, checkpoint, dan hasil eksperimen tidak disimpan dalam Git.

## Training dan prediksi

```bash
python -m algoritma.train --epochs 40 --batch-size 32
python -m algoritma.predict "path/to/clothing.jpg"
```

Training memakai split stratified 70%/15%/15% (train/validation/test), early stopping berdasarkan validation macro-F1, lalu melaporkan test accuracy, macro-F1, classification report, dan confusion matrix. Output ada di `report/runs/laundry_shape_cnn/`.

## Unit test Python

```bash
python -m unittest discover -s algoritma -p "test_*.py" -v
```

Fixture pengujian membuat katalog dummy berformat `styles.csv` + `images/`; fixture tersebut hanya menguji jalur kode, bukan hasil eksperimen dataset Kaggle dan bukan dasar klaim akurasi.

## RTL geometry extractor

`verilog/laundry_geom_ip.v` menerima satu bit masker per beat Avalon-ST. Saat EOP, blok menyelesaikan hitungan baris bagian bawah dengan dua run. Simulasi Icarus:

```bash
iverilog -g2012 -s tb_laundry_geom -o verilog/tb_laundry_geom.out verilog/laundry_geom_ip.v verilog/tb_laundry_geom.v
vvp verilog/tb_laundry_geom.out
```

Peta CSR (word address): 0 status (bit 2 frame done, bit 1 finalizing), 1 area, 2 bounding box `[xmin,xmax,ymin,ymax]`, 3 jumlah baris lower-two-run, 4 lebar/tinggi. Firmware `software_niosv/src/main.c` hanya membaca metadata geometri; belum menjalankan klasifikasi CNN. Hubungkan custom IP di Platform Designer dan pastikan base address muncul sebagai `LAUNDRY_GEOM_IP_0_BASE` di `system.h`.

## Pekerjaan FPGA berikutnya

1. Tetapkan kontrak fixed-point dan Python golden model (skala, lebar accumulator, rounding, saturation, padding, pooling).
2. Implementasikan Conv1–Conv3, ReLU, pooling, dan GAP di RTL.
3. Ekspor bobot/bias ROM dengan skala kuantisasi konsisten.
4. Tambahkan register fitur, FC/ArgMax, serta klasifikasi lima kelas di Nios V.
5. Buat testbench bit-exact Python vs. Verilog.
6. Ukur ALM/M10K/DSP, timing, dan latensi pada board Cyclone V yang dituju.

Poin tersebut adalah roadmap, bukan hasil yang sudah dicapai. Jangan mengklaim akurasi Kaggle, speedup, pemakaian resource, atau timing FPGA sebelum pengukuran aktual dilakukan.
