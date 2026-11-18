# Smart Laundry Sorter — Kaggle Fashion Clothing Catalog + Cyclone V

**Kevin Kidney Sumual | 101022300012**
Dataset sumber: [Fashion Clothing Products Catalog — Kaggle](https://www.kaggle.com/datasets/shivamb/fashion-clothing-products-catalog/data), handle `shivamb/fashion-clothing-products-catalog`.

## Dataset terlihat di folder dataset

Jalankan `python -m algoritma.prepare_dataset`. Downloader memakai KaggleHub, memilih sampai 1.000 gambar per kelas, kemudian menyalin gambar asli ke `dataset/raw/fashion-clothing-products-catalog/images/` bersama `styles.csv`, `manifest.csv`, dan `download_report.json`. Hasil mask 64×64, empat fitur geometri, manifest, distribusi, laporan dan preview masuk ke `dataset/processed/`.

Foto berukuran besar tidak di-commit otomatis ke GitHub. Folder dan dokumentasi ada di repo; menjalankan downloader akan mengisi dataset sebenarnya pada salinan lokal. Ini menghindari unggah arsip massal tanpa mengecek ukuran dan ketentuan redistribusi.

## Lima kelas

| ID | Kelas | articleType |
|---:|---|---|
| 0 | Baju | Shirts, Tshirts, Tops, Kurtas |
| 1 | Celana | Jeans, Trousers, Track Pants, Shorts, Capris |
| 2 | Jaket | Jackets, Sweaters, Sweatshirts, Blazers |
| 3 | Celana Dalam | Briefs, Boxers, Trunk |
| 4 | Kaus Dalam | Innerwear Vests, Camisoles |

Mapping bisa diedit di `algoritma/class_map.json`. Downloader mencoba nama kolom umum untuk ID produk (`id`, `ProductId`, `product_id`) dan tipe pakaian (`articleType`, `ProductType`, `category`, dan variasinya). Nilai kategori tetap harus cocok dengan pemetaan di `class_map.json`; jika tidak tersedia, program berhenti dengan laporan, bukan membuat sampel palsu.

## Instalasi dan download

Gunakan Python 3.10+.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Siapkan autentikasi Kaggle API/KaggleHub dan pastikan akun mendapat akses ke dataset. Jalankan dari root repo:

```bash
python -m algoritma.prepare_dataset
```

Untuk dataset yang sudah diunduh manual:

```bash
python -m algoritma.prepare_dataset --dataset-dir D:/datasets/fashion-clothing-products-catalog
```

Default: gambar sumber minimal 32×32, mask 64×64, maksimum 1.000 gambar per kelas, minimum 50 gambar lolos segmentasi per kelas. Periksa manifest di dataset/raw, distribusi, dan dataset/processed/mask_preview.png sebelum training.

## Training

```bash
python -m algoritma.train --epochs 40 --batch-size 32
python -m algoritma.predict path/to/clothing.jpg
python -m unittest discover -s algoritma -p test_*.py -v
```

CNN Python memakai tiga convolutional blocks, GAP 32 fitur, empat fitur geometri, lalu classifier 36→64→5. Fitur geometri: normalized area, normalized aspect ratio, bounding-box occupancy, lower two-run score. Foto katalog tidak sama dengan pakaian di conveyor; validasi akhir perlu memakai foto conveyor/lightbox.

## Hardware target

- Quartus Prime Lite Edition v25.1+ dengan Cyclone V Device Support.
- ModelSim-Altera v18 atau lebih baru.
- Ashling RiscFree IDE for Altera.

RTL yang tersedia sekarang adalah **geometry extractor Avalon-ST/Avalon-MM, bukan CNN accelerator penuh**. Firmware Nios V hanya membaca telemetry geometri. Conv1–Conv3 RTL, fixed-point/INT8 exporter, FC/ArgMax, dan uji bit-exact belum selesai.

Untuk ModelSim dari folder verilog jalankan `do run_modelsim.do`. Belum ada .qpf/.qsf/pin assignment karena board dan part number Cyclone V belum diketahui. Pastikan Nios V processor IP tersedia di Platform Designer untuk kombinasi Quartus Lite dan Cyclone V Anda; RiscFree IDE terpasang tidak otomatis membuktikan dukungan IP pada semua family/edition. Detail ada di report/TOOLCHAIN_COMPATIBILITY.md.

Tidak ada klaim akurasi dataset, FPGA synthesis, PPA, speedup, atau timing sebelum hasil eksperimen nyata diperoleh.
