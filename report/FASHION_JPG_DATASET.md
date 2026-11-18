# Dataset JPG untuk Smart Laundry CNN

Sumber foto: [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small). Pipeline menggunakan metadata `styles.csv` (`id`, `articleType`) untuk mengelompokkan foto JPG produk ke dalam lima label proyek.

## Pemetaan label proyek

| ID | Folder kelas | Nilai `articleType` yang digunakan |
|---:|---|---|
| 0 | `Baju` | Shirts, Tshirts, Tops, Kurtas |
| 1 | `Celana` | Jeans, Trousers, Track Pants, Shorts, Capris |
| 2 | `Jaket` | Jackets, Sweaters, Sweatshirts, Blazers |
| 3 | `Celana Dalam` | Briefs, Boxers, Trunk |
| 4 | `Kaus Dalam` | Innerwear Vests, Camisoles |

Pemetaan ini adalah penggabungan kategori katalog ke lima kelas laundry proyek; label tersebut bukan lima label bawaan dataset Kaggle. Kategori yang tidak ada dalam mapping dilewati dan dilaporkan, bukan ditebak.

## Struktur folder yang dihasilkan

```text
dataset/fashion_images/
├── train/
│   ├── Baju/*.jpg
│   ├── Celana/*.jpg
│   ├── Jaket/*.jpg
│   ├── Celana Dalam/*.jpg
│   └── Kaus Dalam/*.jpg
├── valid/
│   └── [lima folder kelas berisi JPG]
└── test/
    └── [lima folder kelas berisi JPG]
```

Split default bersifat stratified per kelas: 70% train, 15% valid, dan 15% test. Seed default adalah 42. Default maksimum 100 JPG yang lolos preprocessing per kelas (hingga 500 foto). Jumlah aktual dapat lebih sedikit bila foto hilang, rusak, resolusinya terlalu kecil, atau segmentasinya gagal.

## Menyiapkan foto dan tensor model

Di PC yang memiliki akses internet dan akses Kaggle ke dataset, dari root repository:

```bash
pip install -r requirements.txt
python -m algoritma.prepare_fashion_images --max-per-class 100
```

KaggleHub akan mengunduh dataset ke cache lokal (login/akses Kaggle harus sudah siap). Bila dataset telah diunduh dan diekstrak manual, gunakan:

```bash
python -m algoritma.prepare_fashion_images --dataset-dir "/path/to/fashion-product-images-small"
```

Untuk mengganti split JPG yang sudah ada:

```bash
python -m algoritma.prepare_fashion_images --max-per-class 100 --overwrite
```

Script menyalin file sumber menjadi JPG per kelas, kemudian menjalankan preprocessing proyek `algoritma.preprocessing.preprocess_bgr()`: segmentasi background, pemilihan komponen utama, letterbox menjadi siluet biner 64×64, dan ekstraksi empat fitur geometri. Dataset hasil untuk training ada di `dataset/processed/`:

- `laundry64_train.npz`, `laundry64_valid.npz`, `laundry64_test.npz`
- manifest CSV per split dan `fashion_dataset_report.json`
- preview JPG kontak sampel per split di `dataset/preview/`

Foto JPG tetap disimpan utuh agar bisa dibuka langsung; NPZ menyimpan mask 64×64, vektor geometri, label, dan metadata gambar. Model tidak dilatih dengan RGB 3-channel, melainkan memakai kontrak model laundry yang telah ada: silhouette 1×64×64 + 4 fitur geometri.

## Training dan evaluasi

```bash
python -m algoritma.train --epochs 40 --batch-size 32
```

Training otomatis membaca NPZ train/valid/test. Checkpoint terbaik dipilih berdasarkan validation macro-F1 dengan early stopping; test set hanya digunakan untuk evaluasi setelah model terbaik dipilih. Hasil ditulis ke `report/runs/laundry_shape_cnn/`:

- `best_model.pt`
- `metrics.json` dan `history.json`
- `confusion_matrix.csv/png`
- `training_curves.png`

## Ekspor bobot ke FPGA

Setelah training berhasil:

```bash
python -m algoritma.export_verilog_weights
```

Kalibrasi exporter memakai `laundry64_train.npz`, bukan validation/test. Berkas ROM `.mem` dan metadata kuantisasi ditulis ke `verilog/weights/`. Sebelum digunakan pada board, hasil exporter tetap harus diuji dengan testbench bit-exact Python-versus-Verilog; proses export saja bukan bukti RTL sudah cocok.

## Batasan dataset

Foto Kaggle merupakan foto katalog produk, bukan foto pakaian kotor di atas meja/permukaan laundry proyek. Sebagian produk dapat menampilkan model, manekin, bayangan, atau latar studio. Karena itu mask dapat menangkap siluet model/manekin, bukan hanya pakaian. Periksa contact sheet dan hasil mask, lalu lakukan validasi tambahan pada foto pakaian nyata dengan latar dan posisi sesuai cara kerja sistem. Akurasi test dari katalog tidak boleh dianggap sebagai akurasi lapangan.
