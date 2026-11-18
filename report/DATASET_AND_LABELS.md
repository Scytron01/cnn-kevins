# Dataset dan label

Dataset yang dipakai: [Fashion Clothing Products Catalog — Kaggle](https://www.kaggle.com/datasets/shivamb/fashion-clothing-products-catalog/data), handle `shivamb/fashion-clothing-products-catalog`.

Downloader menggunakan KaggleHub dan mengharapkan metadata katalog berisi kolom `id` dan `articleType`, disertai direktori gambar. Pencarian file CSV dan folder gambar dilakukan secara rekursif agar tidak tergantung satu level ekstraksi archive.

| ID | Kelas laundry | `articleType` yang dipetakan |
|---:|---|---|
| 0 | Baju | Shirts, Tshirts, Tops, Kurtas |
| 1 | Celana | Jeans, Trousers, Track Pants, Shorts, Capris |
| 2 | Jaket / Outer | Jackets, Sweaters, Sweatshirts, Blazers |
| 3 | Celana Dalam | Briefs, Boxers, Trunk |
| 4 | Kaus Dalam | Innerwear Vests, Camisoles |

Mapping dapat diedit pada `algoritma/class_map.json`. Karena kelas laundry adalah pengelompokan proyek (bukan label asli Kaggle), periksa isi `articleType` dan laporan distribusi setelah download. Bila dataset tidak menyediakan kategori target atau hasil segmentasi gagal mencapai minimum, skrip berhenti dan melaporkan kelas yang kurang; tidak ada label buatan yang ditambahkan.

Default downloader memilih maksimum 1.000 gambar per kelas dan menyalin subset beserta `styles.csv`, `manifest.csv`, dan `download_report.json` ke `dataset/raw/fashion-clothing-products-catalog/`. Gambar sumber minimum 32×32. Mask hasil preprocessing berukuran 64×64 dan disimpan dalam `dataset/processed/laundry64.npz`.

Periksa `dataset/processed/mask_preview.png`: foto produk dapat berisi model/mannequin, bayangan, dan latar putih. Evaluasi akhir tetap harus memakai foto pakaian di conveyor/lightbox yang akan digunakan pada purwarupa.
