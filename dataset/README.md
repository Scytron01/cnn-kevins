# Dataset

Sumber data resmi: [Kaggle — Fashion Clothing Products Catalog](https://www.kaggle.com/datasets/shivamb/fashion-clothing-products-catalog/data), handle `shivamb/fashion-clothing-products-catalog`.

Jalankan dari root repo:

```bash
python -m algoritma.prepare_dataset
```

Downloader akan menarik dataset dengan KaggleHub, kemudian menyalin subset class-balanced (maksimum 1.000 gambar per lima kelas laundry) ke:

```text
dataset/raw/fashion-clothing-products-catalog/
├── styles.csv
├── manifest.csv
├── download_report.json
└── images/
    ├── <product-id>.jpg
    └── ...
```

Hasil preprocessing akan berada di `dataset/processed/` sebagai NPZ 64×64, manifest, distribusi kelas, laporan segmentasi, dan `mask_preview.png`.

**Penting:** gambar Kaggle tidak di-commit ke GitHub secara otomatis karena ukuran dan ketentuan redistribusi dataset perlu dihormati. Sesudah menjalankan downloader, gambar dan metadata sumber akan terlihat di folder `dataset/raw/` pada salinan lokal Anda. Folder ini sengaja di-ignore oleh Git; yang ada di GitHub adalah instruksi dan kode downloader.
