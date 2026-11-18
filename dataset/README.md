# Dataset

Dataset utama adalah Kaggle `paramaggarwal/fashion-product-images-small`. Dataset mentah tidak di-commit. Jalankan `python -m algoritma.prepare_dataset` menggunakan akses KaggleHub, atau unduh manual dan berikan `--dataset-dir` ke folder yang berisi `styles.csv` dan `images/`.

Output lokal berada di `dataset/processed/`: `laundry64.npz`, laporan JSON, distribusi CSV, dan `mask_preview.png`. Periksa preview segmentasi dan distribusi lima kelas sebelum training. Dataset dan hasil olahan diabaikan oleh Git.
