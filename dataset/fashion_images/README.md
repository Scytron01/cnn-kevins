# Fashion JPG dataset untuk Laundry CNN

Sumber gambar: [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small).

Jalankan dari root repository (setelah mengaktifkan akses Kaggle):

```bash
pip install -r requirements.txt
python -m algoritma.prepare_fashion_images --max-per-class 100
```

Hasil foto JPG asli dikategorikan ke folder `train/`, `valid/`, dan `test/` berdasarkan lima label proyek laundry. Rincian kelas, rasio split, preprocessing, training, dan ekspor Verilog: lihat [Dataset JPG untuk Smart Laundry CNN](../../report/FASHION_JPG_DATASET.md).
