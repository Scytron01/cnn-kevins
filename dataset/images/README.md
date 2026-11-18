# Dataset dan galeri gambar

## Sumber metadata utama

Dataset utama proyek tetap [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small?resource=download)

## Galeri JPG yang bisa dibuka langsung

Untuk memperoleh file foto produk JPG sungguhan, gunakan dataset pendamping [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) (`paramaggarwal/fashion-product-images-small`). Script `algoritma/download_product_images.py` mengunduh sumber tersebut melalui KaggleHub, memilih contoh dari lima kelas proyek, lalu menyalin JPG ke folder ini:

```text
dataset/images/
  00-baju/<product-id>.jpg
  01-celana/<product-id>.jpg
  02-jaket/<product-id>.jpg
  03-celana-dalam/<product-id>.jpg
  04-kaus-dalam/<product-id>.jpg
  manifest.csv
```

Jalankan dari root repository:

```bash
pip install -r requirements.txt
python -m algoritma.download_product_images --per-class 8
```

Default menghasilkan hingga 8 JPG per kelas (maksimal 40 file). Akun Kaggle harus memiliki akses ke dataset. Gunakan `--force` untuk membuat ulang folder galeri.

**Pemisahan sumber:** foto dari dataset pendamping dipakai sebagai galeri JPG dan contoh input file-based. Dataset katalog `shivamb` tetap menjadi sumber metadata utama. Script ini tidak mengubah pipeline training atau mengklaim foto pendamping sebagai file asli dari katalog `shivamb`.
