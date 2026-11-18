# Dataset dan label

Dataset yang digunakan: Kaggle `paramaggarwal/fashion-product-images-small` dengan metadata `styles.csv` dan direktori gambar `images/`.

| ID | Kelas laundry | `articleType` |
|---:|---|---|
| 0 | Baju | Shirts, Tshirts, Tops, Kurtas |
| 1 | Celana | Jeans, Trousers, Track Pants, Shorts, Capris |
| 2 | Jaket | Jackets, Sweaters, Sweatshirts, Blazers |
| 3 | Celana Dalam | Briefs, Boxers, Trunk |
| 4 | Kaus Dalam | Innerwear Vests, Camisoles |

Mapping dapat diubah pada `algoritma/class_map.json`. Skrip melaporkan distribusi dan gagal dengan pesan jelas bila sampel kelas tidak mencukupi. Jangan menggandakan sampel atau membuat label palsu.

Foto katalog dapat menampilkan model/mannequin, latar putih, bayangan, dan objek menyentuh tepi. Karena itu segmentasi perlu diperiksa melalui `dataset/processed/mask_preview.png`. Foto katalog juga berbeda domain dari pakaian pada conveyor; kumpulkan foto conveyor/lightbox sebelum klaim penggunaan lapangan.
