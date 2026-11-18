# Dataset and labels

Source: Kaggle Fashion Clothing Products Catalog (shivamb/fashion-clothing-products-catalog). The ingestion script locates a CSV with id and articleType columns plus product images.

| ID | Laundry class | articleType |
|---:|---|---|
| 0 | Baju | Shirts, Tshirts, Tops, Kurtas |
| 1 | Celana | Jeans, Trousers, Track Pants, Shorts, Capris |
| 2 | Jaket | Jackets, Sweaters, Sweatshirts, Blazers |
| 3 | Celana Dalam | Briefs, Boxers, Trunk |
| 4 | Kaus Dalam | Innerwear Vests, Camisoles |

Mapping in algoritma/class_map.json. Laundry groups are project labels, not original Kaggle labels. Check class distribution. Script fails if a class has too few samples; it does not fabricate data. Catalog photos may contain mannequins, shadows or white backgrounds; final validation should use conveyor/lightbox photos.
