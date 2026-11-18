# Dataset and labels

Source: Kaggle Fashion Clothing Products Catalog, handle `shivamb/fashion-clothing-products-catalog`. The downloader searches recursively for a catalog CSV and an image directory.

Common ID column aliases supported: `id`, `ProductId`, `product_id`, `itemid`, `articleid`, `styleid`. Common clothing category aliases supported: `articleType`, `ProductType`, `product_type`, `garmentType`, `subcategory`, `category`.

| ID | Laundry class | category values currently mapped |
|---:|---|---|
| 0 | Baju | Shirts, Tshirts, Tops, Kurtas |
| 1 | Celana | Jeans, Trousers, Track Pants, Shorts, Capris |
| 2 | Jaket | Jackets, Sweaters, Sweatshirts, Blazers |
| 3 | Celana Dalam | Briefs, Boxers, Trunk |
| 4 | Kaus Dalam | Innerwear Vests, Camisoles |

Mapping is in `algoritma/class_map.json`. Laundry groups are project labels, not original Kaggle labels. The selected category column's values must match a mapping entry. If no values match, the script prints category examples and stops so the mapping can be corrected rather than fabricating samples.

The default downloader selects up to 1,000 source images per class (seeded) and materializes the subset into `dataset/raw/fashion-clothing-products-catalog/`. Source images smaller than 32×32 are skipped. Processed masks (64×64), four geometry features and labels are written to `dataset/processed/laundry64.npz`, alongside manifests, distribution reports and `mask_preview.png`.

Check the mask preview because catalog photos may include models/mannequins, shadows or white backgrounds. Use conveyor/lightbox photos for final system-domain validation.
