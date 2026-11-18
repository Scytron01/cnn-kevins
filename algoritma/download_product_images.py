"""Download a small, browsable JPG gallery from an image-based Kaggle dataset."""
from __future__ import annotations
import argparse, json, re, shutil
from pathlib import Path
import pandas as pd
import kagglehub

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DEFAULT = ROOT / "dataset" / "images"
DATASET_HANDLE = "paramaggarwal/fashion-product-images-small"
CLASS_MAP_PATH = ROOT / "algoritma" / "class_map.json"

def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")

def locate_dataset(root: Path):
    csvs = list(root.rglob("styles.csv"))
    if not csvs:
        raise FileNotFoundError("styles.csv tidak ditemukan pada hasil download Kaggle.")
    csv_path = csvs[0]
    image_dirs = [p for p in csv_path.parent.rglob("images") if p.is_dir()]
    if not image_dirs:
        raise FileNotFoundError("Folder images/ tidak ditemukan pada dataset Kaggle.")
    return csv_path, image_dirs[0]

def main():
    parser = argparse.ArgumentParser(description="Buat galeri JPG dataset fashion.")
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error("--per-class minimal 1")

    with CLASS_MAP_PATH.open(encoding="utf-8") as f:
        class_map = json.load(f)["article_type_to_class"]
    class_names = ["Baju", "Celana", "Jaket", "Celana Dalam", "Kaus Dalam"]
    downloaded = Path(kagglehub.dataset_download(DATASET_HANDLE))
    csv_path, image_dir = locate_dataset(downloaded)
    frame = pd.read_csv(csv_path)
    if not {"id", "articleType"}.issubset(frame.columns):
        raise ValueError("styles.csv harus memiliki kolom id dan articleType")

    output = args.output.resolve()
    if args.force and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    selected_ids = set()

    for article_type, class_id in class_map.items():
        subset = frame.loc[frame["articleType"].eq(article_type)].sort_values("id")
        class_dir = output / f"{class_id:02d}-{slug(class_names[class_id])}"
        class_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for row in subset.itertuples(index=False):
            product_id = str(getattr(row, "id"))
            if product_id in selected_ids:
                continue
            source = image_dir / f"{product_id}.jpg"
            if not source.is_file():
                continue
            destination = class_dir / source.name
            if not destination.exists():
                shutil.copy2(source, destination)
            selected_ids.add(product_id)
            records.append({"image": destination.relative_to(output).as_posix(),
                            "product_id": product_id, "article_type": article_type,
                            "class_id": class_id, "class_name": class_names[class_id],
                            "source_dataset": DATASET_HANDLE})
            count += 1
            if count >= args.per_class:
                break

    pd.DataFrame(records).to_csv(output / "manifest.csv", index=False)
    print(f"Galeri JPG: {output}")
    print(f"Jumlah JPG: {len(records)}")
    if not records:
        raise RuntimeError("Tidak ada JPG disalin. Periksa akses dan struktur dataset Kaggle.")

if __name__ == "__main__":
    main()
