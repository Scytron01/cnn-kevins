from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from algoritma.preprocessing import preprocess_bgr

ROOT = Path(__file__).resolve().parents[1]
HANDLE = "shivamb/fashion-clothing-products-catalog"
RAW = ROOT / "dataset" / "raw" / "fashion-clothing-products-catalog"
OUT = ROOT / "dataset" / "processed"
MAP = ROOT / "algoritma" / "class_map.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

ID_COLUMNS = ("id", "productid", "product_id", "product id", "itemid", "articleid", "styleid")
TYPE_COLUMNS = ("articletype", "article_type", "producttype", "product_type",
                "garmenttype", "garment_type", "subcategory", "sub_category", "category")


def norm(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def compact(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def norm_id(value: object) -> str:
    text = re.sub(r"\.0$", "", str(value).strip())
    return str(int(text)) if text.isdigit() else text


def find_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lookup = {compact(column): column for column in frame.columns}
    for alias in aliases:
        if compact(alias) in lookup:
            return lookup[compact(alias)]
    return None


def get_map(path: Path):
    config = json.loads(path.read_text(encoding="utf-8"))
    names = config["class_names"]
    mapping = {norm(k): int(v) for k, v in config["article_type_to_class"].items()}
    if len(names) != 5 or set(mapping.values()) - set(range(5)):
        raise ValueError("class_map.json harus memetakan tepat lima kelas.")
    return names, mapping


def locate_catalog(root: Path):
    csv_path, frame, id_col, type_col = None, None, None, None
    for candidate in sorted(root.rglob("*.csv"),
                            key=lambda p: (p.name.lower() not in {"styles.csv", "products.csv"}, len(p.parts))):
        try:
            sample = pd.read_csv(candidate, nrows=5)
        except Exception:
            continue
        candidate_id = find_column(sample, ID_COLUMNS)
        candidate_type = find_column(sample, TYPE_COLUMNS)
        if candidate_id is not None and candidate_type is not None:
            csv_path = candidate
            frame = pd.read_csv(candidate, low_memory=False)
            id_col = find_column(frame, ID_COLUMNS)
            type_col = find_column(frame, TYPE_COLUMNS)
            break
    if frame is None or id_col is None or type_col is None:
        raise FileNotFoundError(
            "CSV katalog tidak ditemukan. Diperlukan kolom ID produk (mis. id/ProductId) "
            "dan tipe/kategori pakaian (mis. articleType/ProductType/category). "
            "Periksa struktur arsip Kaggle dan sesuaikan ID_COLUMNS/TYPE_COLUMNS bila perlu."
        )

    image_dirs = [p for p in root.rglob("*")
                  if p.is_dir() and p.name.lower() in {"images", "image", "images_small", "product_images"}]
    image_dir = image_dirs[0] if image_dirs else root
    return csv_path, image_dir, frame, id_col, type_col


def build_image_index(folder: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        aliases = {path.stem}
        if path.stem.isdigit():
            aliases.add(str(int(path.stem)))
        for alias in aliases:
            if alias not in index or path.suffix.lower() == ".jpg":
                index[alias] = path
    return index


def materialize_dataset(source_root: Path, raw_dir: Path, class_map: Path,
                        max_per_class: int, min_width: int, seed: int,
                        overwrite: bool = False) -> Path:
    """Copy a class-balanced subset of actual Kaggle product photos into dataset/raw."""
    if (raw_dir / "styles.csv").exists() and not overwrite:
        print("[RAW] Menggunakan subset lokal yang sudah ada:", raw_dir)
        return raw_dir

    names, mapping = get_map(class_map)
    csv_path, source_images, frame, id_col, type_col = locate_catalog(source_root)
    frame = frame.copy()
    frame["_source_id"] = frame[id_col].map(norm_id)
    frame["_source_type"] = frame[type_col].map(lambda value: str(value).strip())
    frame["_class_id"] = frame[type_col].map(lambda value: mapping.get(norm(value)))
    mapped = frame[frame["_class_id"].notna()].copy()
    mapped["_class_id"] = mapped["_class_id"].astype(int)
    if mapped.empty:
        available = frame[type_col].dropna().astype(str).value_counts().head(40).to_dict()
        raise RuntimeError(
            "Tidak ada kategori pada kolom '" + str(type_col) +
            "' yang cocok dengan algoritma/class_map.json. Kategori yang ditemukan: " +
            json.dumps(available, ensure_ascii=False)
        )

    source_index = build_image_index(source_images)
    if raw_dir.exists() and overwrite:
        shutil.rmtree(raw_dir)
    images_out = raw_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    selected = []
    skipped = Counter()
    for class_id in range(5):
        group = mapped[mapped["_class_id"] == class_id]
        if not group.empty:
            group = group.sample(frac=1.0, random_state=seed + class_id)
        copied = 0
        for _, row in group.iterrows():
            if copied >= max_per_class:
                break
            source_id = row["_source_id"]
            image_path = source_index.get(source_id)
            if image_path is None:
                skipped["image_missing"] += 1
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                skipped["image_unreadable"] += 1
                continue
            height, width = image.shape[:2]
            if min(width, height) < min_width:
                skipped["below_min_resolution"] += 1
                continue
            filename = source_id + image_path.suffix.lower()
            shutil.copy2(image_path, images_out / filename)
            record = row.drop(labels=["_source_id", "_source_type", "_class_id"]).to_dict()
            record["id"] = source_id
            record["articleType"] = row["_source_type"]
            selected.append((record, class_id, filename, width, height))
            copied += 1

    if not selected:
        raise RuntimeError("Tidak ada gambar dari lima kelas target yang berhasil disalin.")

    styles = pd.DataFrame([item[0] for item in selected])
    styles.to_csv(raw_dir / "styles.csv", index=False)
    manifest = pd.DataFrame([
        {"image_id": item[0]["id"], "articleType": item[0]["articleType"],
         "class_id": item[1], "class_name": names[item[1]], "image_file": item[2],
         "width": item[3], "height": item[4]}
        for item in selected
    ])
    manifest.to_csv(raw_dir / "manifest.csv", index=False)
    report = {
        "source_dataset": HANDLE,
        "source_catalog": str(csv_path),
        "source_images": str(source_images),
        "class_counts": {names[i]: int((manifest["class_id"] == i).sum()) for i in range(5)},
        "skipped": dict(skipped),
        "note": "Curated local subset of source images; not redistribution of the full Kaggle archive."
    }
    (raw_dir / "download_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[RAW] Gambar Kaggle disalin ke:", images_out)
    print("[RAW] Distribusi:", report["class_counts"])
    return raw_dir


def build_dataset(root: Path, out_path: Path, class_map: Path,
                  max_per_class: int = 1000, min_per_class: int = 50,
                  min_width: int = 32):
    """Build the NPZ from an extracted catalog folder. Kept callable for unit tests."""
    names, mapping = get_map(class_map)
    csv_path, images_dir, frame, id_col, type_col = locate_catalog(root)
    frame = frame.copy()
    frame["_id"] = frame[id_col].map(norm_id)
    frame["_type_norm"] = frame[type_col].map(norm)
    frame["_class"] = frame["_type_norm"].map(mapping)
    mapped = frame[frame["_class"].notna()].copy()
    mapped["_class"] = mapped["_class"].astype(int)
    image_index = build_image_index(images_dir)

    masks, geometries, labels, ids, article_types, methods = [], [], [], [], [], []
    accepted, skipped = Counter(), Counter()
    mapped = mapped.sort_values(["_class", "_id"], kind="stable")
    for _, row in mapped.iterrows():
        class_id = int(row["_class"])
        if accepted[class_id] >= max_per_class:
            continue
        image_path = image_index.get(row["_id"])
        if image_path is None:
            skipped["image_missing"] += 1
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            skipped["image_unreadable"] += 1
            continue
        height, width = image.shape[:2]
        if min(width, height) < min_width:
            skipped["below_min_resolution"] += 1
            continue
        result = preprocess_bgr(image)
        if result is None:
            skipped["mask_failed"] += 1
            continue
        masks.append(result.mask)
        geometries.append(result.geometry)
        labels.append(class_id)
        ids.append(row["_id"])
        article_types.append(str(row[type_col]))
        methods.append(result.method)
        accepted[class_id] += 1

    counts = {names[i]: int(accepted[i]) for i in range(5)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "source_dataset": HANDLE,
        "catalog_csv": str(csv_path),
        "images_dir": str(images_dir),
        "class_distribution_after_segmentation": counts,
        "mapped_article_type_counts": {
            str(k): int(v) for k, v in mapped[type_col].value_counts().items()},
        "unmapped_type_examples": {
            str(k): int(v) for k, v in frame.loc[frame["_class"].isna(), type_col].value_counts().head(30).items()},
        "mask_methods": dict(Counter(methods)),
        "skipped": dict(skipped)
    }
    out_path.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([{"class_id": i, "class_name": names[i], "processed": accepted[i]}
                  for i in range(5)]).to_csv(out_path.with_suffix(".distribution.csv"), index=False)
    missing = [names[i] for i in range(5) if accepted[i] < min_per_class]
    if missing:
        raise RuntimeError(
            f"Sampel hasil segmentasi kurang dari {min_per_class} untuk: {missing}. "
            f"Periksa {out_path.with_suffix('.report.json')} dan class_map.json."
        )

    np.savez_compressed(
        out_path,
        X_mask=np.asarray(masks, dtype=np.uint8)[:, None, :, :],
        X_geom=np.asarray(geometries, dtype=np.float32),
        y=np.asarray(labels, dtype=np.int64),
        class_names=np.asarray(names),
        image_ids=np.asarray(ids),
        article_types=np.asarray(article_types),
        mask_methods=np.asarray(methods)
    )
    pd.DataFrame({
        "image_id": ids, "articleType": article_types, "class_id": labels,
        "class_name": [names[value] for value in labels], "mask_method": methods
    }).to_csv(out_path.with_suffix(".manifest.csv"), index=False)
    print("[PROCESSED]", counts, "samples:", len(labels), "saved:", out_path)
    return out_path


def make_preview(npz_path: Path, preview_path: Path, samples_per_class: int = 8):
    archive = np.load(npz_path, allow_pickle=False)
    masks, labels = archive["X_mask"][:, 0], archive["y"]
    names = archive["class_names"].tolist()
    rows = []
    for class_id, class_name in enumerate(names):
        tiles = []
        for index in np.where(labels == class_id)[0][:samples_per_class]:
            tile = cv2.resize(masks[index] * 255, (112, 112), interpolation=cv2.INTER_NEAREST)
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
            cv2.putText(tile, class_name, (3, 105), cv2.FONT_HERSHEY_SIMPLEX, .32, (0, 0, 255), 1)
            tiles.append(tile)
        if tiles:
            while len(tiles) < samples_per_class:
                tiles.append(np.zeros_like(tiles[0]))
            rows.append(np.hstack(tiles))
    if rows:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_path), np.vstack(rows))


def main():
    parser = argparse.ArgumentParser(description=f"Download and prepare Kaggle {HANDLE}")
    parser.add_argument("--dataset-dir", default=None, help="Folder katalog yang sudah diekstrak")
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--class-map", type=Path, default=MAP)
    parser.add_argument("--max-per-class", type=int, default=1000)
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--min-width", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-raw", action="store_true")
    args = parser.parse_args()

    if args.dataset_dir:
        source = Path(args.dataset_dir).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(source)
    else:
        try:
            import kagglehub
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt first to enable KaggleHub.") from exc
        print("[KAGGLE] Downloading", HANDLE)
        source = Path(kagglehub.dataset_download(HANDLE)).resolve()

    raw_root = materialize_dataset(
        source, args.raw_dir, args.class_map, args.max_per_class,
        args.min_width, args.seed, overwrite=args.overwrite_raw)
    out_path = args.out_dir / "laundry64.npz"
    build_dataset(raw_root, out_path, args.class_map,
                  args.max_per_class, args.min_per_class, args.min_width)
    make_preview(out_path, args.out_dir / "mask_preview.png")
    print("[DONE] Raw images:", args.raw_dir / "images")
    print("[DONE] Processed dataset:", out_path)
    print("[DONE] Silhouette preview:", args.out_dir / "mask_preview.png")


if __name__ == "__main__":
    main()
