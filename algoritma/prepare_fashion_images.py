from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from algoritma.preprocessing import preprocess_bgr

ROOT = Path(__file__).resolve().parents[1]
HANDLE = "paramaggarwal/fashion-product-images-small"
DEFAULT_IMAGES = ROOT / "dataset" / "fashion_images"
DEFAULT_PROCESSED = ROOT / "dataset" / "processed"
DEFAULT_PREVIEW = ROOT / "dataset" / "preview"
DEFAULT_CLASS_MAP = ROOT / "algoritma" / "class_map.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def clean_id(value: object) -> str:
    value = str(value).strip()
    if value.endswith(".0") and value[:-2].isdigit():
        value = value[:-2]
    return str(int(value)) if value.isdigit() else value


def find_column(columns, aliases):
    lookup = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in columns}
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]", "", alias.lower())
        if key in lookup:
            return lookup[key]
    return None


def locate_catalog(root: Path):
    csvs = sorted(root.rglob("styles.csv")) or sorted(root.rglob("*.csv"))
    for csv_path in csvs:
        try:
            sample = pd.read_csv(csv_path, nrows=5)
        except Exception:
            continue
        id_col = find_column(sample.columns, ("id", "productId", "product_id", "styleId"))
        type_col = find_column(sample.columns, ("articleType", "article_type", "productType", "category"))
        if id_col and type_col:
            frame = pd.read_csv(csv_path, low_memory=False)
            id_col = find_column(frame.columns, ("id", "productId", "product_id", "styleId"))
            type_col = find_column(frame.columns, ("articleType", "article_type", "productType", "category"))
            image_dirs = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() in {"images", "image"}]
            return csv_path, (image_dirs[0] if image_dirs else root), frame, id_col, type_col
    raise FileNotFoundError("styles.csv dengan kolom id dan articleType tidak ditemukan.")


def build_image_index(image_root: Path):
    index = {}
    for path in image_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        keys = {path.stem}
        if path.stem.isdigit():
            keys.add(str(int(path.stem)))
        for key in keys:
            if key not in index or path.suffix.lower() == ".jpg":
                index[key] = path
    return index


def load_mapping(path: Path):
    config = json.loads(path.read_text(encoding="utf-8"))
    names = config["class_names"]
    mapping = {clean_text(k): int(v) for k, v in config["article_type_to_class"].items()}
    if len(names) != 5 or set(mapping.values()) != set(range(5)):
        raise ValueError("class_map.json harus memetakan tepat lima kelas dengan ID 0..4.")
    return names, mapping


def split_class_records(rows, train_ratio: float, valid_ratio: float, seed: int):
    rng = random.Random(seed)
    split_rows = {"train": [], "valid": [], "test": []}
    by_class = {}
    for row in rows:
        by_class.setdefault(row["class_id"], []).append(row)
    for class_id, class_rows in by_class.items():
        rng.shuffle(class_rows)
        n = len(class_rows)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        n_test = n - n_train - n_valid
        if min(n_train, n_valid, n_test) < 1:
            raise ValueError(f"Kelas {class_id} hanya memiliki {n} gambar valid; tambah --max-per-class.")
        split_rows["train"].extend(class_rows[:n_train])
        split_rows["valid"].extend(class_rows[n_train:n_train + n_valid])
        split_rows["test"].extend(class_rows[n_train + n_valid:])
    for rows_for_split in split_rows.values():
        rng.shuffle(rows_for_split)
    return split_rows


def write_contact_sheet(rows, class_names, path: Path, split_name: str, per_class: int = 5):
    cell_w, cell_h = 150, 185
    class_rows = []
    for class_id, class_name in enumerate(class_names):
        tiles = []
        for row in [r for r in rows if r["class_id"] == class_id][:per_class]:
            image = cv2.imread(str(row["source_path"]))
            if image is None:
                continue
            image = cv2.resize(image, (cell_w, cell_h - 30))
            tile = np.full((cell_h, cell_w, 3), 245, np.uint8)
            tile[:cell_h - 30] = image
            cv2.putText(tile, class_name[:20], (5, cell_h - 17), cv2.FONT_HERSHEY_SIMPLEX, .42, (20, 20, 20), 1)
            cv2.putText(tile, row["image_id"][:18], (5, cell_h - 4), cv2.FONT_HERSHEY_SIMPLEX, .32, (60, 60, 60), 1)
            tiles.append(tile)
        while len(tiles) < per_class:
            tiles.append(np.full((cell_h, cell_w, 3), 235, np.uint8))
        class_rows.append(np.hstack(tiles))
    sheet = np.vstack(class_rows)
    cv2.putText(sheet, f"{split_name.upper()} - ORIGINAL PRODUCT JPG", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 180), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def save_npz(path: Path, rows, class_names):
    if not rows:
        raise RuntimeError(f"Split {path.stem} kosong.")
    np.savez_compressed(
        path,
        X_mask=np.asarray([r["mask"] for r in rows], dtype=np.uint8)[:, None, :, :],
        X_geom=np.asarray([r["geometry"] for r in rows], dtype=np.float32),
        y=np.asarray([r["class_id"] for r in rows], dtype=np.int64),
        class_names=np.asarray(class_names),
        image_ids=np.asarray([r["image_id"] for r in rows]),
        article_types=np.asarray([r["article_type"] for r in rows]),
        image_files=np.asarray([r["relative_path"] for r in rows]),
    )


def main():
    parser = argparse.ArgumentParser(description=f"Prepare JPG Fashion Product Images Small for Laundry CNN ({HANDLE})")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Folder Kaggle yang sudah diekstrak.")
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--max-per-class", type=int, default=100)
    parser.add_argument("--min-per-class", type=int, default=20)
    parser.add_argument("--min-resolution", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_per_class < args.min_per_class or args.min_per_class < 3:
        parser.error("--max-per-class harus >= --min-per-class dan minimal 3.")
    if not (0 < args.train_ratio < 1 and 0 < args.valid_ratio < 1 and args.train_ratio + args.valid_ratio < 1):
        parser.error("Rasio train/valid tidak valid; sisanya menjadi test.")

    class_names, type_to_class = load_mapping(args.class_map)
    if args.dataset_dir:
        source_root = args.dataset_dir.expanduser().resolve()
        if not source_root.is_dir():
            raise FileNotFoundError(source_root)
    else:
        try:
            import kagglehub
        except ImportError as exc:
            raise RuntimeError("Install kagglehub: pip install kagglehub") from exc
        print(f"[KAGGLE] Downloading {HANDLE} ...")
        source_root = Path(kagglehub.dataset_download(HANDLE)).resolve()

    csv_path, image_root, frame, id_col, type_col = locate_catalog(source_root)
    frame["_image_id"] = frame[id_col].map(clean_id)
    frame["_article_type"] = frame[type_col].fillna("").astype(str).str.strip()
    frame["_class_id"] = frame[type_col].map(lambda v: type_to_class.get(clean_text(v)))
    mapped = frame[frame["_class_id"].notna()].copy()
    mapped["_class_id"] = mapped["_class_id"].astype(int)
    image_index = build_image_index(image_root)

    existing_jpgs = [
        path for path in args.output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ] if args.output_dir.exists() else []
    if existing_jpgs and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} sudah berisi foto JPG. Gunakan --overwrite untuk membuat ulang split."
        )
    if args.overwrite:
        for old_image in existing_jpgs:
            old_image.unlink()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.preview_dir.mkdir(parents=True, exist_ok=True)

    records = []
    skipped = Counter()
    unmapped_types = frame.loc[frame["_class_id"].isna(), type_col].fillna("<missing>").astype(str).value_counts()
    for class_id, class_name in enumerate(class_names):
        group = mapped[mapped["_class_id"] == class_id]
        if not group.empty:
            group = group.sample(frac=1.0, random_state=args.seed + class_id)
        accepted = 0
        print(f"[SCAN] {class_name}: {len(group)} metadata rows")
        for _, row in group.iterrows():
            if accepted >= args.max_per_class:
                break
            image_id = row["_image_id"]
            image_path = image_index.get(image_id)
            if image_path is None:
                skipped["image_missing"] += 1
                continue
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                skipped["image_unreadable"] += 1
                continue
            if min(bgr.shape[:2]) < args.min_resolution:
                skipped["below_min_resolution"] += 1
                continue
            processed = preprocess_bgr(bgr, size=64)
            if processed is None:
                skipped["segmentation_failed"] += 1
                continue
            records.append({
                "image_id": image_id, "source_path": image_path, "class_id": class_id,
                "class_name": class_name, "article_type": str(row["_article_type"]),
                "mask": processed.mask, "geometry": processed.geometry, "mask_method": processed.method,
            })
            accepted += 1
        if accepted < args.min_per_class:
            raise RuntimeError(f"{class_name}: hanya {accepted} gambar berhasil diproses; minimum {args.min_per_class}. "
                               "Periksa mapping articleType, kualitas JPG, atau turunkan --min-per-class.")

    splits = split_class_records(records, args.train_ratio, args.valid_ratio, args.seed)
    summary = {
        "source_dataset": HANDLE,
        "source_catalog": str(csv_path),
        "source_images": str(image_root),
        "class_names": class_names,
        "split_ratio": {"train": args.train_ratio, "valid": args.valid_ratio,
                        "test": 1.0 - args.train_ratio - args.valid_ratio},
        "max_per_class": args.max_per_class,
        "seed": args.seed,
        "unmapped_article_type_counts": {str(k): int(v) for k, v in unmapped_types.head(50).items()},
        "skipped": dict(skipped),
        "splits": {},
    }

    for split_name, split_rows in splits.items():
        destination_rows = []
        manifest = []
        for row in split_rows:
            class_dir = args.output_dir / split_name / row["class_name"]
            class_dir.mkdir(parents=True, exist_ok=True)
            destination = class_dir / f"{row['image_id']}.jpg"
            bgr = cv2.imread(str(row["source_path"]), cv2.IMREAD_COLOR)
            if bgr is None or not cv2.imwrite(str(destination), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                skipped["write_failed"] += 1
                continue
            relative = destination.relative_to(ROOT).as_posix()
            output_row = dict(row)
            output_row["relative_path"] = relative
            destination_rows.append(output_row)
            manifest.append({
                "image_id": row["image_id"], "image_file": relative,
                "articleType": row["article_type"], "class_id": row["class_id"],
                "class_name": row["class_name"], "mask_method": row["mask_method"],
            })
        pd.DataFrame(manifest).to_csv(args.processed_dir / f"fashion_{split_name}_manifest.csv", index=False)
        save_npz(args.processed_dir / f"laundry64_{split_name}.npz", destination_rows, class_names)
        counts = Counter(row["class_name"] for row in destination_rows)
        summary["splits"][split_name] = {
            "total": len(destination_rows),
            "class_counts": {name: counts.get(name, 0) for name in class_names},
            "jpg_folder": (args.output_dir / split_name).relative_to(ROOT).as_posix(),
            "npz": (args.processed_dir / f"laundry64_{split_name}.npz").relative_to(ROOT).as_posix(),
        }
        write_contact_sheet(destination_rows, class_names,
                            args.preview_dir / f"fashion_{split_name}_samples.jpg", split_name)

    (args.processed_dir / "fashion_dataset_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] JPG: {args.output_dir}")
    for name, info in summary["splits"].items():
        print(f"  {name}: {info['total']} | {info['class_counts']}")
    print(f"[DONE] Processed tensors: {args.processed_dir}")
    print(f"[DONE] Previews: {args.preview_dir}")


if __name__ == "__main__":
    main()
