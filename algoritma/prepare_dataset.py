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
RAW = ROOT / "dataset/raw/fashion-clothing-products-catalog"
OUT = ROOT / "dataset/processed"
MAP = ROOT / "algoritma/class_map.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

def norm(value):
    return re.sub(r"\s+", " ", str(value).strip()).lower()

def norm_id(value):
    value = re.sub(r"\.0$", "", str(value).strip())
    return str(int(value)) if value.isdigit() else value

def locate(root):
    csv_path, frame = None, None
    for candidate in sorted(root.rglob("*.csv"), key=lambda p: (p.name.lower() != "styles.csv", len(p.parts))):
        try:
            sample = pd.read_csv(candidate, nrows=5)
        except Exception:
            continue
        columns = {str(c).strip().lower() for c in sample.columns}
        if "id" in columns and "articletype" in columns:
            csv_path = candidate
            frame = pd.read_csv(candidate, low_memory=False)
            break
    if frame is None:
        raise FileNotFoundError("CSV katalog dengan kolom id dan articleType tidak ditemukan.")
    dirs = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() in {"images", "image", "images_small"}]
    image_dir = dirs[0] if dirs else root
    return csv_path, image_dir, frame

def get_image_index(folder):
    index = {}
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index[path.stem] = path
            if path.stem.isdigit():
                index[str(int(path.stem))] = path
    return index

def main():
    parser = argparse.ArgumentParser(description="Download dan siapkan Kaggle Fashion Clothing Products Catalog")
    parser.add_argument("--dataset-dir", default=None, help="Folder dataset yang sudah diekstrak")
    parser.add_argument("--max-per-class", type=int, default=1000)
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--min-width", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-raw", action="store_true")
    args = parser.parse_args()

    if args.dataset_dir:
        source = Path(args.dataset_dir).expanduser().resolve()
    else:
        try:
            import kagglehub
        except ImportError as exc:
            raise RuntimeError("Jalankan pip install -r requirements.txt terlebih dahulu.") from exc
        print("[KAGGLE] Mengunduh " + HANDLE)
        source = Path(kagglehub.dataset_download(HANDLE)).resolve()

    config = json.loads(MAP.read_text(encoding="utf-8"))
    names = config["class_names"]
    class_map = {norm(k): int(v) for k, v in config["article_type_to_class"].items()}
    csv_path, source_images, frame = locate(source)
    columns = {str(c).strip().lower(): c for c in frame.columns}
    id_col, type_col = columns["id"], columns["articletype"]
    frame["_id"] = frame[id_col].map(norm_id)
    frame["_class"] = frame[type_col].map(lambda x: class_map.get(norm(x)))
    frame = frame[frame["_class"].notna()].copy()
    frame["_class"] = frame["_class"].astype(int)
    image_index = get_image_index(source_images)

    if (RAW / "styles.csv").exists() and not args.overwrite_raw:
        print("[RAW] Memakai subset yang sudah ada:", RAW)
    else:
        if RAW.exists():
            shutil.rmtree(RAW)
        image_out = RAW / "images"
        image_out.mkdir(parents=True, exist_ok=True)
        selected, skipped = [], Counter()
        for class_id in range(5):
            group = frame[frame["_class"] == class_id].sample(frac=1, random_state=args.seed + class_id)
            count = 0
            for _, row in group.iterrows():
                if count >= args.max_per_class:
                    break
                source_image = image_index.get(row["_id"])
                if source_image is None:
                    skipped["image_missing"] += 1
                    continue
                image = cv2.imread(str(source_image))
                if image is None:
                    skipped["image_unreadable"] += 1
                    continue
                height, width = image.shape[:2]
                if min(width, height) < args.min_width:
                    skipped["below_min_resolution"] += 1
                    continue
                filename = row["_id"] + source_image.suffix.lower()
                shutil.copy2(source_image, image_out / filename)
                record = row.drop(labels=["_id", "_class"]).to_dict()
                record["id"] = row["_id"]
                selected.append((record, class_id, filename, width, height))
                count += 1
        if not selected:
            raise RuntimeError("Tidak ada gambar dari kategori target. Periksa schema katalog dan class_map.json.")
        pd.DataFrame([x[0] for x in selected]).to_csv(RAW / "styles.csv", index=False)
        manifest = pd.DataFrame([{
            "image_id": x[0]["id"], "articleType": x[0].get(type_col, ""),
            "class_id": x[1], "class_name": names[x[1]], "image_file": x[2],
            "width": x[3], "height": x[4]} for x in selected])
        manifest.to_csv(RAW / "manifest.csv", index=False)
        report = {
            "source_dataset": HANDLE, "source_catalog": str(csv_path),
            "class_counts": {names[i]: int((manifest["class_id"] == i).sum()) for i in range(5)},
            "skipped": dict(skipped),
            "note": "Subset gambar sumber untuk penggunaan lokal; bukan redistribusi full Kaggle archive."
        }
        (RAW / "download_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("[RAW] Gambar sumber disalin ke", image_out)
        print("[RAW] Distribusi:", report["class_counts"])

    local_csv, local_images, local_frame = locate(RAW)
    local_columns = {str(c).strip().lower(): c for c in local_frame.columns}
    local_id_col, local_type_col = local_columns["id"], local_columns["articletype"]
    local_frame["_id"] = local_frame[local_id_col].map(norm_id)
    local_frame["_class"] = local_frame[local_type_col].map(lambda x: class_map.get(norm(x)))
    local_frame = local_frame[local_frame["_class"].notna()].copy()
    local_frame["_class"] = local_frame["_class"].astype(int)
    local_index = get_image_index(local_images)

    masks, geometries, labels, ids, types, methods = [], [], [], [], [], []
    counts, skipped = Counter(), Counter()
    for _, row in local_frame.sort_values(["_class", "_id"]).iterrows():
        class_id = int(row["_class"])
        if counts[class_id] >= args.max_per_class:
            continue
        image_path = local_index.get(row["_id"])
        if image_path is None:
            skipped["image_missing"] += 1
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            skipped["image_unreadable"] += 1
            continue
        height, width = image.shape[:2]
        if min(width, height) < args.min_width:
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
        types.append(str(row[local_type_col]))
        methods.append(result.method)
        counts[class_id] += 1

    distribution = {names[i]: int(counts[i]) for i in range(5)}
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "source_dataset": HANDLE, "class_distribution_after_segmentation": distribution,
        "mask_methods": dict(Counter(methods)), "skipped": dict(skipped),
        "raw_folder": str(RAW), "minimum_source_resolution": args.min_width
    }
    (OUT / "laundry64.report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame([{"class_id": i, "class_name": names[i], "processed": counts[i]} for i in range(5)]).to_csv(
        OUT / "laundry64.distribution.csv", index=False)
    missing = [names[i] for i in range(5) if counts[i] < args.min_per_class]
    if missing:
        raise RuntimeError("Sampel hasil segmentasi kurang untuk: " + ", ".join(missing) +
                           ". Periksa laporan dan mapping; jangan membuat sampel palsu.")

    np.savez_compressed(
        OUT / "laundry64.npz",
        X_mask=np.asarray(masks, dtype=np.uint8)[:, None],
        X_geom=np.asarray(geometries, dtype=np.float32),
        y=np.asarray(labels, dtype=np.int64),
        class_names=np.asarray(names),
        image_ids=np.asarray(ids),
        article_types=np.asarray(types),
        mask_methods=np.asarray(methods))
    pd.DataFrame({
        "image_id": ids, "articleType": types, "class_id": labels,
        "class_name": [names[x] for x in labels], "mask_method": methods
    }).to_csv(OUT / "laundry64.manifest.csv", index=False)

    archive = np.load(OUT / "laundry64.npz", allow_pickle=False)
    preview_rows = []
    for class_id, class_name in enumerate(names):
        tiles = []
        for idx in np.where(archive["y"] == class_id)[0][:8]:
            tile = cv2.resize(archive["X_mask"][idx, 0] * 255, (112, 112), interpolation=cv2.INTER_NEAREST)
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
            cv2.putText(tile, class_name, (3, 105), cv2.FONT_HERSHEY_SIMPLEX, .32, (0, 0, 255), 1)
            tiles.append(tile)
        if tiles:
            while len(tiles) < 8:
                tiles.append(np.zeros_like(tiles[0]))
            preview_rows.append(np.hstack(tiles))
    if preview_rows:
        cv2.imwrite(str(OUT / "mask_preview.png"), np.vstack(preview_rows))
    print("[DONE] Raw images:", RAW / "images")
    print("[DONE] Processed dataset:", OUT / "laundry64.npz")
    print("[DONE] Preview:", OUT / "mask_preview.png")

if __name__ == "__main__":
    main()
