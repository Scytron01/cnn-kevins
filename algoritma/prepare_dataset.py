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
DATASET_HANDLE = "shivamb/fashion-clothing-products-catalog"
DEFAULT_MAP = ROOT / "algoritma" / "class_map.json"
DEFAULT_RAW = ROOT / "dataset" / "raw" / "fashion-clothing-products-catalog"
DEFAULT_OUT = ROOT / "dataset" / "processed" / "laundry64.npz"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def norm(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def normalize_id(value: object) -> str:
    text = re.sub(r"\.0$", "", str(value).strip())
    # CSV readers sometimes parse numeric product IDs and drop leading zeroes.
    return str(int(text)) if text.isdigit() else text


def load_map(path: Path):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    names = cfg["class_names"]
    mapping = {norm(k): int(v) for k, v in cfg["article_type_to_class"].items()}
    if len(names) != 5 or set(mapping.values()) - set(range(5)):
        raise ValueError("class_map.json harus mendefinisikan lima kelas valid.")
    return names, mapping


def resolve_dataset(dataset_dir: str | None) -> Path:
    if dataset_dir:
        root = Path(dataset_dir).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        return root
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("kagglehub belum tersedia. Jalankan: pip install -r requirements.txt") from exc
    print(f"[KAGGLE] Download {DATASET_HANDLE}")
    print("[KAGGLE] Pastikan akun Kaggle memiliki akses dan autentikasi API telah disiapkan.")
    return Path(kagglehub.dataset_download(DATASET_HANDLE)).resolve()


def locate_catalog(root: Path):
    candidates = list(root.rglob("*.csv"))
    preferred = sorted(candidates, key=lambda p: (p.name.lower() != "styles.csv", len(p.parts)))
    csv_path, frame = None, None
    for candidate in preferred:
        try:
            probe = pd.read_csv(candidate, nrows=5)
        except Exception:
            continue
        cols = {str(c).strip().lower() for c in probe.columns}
        if {"id", "articletype"}.issubset(cols):
            csv_path = candidate
            frame = pd.read_csv(candidate, low_memory=False)
            break
    if csv_path is None or frame is None:
        raise FileNotFoundError("Tidak menemukan CSV katalog dengan kolom id dan articleType.")

    image_dirs = [p for p in root.rglob("*")
                  if p.is_dir() and p.name.lower() in {"images", "image", "images_small"}]
    image_dir = image_dirs[0] if image_dirs else root
    if not any(p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in image_dir.rglob("*")):
        image_dir = root
    return csv_path, image_dir, frame


def image_index(image_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    files = sorted(p for p in image_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    for path in files:
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
    """Copy a balanced target-class subset into dataset/raw for local inspection.

    The full Kaggle archive is not copied into GitHub. Only selected source
    images are materialized locally; Git ignores the raw image payload.
    """
    names, mapping = load_map(class_map)
    if (raw_dir / "styles.csv").exists() and not overwrite:
        print(f"[RAW] Dataset lokal sudah ada: {raw_dir}; gunakan --overwrite-raw untuk membangun ulang.")
        return raw_dir

    csv_path, source_images, frame = locate_catalog(source_root)
    columns = {str(c).strip().lower(): c for c in frame.columns}
    id_col, type_col = columns["id"], columns["articletype"]
    frame = frame.copy()
    frame["_source_id"] = frame[id_col].map(normalize_id)
    frame["_class_id"] = frame[type_col].map(lambda x: mapping.get(norm(x)))
    frame = frame[frame["_class_id"].notna()].copy()
    frame["_class_id"] = frame["_class_id"].astype(int)
    lookup = image_index(source_images)

    if raw_dir.exists() and overwrite:
        shutil.rmtree(raw_dir)
    images_out = raw_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    selected, skipped = [], Counter()
    for class_id in range(5):
        group = frame[frame["_class_id"] == class_id]
        if len(group):
            group = group.sample(frac=1.0, random_state=seed + class_id)
        accepted = 0
        for _, row in group.iterrows():
            if accepted >= max_per_class:
                break
            source_id = row["_source_id"]
            image_path = lookup.get(source_id)
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
            target_name = f"{source_id}{image_path.suffix.lower()}"
            shutil.copy2(image_path, images_out / target_name)
            record = row.drop(labels=["_source_id", "_class_id"]).to_dict()
            record["id"] = source_id
            selected.append((record, class_id, target_name, width, height))
            accepted += 1

    if not selected:
        raise RuntimeError("Tidak ada gambar dari lima kelas target yang berhasil disalin.")

    styles = pd.DataFrame([item[0] for item in selected])
    styles.to_csv(raw_dir / "styles.csv", index=False)
    manifest = pd.DataFrame([
        {"image_id": item[0]["id"], "articleType": item[0].get(type_col, ""),
         "class_id": item[1], "class_name": names[item[1]], "image_file": item[2],
         "width": item[3], "height": item[4]}
        for item in selected
    ])
    manifest.to_csv(raw_dir / "manifest.csv", index=False)
    info = {
        "source_dataset": DATASET_HANDLE,
        "source_csv": str(csv_path),
        "source_images": str(source_images),
        "materialized_dir": str(raw_dir),
        "class_counts": {names[i]: int((manifest["class_id"] == i).sum()) for i in range(5)},
        "skipped": dict(skipped),
        "note": "Curated local subset of source images; not a redistributed full Kaggle dataset."
    }
    (raw_dir / "download_report.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[RAW] Subset Kaggle disalin ke", raw_dir)
    print("[RAW] Distribusi:", info["class_counts"])
    return raw_dir


def build_dataset(root: Path, out_path: Path, class_map: Path,
                  max_per_class: int, min_per_class: int, min_width: int):
    names, mapping = load_map(class_map)
    csv_path, images_dir, frame = locate_catalog(root)
    columns = {str(c).strip().lower(): c for c in frame.columns}
    id_col, type_col = columns["id"], columns["articletype"]
    frame["_id"] = frame[id_col].map(normalize_id)
    frame["_type_norm"] = frame[type_col].map(norm)
    frame["_class"] = frame["_type_norm"].map(mapping)
    mapped = frame[frame["_class"].notna()].copy()
    mapped["_class"] = mapped["_class"].astype(int)
    lookup = image_index(images_dir)

    masks, geoms, labels, ids, types, methods = [], [], [], [], [], []
    accepted, skipped = Counter(), Counter()
    mapped = mapped.sort_values(["_class", "_id"], kind="stable")
    for _, row in mapped.iterrows():
        class_id = int(row["_class"])
        if accepted[class_id] >= max_per_class:
            continue
        image_path = lookup.get(row["_id"])
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
        geoms.append(result.geometry)
        labels.append(class_id)
        ids.append(row["_id"])
        types.append(str(row[type_col]))
        methods.append(result.method)
        accepted[class_id] += 1

    counts = {names[i]: int(accepted[i]) for i in range(5)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "source_dataset": DATASET_HANDLE,
        "catalog_csv": str(csv_path),
        "images_dir": str(images_dir),
        "class_distribution_after_segmentation": counts,
        "mapped_article_type_counts": {
            str(k): int(v) for k, v in mapped[type_col].value_counts().items()},
        "unmapped_article_type_examples": {
            str(k): int(v) for k, v in frame.loc[frame["_class"].isna(), type_col].value_counts().head(30).items()},
        "mask_methods": dict(Counter(methods)),
        "skipped": dict(skipped),
    }
    out_path.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([{"class_id": i, "class_name": names[i], "processed": accepted[i]}
                  for i in range(5)]).to_csv(out_path.with_suffix(".distribution.csv"), index=False)

    low = [names[i] for i in range(5) if accepted[i] < min_per_class]
    if low:
        raise RuntimeError(
            f"Sampel hasil segmentasi di bawah minimum {min_per_class}: {low}. "
            f"Baca laporan {out_path.with_suffix('.report.json')}.")

    np.savez_compressed(
        out_path,
        X_mask=np.asarray(masks, dtype=np.uint8)[:, None, :, :],
        X_geom=np.asarray(geoms, dtype=np.float32),
        y=np.asarray(labels, dtype=np.int64),
        class_names=np.asarray(names),
        image_ids=np.asarray(ids),
        article_types=np.asarray(types),
        mask_methods=np.asarray(methods))
    pd.DataFrame({
        "image_id": ids, "articleType": types, "class_id": labels,
        "class_name": [names[x] for x in labels], "mask_method": methods
    }).to_csv(out_path.with_suffix(".manifest.csv"), index=False)
    print("[PROCESSED]", counts, "shape=", len(labels), "saved=", out_path)
    return out_path


def make_preview(npz_path: Path, out_path: Path, samples_per_class: int = 8):
    archive = np.load(npz_path, allow_pickle=False)
    masks, labels, names = archive["X_mask"][:, 0], archive["y"], archive["class_names"].tolist()
    rows = []
    for class_id, class_name in enumerate(names):
        tiles = []
        for idx in np.where(labels == class_id)[0][:samples_per_class]:
            tile = cv2.resize(masks[idx] * 255, (112, 112), interpolation=cv2.INTER_NEAREST)
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
            cv2.putText(tile, class_name[:16], (3, 105), cv2.FONT_HERSHEY_SIMPLEX, .32, (0, 0, 255), 1)
            tiles.append(tile)
        if tiles:
            while len(tiles) < samples_per_class:
                tiles.append(np.zeros_like(tiles[0]))
            rows.append(np.hstack(tiles))
    if rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), np.vstack(rows))


def main():
    parser = argparse.ArgumentParser(description=f"Download and prepare Kaggle {DATASET_HANDLE}")
    parser.add_argument("--dataset-dir", default=None,
                        help="Folder dataset lengkap lokal (berisi CSV katalog dan folder gambar)")
    parser.add_argument("--class-map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW,
                        help="Subset dataset yang akan terlihat di dataset/raw/")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-per-class", type=int, default=1000,
                        help="Maksimum gambar sumber per kelas yang disalin")
    parser.add_argument("--min-per-class", type=int, default=50,
                        help="Minimum gambar lolos segmentasi per kelas")
    parser.add_argument("--min-width", type=int, default=32,
                        help="Lebar dan tinggi minimum gambar sumber")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-raw", action="store_true",
                        help="Bangun ulang subset lokal dataset/raw yang sudah ada")
    args = parser.parse_args()

    source_root = resolve_dataset(args.dataset_dir)
    raw_root = materialize_dataset(
        source_root, args.raw_dir, args.class_map, args.max_per_class,
        args.min_width, args.seed, overwrite=args.overwrite_raw)
    out = build_dataset(raw_root, args.out, args.class_map,
                        max_per_class=args.max_per_class,
                        min_per_class=args.min_per_class,
                        min_width=args.min_width)
    make_preview(out, out.parent / "mask_preview.png")
    print("[DONE] Periksa dataset/raw/.../images dan dataset/processed/mask_preview.png.")


if __name__ == "__main__":
    main()
