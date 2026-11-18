from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from algoritma.model import LaundryShapeCNN

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "dataset" / "processed"


class MaskDataset(Dataset):
    def __init__(self, archive, augment=False):
        self.masks = torch.from_numpy(archive["X_mask"].astype(np.float32))
        self.geometries = torch.from_numpy(archive["X_geom"].astype(np.float32))
        self.labels = torch.from_numpy(archive["y"].astype(np.int64))
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        mask = self.masks[index].clone()
        if self.augment and torch.rand(()) < 0.5:
            mask = torch.flip(mask, dims=[-1])
        return mask, self.geometries[index], self.labels[index]


def read_archive(path):
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split tidak ditemukan: {path}")
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key].copy() for key in data.files}
    required = {"X_mask", "X_geom", "y", "class_names"}
    if not required.issubset(result):
        raise ValueError(f"{path} harus memiliki {sorted(required)}")
    if result["X_mask"].ndim != 4 or result["X_mask"].shape[1:] != (1, 64, 64):
        raise ValueError(f"X_mask pada {path} harus berbentuk (N,1,64,64)")
    if result["X_geom"].shape != (len(result["y"]), 4):
        raise ValueError(f"X_geom pada {path} harus berbentuk (N,4)")
    if len(result["X_mask"]) != len(result["y"]):
        raise ValueError(f"Jumlah gambar dan label berbeda pada {path}")
    return result


def load_splits(args):
    train_path = args.train_data or PROCESSED / "laundry64_train.npz"
    valid_path = args.valid_data or PROCESSED / "laundry64_valid.npz"
    test_path = args.test_data or PROCESSED / "laundry64_test.npz"

    if train_path.is_file() and valid_path.is_file() and test_path.is_file():
        train_data, valid_data, test_data = [
            read_archive(path) for path in (train_path, valid_path, test_path)
        ]
        split_mode = "predefined-train-valid-test"
    else:
        if not args.data.is_file():
            raise FileNotFoundError(
                "Split NPZ tidak ditemukan. Jalankan "
                "'python -m algoritma.prepare_fashion_images' terlebih dahulu."
            )
        combined = read_archive(args.data)
        labels = combined["y"].astype(np.int64)
        indices = np.arange(len(labels))
        train_idx, rest_idx = train_test_split(
            indices, test_size=0.30, random_state=args.seed, stratify=labels
        )
        valid_idx, test_idx = train_test_split(
            rest_idx, test_size=0.50, random_state=args.seed, stratify=labels[rest_idx]
        )

        def subset(indexes):
            return {
                "X_mask": combined["X_mask"][indexes],
                "X_geom": combined["X_geom"][indexes],
                "y": combined["y"][indexes],
                "class_names": combined["class_names"],
            }

        train_data, valid_data, test_data = (
            subset(train_idx), subset(valid_idx), subset(test_idx)
        )
        split_mode = "legacy-random-split"

    class_names = train_data["class_names"].tolist()
    if len(class_names) != 5:
        raise ValueError(f"Model laundry mempunyai 5 output; dataset memiliki {len(class_names)} kelas.")
    for split_name, data in (
        ("train", train_data), ("valid", valid_data), ("test", test_data)
    ):
        if data["class_names"].tolist() != class_names:
            raise ValueError(f"Urutan class_names pada split {split_name} berbeda.")
        if data["X_mask"].shape[1:] != (1, 64, 64):
            raise ValueError(f"Mask pada split {split_name} tidak berukuran 1x64x64.")
        if data["X_geom"].shape != (len(data["y"]), 4):
            raise ValueError(f"Fitur geometri pada split {split_name} tidak berbentuk (N,4).")
        counts = np.bincount(data["y"].astype(np.int64), minlength=5)
        if len(data["y"]) == 0 or np.any(counts == 0):
            raise ValueError(f"Semua lima kelas wajib ada pada split {split_name}; count={counts.tolist()}")
    return train_data, valid_data, test_data, class_names, split_mode


def evaluate(model, loader, device):
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for masks, geometry, labels in loader:
            logits = model(masks.to(device), geometry.to(device))
            actual.extend(labels.numpy().tolist())
            predicted.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    return np.asarray(actual), np.asarray(predicted)


def main():
    parser = argparse.ArgumentParser(description="Train LaundryShapeCNN dari folder JPG train/valid/test.")
    parser.add_argument("--data", type=Path, default=PROCESSED / "laundry64.npz",
                        help="Fallback arsip NPZ lama bila split NPZ tidak tersedia.")
    parser.add_argument("--train-data", type=Path, default=None)
    parser.add_argument("--valid-data", type=Path, default=None)
    parser.add_argument("--test-data", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "report/runs/laundry_shape_cnn")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1:
        parser.error("epochs, batch-size, dan patience harus positif.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_data, valid_data, test_data, class_names, split_mode = load_splits(args)

    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diminta tetapi tidak tersedia.")
    device = torch.device("cuda" if use_cuda else "cpu")

    train_ds = MaskDataset(train_data, augment=True)
    valid_ds = MaskDataset(valid_data)
    test_ds = MaskDataset(test_data)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = LaundryShapeCNN(num_classes=5).to(device)
    counts = np.bincount(train_data["y"].astype(np.int64), minlength=5)
    class_weights = len(train_ds) / (5 * np.maximum(counts, 1))
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / "best_model.pt"
    best_valid_f1 = -1.0
    stale_epochs = 0
    history = []
    print(f"[TRAIN] mode={split_mode}, device={device}, classes={class_names}")
    print(f"[SPLIT] train={len(train_ds)}, valid={len(valid_ds)}, test={len(test_ds)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for masks, geometry, labels in train_loader:
            masks = masks.to(device)
            geometry = geometry.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(masks, geometry), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)

        y_valid, pred_valid = evaluate(model, valid_loader, device)
        valid_accuracy = float(accuracy_score(y_valid, pred_valid))
        valid_macro_f1 = float(f1_score(y_valid, pred_valid, average="macro", zero_division=0))
        train_loss = total_loss / len(train_ds)
        scheduler.step(valid_macro_f1)
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "valid_accuracy": valid_accuracy, "valid_macro_f1": valid_macro_f1
        })
        print(f"Epoch {epoch:02d}/{args.epochs} | loss={train_loss:.4f} "
              f"valid_acc={valid_accuracy:.4f} valid_macro_f1={valid_macro_f1:.4f}")

        if valid_macro_f1 > best_valid_f1:
            best_valid_f1 = valid_macro_f1
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "input_size": 64,
                "geometry_features": [
                    "area_norm", "aspect_norm", "bbox_occupancy", "lower_two_run_score"
                ],
                "architecture": "LaundryShapeCNN-v1",
                "dataset": "paramaggarwal/fashion-product-images-small",
            }, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("[EARLY STOP] Validation macro-F1 tidak membaik.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    y_test, pred_test = evaluate(model, test_loader, device)
    matrix = confusion_matrix(y_test, pred_test, labels=list(range(5)))
    metrics = {
        "dataset": "paramaggarwal/fashion-product-images-small",
        "split_mode": split_mode,
        "test_accuracy": float(accuracy_score(y_test, pred_test)),
        "test_macro_f1": float(f1_score(y_test, pred_test, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_test, pred_test, labels=list(range(5)), target_names=class_names,
            output_dict=True, zero_division=0
        ),
        "class_names": class_names,
        "split_sizes": {
            "train": len(train_ds), "valid": len(valid_ds), "test": len(test_ds)
        },
        "seed": args.seed,
        "device": str(device),
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.out_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    np.savetxt(args.out_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(matrix, interpolation="nearest")
    ax.set_title("Laundry Shape-CNN — Test Confusion Matrix")
    ax.set_xticks(range(5), class_names, rotation=35, ha="right")
    ax.set_yticks(range(5), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for row in range(5):
        for column in range(5):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(args.out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot([item["epoch"] for item in history],
            [item["train_loss"] for item in history], label="Train loss")
    ax.plot([item["epoch"] for item in history],
            [item["valid_accuracy"] for item in history], label="Validation accuracy")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "training_curves.png", dpi=150)
    plt.close(fig)

    print(f"[TEST] accuracy={metrics['test_accuracy']:.4f}, "
          f"macro_f1={metrics['test_macro_f1']:.4f}")
    print(f"[SAVED] {checkpoint_path}")


if __name__ == "__main__":
    main()
