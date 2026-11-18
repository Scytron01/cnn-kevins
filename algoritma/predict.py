from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from algoritma.model import LaundryShapeCNN
from algoritma.preprocessing import preprocess_bgr

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description="Prediksi jenis pakaian dari satu foto")
    p.add_argument("image",type=Path)
    p.add_argument("--model",type=Path,default=ROOT/"report/runs/laundry_shape_cnn/best_model.pt")
    a=p.parse_args()
    if not a.image.is_file() or not a.model.is_file(): raise FileNotFoundError("Gambar atau checkpoint tidak ditemukan.")
    result=preprocess_bgr(cv2.imread(str(a.image)))
    if result is None: raise RuntimeError("Segmentasi gagal; gunakan foto produk dengan kontras latar lebih baik.")
    ckpt=torch.load(a.model,map_location="cpu",weights_only=False)
    names=ckpt["class_names"]; model=LaundryShapeCNN(len(names)); model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    mask=torch.from_numpy(result.mask[None,None].astype(np.float32))
    geom=torch.from_numpy(result.geometry[None].astype(np.float32))
    with torch.no_grad(): probs=torch.softmax(model(mask,geom),dim=1)[0].numpy()
    best=int(probs.argmax())
    print(f"Prediksi: {names[best]} | confidence={probs[best]:.3f} | segmentation={result.method}")
    for name,score in sorted(zip(names,probs),key=lambda item:-item[1]): print(f"  {name:16s} {score:.4f}")


if __name__=="__main__": main()
