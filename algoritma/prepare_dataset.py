from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from algoritma.preprocessing import preprocess_bgr

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_MAP=ROOT/"algoritma"/"class_map.json"


def norm(value):
    return re.sub(r"\s+"," ",str(value).strip()).lower()


def load_map(path):
    cfg=json.loads(path.read_text(encoding="utf-8"))
    names=cfg["class_names"]
    mapping={norm(k):int(v) for k,v in cfg["article_type_to_class"].items()}
    if len(names)!=5 or any(v<0 or v>=5 for v in mapping.values()):
        raise ValueError("class_map.json harus mendefinisikan lima kelas valid.")
    return names,mapping


def resolve_dataset(dataset_dir):
    if dataset_dir:
        root=Path(dataset_dir).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        return root
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Instal requirements.txt agar kagglehub tersedia.") from exc
    print("[KAGGLE] Download paramaggarwal/fashion-product-images-small")
    return Path(kagglehub.dataset_download("paramaggarwal/fashion-product-images-small"))


def locate_catalog(root):
    csvs=list(root.rglob("styles.csv"))
    if not csvs:
        raise FileNotFoundError(f"styles.csv tidak ditemukan: {root}")
    csv=csvs[0]
    dirs=[p for p in root.rglob("images") if p.is_dir()]
    return csv,dirs[0] if dirs else csv.parent


def build_dataset(root,out_path,class_map,max_per_class,min_per_class,min_width):
    names,mapping=load_map(class_map)
    csv_path,image_dir=locate_catalog(root)
    frame=pd.read_csv(csv_path,dtype={"id":"string"},low_memory=False)
    if not {"id","articleType"}.issubset(frame.columns):
        raise ValueError("styles.csv harus berisi kolom id dan articleType.")
    frame["_id"]=frame["id"].astype(str).str.replace(r"\.0$","",regex=True)
    frame["_type"]=frame["articleType"].map(norm)
    frame["_class"]=frame["_type"].map(mapping)
    mapped=frame[frame["_class"].notna()].copy()
    mapped["_class"]=mapped["_class"].astype(int)
    image_index={p.stem:p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}}
    masks=[]; geoms=[]; labels=[]; ids=[]; types=[]; methods=[]
    accepted=Counter(); skipped=Counter()
    mapped=mapped.sort_values(["_class","_id"],kind="stable")
    for _,row in mapped.iterrows():
        cls=int(row["_class"])
        if accepted[cls]>=max_per_class:
            continue
        path=image_index.get(row["_id"])
        if path is None:
            skipped["image_missing"]+=1; continue
        img=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if img is None:
            skipped["image_unreadable"]+=1; continue
        h,w=img.shape[:2]
        if min(h,w)<min_width:
            skipped["below_min_resolution"]+=1; continue
        result=preprocess_bgr(img)
        if result is None:
            skipped["mask_failed"]+=1; continue
        masks.append(result.mask); geoms.append(result.geometry); labels.append(cls)
        ids.append(row["_id"]); types.append(str(row["articleType"])); methods.append(result.method)
        accepted[cls]+=1
    counts={names[i]:int(accepted[i]) for i in range(5)}
    report={
      "dataset":"paramaggarwal/fashion-product-images-small",
      "dataset_root":str(root),"styles_csv":str(csv_path),"image_dir":str(image_dir),
      "class_distribution_after_processing":counts,
      "mapped_article_type_counts":{str(k):int(v) for k,v in mapped["articleType"].value_counts().items()},
      "unmapped_article_type_examples":{str(k):int(v) for k,v in frame.loc[frame["_class"].isna(),"articleType"].value_counts().head(30).items()},
      "mask_methods":dict(Counter(methods)),"skipped":dict(skipped)
    }
    out_path.parent.mkdir(parents=True,exist_ok=True)
    report_path=out_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    pd.DataFrame([{"class_id":i,"class_name":names[i],"processed":accepted[i]} for i in range(5)]).to_csv(out_path.with_suffix(".distribution.csv"),index=False)
    low=[names[i] for i in range(5) if accepted[i]<min_per_class]
    if low:
        raise RuntimeError(f"Sampel per kelas belum mencapai minimum {min_per_class}: {low}. Periksa report dan algoritma/class_map.json.")
    np.savez_compressed(out_path,X_mask=np.asarray(masks,dtype=np.uint8)[:,None],
        X_geom=np.asarray(geoms,dtype=np.float32),y=np.asarray(labels,dtype=np.int64),
        class_names=np.asarray(names),image_ids=np.asarray(ids),article_types=np.asarray(types),mask_methods=np.asarray(methods))
    print("[DATASET]",counts,"shape=",len(labels),"saved=",out_path)
    return out_path


def make_preview(npz_path,out_path,samples_per_class=8):
    archive=np.load(npz_path,allow_pickle=False); masks=archive["X_mask"][:,0]; labels=archive["y"]; names=archive["class_names"].tolist()
    rows=[]
    for cls,name in enumerate(names):
        tiles=[]
        for idx in np.where(labels==cls)[0][:samples_per_class]:
            tile=cv2.resize(masks[idx]*255,(112,112),interpolation=cv2.INTER_NEAREST)
            tile=cv2.cvtColor(tile,cv2.COLOR_GRAY2BGR)
            cv2.putText(tile,name[:16],(3,105),cv2.FONT_HERSHEY_SIMPLEX,.32,(0,0,255),1)
            tiles.append(tile)
        if tiles:
            while len(tiles)<samples_per_class: tiles.append(np.zeros_like(tiles[0]))
            rows.append(np.hstack(tiles))
    if rows:
        out_path.parent.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(out_path),np.vstack(rows))


def main():
    p=argparse.ArgumentParser(description="Fashion Product Images Small -> laundry silhouette dataset")
    p.add_argument("--dataset-dir",default=None,help="Folder berisi styles.csv dan images/")
    p.add_argument("--no-download",action="store_true")
    p.add_argument("--class-map",type=Path,default=DEFAULT_MAP)
    p.add_argument("--out",type=Path,default=ROOT/"dataset/processed/laundry64.npz")
    p.add_argument("--max-per-class",type=int,default=2500)
    p.add_argument("--min-per-class",type=int,default=20)
    p.add_argument("--min-width",type=int,default=32)
    args=p.parse_args()
    root=resolve_dataset(args.dataset_dir) if not args.no_download else Path(args.dataset_dir).expanduser().resolve()
    out=build_dataset(root,args.out,args.class_map,args.max_per_class,args.min_per_class,args.min_width)
    make_preview(out,out.parent/"mask_preview.png")


if __name__=="__main__":
    main()
