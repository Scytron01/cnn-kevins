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
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Dataset

from algoritma.model import LaundryShapeCNN

ROOT=Path(__file__).resolve().parents[1]


class MaskDataset(Dataset):
    def __init__(self,masks,geometries,labels,augment=False):
        self.masks=torch.from_numpy(masks.astype(np.float32))
        self.geometries=torch.from_numpy(geometries.astype(np.float32))
        self.labels=torch.from_numpy(labels.astype(np.int64))
        self.augment=augment

    def __len__(self): return len(self.labels)

    def __getitem__(self,i):
        mask=self.masks[i].clone()
        if self.augment and torch.rand(())<.5:
            mask=torch.flip(mask,[-1])
        return mask,self.geometries[i],self.labels[i]


def evaluate(model,loader,device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for mask,geom,y in loader:
            pred=model(mask.to(device),geom.to(device)).argmax(1).cpu().numpy()
            ys.extend(y.numpy().tolist()); ps.extend(pred.tolist())
    return np.asarray(ys),np.asarray(ps)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=ROOT/"dataset/processed/laundry64.npz")
    p.add_argument("--out-dir",type=Path,default=ROOT/"report/runs/laundry_shape_cnn")
    p.add_argument("--epochs",type=int,default=40); p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--lr",type=float,default=1e-3); p.add_argument("--patience",type=int,default=7)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    a=p.parse_args()
    if not a.data.exists(): raise FileNotFoundError(f"{a.data}; jalankan algoritma.prepare_dataset terlebih dahulu.")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    data=np.load(a.data,allow_pickle=False)
    masks=data["X_mask"].astype(np.float32); geom=data["X_geom"].astype(np.float32); labels=data["y"].astype(np.int64)
    names=data["class_names"].tolist(); nclass=len(names)
    if masks.shape[1:]!=(1,64,64) or geom.shape!=(len(labels),4): raise ValueError("Shape data tidak sesuai kontrak model.")
    counts=np.bincount(labels,minlength=nclass)
    if len(labels)!=len(masks) or np.any(counts<10): raise ValueError(f"Minimal 10 sampel/kelas untuk split stratified: {dict(zip(names,counts.tolist()))}")
    all_idx=np.arange(len(labels))
    train_idx,temp_idx=train_test_split(all_idx,test_size=.30,random_state=a.seed,stratify=labels)
    val_idx,test_idx=train_test_split(temp_idx,test_size=.50,random_state=a.seed,stratify=labels[temp_idx])
    train=MaskDataset(masks[train_idx],geom[train_idx],labels[train_idx],True)
    val=MaskDataset(masks[val_idx],geom[val_idx],labels[val_idx]); test=MaskDataset(masks[test_idx],geom[test_idx],labels[test_idx])
    train_loader=DataLoader(train,batch_size=a.batch_size,shuffle=True)
    val_loader=DataLoader(val,batch_size=a.batch_size); test_loader=DataLoader(test,batch_size=a.batch_size)
    device=torch.device("cuda" if (a.device=="cuda" or (a.device=="auto" and torch.cuda.is_available())) else "cpu")
    model=LaundryShapeCNN(nclass).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",patience=3,factor=.5)
    loss_fn=torch.nn.CrossEntropyLoss()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    best=-1.; stale=0; history=[]; best_path=a.out_dir/"best_model.pt"
    print(f"[TRAIN] device={device}, samples={len(labels)}, split={len(train)}/{len(val)}/{len(test)}")
    for epoch in range(1,a.epochs+1):
        model.train(); total=0.
        for m,g,y in train_loader:
            m,g,y=m.to(device),g.to(device),y.to(device)
            opt.zero_grad(set_to_none=True); logits=model(m,g); loss=loss_fn(logits,y)
            loss.backward(); opt.step(); total+=loss.item()*len(y)
        vy,vp=evaluate(model,val_loader,device)
        vacc=float(accuracy_score(vy,vp)); vf1=float(f1_score(vy,vp,average="macro",zero_division=0))
        scheduler.step(vf1)
        history.append({"epoch":epoch,"train_loss":total/len(train),"val_accuracy":vacc,"val_macro_f1":vf1})
        print(f"Epoch {epoch}/{a.epochs}: loss={total/len(train):.4f} val_acc={vacc:.4f} val_macro_f1={vf1:.4f}")
        if vf1>best:
            best=vf1; stale=0
            torch.save({"model_state_dict":model.state_dict(),"class_names":names,"input_size":64,
                        "geometry_features":["area_norm","aspect_norm","bbox_occupancy","lower_two_run_score"]},best_path)
        else:
            stale+=1
            if stale>=a.patience: print("[EARLY STOP]"); break
    ckpt=torch.load(best_path,map_location=device,weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    y,pred=evaluate(model,test_loader,device)
    cm=confusion_matrix(y,pred,labels=list(range(nclass)))
    metrics={"test_accuracy":float(accuracy_score(y,pred)),
      "test_macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
      "classification_report":classification_report(y,pred,labels=list(range(nclass)),target_names=names,output_dict=True,zero_division=0),
      "class_counts":dict(zip(names,counts.tolist())),"split_sizes":{"train":len(train),"validation":len(val),"test":len(test)},
      "seed":a.seed,"device":str(device)}
    (a.out_dir/"metrics.json").write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding="utf-8")
    (a.out_dir/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    np.savetxt(a.out_dir/"confusion_matrix.csv",cm,delimiter=",",fmt="%d")
    fig,ax=plt.subplots(figsize=(8,7)); ax.imshow(cm,interpolation="nearest"); ax.set_title("Laundry Shape-CNN")
    ax.set_xticks(range(nclass),names,rotation=35,ha="right"); ax.set_yticks(range(nclass),names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(nclass):
        for j in range(nclass): ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    fig.tight_layout(); fig.savefig(a.out_dir/"confusion_matrix.png",dpi=150); plt.close(fig)
    print(f"[TEST] accuracy={metrics['test_accuracy']:.4f} macro_f1={metrics['test_macro_f1']:.4f}")
    print("[SAVED]",best_path)


if __name__=="__main__": main()
