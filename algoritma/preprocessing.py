from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    mask: np.ndarray
    geometry: np.ndarray
    method: str
    foreground_ratio: float


def _largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return np.zeros_like(binary)
    h, w = binary.shape
    cx, cy = w / 2.0, h / 2.0
    best_idx, best_score = 0, -1.0
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < max(8, int(h * w * 0.001)):
            continue
        px, py = centers[i]
        distance = ((px-cx)/max(w,1))**2 + ((py-cy)/max(h,1))**2
        score = area / (1.0 + 2.0 * distance)
        if score > best_score:
            best_idx, best_score = i, score
    return (labels == best_idx).astype(np.uint8) if best_idx else np.zeros_like(binary)


def _border_mask(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    band = max(2, min(h, w)//16)
    border = np.concatenate([
        lab[:band].reshape(-1,3), lab[-band:].reshape(-1,3),
        lab[:, :band].reshape(-1,3), lab[:, -band:].reshape(-1,3)
    ])
    bg = np.median(border, axis=0)
    distance = np.sqrt(((lab-bg[None,None,:])**2).sum(axis=2))
    raw = (distance > 14.0).astype(np.uint8)*255
    kernel = np.ones((3,3), np.uint8)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel, iterations=2)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel, iterations=1)
    return _largest_component(raw)


def _grabcut(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    if min(h,w) < 8:
        return np.zeros((h,w), np.uint8)
    margin_x, margin_y = max(1,int(w*.04)), max(1,int(h*.04))
    rect = (margin_x, margin_y, max(2,w-2*margin_x), max(2,h-2*margin_y))
    mask = np.zeros((h,w), np.uint8)
    bgd, fgd = np.zeros((1,65),np.float64), np.zeros((1,65),np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        fg = np.isin(mask, [cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8)
        return _largest_component(fg)
    except cv2.error:
        return np.zeros((h,w), np.uint8)


def _letterbox(mask: np.ndarray, size: int=64, margin: int=2) -> np.ndarray:
    points = cv2.findNonZero((mask>0).astype(np.uint8))
    canvas = np.zeros((size,size), np.uint8)
    if points is None:
        return canvas
    x,y,w,h = cv2.boundingRect(points)
    crop = mask[y:y+h,x:x+w]
    scale = (size-2*margin)/float(max(w,h))
    nw = max(1,min(size-2*margin,int(round(w*scale))))
    nh = max(1,min(size-2*margin,int(round(h*scale))))
    resized = cv2.resize(crop,(nw,nh),interpolation=cv2.INTER_NEAREST)
    ox,oy=(size-nw)//2,(size-nh)//2
    canvas[oy:oy+nh,ox:ox+nw]=(resized>0).astype(np.uint8)
    return canvas


def geometry_features(mask: np.ndarray) -> np.ndarray:
    """[area_norm, aspect_norm, bbox_occupancy, lower_two_run_score]."""
    binary=(mask>0).astype(np.uint8)
    area=int(binary.sum())
    if area==0:
        return np.zeros(4,dtype=np.float32)
    ys,xs=np.where(binary)
    xmin,xmax,ymin,ymax=int(xs.min()),int(xs.max()),int(ys.min()),int(ys.max())
    width,height=xmax-xmin+1,ymax-ymin+1
    aspect=min(width/max(height,1),2.0)/2.0
    occupancy=min(area/max(width*height,1),1.0)
    start=ymin+int(np.floor(height*9/16))
    end=ymin+int(np.floor(height*14/16))
    rows=split=0
    for y in range(start,min(end+1,ymax+1)):
        row=binary[y,xmin:xmax+1]
        runs=int(row[0]==1)+int(np.sum((row[1:]==1)&(row[:-1]==0)))
        rows+=1
        split+=int(runs==2)
    return np.asarray([area/mask.size,aspect,occupancy,split/max(rows,1)],dtype=np.float32)


def preprocess_bgr(bgr: np.ndarray, size: int=64,
                   min_area_ratio: float=.012, max_area_ratio: float=.88) -> Optional[PreprocessResult]:
    if bgr is None or bgr.ndim!=3 or bgr.shape[2]!=3:
        return None
    h,w=bgr.shape[:2]
    if min(h,w)<32:
        return None
    mask=_border_mask(bgr)
    ratio=float(mask.mean())
    method="border_lab"
    if ratio<min_area_ratio or ratio>max_area_ratio:
        mask=_grabcut(bgr)
        ratio=float(mask.mean())
        method="grabcut"
    if ratio<min_area_ratio or ratio>max_area_ratio:
        return None
    mask=_letterbox(mask,size)
    if int(mask.sum())<20:
        return None
    return PreprocessResult(mask,geometry_features(mask),method,ratio)


def preprocess_path(path: str, size: int=64) -> Optional[PreprocessResult]:
    return preprocess_bgr(cv2.imread(str(path),cv2.IMREAD_COLOR),size=size)
