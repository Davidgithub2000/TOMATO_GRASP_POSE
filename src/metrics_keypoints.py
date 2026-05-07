"""
Additional keypoint metrics for tomato pose estimation.

The built-in Ultralytics validation already reports box and pose mAP.
This module adds simple PCK and NME metrics by matching predicted boxes to
YOLO-format labels on a validation/test split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


@dataclass
class KptMetrics:
    n_gt: int
    n_matched: int
    pck_005: float
    pck_010: float
    nme: float


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _load_yaml(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    x, y, w, h = box
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dtype=np.float32)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)


def _read_label(label_path: Path, nkpt: int = 4) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not label_path.exists():
        return []
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        vals = np.array([float(x) for x in parts], dtype=np.float32)
        if vals.size < 5 + nkpt * 3:
            continue
        box = vals[1:5]
        kpts = vals[5:5 + nkpt * 3].reshape(nkpt, 3)
        rows.append((box, kpts[:, :2], kpts[:, 2]))
    return rows


def evaluate_pck_nme(
    weights: str | Path,
    data_yaml: str | Path,
    split: str = "val",
    imgsz: int = 768,
    conf: float = 0.25,
    iou_match: float = 0.5,
    device: str | int = 0,
    max_images: int | None = None,
) -> KptMetrics:
    data = _load_yaml(data_yaml)
    root = Path(data.get("path", "."))
    img_dir = Path(data.get(split, f"images/{split}"))
    if not img_dir.is_absolute():
        img_dir = root / img_dir
    label_dir = Path(str(img_dir).replace(str(Path("images")), str(Path("labels"))))

    images = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    images = sorted(images)
    if max_images:
        images = images[:max_images]

    model = YOLO(str(weights))

    all_errors = []
    pck005 = []
    pck010 = []
    n_gt = 0
    n_matched = 0

    for img_path in images:
        h_img, w_img = cv2.imread(str(img_path)).shape[:2]
        label_path = label_dir / (img_path.stem + ".txt")
        gts = _read_label(label_path, nkpt=4)
        n_gt += len(gts)
        if not gts:
            continue

        result = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        )[0]

        if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
            continue

        pred_boxes = result.boxes.xywhn.cpu().numpy()
        pred_kpts = result.keypoints.xyn.cpu().numpy()

        used = set()
        for gt_box, gt_kpts, gt_vis in gts:
            gt_xyxy = _xywh_to_xyxy(gt_box)
            best_j, best_iou = -1, 0.0
            for j, pb in enumerate(pred_boxes):
                if j in used:
                    continue
                piou = _iou(gt_xyxy, _xywh_to_xyxy(pb))
                if piou > best_iou:
                    best_j, best_iou = j, piou

            if best_j < 0 or best_iou < iou_match:
                continue

            used.add(best_j)
            n_matched += 1
            pk = pred_kpts[best_j, :4, :2]
            visible = gt_vis > 0
            if visible.sum() == 0:
                continue

            diag = float(np.sqrt(gt_box[2] ** 2 + gt_box[3] ** 2) + 1e-9)
            err = np.linalg.norm(pk[visible] - gt_kpts[visible], axis=1) / diag
            all_errors.extend(err.tolist())
            pck005.extend((err <= 0.05).astype(np.float32).tolist())
            pck010.extend((err <= 0.10).astype(np.float32).tolist())

    return KptMetrics(
        n_gt=n_gt,
        n_matched=n_matched,
        pck_005=float(np.mean(pck005)) if pck005 else 0.0,
        pck_010=float(np.mean(pck010)) if pck010 else 0.0,
        nme=float(np.mean(all_errors)) if all_errors else 0.0,
    )
