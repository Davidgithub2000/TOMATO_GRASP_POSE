from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


KPT_ORDER = ["tomato_calyx", "tomato_bottom", "tomato_left", "tomato_right"]
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def polygon_bbox(poly: np.ndarray) -> Tuple[float, float, float, float]:
    x1, y1 = poly[:, 0].min(), poly[:, 1].min()
    x2, y2 = poly[:, 0].max(), poly[:, 1].max()
    return x1, y1, x2, y2


def point_inside(poly: np.ndarray, pt: Tuple[float, float]) -> bool:
    return cv2.pointPolygonTest(poly.astype(np.float32), pt, False) >= 0


def point_dist_to_center(poly: np.ndarray, pt: Tuple[float, float]) -> float:
    c = poly.mean(axis=0)
    return float(np.linalg.norm(np.array(pt) - c))


def convert_one(json_path: Path, img_path: Path, object_labels: set[str]):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    h, w = img.shape[:2]

    polygons = []
    points_by_label: Dict[str, List[Tuple[float, float]]] = {k: [] for k in KPT_ORDER}

    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        pts = shape.get("points", [])
        if not pts:
            continue
        if label in object_labels and len(pts) >= 3:
            polygons.append(np.array(pts, dtype=np.float32))
        elif label in points_by_label:
            points_by_label[label].append(tuple(map(float, pts[0])))

    rows = []
    for poly in polygons:
        x1, y1, x2, y2 = polygon_bbox(poly)
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        cx = x1 + bw / 2
        cy = y1 + bh / 2

        kpts = []
        for name in KPT_ORDER:
            candidates = points_by_label.get(name, [])
            inside = [p for p in candidates if point_inside(poly, p)]
            if inside:
                # If multiple points of the same label fall inside, choose closest to polygon center.
                chosen = min(inside, key=lambda p: point_dist_to_center(poly, p))
                kpts.extend([chosen[0] / w, chosen[1] / h, 2])
            elif candidates:
                chosen = min(candidates, key=lambda p: point_dist_to_center(poly, p))
                kpts.extend([chosen[0] / w, chosen[1] / h, 2])
            else:
                kpts.extend([0.0, 0.0, 0])

        row = [
            0,
            cx / w,
            cy / h,
            bw / w,
            bh / h,
            *kpts,
        ]
        row = [max(0.0, min(1.0, float(v))) if i > 0 and (i - 5) % 3 != 2 else v for i, v in enumerate(row)]
        rows.append(row)

    return rows


def write_yaml(out: Path):
    yaml_text = f"""path: {out.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: tomato

kpt_shape: [4, 3]
flip_idx: [0, 1, 3, 2]

kpt_names:
  0:
    - tomato_calyx
    - tomato_bottom
    - tomato_left
    - tomato_right
"""
    (out / "tomato_pose.yaml").write_text(yaml_text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser("Convert Labelme tomato polygons + 4 keypoints to YOLO-pose.")
    ap.add_argument("--labelme-dir", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--object-labels", default="red,tomato,tomato_polygon")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--test-ratio", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy-images", action="store_true")
    args = ap.parse_args()

    labelme_dir = Path(args.labelme_dir)
    images_dir = Path(args.images_dir)
    out = Path(args.out)
    object_labels = {x.strip() for x in args.object_labels.split(",") if x.strip()}

    for split in ["train", "val", "test"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    jsons = sorted(labelme_dir.glob("*.json"))
    random.seed(args.seed)
    random.shuffle(jsons)

    n = len(jsons)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)
    split_map = {}
    for i, jp in enumerate(jsons):
        if i < n_test:
            split_map[jp] = "test"
        elif i < n_test + n_val:
            split_map[jp] = "val"
        else:
            split_map[jp] = "train"

    for jp in tqdm(jsons, desc="Converting"):
        img_path = find_image(images_dir, jp.stem)
        if img_path is None:
            print(f"[WARN] image not found for {jp.name}")
            continue
        split = split_map[jp]
        rows = convert_one(jp, img_path, object_labels)

        label_out = out / "labels" / split / f"{jp.stem}.txt"
        with open(label_out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(" ".join(str(int(v)) if i == 0 else f"{float(v):.6f}" for i, v in enumerate(row)) + "\n")

        img_out = out / "images" / split / img_path.name
        if args.copy_images:
            shutil.copy2(img_path, img_out)
        else:
            if not img_out.exists():
                shutil.copy2(img_path, img_out)

    write_yaml(out)
    print(f"[DONE] dataset saved to: {out}")
    print(f"[DONE] yaml: {out / 'tomato_pose.yaml'}")


if __name__ == "__main__":
    main()
