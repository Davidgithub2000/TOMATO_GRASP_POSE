from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser("Convert YOLO-pose tomato labels to COCO keypoints JSON.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_yaml = Path(args.data)
    data = load_yaml(data_yaml)
    root = Path(data["path"])
    if not root.is_absolute():
        root = data_yaml.parent / root

    img_dir = Path(data[args.split])
    if not img_dir.is_absolute():
        img_dir = root / img_dir
    label_dir = Path(str(img_dir).replace(str(Path("images")), str(Path("labels"))))

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{
            "id": 1,
            "name": "tomato",
            "supercategory": "fruit",
            "keypoints": ["tomato_calyx", "tomato_bottom", "tomato_left", "tomato_right"],
            "skeleton": [[1, 2], [3, 4]],
        }],
    }

    ann_id = 1
    img_id = 1
    for img_path in sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 17:
                    continue
                vals = [float(x) for x in parts]
                x, y, bw, bh = vals[1:5]
                x1 = (x - bw / 2) * w
                y1 = (y - bh / 2) * h
                keypoints = []
                num_kpts = 0
                for i in range(4):
                    kx, ky, kv = vals[5 + i * 3: 8 + i * 3]
                    keypoints.extend([kx * w, ky * h, int(kv)])
                    if kv > 0:
                        num_kpts += 1
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x1, y1, bw * w, bh * h],
                    "area": bw * w * bh * h,
                    "iscrowd": 0,
                    "num_keypoints": num_kpts,
                    "keypoints": keypoints,
                })
                ann_id += 1
        img_id += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] saved COCO keypoints json: {out}")


if __name__ == "__main__":
    main()
