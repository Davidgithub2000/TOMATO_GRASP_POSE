from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import yaml


COLORS = {
    0: (0, 255, 255),  # calyx
    1: (255, 0, 255),  # bottom
    2: (255, 0, 0),    # left
    3: (0, 0, 255),    # right
}
NAMES = ["calyx", "bottom", "left", "right"]


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser("Visualize YOLO-pose labels.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--num", type=int, default=30)
    ap.add_argument("--out", default="debug_vis")
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

    imgs = [p for p in img_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    random.seed(42)
    random.shuffle(imgs)
    imgs = imgs[: args.num]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 17:
                    continue
                vals = [float(x) for x in parts]
                x, y, bw, bh = vals[1:5]
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 0), 2)

                kpts = vals[5:]
                for i in range(4):
                    kx, ky, kv = kpts[i * 3:(i + 1) * 3]
                    if kv <= 0:
                        continue
                    px, py = int(kx * w), int(ky * h)
                    cv2.circle(img, (px, py), 5, COLORS[i], -1)
                    cv2.putText(img, NAMES[i], (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[i], 1)

        cv2.imwrite(str(out_dir / img_path.name), img)

    print(f"[DONE] saved visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
