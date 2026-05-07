from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def group_key(path: Path, sep: str = "_", n: int = 2) -> str:
    parts = path.stem.split(sep)
    return sep.join(parts[:n]) if len(parts) >= n else path.stem


def main():
    ap = argparse.ArgumentParser("Split images/labels by group to reduce sequence leakage.")
    ap.add_argument("--src-root", required=True, help="Root containing images/all and labels/all.")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--group-parts", type=int, default=2)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src_root)
    out = Path(args.out_root)
    images = sorted([p for p in (src / "images" / "all").rglob("*") if p.suffix.lower() in IMG_EXTS])

    groups = {}
    for img in images:
        groups.setdefault(group_key(img, n=args.group_parts), []).append(img)

    keys = sorted(groups)
    random.seed(args.seed)
    random.shuffle(keys)

    n = len(keys)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)

    split_for = {}
    for i, k in enumerate(keys):
        if i < n_test:
            split_for[k] = "test"
        elif i < n_test + n_val:
            split_for[k] = "val"
        else:
            split_for[k] = "train"

    for split in ["train", "val", "test"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    for k, imgs in groups.items():
        split = split_for[k]
        for img in imgs:
            label = src / "labels" / "all" / f"{img.stem}.txt"
            shutil.copy2(img, out / "images" / split / img.name)
            if label.exists():
                shutil.copy2(label, out / "labels" / split / label.name)

    print(f"[DONE] split saved to {out}")
    print({s: sum(1 for v in split_for.values() if v == s) for s in ["train", "val", "test"]})


if __name__ == "__main__":
    main()
