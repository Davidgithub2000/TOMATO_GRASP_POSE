from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_labels(label_dir: Path, nkpt: int = 4, ndim: int = 3):
    n_files = 0
    n_objects = 0
    errors = []

    for txt in sorted(label_dir.rglob("*.txt")):
        n_files += 1
        lines = txt.read_text(encoding="utf-8").strip().splitlines()
        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue
            parts = line.split()
            expected = 5 + nkpt * ndim
            if len(parts) != expected:
                errors.append(f"{txt}:{i} expected {expected} values, got {len(parts)}")
                continue
            vals = [float(x) for x in parts]
            n_objects += 1
            if int(vals[0]) != 0:
                errors.append(f"{txt}:{i} class id should be 0 for tomato.")
            coords = vals[1:]
            for j, v in enumerate(coords):
                # visibility can be 0/1/2, coords should be normalized
                if (j - 4) % 3 == 2 and j >= 4:
                    if v not in (0, 1, 2):
                        errors.append(f"{txt}:{i} visibility value should be 0/1/2, got {v}")
                else:
                    if not (0.0 <= v <= 1.0):
                        errors.append(f"{txt}:{i} normalized coordinate out of range: {v}")

    return n_files, n_objects, errors


def main():
    ap = argparse.ArgumentParser("Check tomato YOLO-pose dataset.")
    ap.add_argument("--data", default=r"D:\code\Tomato_Grasp_Project\2026\yolopose改进项目20260429\tomato_yolo11_pose_improvement_v3\configs\tomato_pose.yaml")
    args = ap.parse_args()

    data_yaml = Path(args.data)
    data = load_yaml(data_yaml)
    root = Path(data["path"])
    if not root.is_absolute():
        root = data_yaml.parent / root

    print(f"[DATA] root: {root}")
    print(f"[DATA] kpt_shape: {data.get('kpt_shape')}")
    print(f"[DATA] flip_idx: {data.get('flip_idx')}")

    if data.get("kpt_shape") != [4, 3]:
        print("[WARN] Expected kpt_shape: [4, 3]")
    if data.get("flip_idx") != [0, 1, 3, 2]:
        print("[WARN] Expected flip_idx: [0, 1, 3, 2] for calyx/bottom/left/right.")

    for split in ["train", "val", "test"]:
        if split not in data or not data.get(split):
            continue
        img_dir = Path(data[split])
        if not img_dir.is_absolute():
            img_dir = root / img_dir
        label_dir = Path(str(img_dir).replace(str(Path("images")), str(Path("labels"))))
        print(f"\n[{split}] images: {img_dir}")
        print(f"[{split}] labels: {label_dir}")
        if not img_dir.exists():
            print(f"[ERROR] image dir not found: {img_dir}")
            continue
        if not label_dir.exists():
            print(f"[ERROR] label dir not found: {label_dir}")
            continue

        n_files, n_objects, errors = check_labels(label_dir)
        print(f"[{split}] label files: {n_files}, objects: {n_objects}, errors: {len(errors)}")
        for e in errors[:20]:
            print("[ERROR]", e)
        if len(errors) > 20:
            print(f"... {len(errors) - 20} more errors")

    print("\n[DONE] dataset check finished.")


if __name__ == "__main__":
    main()
