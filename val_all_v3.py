from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO
import yaml

from src.register_modules import register_tomato_modules
from src.metrics_keypoints import evaluate_pck_nme


ROOT = Path(__file__).resolve().parent


def find_best_weights(project: Path):
    return sorted(project.rglob("weights/best.pt"))


def metric_value(metrics, name: str, default=0.0):
    try:
        return float(metrics.results_dict.get(name, default))
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser("Validate all best.pt files and export a CSV table.")
    ap.add_argument("--project", default="Tomato_Pose_Runs_v3_Attention")
    ap.add_argument("--data", default="configs/tomato_pose.yaml")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=0)
    ap.add_argument("--out", default="tomato_v3_metrics.csv")
    ap.add_argument("--with-pck", action="store_true", default=True,help="Also compute PCK/NME by prediction-label matching.")
    args = ap.parse_args()

    register_tomato_modules()

    project = Path(args.project)
    if not project.is_absolute():
        project = ROOT / project
    weights = find_best_weights(project)
    if not weights:
        raise FileNotFoundError(f"No best.pt found under {project}")

    rows = []
    for w in weights:
        print(f"[VAL] {w}")
        model = YOLO(str(w))
        metrics = model.val(
            data=str(ROOT / args.data if not Path(args.data).is_absolute() else args.data),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            plots=False,
            verbose=False,
        )

        row = {
            "run": w.parents[1].name,
            "weights": str(w),
            "fitness": getattr(metrics, "fitness", 0.0),
        }
        for k, v in getattr(metrics, "results_dict", {}).items():
            row[k] = v

        if args.with_pck:
            extra = evaluate_pck_nme(
                weights=w,
                data_yaml=ROOT / args.data if not Path(args.data).is_absolute() else args.data,
                split="val",
                imgsz=args.imgsz,
                device=args.device,
            )
            row.update({
                "n_gt": extra.n_gt,
                "n_matched": extra.n_matched,
                "PCK@0.05": extra.pck_005,
                "PCK@0.10": extra.pck_010,
                "NME": extra.nme,
            })

        rows.append(row)

    out = ROOT / args.out
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] saved: {out}")


if __name__ == "__main__":
    main()
