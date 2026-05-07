from __future__ import annotations

import argparse
from pathlib import Path
import csv

import pandas as pd


ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser("Collect Ultralytics results.csv files.")
    ap.add_argument("--project", default="Tomato_Pose_Runs_v3_Ablation")
    ap.add_argument("--out", default="tomato_v3_results_summary.csv")
    args = ap.parse_args()

    project = Path(args.project)
    if not project.is_absolute():
        project = ROOT / project

    rows = []
    for csv_path in sorted(project.rglob("results.csv")):
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            best = df.iloc[df["metrics/pose(B)_mAP50-95"].idxmax()] if "metrics/pose(B)_mAP50-95" in df else df.iloc[-1]
            row = {"run": csv_path.parent.name, "results_csv": str(csv_path)}
            for col in df.columns:
                row[col] = best[col]
            rows.append(row)
        except Exception as e:
            rows.append({"run": csv_path.parent.name, "results_csv": str(csv_path), "error": str(e)})

    if not rows:
        print(f"No results.csv found under {project}")
        return

    out = ROOT / args.out
    fieldnames = sorted({k for r in rows for k in r})
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] saved summary: {out}")


if __name__ == "__main__":
    main()
