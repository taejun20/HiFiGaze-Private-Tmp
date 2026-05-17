"""
Plot Study1 GT gaze locations (cm) for each subject.

Writes per subject under the checkpoint directory (or --out-dir):
  - RGB_500_p{X}_pretrain_scatter.png (GT gaze points in cm)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from pathlib import Path

_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _ROOT)
sys.path.append(os.path.abspath(os.path.join(_ROOT, "..")))

import numpy as np
import torch

warnings.filterwarnings(
    "ignore",
    message=".*You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)

SUBJECT = [2, 6, 7, 8, 10, 11, 12, 13, 15, 16, 19, 25]

WandB_PROJECT = "260507_highres_pretrain"

seed = 2026
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

BATCH_SIZE = 30

os.makedirs("model_checkpoints", exist_ok=True)
CHECKPOINT_DIR = Path(f"model_checkpoints/{WandB_PROJECT}")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SX_CM = 7.1000 / 0.430
SY_CM = 14.3915 / 0.8716
FRAME_W_CM = 7.1
FRAME_H_CM = 14.3915

LEFT_EYE_INNER = 362
LEFT_EYE_OUTER = 263
RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133

def _save_gt_scatter(gt_x: list[float], gt_y: list[float], out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_w = 8.0
    fig_h = fig_w * (FRAME_H_CM / FRAME_W_CM)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.scatter(gt_x, gt_y, s=18, c="#2ca02c", alpha=0.35, edgecolors="none")
    ax.set_xlim(0.0, FRAME_W_CM)
    ax.set_ylim(FRAME_H_CM, 0.0)
    ax.set_title(title)
    ax.set_xlabel("gt_x_cm (cm from left)")
    ax.set_ylabel("gt_y_cm (cm from top)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_all_subjects_gt(subjects: list[int], *, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for sid in subjects:
        metadata_dir = Path(f"study1_rawdata_processed/p{sid}/json")
        metadata_files = sorted(metadata_dir.glob("*.json"))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata json files found under: {metadata_dir}")

        gt_xs: list[float] = []
        gt_ys: list[float] = []
        skipped = 0

        for p in metadata_files:
            try:
                with open(p, "r") as f:
                    d = json.load(f)
                gx = float(d["gt_x_px"]) * 0.001
                gy = float(d["gt_y_px"]) * 0.001
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                skipped += 1
                continue

            # Convert 1000px-units to cm for plotting.
            gt_xs.append(gx * SX_CM)
            gt_ys.append(gy * SY_CM)

        plot_path = out_dir / f"RGB_500_p{sid}_pretrain_scatter.png"
        _save_gt_scatter(
            gt_xs,
            gt_ys,
            plot_path,
            title=f"RGB_500 p{sid} GT gaze in cm (n={len(gt_xs)}, skipped={skipped})",
        )
        print(f"p{sid}: plotted {len(gt_xs)} GT points (skipped={skipped}) -> {plot_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Plot Study1 GT gaze scatter per subject (no model eval)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for PNGs (default: CHECKPOINT_DIR)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve() if args.out_dir else CHECKPOINT_DIR
    print(f"Output directory: {out_dir}")
    plot_all_subjects_gt(SUBJECT, out_dir=out_dir)


if __name__ == "__main__":
    main()
