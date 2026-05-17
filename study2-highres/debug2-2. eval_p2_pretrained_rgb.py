"""
Evaluate pretrained RGB gaze model (from 1-1.rgb_pretrain.py) on all frames for subject p2.

Writes per-sample CSV and a scatter of predicted gaze points in phone-frame cm coordinates.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.append(str(_ROOT.parent))

import cv2
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.GazeModel_RGB import GazeModel_RGB

warnings.filterwarnings(
    "ignore",
    message=".*You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)

# Same as 1-1.rgb_pretrain.py
LEFT_EYE_INNER = 362
LEFT_EYE_OUTER = 263
RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133

SEED = 2026
SX_CM = 7.1000 / 0.430
SY_CM = 14.3915 / 0.8716

# Same rectangular frame as debug2-1 plot_scatter_pretrain.py
FRAME_W_CM = 7.1
FRAME_H_CM = 14.3915

EVAL_SUBJECT = 2

DEFAULT_CKPT = (
    _ROOT
    / "model_checkpoints"
    / "pretrained"
    / "RGB_train_p2,6,7,8,10,11,12,13,15,16,19,25_Epoch12_TrainLoss0.977.pt"
)


def _stem_from_checkpoint(ckpt_path: Path) -> str:
    """RGB_train_... without .pt"""
    name = ckpt_path.name
    if name.endswith(".pt"):
        name = name[:-3]
    return name


def _default_out_names(ckpt_path: Path) -> tuple[Path, Path]:
    stem = _stem_from_checkpoint(ckpt_path)
    out_dir = ckpt_path.parent
    csv_name = f"Eval_p{EVAL_SUBJECT}_{stem}.csv"
    plot_name = f"PLOT_Eval_p{EVAL_SUBJECT}_{stem}.png"
    return out_dir / csv_name, out_dir / plot_name


class GazePTDataset(torch.utils.data.Dataset):
    """Study1 preprocessed data — matches 1-1.rgb_pretrain.py."""

    def __init__(self, subjects: list[int]):
        self.data_samples: list[dict] = []
        for subject in subjects:
            eyepatch_dir = Path(f"study1_rawdata_processed/p{subject}/preprocessed_eyecrop")
            landmark_dir = Path(f"study1_rawdata_processed/p{subject}/preprocessed_landmark")
            metadata_dir = Path(f"study1_rawdata_processed/p{subject}/json")

            for metadata_file in sorted(metadata_dir.glob("*.json")):
                frame_num = metadata_file.stem.replace(".json", "")
                l_eye_file = eyepatch_dir / f"{frame_num}_l.png"
                r_eye_file = eyepatch_dir / f"{frame_num}_r.png"
                landmark_file = landmark_dir / f"{frame_num}.json"

                if l_eye_file.exists() and r_eye_file.exists() and landmark_file.exists():
                    self.data_samples.append(
                        {
                            "subject": subject,
                            "frame_num": frame_num,
                            "l_eye_path": l_eye_file,
                            "r_eye_path": r_eye_file,
                            "landmark_path": landmark_file,
                            "metadata_path": metadata_file,
                        }
                    )

        assert len(self.data_samples) > 0, f"No data samples found for subjects {subjects}"
        print(f"Loaded {len(self.data_samples)} data samples from {len(subjects)} subjects")

    def __len__(self) -> int:
        return len(self.data_samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data_samples[idx]

        l_eye_img = cv2.imread(str(sample["l_eye_path"]))
        r_eye_img = cv2.imread(str(sample["r_eye_path"]))
        l_eye_img = cv2.cvtColor(l_eye_img, cv2.COLOR_BGR2RGB)
        r_eye_img = cv2.cvtColor(r_eye_img, cv2.COLOR_BGR2RGB)
        l_eye_img = cv2.resize(l_eye_img, (1400, 700), interpolation=cv2.INTER_LINEAR)
        r_eye_img = cv2.resize(r_eye_img, (1400, 700), interpolation=cv2.INTER_LINEAR)

        l_eye = torch.from_numpy(l_eye_img).permute(2, 0, 1).contiguous().float().mul_(1.0 / 255.0)
        r_eye = torch.from_numpy(r_eye_img).permute(2, 0, 1).contiguous().float().mul_(1.0 / 255.0)

        with open(sample["landmark_path"], "r") as f:
            landmark_data = json.load(f)

        lmk = torch.tensor(
            [
                landmark_data[f"{LEFT_EYE_INNER}x"],
                landmark_data[f"{LEFT_EYE_INNER}y"],
                landmark_data[f"{LEFT_EYE_OUTER}x"],
                landmark_data[f"{LEFT_EYE_OUTER}y"],
                landmark_data[f"{RIGHT_EYE_INNER}x"],
                landmark_data[f"{RIGHT_EYE_INNER}y"],
                landmark_data[f"{RIGHT_EYE_OUTER}x"],
                landmark_data[f"{RIGHT_EYE_OUTER}y"],
            ],
            dtype=torch.float32,
        )

        with open(sample["metadata_path"], "r") as f:
            metadata_data = json.load(f)
        gt = torch.tensor(
            [
                float(metadata_data["gt_x_px"]) * 0.001,
                float(metadata_data["gt_y_px"]) * 0.001,
            ],
            dtype=torch.float32,
        )

        return {
            "subject": sample["subject"],
            "frame_num": sample["frame_num"],
            "l_eye": l_eye,
            "r_eye": r_eye,
            "lmk": lmk,
            "gt": gt,
        }


def _collate(batch: list[dict]) -> dict:
    return {
        "subject": torch.tensor([b["subject"] for b in batch], dtype=torch.long),
        "frame_num": [b["frame_num"] for b in batch],
        "l_eye": torch.stack([b["l_eye"] for b in batch], dim=0),
        "r_eye": torch.stack([b["r_eye"] for b in batch], dim=0),
        "lmk": torch.stack([b["lmk"] for b in batch], dim=0),
        "gt": torch.stack([b["gt"] for b in batch], dim=0),
    }


def _save_pred_scatter(pred_x: list[float], pred_y: list[float], out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_w = 8.0
    fig_h = fig_w * (FRAME_H_CM / FRAME_W_CM)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.scatter(pred_x, pred_y, s=18, c="#1f77b4", alpha=0.35, edgecolors="none")
    ax.set_xlim(0.0, FRAME_W_CM)
    ax.set_ylim(FRAME_H_CM, 0.0)
    ax.set_title(title)
    ax.set_xlabel("pred_x_cm (cm from left)")
    ax.set_ylabel("pred_y_cm (cm from top)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval pretrained RGB on all p2 Study1 samples")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CKPT,
        help="Path to RGB_train_*.pt checkpoint",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Output CSV path (default: next to checkpoint)",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="Output PNG path (default: next to checkpoint)",
    )
    parser.add_argument("-g", type=int, default=0, help="CUDA device index (ignored if no CUDA)")
    parser.add_argument("--batch-size", type=int, default=30)
    args = parser.parse_args()

    ckpt = args.checkpoint.resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    default_csv, default_plot = _default_out_names(ckpt)
    csv_out = args.csv_out.resolve() if args.csv_out else default_csv
    plot_out = args.plot_out.resolve() if args.plot_out else default_plot

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(f"cuda:{args.g}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.g)

    dataset = GazePTDataset([EVAL_SUBJECT])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=min(6, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
    )

    model = GazeModel_RGB().to(device)
    checkpoint = torch.load(str(ckpt), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows: list[dict[str, str | float | int]] = []
    all_px: list[float] = []
    all_py: list[float] = []

    with torch.no_grad():
        for batch in loader:
            l_eye = batch["l_eye"].to(device)
            r_eye = batch["r_eye"].to(device)
            lmk = batch["lmk"].to(device)
            gt = batch["gt"].to(device)

            pred = model(l_eye, r_eye, lmk)
            pred_cm = torch.stack(
                [pred[:, 0] * SX_CM, pred[:, 1] * SY_CM],
                dim=1,
            )
            gt_cm = torch.stack(
                [gt[:, 0] * SX_CM, gt[:, 1] * SY_CM],
                dim=1,
            )
            diff = pred_cm - gt_cm
            loss_cm = torch.norm(diff, dim=1)

            subjects = batch["subject"].tolist()
            frames = batch["frame_num"]
            for i in range(pred_cm.shape[0]):
                px = float(pred_cm[i, 0].item())
                py = float(pred_cm[i, 1].item())
                gx = float(gt_cm[i, 0].item())
                gy = float(gt_cm[i, 1].item())
                lc = float(loss_cm[i].item())
                rows.append(
                    {
                        "subject": int(subjects[i]),
                        "frame_num": str(frames[i]),
                        "pred_x_cm": px,
                        "pred_y_cm": py,
                        "gt_x_cm": gx,
                        "gt_y_cm": gy,
                        "loss_cm": lc,
                    }
                )
                all_px.append(px)
                all_py.append(py)

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subject", "frame_num", "pred_x_cm", "pred_y_cm", "gt_x_cm", "gt_y_cm", "loss_cm"]
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    stem = _stem_from_checkpoint(ckpt)
    _save_pred_scatter(
        all_px,
        all_py,
        plot_out,
        title=f"p{EVAL_SUBJECT} pretrained RGB ({stem})\n{len(rows)} predictions",
    )

    mean_loss = sum(r["loss_cm"] for r in rows) / len(rows) if rows else math.nan
    print(f"Wrote {len(rows)} rows -> {csv_out}")
    print(f"Mean loss_cm: {mean_loss:.4f}")
    print(f"Plot -> {plot_out}")


if __name__ == "__main__":
    main()
