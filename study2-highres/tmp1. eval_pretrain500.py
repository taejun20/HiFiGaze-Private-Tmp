"""
Load RGB_500 pretrained checkpoint (500×250 eye crops) and evaluate each training subject.

Writes per subject under the checkpoint directory (or --out-dir):
  - RGB_500_p{X}_eval.csv
  - RGB_500_p{X}_pretrain_scatter.png
"""
from __future__ import annotations  

import argparse
import csv
import json
import math
import os
import random
import sys
import warnings
from pathlib import Path
_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _ROOT)
sys.path.append(os.path.abspath(os.path.join(_ROOT, "..")))
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.GazeModel_RGB import GazeModel_RGB
warnings.filterwarnings(
    "ignore",
    message=".*You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)

TYPE = "RGB_500"

SUBJECT = [2,6,7,8,10,11,12,13,15,16,19,25] # 12 subjects. Excluding the 10 participants who have also participated in study 2.
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

DEFAULT_CKPT = CHECKPOINT_DIR / (
    "RGB_500_train_p2,6,7,8,10,11,12,13,15,16,19,25_Epoch12_TrainLoss1.897.pt"
)

SX_CM = 7.1000 / 0.430
SY_CM = 14.3915 / 0.8716
FRAME_W_CM = 7.1
FRAME_H_CM = 14.3915

LEFT_EYE_INNER = 362
LEFT_EYE_OUTER = 263
RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133


class GazePTDataset(torch.utils.data.Dataset):
    def __init__(self, subjects):
        self.data_samples = []
        for subject in subjects:
            eyepatch_dir = Path(f"study1_rawdata_processed/p{subject}/preprocessed_eyecrop")
            landmark_dir = Path(f"study1_rawdata_processed/p{subject}/preprocessed_landmark")
            metadata_dir = Path(f"study1_rawdata_processed/p{subject}/json")

            metadata_files = sorted(metadata_dir.glob("*.json"))

            for metadata_file in metadata_files:
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

        assert len(self.data_samples) > 0, f"No data samples found in preprocessed for subjects {subjects}"
        print(f"Loaded {len(self.data_samples)} data samples from {len(subjects)} subjects")

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]

        l_eye_img = cv2.imread(str(sample["l_eye_path"]))
        r_eye_img = cv2.imread(str(sample["r_eye_path"]))

        l_eye_img = cv2.cvtColor(l_eye_img, cv2.COLOR_BGR2RGB)
        r_eye_img = cv2.cvtColor(r_eye_img, cv2.COLOR_BGR2RGB)

        l_eye_img = cv2.resize(l_eye_img, (500, 250), interpolation=cv2.INTER_LINEAR)
        r_eye_img = cv2.resize(r_eye_img, (500, 250), interpolation=cv2.INTER_LINEAR)

        l_eye = torch.from_numpy(l_eye_img).permute(2, 0, 1).contiguous().float().mul_(1.0 / 255.0)
        r_eye = torch.from_numpy(r_eye_img).permute(2, 0, 1).contiguous().float().mul_(1.0 / 255.0)

        with open(sample["landmark_path"], "r") as f:
            landmark_data = json.load(f)

        lmk = torch.tensor([
                landmark_data[f"{LEFT_EYE_INNER}x"],
                landmark_data[f"{LEFT_EYE_INNER}y"],
                landmark_data[f"{LEFT_EYE_OUTER}x"],
                landmark_data[f"{LEFT_EYE_OUTER}y"],
                landmark_data[f"{RIGHT_EYE_INNER}x"],
                landmark_data[f"{RIGHT_EYE_INNER}y"],
                landmark_data[f"{RIGHT_EYE_OUTER}x"],
                landmark_data[f"{RIGHT_EYE_OUTER}y"],
            ], dtype=torch.float32)

        with open(sample["metadata_path"], "r") as f:
            metadata_data = json.load(f)

        gt = torch.tensor([
            float(metadata_data["gt_x_px"]) * 0.001, # convert to unit of 1000 px to normalize the value in 0-1 range.
            float(metadata_data["gt_y_px"]) * 0.001, # convert to unit of 1000 px to normalize the value in 0-1 range.
        ], dtype=torch.float32)

        return {
            "subject": sample["subject"],
            "frame_num": sample["frame_num"],
            "l_eye": l_eye,
            "r_eye": r_eye,
            "lmk": lmk,
            "gt": gt,
        }


def _collate_eval(batch: list[dict]) -> dict:
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


def _load_model_weights(model: GazeModel_RGB, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)


def evaluate_all_subjects(
    subjects: list[int],
    device: torch.device,
    *,
    checkpoint_path: Path,
    out_dir: Path,
    batch_size: int,
) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path.resolve()}")

    model = GazeModel_RGB().to(device)
    _load_model_weights(model, checkpoint_path, device)
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)

    for sid in subjects:
        dataset = GazePTDataset([sid])
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=min(6, os.cpu_count() or 1),
            pin_memory=torch.cuda.is_available(),
            collate_fn=_collate_eval,
        )

        rows: list[dict] = []
        pred_xs: list[float] = []
        pred_ys: list[float] = []

        with torch.no_grad():
            for batch in loader:
                l_eye = batch["l_eye"].to(device)
                r_eye = batch["r_eye"].to(device)
                lmk = batch["lmk"].to(device)
                gt = batch["gt"].to(device)

                pred = model(l_eye, r_eye, lmk)
                pred_cm = torch.stack([pred[:, 0] * SX_CM, pred[:, 1] * SY_CM], dim=1)
                gt_cm = torch.stack([gt[:, 0] * SX_CM, gt[:, 1] * SY_CM], dim=1)
                loss_cm = torch.norm(pred_cm - gt_cm, dim=1)

                subj_list = batch["subject"].tolist()
                frames = batch["frame_num"]
                for i in range(pred_cm.shape[0]):
                    px = float(pred_cm[i, 0].item())
                    py = float(pred_cm[i, 1].item())
                    pred_xs.append(px)
                    pred_ys.append(py)
                    rows.append(
                        {
                            "subject": int(subj_list[i]),
                            "frame_num": str(frames[i]),
                            "gt_x_cm": float(gt_cm[i, 0].item()),
                            "gt_y_cm": float(gt_cm[i, 1].item()),
                            "pred_x_cm": px,
                            "pred_y_cm": py,
                            "loss_cm": float(loss_cm[i].item()),
                        }
                    )

        csv_path = out_dir / f"{TYPE}_p{sid}_eval.csv"
        fieldnames = ["subject", "frame_num", "gt_x_cm", "gt_y_cm", "pred_x_cm", "pred_y_cm", "loss_cm"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        plot_path = out_dir / f"{TYPE}_p{sid}_pretrain_scatter.png"
        _save_pred_scatter(
            pred_xs,
            pred_ys,
            plot_path,
            title=f"{TYPE} p{sid} pretrained preds (n={len(rows)})",
        )

        mean_l = sum(r["loss_cm"] for r in rows) / len(rows) if rows else math.nan
        print(f"p{sid}: {len(rows)} rows -> {csv_path.name}; mean loss_cm={mean_l:.4f}; {plot_path.name}")


def main():
    parser = argparse.ArgumentParser(description=f"Eval {TYPE} pretrained model per subject")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CKPT,
        help=f"Path to {TYPE}_train_...pt",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for CSV and PNG (default: same as checkpoint parent)",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("-g", type=int, default=0, help="CUDA device index")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.g}")
        torch.cuda.set_device(args.g)
    else:
        device = torch.device("cpu")
        print("Warning: CUDA not available, using CPU")

    print("CUDA available:", torch.cuda.is_available())
    print(f"Using device: {device}")

    ckpt = args.checkpoint.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else ckpt.parent
    print(f"\nEvaluating checkpoint: {ckpt}")
    print(f"Output directory: {out_dir}")
    evaluate_all_subjects(SUBJECT, device, checkpoint_path=ckpt, out_dir=out_dir, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
