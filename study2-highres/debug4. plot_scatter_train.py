from __future__ import annotations

import argparse
import math
from pathlib import Path

import json
import torch
from torch.utils.data import DataLoader

from models.GazeModel_RGB import GazeModel_RGB

# ---- User-configurable globals ----
PARTICIPANT = 7
RESOLUTION = "sd"
CHECKPOINT_DIR = Path("model_checkpoints/260508_108MP")
CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "RGB_sd_train_p7,8,9,10,1,2,3_val_p4,5_eval_p6_Epoch9_ValError1.648_EvalError1.666.pt"
)
OUT_PATH = f"scatter_train_p{PARTICIPANT}_{RESOLUTION}.png"

# RGB_108mp_train_p7,8,9,10,1,2,3_val_p4,5_eval_p6_Epoch3_ValError1.679_EvalError1.422.pt
# RGB_qqvga_train_p7,8,9,10,1,2,3_val_p4,5_eval_p6_Epoch7_ValError2.060_EvalError2.317.pt
# RGB_sd_train_p7,8,9,10,1,2,3_val_p4,5_eval_p6_Epoch9_ValError1.648_EvalError1.666.pt

# Frame in centimeters, origin at top-left.
FRAME_W_CM = 7.1
FRAME_H_CM = 14.3915

LEFT_EYE_INNER = 362
LEFT_EYE_OUTER = 263
RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133


def _clip01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    h = h % 1.0
    s = _clip01(s)
    v = _clip01(v)
    if s == 0.0:
        return (v, v, v)
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    if i == 0:
        return (v, t, p)
    if i == 1:
        return (q, v, p)
    if i == 2:
        return (p, v, t)
    if i == 3:
        return (p, q, v)
    if i == 4:
        return (t, p, v)
    return (v, p, q)


def _morton2(x: int, y: int) -> int:
    def _part1by1(n: int) -> int:
        n &= 0xFFFF
        n = (n | (n << 8)) & 0x00FF00FF
        n = (n | (n << 4)) & 0x0F0F0F0F
        n = (n | (n << 2)) & 0x33333333
        n = (n | (n << 1)) & 0x55555555
        return n

    return _part1by1(x) | (_part1by1(y) << 1)


def _gt_bin_key(
    gt_x_cm: float,
    gt_y_cm: float,
    *,
    nx: int,
    ny: int,
    width_cm: float,
    height_cm: float,
) -> tuple[int, int] | None:
    if not (math.isfinite(gt_x_cm) and math.isfinite(gt_y_cm)):
        return None
    if gt_x_cm < 0.0 or gt_x_cm > width_cm or gt_y_cm < 0.0 or gt_y_cm > height_cm:
        return None
    ix = int(math.floor(gt_x_cm / width_cm * nx))
    iy = int(math.floor(gt_y_cm / height_cm * ny))
    if ix < 0:
        ix = 0
    elif ix >= nx:
        ix = nx - 1
    if iy < 0:
        iy = 0
    elif iy >= ny:
        iy = ny - 1
    return (ix, iy)


def _color_for_gt_bin(ix: int, iy: int) -> tuple[float, float, float]:
    phi = 0.6180339887498949
    code = _morton2(ix, iy)
    hue = (code * phi) % 1.0
    return _hsv_to_rgb(hue, 0.90, 0.95)


def _save_scatter(
    pred_by_gtbin: dict[tuple[int, int], list[tuple[float, float]]],
    gt_anchor_by_gtbin: dict[tuple[int, int], tuple[float, float]],
    *,
    out_path: Path,
    width_cm: float,
    height_cm: float,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_w = 8.0
    fig_h = fig_w * (height_cm / width_cm)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)

    pred_size = 12
    for (ix, iy), pts in pred_by_gtbin.items():
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        color = _color_for_gt_bin(ix, iy)
        ax.scatter(xs, ys, s=pred_size, c=[color], alpha=0.8, edgecolors="none")

    for (ix, iy), (gt_x, gt_y) in gt_anchor_by_gtbin.items():
        color = _color_for_gt_bin(ix, iy)
        ax.scatter(
            [gt_x],
            [gt_y],
            s=pred_size * 9,
            c=[color],
            alpha=1.0,
            marker="o",
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

    ax.set_xlim(0.0, width_cm)
    ax.set_ylim(height_cm, 0.0)
    ax.set_title(title)
    ax.set_xlabel("pred_x_cm (cm from left)")
    ax.set_ylabel("pred_y_cm (cm from top)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


class GazePTDataset(torch.utils.data.Dataset):
    def __init__(self, subject: int, res: str):
        self.subject = subject
        self.res = res
        self.data_samples: list[dict[str, Path | str | int]] = []

        eyepatch_dir = Path(f"study2_rawdata_processed/p{subject}/preprocessed/{res}_frames/eyecrop")
        landmark_dir = Path(f"study2_rawdata_processed/p{subject}/preprocessed/{res}_frames/landmark")
        metadata_dir = Path(f"study2_rawdata_processed/p{subject}/json")

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

        if not self.data_samples:
            raise FileNotFoundError(
                f"No samples for p{subject} res={res}. "
                f"Checked {eyepatch_dir}, {landmark_dir}, {metadata_dir}"
            )

        print(f"Loaded {len(self.data_samples)} samples for p{subject} ({res})")

    def __len__(self) -> int:
        return len(self.data_samples)

    def __getitem__(self, idx: int):
        sample = self.data_samples[idx]

        # Prefer OpenCV if available, otherwise fall back to Pillow.
        l_eye: torch.Tensor
        r_eye: torch.Tensor
        try:
            import cv2  # type: ignore

            l_eye_img = cv2.imread(str(sample["l_eye_path"]))
            r_eye_img = cv2.imread(str(sample["r_eye_path"]))
            if l_eye_img is None or r_eye_img is None:
                raise FileNotFoundError(f"Missing eye images for frame={sample['frame_num']}")

            l_eye_img = cv2.cvtColor(l_eye_img, cv2.COLOR_BGR2RGB)
            r_eye_img = cv2.cvtColor(r_eye_img, cv2.COLOR_BGR2RGB)

            l_eye_img = cv2.resize(l_eye_img, (1400, 700), interpolation=cv2.INTER_LINEAR)
            r_eye_img = cv2.resize(r_eye_img, (1400, 700), interpolation=cv2.INTER_LINEAR)

            l_eye = (
                torch.from_numpy(l_eye_img).permute(2, 0, 1).contiguous().float().mul_(1.0 / 255.0)
            )
            r_eye = (
                torch.from_numpy(r_eye_img).permute(2, 0, 1).contiguous().float().mul_(1.0 / 255.0)
            )
        except ModuleNotFoundError:
            from PIL import Image

            def _load_rgb(p: Path) -> torch.Tensor:
                img = Image.open(p).convert("RGB").resize((1400, 700))
                # HWC uint8 -> CHW float in [0,1]
                data = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
                data = data.view(700, 1400, 3).permute(2, 0, 1).contiguous()
                return data.float().mul_(1.0 / 255.0)

            l_eye = _load_rgb(sample["l_eye_path"])  # type: ignore[arg-type]
            r_eye = _load_rgb(sample["r_eye_path"])  # type: ignore[arg-type]

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

        return {"l_eye": l_eye, "r_eye": r_eye, "lmk": lmk, "gt": gt}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=PARTICIPANT)
    parser.add_argument("--res", type=str, default=RESOLUTION)
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=0,
        help="CUDA device index to use (set -1 for CPU).",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_PATH,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--nx", type=int, default=19)
    parser.add_argument("--ny", type=int, default=39)
    args = parser.parse_args()

    if not args.ckpt.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.ckpt.resolve()}")

    if args.gpu is not None and int(args.gpu) >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    model = GazeModel_RGB().to(device)
    ckpt = torch.load(str(args.ckpt), map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}")

    dataset = GazePTDataset(int(args.p), str(args.res))
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)

    pred_by_gtbin: dict[tuple[int, int], list[tuple[float, float]]] = {}
    gt_sums: dict[tuple[int, int], tuple[float, float, int]] = {}

    with torch.no_grad():
        for batch in loader:
            l_eye = batch["l_eye"].to(device)
            r_eye = batch["r_eye"].to(device)
            lmk = batch["lmk"].to(device)
            gt = batch["gt"].to(device)

            pred = model(l_eye, r_eye, lmk)

            pred_cm = torch.stack(
                [pred[:, 0] * (7.1000 / 0.430), pred[:, 1] * (14.3915 / 0.8716)], dim=1
            )
            gt_cm = torch.stack([gt[:, 0] * (7.1000 / 0.430), gt[:, 1] * (14.3915 / 0.8716)], dim=1)

            pred_cm = pred_cm.detach().cpu().tolist()
            gt_cm = gt_cm.detach().cpu().tolist()

            for (px, py), (gx, gy) in zip(pred_cm, gt_cm, strict=False):
                if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(gx) and math.isfinite(gy)):
                    continue
                if px < 0.0 or px > FRAME_W_CM or py < 0.0 or py > FRAME_H_CM:
                    continue

                key = _gt_bin_key(
                    gx,
                    gy,
                    nx=int(args.nx),
                    ny=int(args.ny),
                    width_cm=FRAME_W_CM,
                    height_cm=FRAME_H_CM,
                )
                if key is None:
                    continue
                pred_by_gtbin.setdefault(key, []).append((px, py))
                sx, sy, n = gt_sums.get(key, (0.0, 0.0, 0))
                gt_sums[key] = (sx + gx, sy + gy, n + 1)

    if not pred_by_gtbin:
        raise RuntimeError("No prediction points collected; check preprocessing paths.")

    gt_anchor_by_gtbin: dict[tuple[int, int], tuple[float, float]] = {}
    for key, (sx, sy, n) in gt_sums.items():
        if n > 0:
            gt_anchor_by_gtbin[key] = (sx / n, sy / n)

    _save_scatter(
        pred_by_gtbin,
        gt_anchor_by_gtbin,
        out_path=args.out,
        width_cm=FRAME_W_CM,
        height_cm=FRAME_H_CM,
        title=f"p{int(args.p)} ({args.res}) preds colored by GT (bins {args.nx}x{args.ny})",
    )
    print(f"Saved: {args.out.resolve()} (gt_bins={len(pred_by_gtbin)})")


if __name__ == "__main__":
    main()
