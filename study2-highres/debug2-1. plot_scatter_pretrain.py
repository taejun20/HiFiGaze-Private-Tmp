from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path


RESOLUTIONS = ["qqvga", "qvga", "sd", "hd", "fhd", "2k", "4k", "6k", "8k", "54mp", "108mp"]
PARTICIPANTS = list(range(1, 11))

# Frame in centimeters, origin at top-left.
FRAME_W_CM = 7.1
FRAME_H_CM = 14.3915

@dataclass(frozen=True)
class EvalFile:
    resolution: str
    eval_p: int
    epoch: int
    path: Path


_EVAL_RE = re.compile(
    r"^Eval_LMK_(?P<res>[^_]+)_(?P<run_tag>train_p[^_]+_val_p[^_]+_eval_p[^_]+)_Epoch(?P<epoch>\d+)\.csv$"
)


def _parse_eval_p_from_run_tag(run_tag: str) -> int | None:
    m = re.search(r"_eval_p(?P<eval>[^_]+)$", run_tag)
    if not m:
        return None
    eval_part = m.group("eval").strip()
    try:
        return int(eval_part.split(",")[0])
    except ValueError:
        return None


def _collect_eval_files(eval_dir: Path) -> list[EvalFile]:
    files: list[EvalFile] = []
    for p in eval_dir.rglob("Eval_LMK_*.csv"):
        m = _EVAL_RE.match(p.name)
        if not m:
            continue
        res = m.group("res")
        if res not in RESOLUTIONS:
            continue
        run_tag = m.group("run_tag")
        eval_p = _parse_eval_p_from_run_tag(run_tag)
        if eval_p is None:
            continue
        try:
            epoch = int(m.group("epoch"))
        except ValueError:
            continue
        files.append(EvalFile(resolution=res, eval_p=eval_p, epoch=epoch, path=p))
    return files


def _clip01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    # h in [0,1), s/v in [0,1]
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
    # Interleave bits (up to 16 bits each is plenty here).
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
    """
    Deterministic, high-contrast color assignment for GT bins.
    Uses Morton code + golden ratio to decorrelate adjacent bins.
    """
    phi = 0.6180339887498949  # golden ratio conjugate
    code = _morton2(ix, iy)
    hue = (code * phi) % 1.0
    return _hsv_to_rgb(hue, 0.90, 0.95)


def _save_heatmap(
    points_by_gtbin: dict[tuple[int, int], list[tuple[float, float]]],
    gt_anchor_by_gtbin: dict[tuple[int, int], tuple[float, float]],
    *,
    out_path: Path,
    width_cm: float,
    height_cm: float,
    title: str,
    vmin: float | None,
    vmax: float | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_w = 8.0
    fig_h = fig_w * (height_cm / width_cm)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)

    # Scatter predicted points, colored by discretized GT position.
    pred_size = 18
    for (ix, iy), pts in points_by_gtbin.items():
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        color = _color_for_gt_bin(ix, iy)
        ax.scatter(
            xs,
            ys,
            s=pred_size,
            c=[color],
            alpha=0.8,
            edgecolors="none",
        )

    # Draw GT anchor positions (same color, 9x larger circles).
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

    # Frame extents, origin at top-left.
    ax.set_xlim(0.0, width_cm)
    ax.set_ylim(height_cm, 0.0)
    ax.set_title(title)
    ax.set_xlabel("pred_x_cm (cm from left)")
    ax.set_ylabel("pred_y_cm (cm from top)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("model_checkpoints/best"),
        help="Directory containing Eval_LMK_*.csv (searched recursively).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("spatial_heatmaps"))
    parser.add_argument("--nx", type=int, default=19, help="Discretization bins along width for GT grouping.")
    parser.add_argument("--ny", type=int, default=39, help="Discretization bins along height for GT grouping.")
    args = parser.parse_args()

    eval_dir: Path = args.eval_dir
    if not eval_dir.exists():
        raise FileNotFoundError(f"Missing eval dir: {eval_dir.resolve()}")

    files = _collect_eval_files(eval_dir)
    if not files:
        raise FileNotFoundError(f"No Eval_LMK_*.csv found under: {eval_dir.resolve()}")

    by_key: dict[tuple[str, int], EvalFile] = {}
    dupes = 0
    for ef in files:
        key = (ef.resolution, ef.eval_p)
        # Prefer the latest epoch if duplicates exist (shouldn't happen for "best/").
        cur = by_key.get(key)
        if cur is None or ef.epoch > cur.epoch:
            if cur is not None:
                dupes += 1
            by_key[key] = ef

    print(f"Found Eval CSVs: {len(files)} (unique res+participant: {len(by_key)}; dupes replaced: {dupes})")

    for res in RESOLUTIONS:
        used_p: list[int] = []
        points_by_gtbin: dict[tuple[int, int], list[tuple[float, float]]] = {}
        gt_sums: dict[tuple[int, int], tuple[float, float, int]] = {}
        for pid in PARTICIPANTS:
            ef = by_key.get((res, pid))
            if ef is None:
                continue
            with open(ef.path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    try:
                        gt_x = float(r["gt_x_cm"])
                        gt_y = float(r["gt_y_cm"])
                        pred_x = float(r["pred_x_cm"])
                        pred_y = float(r["pred_y_cm"])
                    except (KeyError, TypeError, ValueError):
                        continue

                    if not (
                        math.isfinite(gt_x)
                        and math.isfinite(gt_y)
                        and math.isfinite(pred_x)
                        and math.isfinite(pred_y)
                    ):
                        continue
                    if pred_x < 0.0 or pred_x > FRAME_W_CM or pred_y < 0.0 or pred_y > FRAME_H_CM:
                        continue

                    key = _gt_bin_key(
                        gt_x,
                        gt_y,
                        nx=args.nx,
                        ny=args.ny,
                        width_cm=FRAME_W_CM,
                        height_cm=FRAME_H_CM,
                    )
                    if key is None:
                        continue
                    points_by_gtbin.setdefault(key, []).append((pred_x, pred_y))
                    sx, sy, n = gt_sums.get(key, (0.0, 0.0, 0))
                    gt_sums[key] = (sx + gt_x, sy + gt_y, n + 1)
            used_p.append(pid)

        if not points_by_gtbin:
            print(f"[{res}] no participants found; skipping")
            continue

        gt_anchor_by_gtbin: dict[tuple[int, int], tuple[float, float]] = {}
        for key, (sx, sy, n) in gt_sums.items():
            if n <= 0:
                continue
            gt_anchor_by_gtbin[key] = (sx / n, sy / n)

        out_path = args.out_dir / f"scatter_LMK_{res}.png"
        _save_heatmap(
            points_by_gtbin,
            gt_anchor_by_gtbin,
            out_path=out_path,
            width_cm=FRAME_W_CM,
            height_cm=FRAME_H_CM,
            title=f"LMK {res}: pred scatter colored by GT bin (participants {len(used_p)})",
            vmin=None,
            vmax=None,
        )
        print(f"[{res}] saved {out_path} (participants: {used_p}; gt_bins: {len(points_by_gtbin)})")


if __name__ == "__main__":
    main()
