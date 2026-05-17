from __future__ import annotations

import argparse
import colorsys
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

RESOLUTIONS = ["qqvga", "qvga", "sd", "hd", "fhd", "2k", "4k", "6k", "8k", "54mp", "108mp"]
PARTICIPANTS = list(range(1, 11))

CHECKPOINT_DIR_DEFAULT = Path("model_checkpoints/260512_108MP")

# Frame in centimeters, origin at top-left.
FRAME_W_CM = 7.1
FRAME_H_CM = 14.3915

PHI = 0.6180339887498949  # spread hues across pairs


@dataclass(frozen=True)
class BestRun:
    resolution: str
    eval_p: int
    run_tag: str
    epoch: int
    val_error: float
    ckpt_path: Path


_CKPT_RE = re.compile(
    r"^RGB_(?P<res>[^_]+)_(?P<run_tag>train_p[^_]+_val_p[^_]+_eval_p[^_]+)_"
    r"Epoch(?P<epoch>\d+)_ValError(?P<valerr>\d+\.\d+)_EvalError(?P<evalerr>\d+\.\d+)\.pt$"
)


def _parse_eval_p(run_tag: str) -> int | None:
    m = re.search(r"_eval_p(?P<eval>[^_]+)$", run_tag)
    if not m:
        return None
    eval_part = m.group("eval").strip()
    try:
        return int(eval_part.split(",")[0])
    except ValueError:
        return None


def _collect_best_runs(checkpoint_dir: Path) -> dict[tuple[str, int], BestRun]:
    best: dict[tuple[str, int], BestRun] = {}
    for p in checkpoint_dir.iterdir():
        if not p.is_file() or p.suffix != ".pt":
            continue
        m = _CKPT_RE.match(p.name)
        if not m:
            continue
        res = m.group("res")
        if res not in RESOLUTIONS:
            continue
        run_tag = m.group("run_tag")
        eval_p = _parse_eval_p(run_tag)
        if eval_p is None:
            continue
        try:
            epoch = int(m.group("epoch"))
            valerr = float(m.group("valerr"))
        except ValueError:
            continue
        key = (res, eval_p)
        cur = best.get(key)
        cand = BestRun(
            resolution=res,
            eval_p=eval_p,
            run_tag=run_tag,
            epoch=epoch,
            val_error=valerr,
            ckpt_path=p,
        )
        if cur is None or cand.val_error < cur.val_error:
            best[key] = cand
    return best


def _pair_color(i: int, n: int) -> tuple[float, float, float]:
    """RGB in [0,1], visually separated per index."""
    if n <= 0:
        return (0.5, 0.5, 0.5)
    h = ((i * PHI) % 1.0)
    return colorsys.hsv_to_rgb(h, 0.85, 0.92)


def _save_scatter_pairs(
    pairs: list[tuple[float, float, float, float]],
    *,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_w = 8.0
    fig_h = fig_w * (FRAME_H_CM / FRAME_W_CM)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)

    n = len(pairs)
    for i, (px, py, gx, gy) in enumerate(pairs):
        c = _pair_color(i, n)
        ax.scatter([px], [py], s=14, c=[c], alpha=0.85, edgecolors="none", zorder=1)
        ax.scatter(
            [gx],
            [gy],
            s=36,
            c=[c],
            alpha=0.95,
            marker="o",
            edgecolors="black",
            linewidths=0.35,
            zorder=2,
        )

    ax.set_xlim(0.0, FRAME_W_CM)
    ax.set_ylim(FRAME_H_CM, 0.0)
    ax.set_title(title)
    ax.set_xlabel("x_cm (cm from left)")
    ax.set_ylabel("y_cm (cm from top)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _load_eval_pairs(csv_path: Path, eval_p: int) -> list[tuple[float, float, float, float]]:
    pairs: list[tuple[float, float, float, float]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return pairs
        need = {"subject", "pred_x_cm", "pred_y_cm", "gt_x_cm", "gt_y_cm"}
        if not need.issubset(set(reader.fieldnames)):
            return pairs
        for row in reader:
            try:
                sid = int(row["subject"])
            except (TypeError, ValueError, KeyError):
                continue
            if sid != eval_p:
                continue
            try:
                px = float(row["pred_x_cm"])
                py = float(row["pred_y_cm"])
                gx = float(row["gt_x_cm"])
                gy = float(row["gt_y_cm"])
            except (TypeError, ValueError, KeyError):
                continue
            if not all(math.isfinite(v) for v in (px, py, gx, gy)):
                continue
            pairs.append((px, py, gx, gy))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Best ValError checkpoint per (res, eval_p) -> Eval_RGB CSV -> pred/GT scatter (unique color per frame)."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=CHECKPOINT_DIR_DEFAULT,
        help="Directory with RGB_*_Epoch*_ValError*_EvalError*.pt",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("spatial_heatmaps1400"))
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir.resolve()
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Missing checkpoint dir: {ckpt_dir}")

    best = _collect_best_runs(ckpt_dir)
    print(f"Best runs indexed: {len(best)} (from {ckpt_dir})")

    out_dir = args.out_dir.resolve()
    n_written = 0
    n_missing_csv = 0

    for res in RESOLUTIONS:
        for pid in PARTICIPANTS:
            br = best.get((res, pid))
            if br is None:
                continue
            csv_name = f"Eval_RGB_{res}_{br.run_tag}_Epoch{br.epoch}.csv"
            csv_path = ckpt_dir / csv_name
            if not csv_path.is_file():
                n_missing_csv += 1
                print(f"[WARN] Missing eval CSV for best run: {csv_path.name}")
                continue

            pairs = _load_eval_pairs(csv_path, br.eval_p)
            if not pairs:
                print(f"[WARN] No rows for p{pid} in {csv_name}")
                continue

            out_path = out_dir / f"scatter_RGB_{res}_p{pid}.png"
            title = (
                f"RGB {res} p{pid} (best ValError={br.val_error:.3f}, epoch={br.epoch}, n={len(pairs)})\n"
                f"{br.run_tag}"
            )
            _save_scatter_pairs(pairs, out_path=out_path, title=title)
            n_written += 1
            print(f"[OK] {out_path.name} ({len(pairs)} pairs)")

    print(f"Done. Wrote {n_written} plots -> {out_dir}")
    if n_missing_csv:
        print(f"Missing CSV count: {n_missing_csv}")


if __name__ == "__main__":
    main()
