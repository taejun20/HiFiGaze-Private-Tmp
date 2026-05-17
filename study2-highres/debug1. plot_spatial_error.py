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
    r"^Eval_RGB_(?P<res>[^_]+)_(?P<run_tag>train_p[^_]+_val_p[^_]+_eval_p[^_]+)_Epoch(?P<epoch>\d+)\.csv$"
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
    for p in eval_dir.rglob("Eval_RGB_*.csv"):
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


def _bin_mean_loss(
    rows: list[dict[str, str]],
    *,
    nx: int,
    ny: int,
    width_cm: float,
    height_cm: float,
) -> list[list[float | None]]:
    """
    Return (ny, nx) grid: mean loss per spatial bin (None where no samples).
    Coordinates are interpreted from the top-left origin in centimeters.
    """
    sums: list[list[float]] = [[0.0 for _ in range(nx)] for _ in range(ny)]
    counts: list[list[int]] = [[0 for _ in range(nx)] for _ in range(ny)]

    for r in rows:
        try:
            x = float(r["gt_x_cm"])
            y = float(r["gt_y_cm"])
            loss = float(r["loss_cm"])
        except (KeyError, TypeError, ValueError):
            continue

        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(loss)):
            continue
        if x < 0.0 or x > width_cm or y < 0.0 or y > height_cm:
            continue

        ix = int(math.floor(x / width_cm * nx))
        iy = int(math.floor(y / height_cm * ny))
        if ix < 0:
            ix = 0
        elif ix >= nx:
            ix = nx - 1
        if iy < 0:
            iy = 0
        elif iy >= ny:
            iy = ny - 1

        sums[iy][ix] += loss
        counts[iy][ix] += 1

    out: list[list[float | None]] = [[None for _ in range(nx)] for _ in range(ny)]
    for y in range(ny):
        for x in range(nx):
            c = counts[y][x]
            if c > 0:
                out[y][x] = sums[y][x] / c
    return out


def _save_heatmap(
    heat: list[list[float | None]],
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

    fig_w = 6.0
    fig_h = fig_w * (height_cm / width_cm)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)

    if vmin is None or vmax is None:
        vals: list[float] = []
        for row in heat:
            for v in row:
                if v is None:
                    continue
                if math.isfinite(v):
                    vals.append(float(v))
        if vals:
            vals.sort()
            p5 = vals[int(0.05 * (len(vals) - 1))]
            p95 = vals[int(0.95 * (len(vals) - 1))]
            vmin, vmax = float(p5), float(p95)
        else:
            vmin, vmax = 0.0, 1.0

    ny = len(heat)
    nx = len(heat[0]) if ny > 0 else 0

    xs: list[float] = []
    ys: list[float] = []
    cs: list[float] = []
    for iy, row in enumerate(heat):
        for ix, v in enumerate(row):
            if v is None or not math.isfinite(v):
                continue
            # Bin center coordinates (cm from top-left).
            x = (ix + 0.5) * (width_cm / nx)
            y = (iy + 0.5) * (height_cm / ny)
            xs.append(x)
            ys.append(y)
            cs.append(float(v))

    # Draw as larger colored points (instead of dense image).
    sc = ax.scatter(
        xs,
        ys,
        c=cs,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        s=800,  # 4x bigger points
        marker="s",
        linewidths=0.0,
    )

    # Label each point with 0.1 precision, slightly above it.
    y_off = (height_cm / ny) * 1.0 if ny else 0.0
    for x, y, v in zip(xs, ys, cs, strict=False):
        ax.text(
            x,
            y - y_off,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=30,  # 10x larger text
            color="black",
        )

    # Frame extents, origin at top-left.
    ax.set_xlim(0.0, width_cm)
    ax.set_ylim(height_cm, 0.0)
    ax.set_title(title)
    ax.set_xlabel("gt_x_cm (cm from left)")
    ax.set_ylabel("gt_y_cm (cm from top)")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("mean loss_cm")

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
        help="Directory containing Eval_RGB_*.csv (searched recursively).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("spatial_heatmaps"))
    parser.add_argument("--nx", type=int, default=71, help="Number of bins along width (x).")
    parser.add_argument("--ny", type=int, default=144, help="Number of bins along height (y).")
    parser.add_argument("--min-participants", type=int, default=1, help="Min participants per-bin to keep.")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    args = parser.parse_args()

    eval_dir: Path = args.eval_dir
    if not eval_dir.exists():
        raise FileNotFoundError(f"Missing eval dir: {eval_dir.resolve()}")

    files = _collect_eval_files(eval_dir)
    if not files:
        raise FileNotFoundError(f"No Eval_RGB_*.csv found under: {eval_dir.resolve()}")

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
        participant_maps: list[list[list[float | None]]] = []
        used_p: list[int] = []
        for pid in PARTICIPANTS:
            ef = by_key.get((res, pid))
            if ef is None:
                continue
            with open(ef.path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            heat_p = _bin_mean_loss(
                rows,
                nx=args.nx,
                ny=args.ny,
                width_cm=FRAME_W_CM,
                height_cm=FRAME_H_CM,
            )
            participant_maps.append(heat_p)
            used_p.append(pid)

        if not participant_maps:
            print(f"[{res}] no participants found; skipping")
            continue

        # Aggregate participant means (mean of means), ignoring missing bins.
        agg_sums: list[list[float]] = [[0.0 for _ in range(args.nx)] for _ in range(args.ny)]
        agg_counts: list[list[int]] = [[0 for _ in range(args.nx)] for _ in range(args.ny)]
        for heat_p in participant_maps:
            for y in range(args.ny):
                row_p = heat_p[y]
                row_s = agg_sums[y]
                row_c = agg_counts[y]
                for x in range(args.nx):
                    v = row_p[x]
                    if v is None:
                        continue
                    if not math.isfinite(v):
                        continue
                    row_s[x] += float(v)
                    row_c[x] += 1

        agg: list[list[float | None]] = [[None for _ in range(args.nx)] for _ in range(args.ny)]
        min_p = int(args.min_participants)
        for y in range(args.ny):
            for x in range(args.nx):
                c = agg_counts[y][x]
                if c >= min_p:
                    agg[y][x] = agg_sums[y][x] / c

        out_path = args.out_dir / f"heatmap_RGB_{res}.png"
        _save_heatmap(
            agg,
            out_path=out_path,
            width_cm=FRAME_W_CM,
            height_cm=FRAME_H_CM,
            title=f"RGB {res}: mean loss_cm (participants {len(used_p)})",
            vmin=args.vmin,
            vmax=args.vmax,
        )
        print(f"[{res}] saved {out_path} (participants: {used_p})")


if __name__ == "__main__":
    main()
