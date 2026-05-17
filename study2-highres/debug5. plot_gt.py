from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _iter_metadata_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for pid in range(1, 11):
        pdir = root / f"p{pid}" / "json"
        if not pdir.is_dir():
            continue
        files.extend(sorted(pdir.glob("*.json")))
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("study2_rawdata_processed"),
        help="Root directory containing p1..p10/json/*.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("gt_xy_hist.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=200,
        help="If too many unique pairs, plot only the top-K labels (by count).",
    )
    parser.add_argument(
        "--round-digits",
        type=int,
        default=-1,
        help="Round gt_x_px/gt_y_px to this many decimals before counting (use -1 for no rounding).",
    )
    args = parser.parse_args()

    root: Path = args.root
    files = _iter_metadata_files(root)
    if not files:
        raise FileNotFoundError(f"No json files found under: {root.resolve()}")

    counts: Counter[tuple[float, float]] = Counter()
    skipped = 0
    digits = int(args.round_digits)
    for p in files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            gx = float(d["gt_x_px"])
            gy = float(d["gt_y_px"])
            if digits >= 0:
                gx = round(gx, digits)
                gy = round(gy, digits)
        except Exception:
            skipped += 1
            continue
        counts[(gx, gy)] += 1

    if not counts:
        raise RuntimeError(f"Loaded {len(files)} files but found no valid gt pairs (skipped={skipped}).")

    items = counts.most_common()
    total_unique = len(items)
    max_labels = int(args.max_labels)
    if max_labels > 0 and total_unique > max_labels:
        items = items[:max_labels]

    if digits >= 0:
        fmt = f"{{:.{digits}f}}"
        labels = [f"({fmt.format(x)},{fmt.format(y)})" for (x, y), _c in items]
    else:
        # Show exact float identity used for counting.
        labels = [f"({repr(x)},{repr(y)})" for (x, y), _c in items]
    values = [_c for (_xy, _c) in items]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_w = max(12.0, 0.12 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 6.0), dpi=200)
    ax.bar(range(len(labels)), values)

    ax.set_title(
        f"GT (gt_x_px, gt_y_px) pair counts (files={len(files)}, unique={total_unique}, skipped={skipped})"
        + (f" [top {len(labels)}]" if len(labels) != total_unique else "")
    )
    ax.set_xlabel("(gt_x_px, gt_y_px)")
    ax.set_ylabel("count")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)

    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)

    print(f"Saved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
