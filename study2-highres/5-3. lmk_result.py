from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


CHECKPOINT_DIR = Path("model_checkpoints/260508_108MP")
OUT_CSV = Path("LMK_study2_result.csv")

# Column order requested by user
RESOLUTIONS_OUT = ["qqvga", "qvga", "sd", "hd", "fhd", "2k", "4k", "6k", "8k", "54mp", "108mp"]

# Resolutions that can appear in checkpoint filenames (we'll skip anything else)
RESOLUTIONS_IN = {"108mp", "4k", "sd", "qqvga", "qvga", "54mp", "8k", "6k", "2k", "fhd", "hd"}


@dataclass(frozen=True)
class BestRun:
    resolution: str
    eval_p: int
    run_tag: str  # train_..._val_..._eval_...
    epoch: int
    val_error: float


_CKPT_RE = re.compile(
    r"^LMK_(?P<res>[^_]+)_(?P<run_tag>train_p[^_]+_val_p[^_]+_eval_p[^_]+)_"
    r"Epoch(?P<epoch>\d+)_ValError(?P<valerr>\d+\.\d+)_EvalError(?P<evalerr>\d+\.\d+)\.pt$"
)


def _parse_eval_p(run_tag: str) -> int | None:
    # run_tag like: train_p1,2,3_val_p4,5_eval_p6
    m = re.search(r"_eval_p(?P<eval>[^_]+)$", run_tag)
    if not m:
        return None
    eval_part = m.group("eval").strip()
    # expected single subject, but be tolerant
    try:
        return int(eval_part.split(",")[0])
    except ValueError:
        return None


def _mean_loss_cm(eval_csv_path: Path) -> float | None:
    """
    Average loss_cm column over all rows in an Eval_*.csv.
    Returns None if file missing, column missing, or no numeric rows.
    """
    if not eval_csv_path.is_file():
        return None

    total = 0.0
    n = 0
    with open(eval_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "loss_cm" not in reader.fieldnames:
            return None
        for row in reader:
            v = row.get("loss_cm")
            if v is None:
                continue
            try:
                total += float(v)
            except ValueError:
                continue
            n += 1
    if n == 0:
        return None
    return total / n


def main() -> None:
    if not CHECKPOINT_DIR.is_dir():
        raise FileNotFoundError(f"Missing CHECKPOINT_DIR: {CHECKPOINT_DIR.resolve()}")

    # Step 1: pick best (lowest ValError) per (res, eval_p)
    best: dict[tuple[str, int], BestRun] = {}

    for p in CHECKPOINT_DIR.iterdir():
        if not p.is_file() or p.suffix != ".pt":
            continue

        m = _CKPT_RE.match(p.name)
        if not m:
            continue

        res = m.group("res")
        if res not in RESOLUTIONS_IN:
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
        candidate = BestRun(resolution=res, eval_p=eval_p, run_tag=run_tag, epoch=epoch, val_error=valerr)

        if cur is None or candidate.val_error < cur.val_error:
            best[key] = candidate

    # Step 2: compute final metric per (res, eval_p) from corresponding Eval_*.csv
    # Output grid: rows p1..p10, cols RESOLUTIONS_OUT
    results: dict[tuple[int, str], float] = {}

    for (res, eval_p), run in best.items():
        eval_csv_name = f"Eval_LMK_{res}_{run.run_tag}_Epoch{run.epoch}.csv"
        eval_csv_path = CHECKPOINT_DIR / eval_csv_name
        mean_loss = _mean_loss_cm(eval_csv_path)
        if mean_loss is None:
            continue
        results[(eval_p, res)] = mean_loss

    # Step 3: write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["participant", *RESOLUTIONS_OUT])
        for pid in range(1, 11):
            row = [f"p{pid}"]
            for res in RESOLUTIONS_OUT:
                v = results.get((pid, res))
                row.append("" if v is None else f"{v:.6f}")
            writer.writerow(row)

    # Step 4: print table to terminal
    table: list[list[str]] = []
    table.append(["participant", *RESOLUTIONS_OUT])
    for pid in range(1, 11):
        row = [f"p{pid}"]
        for res in RESOLUTIONS_OUT:
            v = results.get((pid, res))
            row.append("" if v is None else f"{v:.6f}")
        table.append(row)

    col_widths = [
        max(len(table[r][c]) for r in range(len(table))) for c in range(len(table[0]))
    ]

    print("\nFinal table (mean loss_cm; empty = missing):")
    for ridx, row in enumerate(table):
        line = "  ".join(row[c].rjust(col_widths[c]) for c in range(len(row)))
        print(line)
        if ridx == 0:
            print("  ".join("-" * col_widths[c] for c in range(len(row))))

    print(f"Saved: {OUT_CSV.resolve()}")
    print(f"Found best checkpoints: {len(best)}")
    print(f"Computed results: {len(results)}")


if __name__ == "__main__":
    main()

