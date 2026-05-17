from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


# Match training script WandB_PROJECT / checkpoint root (macOS may treat 4k/4K the same).
CHECKPOINT_DIR = Path("model_checkpoints/260513_4K")
OUT_CSV = Path("rgbt_result_4k.csv")

# Column order (same as 6-1. rgb_train_and_eval_4k.py)
RESOLUTIONS = ["qqvga", "qvga", "sd", "hd", "fhd", "2k", "4k"]

# Every participant that appears in some fold's eval set (row order for the table)
PARTICIPANT_IDS = [1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26]


@dataclass(frozen=True)
class BestRun:
    resolution: str
    eval_key: tuple[int, ...]  # sorted eval subject ids for this run
    run_tag: str
    epoch: int
    val_error: float


_CKPT_RE = re.compile(
    r"^RGBT4K_(?P<res>[^_]+)_(?P<run_tag>.+)_Epoch(?P<epoch>\d+)_"
    r"ValError(?P<valerr>\d+\.\d+)_EvalError(?P<evalerr>\d+\.\d+)\.pt$"
)


def _parse_eval_key(run_tag: str) -> tuple[int, ...] | None:
    """run_tag like: train_p..._val_p..._eval_p1,2,6 -> (1, 2, 6)."""
    m = re.search(r"_eval_p(?P<eval>[\d,]+)$", run_tag)
    if not m:
        return None
    parts = [p.strip() for p in m.group("eval").split(",") if p.strip()]
    try:
        ids = sorted(int(p) for p in parts)
    except ValueError:
        return None
    if not ids:
        return None
    return tuple(ids)


def _per_subject_mean_loss_cm(eval_csv_path: Path) -> dict[int, float]:
    """
    Mean loss_cm per subject id from an Eval_*.csv.
    Subjects with no numeric rows are omitted.
    """
    out: dict[int, float] = {}
    if not eval_csv_path.is_file():
        return out

    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)

    with open(eval_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "loss_cm" not in reader.fieldnames or "subject" not in reader.fieldnames:
            return out
        for row in reader:
            subj_raw = row.get("subject")
            loss_raw = row.get("loss_cm")
            if subj_raw is None or loss_raw is None:
                continue
            try:
                sid = int(subj_raw)
                loss = float(loss_raw)
            except ValueError:
                continue
            sums[sid] += loss
            counts[sid] += 1

    for sid, s in sums.items():
        n = counts[sid]
        if n > 0:
            out[sid] = s / n
    return out


def main() -> None:
    if not CHECKPOINT_DIR.is_dir():
        raise FileNotFoundError(f"Missing CHECKPOINT_DIR: {CHECKPOINT_DIR.resolve()}")

    # Best (lowest ValError) per (resolution, eval split)
    best: dict[tuple[str, tuple[int, ...]], BestRun] = {}

    for p in CHECKPOINT_DIR.iterdir():
        if not p.is_file() or p.suffix != ".pt":
            continue
        m = _CKPT_RE.match(p.name)
        if not m:
            continue

        res = m.group("res")
        if res not in RESOLUTIONS:
            continue

        run_tag = m.group("run_tag")
        eval_key = _parse_eval_key(run_tag)
        if eval_key is None:
            continue

        try:
            epoch = int(m.group("epoch"))
            valerr = float(m.group("valerr"))
        except ValueError:
            continue

        key = (res, eval_key)
        candidate = BestRun(
            resolution=res,
            eval_key=eval_key,
            run_tag=run_tag,
            epoch=epoch,
            val_error=valerr,
        )
        cur = best.get(key)
        if cur is None or candidate.val_error < cur.val_error:
            best[key] = candidate

    results: dict[tuple[int, str], float] = {}

    for (res, eval_key), run in best.items():
        eval_csv_name = f"Eval_RGBT4K_{res}_{run.run_tag}_Epoch{run.epoch}.csv"
        eval_csv_path = CHECKPOINT_DIR / eval_csv_name
        per_subj = _per_subject_mean_loss_cm(eval_csv_path)
        for sid in eval_key:
            v = per_subj.get(sid)
            if v is not None:
                results[(sid, res)] = v

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["participant", *RESOLUTIONS])
        for pid in PARTICIPANT_IDS:
            row = [f"p{pid}"]
            for res in RESOLUTIONS:
                v = results.get((pid, res))
                row.append("" if v is None else f"{v:.6f}")
            writer.writerow(row)

    table: list[list[str]] = []
    table.append(["participant", *RESOLUTIONS])
    for pid in PARTICIPANT_IDS:
        row = [f"p{pid}"]
        for res in RESOLUTIONS:
            v = results.get((pid, res))
            row.append("" if v is None else f"{v:.6f}")
        table.append(row)

    col_widths = [
        max(len(table[r][c]) for r in range(len(table))) for c in range(len(table[0]))
    ]

    print("\nFinal table (mean loss_cm per participant; empty = missing):")
    for ridx, row in enumerate(table):
        line = "  ".join(row[c].rjust(col_widths[c]) for c in range(len(row)))
        print(line)
        if ridx == 0:
            print("  ".join("-" * col_widths[c] for c in range(len(row))))

    print(f"Saved: {OUT_CSV.resolve()}")
    print(f"Found best (res, eval_split) runs: {len(best)}")
    print(f"Filled cells: {len(results)}")


if __name__ == "__main__":
    main()
