from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path


CHECKPOINT_DIR = Path("model_checkpoints/260512_108MP")
BEST_DIR = Path("model_checkpoints/best")

# Allowed resolution tags in checkpoint filenames
RESOLUTIONS = ["qqvga", "qvga", "sd", "hd", "fhd", "2k", "4k", "6k", "8k", "54mp", "108mp"]


@dataclass(frozen=True)
class BestRun:
    resolution: str
    eval_p: int
    run_tag: str
    epoch: int
    val_error: float
    source_path: Path


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


def main() -> None:
    if not CHECKPOINT_DIR.is_dir():
        raise FileNotFoundError(f"Missing CHECKPOINT_DIR: {CHECKPOINT_DIR.resolve()}")

    best: dict[tuple[str, int], BestRun] = {}

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
        candidate = BestRun(
            resolution=res,
            eval_p=eval_p,
            run_tag=run_tag,
            epoch=epoch,
            val_error=valerr,
            source_path=p.resolve(),
        )

        if cur is None or candidate.val_error < cur.val_error:
            best[key] = candidate

    BEST_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0
    for run in sorted(best.values(), key=lambda r: (r.resolution, r.eval_p)):
        dest = BEST_DIR / run.source_path.name
        shutil.copy2(run.source_path, dest)
        copied += 1
        print(f"Copied: {run.source_path.name} -> {dest}")

    print(f"\nCopied {copied} checkpoint(s) to {BEST_DIR.resolve()}")
    print(f"Best keys (res × eval_p): {len(best)}")


if __name__ == "__main__":
    main()
