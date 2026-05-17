from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


SUBJECTS = [1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26]

# Resume: skip through p9; start at p10, then 11, ...
RESUME_START_SUBJECT = 10

def separable_gaussian_blur(img: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    # Keep behavior consistent with training code; ensure kernel_size is odd for symmetric kernel.
    if kernel_size % 2 == 0:
        kernel_size += 1
    k = (kernel_size - 1) // 2
    x = np.linspace(-k, k, kernel_size)
    kernel_1d = np.exp(-(x**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    kernel_h = kernel_1d.reshape(1, -1)
    blurred_h = cv2.filter2D(img, -1, kernel_h)

    kernel_v = kernel_1d.reshape(-1, 1)
    blurred = cv2.filter2D(blurred_h, -1, kernel_v)
    return blurred


def build_template(background_a: np.ndarray, background_b: np.ndarray, dissolve: float) -> np.ndarray:
    # Match @6-2-2. rgbt_train_and_eval_4k.py: blend -> blur -> downsample.
    template = cv2.addWeighted(background_a, 1.0 - dissolve, background_b, dissolve, 0)
    template = separable_gaussian_blur(template, kernel_size=300, sigma=50)
    template = cv2.resize(template, (60, 120), interpolation=cv2.INTER_LINEAR)  # (width, height)
    return template


def main() -> None:
    root = Path("study1_rawdata_processed")
    screen_dir = root / "screen"
    if not screen_dir.is_dir():
        raise FileNotFoundError(f"Missing screen directory: {screen_dir.resolve()}")

    for subject in SUBJECTS:
        if subject < RESUME_START_SUBJECT:
            continue
        json_dir = root / f"p{subject}" / "json"
        if not json_dir.is_dir():
            raise FileNotFoundError(f"Missing json dir: {json_dir.resolve()}")

        out_dir = root / f"p{subject}" / "preprocessed" / "template"
        out_dir.mkdir(parents=True, exist_ok=True)

        json_files = sorted(json_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No json files found in: {json_dir.resolve()}")

        for json_path in tqdm(json_files, desc=f"p{subject}", unit="frame"):
            out_path = out_dir / f"{json_path.stem}.png"
            if out_path.exists():
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            try:
                background_a_id = str(int(metadata["backgroundA"]))
                background_b_id = str(int(metadata["backgroundB"]))
                dissolve = float(metadata["dissolve"])
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"Bad metadata in {json_path}: {e}") from e

            background_a_path = screen_dir / f"{background_a_id}.jpg"
            background_b_path = screen_dir / f"{background_b_id}.jpg"

            background_a = cv2.imread(str(background_a_path))
            background_b = cv2.imread(str(background_b_path))
            if background_a is None or background_b is None:
                raise FileNotFoundError(
                    f"Could not load backgrounds for {json_path.name}: {background_a_path}, {background_b_path}"
                )

            template = build_template(background_a, background_b, dissolve)

            ok = cv2.imwrite(str(out_path), template)
            if not ok:
                raise OSError(f"Failed to write: {out_path.resolve()}")


if __name__ == "__main__":
    main()
