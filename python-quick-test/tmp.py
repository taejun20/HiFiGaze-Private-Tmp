from __future__ import annotations

from pathlib import Path

import sys


def main() -> None:
    root = Path("study2_backgrounds")
    if not root.is_dir():
        raise FileNotFoundError(f"Missing directory: {root.resolve()}")

    jpgs = sorted(list(root.glob("*.jpg")) + list(root.glob("*.JPG")))
    if not jpgs:
        raise FileNotFoundError(f"No JPGs found in: {root.resolve()}")

    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "OpenCV (cv2) is required. Install it with `python -m pip install opencv-python`."
        ) from e

    threshold = 255 / 2  # 127.5
    dark = 0
    bright = 0
    unreadable = 0

    for p in jpgs:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            unreadable += 1
            continue

        mean_intensity = float(img.mean())
        if mean_intensity < threshold:
            dark += 1
        else:
            bright += 1

    total = len(jpgs)
    print(f"Total JPGs: {total}")
    print(f"Dark (<{threshold:.1f} mean intensity): {dark}")
    print(f"Bright (≥{threshold:.1f} mean intensity): {bright}")
    if unreadable:
        print(f"Unreadable (cv2.imread returned None): {unreadable}", file=sys.stderr)


if __name__ == "__main__":
    main()
