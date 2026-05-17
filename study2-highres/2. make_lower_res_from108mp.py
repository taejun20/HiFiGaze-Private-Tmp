from __future__ import annotations

from pathlib import Path
import shutil

import cv2

RESOLUTIONS = {
    "108mp": (12000, 9000),  # native: copied verbatim; size tuple unused
    "54mp": (8485, 6364),
    "8k": (5760, 4320),
    "6k": (4213, 3160),
    "4k": (2880, 2160),
    "2k": (1920, 1440),
    "fhd": (1440, 1080),
    "hd": (960, 720),
    "sd": (640, 480),
    "qvga": (320, 240),
    "qqvga": (160, 120)
}

PARTICIPANTS = [
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
    "p6",
    "p7",
    "p8",
    "p9",
    "p10"
]

def resize_all_frames(dataset_root: Path) -> None:
    """
    For each participant: create sibling folders per RESOLUTIONS.
    For "108mp", duplicate frames/ verbatim into 108mp_frames/ (no resize).
    For other keys, write resized JPG copies.
    """
    for participant in PARTICIPANTS:
        participant_dir = dataset_root / participant
        frames_dir = participant_dir / "frames"

        if not frames_dir.exists():
            print(f"[WARN] Missing frames directory: {frames_dir}")
            continue

        output_dirs = {}
        for key in RESOLUTIONS:
            out_dir = participant_dir / f"{key}_frames"
            out_dir.mkdir(parents=True, exist_ok=True)
            output_dirs[key] = out_dir
            if key == "108mp":
                for src in sorted(frames_dir.iterdir()):
                    if src.is_file():
                        shutil.copy2(src, out_dir / src.name)
                n_full = sum(1 for p in out_dir.iterdir() if p.is_file())
                print(f"[{participant}] Copied frames -> 108mp_frames/ ({n_full} files)")

        image_paths = sorted(frames_dir.glob("*.jpg"))
        if not image_paths:
            print(f"[WARN] No JPG files found in: {frames_dir}")
            continue

        print(f"\n[{participant}] Processing {len(image_paths)} images...")

        for idx, image_path in enumerate[Path](image_paths, start=1):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"[WARN] Could not read: {image_path}")
                continue

            for key, (w, h) in RESOLUTIONS.items():
                if key == "108mp":
                    continue
                resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
                out_path = output_dirs[key] / image_path.name
                ok = cv2.imwrite(str(out_path), resized)
                if not ok:
                    print(f"[WARN] Failed to write: {out_path}")

            if idx % 200 == 0 or idx == len(image_paths):
                print(f"[{participant}] {idx}/{len(image_paths)} done")

        print(f"[{participant}] Completed.")


def main() -> None:
    # Assumes this script is run from the "study2-highres" directory.
    # Adjust the path if your layout is different.
    dataset_root = Path("study2_rawdata_processed")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root.resolve()}")

    resize_all_frames(dataset_root)
    print("\nAll participants processed.")


if __name__ == "__main__":
    main()
