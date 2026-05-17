from __future__ import annotations

from pathlib import Path
import shutil

import cv2
from tqdm import tqdm

RESOLUTIONS = {
    "4k": (2160, 3840),
    "2k": (1440, 2560),
    "fhd": (1080, 1920),
    "hd": (720, 1280),
    "sd": (360, 640),
    "qvga": (180, 320),
    "qqvga": (90, 160)
}

PARTICIPANTS = [1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26]

def resize_all_frames(dataset_root: Path) -> None:
    for participant in PARTICIPANTS:
        participant_dir = dataset_root / f"p{participant}"
        frames_dir = participant_dir / "frames"

        if not frames_dir.exists():
            print(f"[WARN] Missing frames directory: {frames_dir}")
            continue

        output_dirs = {}
        for key in RESOLUTIONS:
            out_dir = participant_dir / f"{key}_frames"
            out_dir.mkdir(parents=True, exist_ok=True)
            output_dirs[key] = out_dir
            if key == "4k":
                for src in sorted(frames_dir.iterdir()):
                    if src.is_file():
                        shutil.copy2(src, out_dir / src.name)
                n_full = sum(1 for p in out_dir.iterdir() if p.is_file())
                print(f"[{participant}] Copied frames -> 4k_frames/ ({n_full} files)")

        image_paths = sorted(frames_dir.glob("*.jpg"))
        if not image_paths:
            print(f"[WARN] No JPG files found in: {frames_dir}")
            continue

        print(f"\n[{participant}] Processing {len(image_paths)} images...")

        for idx, image_path in enumerate(
            tqdm(image_paths, desc=f"p{participant}", unit="img"), start=1
        ):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"[WARN] Could not read: {image_path}")
                continue

            for key, (w, h) in RESOLUTIONS.items():
                if key == "4k":
                    continue
                resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
                out_path = output_dirs[key] / image_path.name
                ok = cv2.imwrite(str(out_path), resized)
                if not ok:
                    print(f"[WARN] Failed to write: {out_path}")

        print(f"[{participant}] Completed.")


def main() -> None:
    # Assumes this script is run from the "study2-highres" directory.
    # Adjust the path if your layout is different.
    dataset_root = Path("study1_rawdata_processed")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root.resolve()}")

    resize_all_frames(dataset_root)
    print("\nAll participants processed.")


if __name__ == "__main__":
    main()
