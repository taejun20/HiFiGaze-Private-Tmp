from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> None:
    img_dir = Path(__file__).resolve().parent / "img"
    sigma = 50

    for i in range(1, 10):
        src = img_dir / f"img{i}.jpg"
        dst = img_dir / f"img{i}_blurred.jpg"

        if not src.exists():
            raise FileNotFoundError(f"Missing input: {src}")

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-vf",
            f"gblur=sigma={sigma}:steps=2",
            "-q:v",
            "2",
            str(dst),
        ]
        subprocess.run(cmd, check=True)
        print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
