import argparse
import json
import math
from itertools import product
from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

RESOLUTIONS = ["108mp", "54mp", "8k", "6k", "4k", "2k", "fhd", "hd", "sd", "qvga", "qqvga"]

SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# MediaPipe indices
LEFT_IRIS_INNER = 476
LEFT_IRIS_OUTER = 474
LEFT_IRIS_CENTER = 473

RIGHT_IRIS_INNER = 469
RIGHT_IRIS_OUTER = 471
RIGHT_IRIS_CENTER = 468

# Augmentation grid: scale1..5, x1..5, y1..5 -> 125 variants per eye
SCALE_SHIFT = 0.1
SCALE_FACTORS = [2.8 - SCALE_SHIFT * 2, 2.8 - SCALE_SHIFT, 2.8, 2.8 + SCALE_SHIFT, 2.8 + SCALE_SHIFT * 2]
SHIFT = 10
CENTER_OFFSETS = [-2 * SHIFT, -SHIFT, 0, SHIFT, 2 * SHIFT]


def create_face_landmarker(model_path: Path) -> vision.FaceLandmarker:
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.FaceLandmarker.create_from_options(options)


def find_iris_width(frame_shape, landmarks, is_left: bool):
    if is_left:
        iris_inner_mp = (landmarks[LEFT_IRIS_INNER].x, landmarks[LEFT_IRIS_INNER].y)
        iris_outer_mp = (landmarks[LEFT_IRIS_OUTER].x, landmarks[LEFT_IRIS_OUTER].y)
    else:
        iris_inner_mp = (landmarks[RIGHT_IRIS_INNER].x, landmarks[RIGHT_IRIS_INNER].y)
        iris_outer_mp = (landmarks[RIGHT_IRIS_OUTER].x, landmarks[RIGHT_IRIS_OUTER].y)

    frame_height, frame_width = frame_shape[:2]
    iris_width = round(
        math.hypot(
            (iris_outer_mp[0] - iris_inner_mp[0]) * frame_width,
            (iris_outer_mp[1] - iris_inner_mp[1]) * frame_height,
        )
    )
    return round(iris_width * 0.92)


def crop_eye(frame, iris_center_frame, irisbox_width: int, scale_factor: float):
    frame_height, frame_width = frame.shape[:2]
    eyepatch_width = round(irisbox_width * scale_factor)
    eyepatch_height = eyepatch_width // 2

    top = iris_center_frame[1] - eyepatch_height // 2
    bottom = iris_center_frame[1] + eyepatch_height // 2
    left = iris_center_frame[0] - eyepatch_width // 2
    right = iris_center_frame[0] + eyepatch_width // 2

    if top < 0 or bottom > frame_height or left < 0 or right > frame_width:
        return None

    return frame[top:bottom, left:right]


def make_landmark_json_dict(landmarks):
    out = {}
    for i in range(478):
        out[f"{i}x"] = landmarks[i].x
        out[f"{i}y"] = landmarks[i].y
    return out


def augment_combo_dir(eyecrop_root: Path, side: str, scale_i: int, x_i: int, y_i: int) -> Path:
    return eyecrop_root / side / f"scale{scale_i}_x{x_i}_y{y_i}"


def process_frame(
    frame_path: Path,
    eyecrop_root: Path,
    landmark_output_dir: Path,
    detector,
    *,
    skip_existing: bool,
):
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return False, "image_read_fail", {}

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    results = detector.detect(mp_image)
    if not results.face_landmarks:
        return False, "no_face", {}

    landmarks = results.face_landmarks[0]
    frame_height, frame_width = frame.shape[:2]

    l_irisbox_width = find_iris_width(frame.shape, landmarks, is_left=True)
    r_irisbox_width = find_iris_width(frame.shape, landmarks, is_left=False)

    l_iris_center_frame = (
        round(landmarks[LEFT_IRIS_CENTER].x * frame_width),
        round(landmarks[LEFT_IRIS_CENTER].y * frame_height),
    )
    r_iris_center_frame = (
        round(landmarks[RIGHT_IRIS_CENTER].x * frame_width),
        round(landmarks[RIGHT_IRIS_CENTER].y * frame_height),
    )

    stem = frame_path.stem
    landmark_path = landmark_output_dir / f"{stem}.json"
    if not skip_existing or not landmark_path.is_file():
        landmark_output_dir.mkdir(parents=True, exist_ok=True)
        with open(landmark_path, "w", encoding="utf-8") as f:
            json.dump(make_landmark_json_dict(landmarks), f, indent=2)

    aug_stats = {"left_ok": 0, "left_oob": 0, "right_ok": 0, "right_oob": 0}

    for scale_i, x_i, y_i in product(range(1, 6), range(1, 6), range(1, 6)):
        scale_factor = SCALE_FACTORS[scale_i - 1]
        dx = CENTER_OFFSETS[x_i - 1]
        dy = CENTER_OFFSETS[y_i - 1]

        l_center = (l_iris_center_frame[0] + dx, l_iris_center_frame[1] + dy)
        r_center = (r_iris_center_frame[0] + dx, r_iris_center_frame[1] + dy)

        left_dir = augment_combo_dir(eyecrop_root, "left", scale_i, x_i, y_i)
        right_dir = augment_combo_dir(eyecrop_root, "right", scale_i, x_i, y_i)
        left_path = left_dir / f"{stem}_l.png"
        right_path = right_dir / f"{stem}_r.png"

        if skip_existing and left_path.is_file() and right_path.is_file():
            aug_stats["left_ok"] += 1
            aug_stats["right_ok"] += 1
            continue

        l_eye_crop = crop_eye(frame, l_center, l_irisbox_width, scale_factor)
        r_eye_crop = crop_eye(frame, r_center, r_irisbox_width, scale_factor)

        if l_eye_crop is None:
            aug_stats["left_oob"] += 1
        else:
            l_eye_crop = cv2.flip(l_eye_crop, 1)
            left_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(left_path), l_eye_crop)
            aug_stats["left_ok"] += 1

        if r_eye_crop is None:
            aug_stats["right_oob"] += 1
        else:
            right_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(right_path), r_eye_crop)
            aug_stats["right_ok"] += 1

    # Frame counts as ok if at least one augment variant succeeded per eye
    if aug_stats["left_ok"] == 0 or aug_stats["right_ok"] == 0:
        return False, "crop_oob", aug_stats
    return True, "ok", aug_stats


def process_resolution(subject_dir: Path, resolution: str, detector, *, skip_existing: bool):
    frames_dir = subject_dir / f"{resolution}_frames"
    preprocessed_resolution_dir = subject_dir / "preprocessed_augmented" / f"{resolution}_frames"
    eyecrop_root = preprocessed_resolution_dir / "eyecrop"
    landmark_output_dir = preprocessed_resolution_dir / "landmark"

    if not frames_dir.exists():
        print(f"[SKIP] {subject_dir.name}/{resolution}: no frames directory")
        return

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        print(f"[SKIP] {subject_dir.name}/{resolution}: no jpg frames")
        return

    stats = {
        "ok": 0,
        "image_read_fail": 0,
        "no_face": 0,
        "crop_oob": 0,
        "left_crops": 0,
        "left_oob": 0,
        "right_crops": 0,
        "right_oob": 0,
    }
    for frame_path in tqdm(frame_paths, desc=f"{subject_dir.name}/{resolution}", leave=False):
        ok, reason, aug_stats = process_frame(
            frame_path,
            eyecrop_root,
            landmark_output_dir,
            detector,
            skip_existing=skip_existing,
        )
        if ok:
            stats["ok"] += 1
        else:
            stats[reason] += 1
        stats["left_crops"] += aug_stats.get("left_ok", 0)
        stats["left_oob"] += aug_stats.get("left_oob", 0)
        stats["right_crops"] += aug_stats.get("right_ok", 0)
        stats["right_oob"] += aug_stats.get("right_oob", 0)

    print(
        f"[DONE] {subject_dir.name}/{resolution}: total={len(frame_paths)} "
        f"ok={stats['ok']} no_face={stats['no_face']} crop_oob={stats['crop_oob']} "
        f"read_fail={stats['image_read_fail']} "
        f"left_crops={stats['left_crops']} left_oob={stats['left_oob']} "
        f"right_crops={stats['right_crops']} right_oob={stats['right_oob']}"
    )


def process_subject(subject_dir: Path, detector, *, skip_existing: bool):
    preprocessed_root = subject_dir / "preprocessed_augmented"
    preprocessed_root.mkdir(parents=True, exist_ok=True)

    for resolution in RESOLUTIONS:
        process_resolution(subject_dir, resolution, detector, skip_existing=skip_existing)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create augmented study2 eyecrops (125 scale/x/y variants per eye) and "
            "478-point landmark json files under preprocessed_augmented/."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("study2_rawdata_processed"),
        help="Root directory containing p*/ folders (default: study2_rawdata_processed).",
    )
    parser.add_argument(
        "-s",
        "--subjects",
        nargs="*",
        type=int,
        default=SUBJECTS,
        help="Subjects to process (e.g. -s 1 2 6).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("face_landmarker.task"),
        help="Path to MediaPipe face landmarker task file.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-write outputs even when eyecrop pngs already exist.",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")

    detector = create_face_landmarker(args.model)
    skip_existing = not args.no_skip_existing

    print(f"Data root: {args.data_root}")
    print(f"Subjects: {args.subjects}")
    print(f"Scale factors: {SCALE_FACTORS}")
    print(f"Center offsets (px): {CENTER_OFFSETS}")
    print(f"Output: preprocessed_augmented/{{res}}_frames/eyecrop/{{left,right}}/scale{{1-5}}_x{{1-5}}_y{{1-5}}/")
    for subject in args.subjects:
        subject_dir = args.data_root / f"p{subject}"
        if not subject_dir.exists():
            print(f"[SKIP] p{subject}: subject directory not found")
            continue
        process_subject(subject_dir, detector, skip_existing=skip_existing)

    print("All done.")


if __name__ == "__main__":
    main()
