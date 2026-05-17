import argparse
import json
import math
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

#RESOLUTIONS = ["108mp","54mp","8k","6k","4k","2k","fhd","hd","sd","qvga","qqvga"]
RESOLUTIONS = ["108mp","qvga","qqvga"]

SUBJECTS = [1,2,3,4,5,6,7,8,9,10]

# MediaPipe indices
LEFT_IRIS_INNER = 476
LEFT_IRIS_OUTER = 474
LEFT_IRIS_CENTER = 473

RIGHT_IRIS_INNER = 469
RIGHT_IRIS_OUTER = 471
RIGHT_IRIS_CENTER = 468

EYE_CROP_SCALE_FACTOR = 2.8

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

    irisbox_width = round(iris_width * 0.92)
    return irisbox_width


def crop_eye(frame: np.ndarray, iris_center_frame, irisbox_width: int):
    frame_height, frame_width = frame.shape[:2]
    eyepatch_width = round(irisbox_width * EYE_CROP_SCALE_FACTOR)
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


def process_frame(frame_path: Path, eyecrop_output_dir: Path, landmark_output_dir: Path, detector):
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return False, "image_read_fail"

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    results = detector.detect(mp_image)
    if not results.face_landmarks:
        return False, "no_face"

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

    # Always use MediaPipe iris centers for eye cropping.
    l_crop_center = l_iris_center_frame
    r_crop_center = r_iris_center_frame

    l_eye_crop = crop_eye(frame, l_crop_center, l_irisbox_width)
    r_eye_crop = crop_eye(frame, r_crop_center, r_irisbox_width)
    if l_eye_crop is None or r_eye_crop is None:
        return False, "crop_oob"

    # Match reference pipeline orientation.
    l_eye_crop = cv2.flip(l_eye_crop, 1)

    stem = frame_path.stem
    eyecrop_left_path = eyecrop_output_dir / f"{stem}_l.png"
    eyecrop_right_path = eyecrop_output_dir / f"{stem}_r.png"
    json_path = landmark_output_dir / f"{stem}.json"

    cv2.imwrite(str(eyecrop_left_path), l_eye_crop)
    cv2.imwrite(str(eyecrop_right_path), r_eye_crop)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(make_landmark_json_dict(landmarks), f, indent=2)

    return True, "ok"

def process_resolution(subject_dir: Path, resolution: str, detector):
    frames_dir = subject_dir / f"{resolution}_frames"
    preprocessed_resolution_dir = subject_dir / "preprocessed" / f"{resolution}_frames"
    eyecrop_output_dir = preprocessed_resolution_dir / "eyecrop"
    landmark_output_dir = preprocessed_resolution_dir / "landmark"

    if not frames_dir.exists():
        print(f"[SKIP] {subject_dir.name}/{resolution}: no frames directory")
        return

    eyecrop_output_dir.mkdir(parents=True, exist_ok=True)
    landmark_output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        print(f"[SKIP] {subject_dir.name}/{resolution}: no jpg frames")
        return

    stats = {"ok": 0, "image_read_fail": 0, "no_face": 0, "crop_oob": 0}
    for frame_path in tqdm(frame_paths, desc=f"{subject_dir.name}/{resolution}", leave=False):
        ok, reason = process_frame(frame_path, eyecrop_output_dir, landmark_output_dir, detector)
        if ok:
            stats["ok"] += 1
        else:
            stats[reason] += 1

    print(
        f"[DONE] {subject_dir.name}/{resolution}: total={len(frame_paths)} "
        f"ok={stats['ok']} no_face={stats['no_face']} crop_oob={stats['crop_oob']} read_fail={stats['image_read_fail']}"
    )

def process_subject(subject_dir: Path, detector):
    preprocessed_root = subject_dir / "preprocessed"
    preprocessed_root.mkdir(parents=True, exist_ok=True)

    for resolution in RESOLUTIONS:
        process_resolution(subject_dir, resolution, detector)


def main():
    parser = argparse.ArgumentParser(description="Create study2 eyecrops and 478-point landmark json files for all resolutions.")
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
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")

    detector = create_face_landmarker(args.model)

    print(f"Data root: {args.data_root}")
    print(f"Subjects: {args.subjects}")
    for subject in args.subjects:
        subject_dir = args.data_root / f"p{subject}"
        if not subject_dir.exists():
            print(f"[SKIP] p{subject}: subject directory not found")
            continue
        process_subject(subject_dir, detector)

    print("All done.")


if __name__ == "__main__":
    main()
