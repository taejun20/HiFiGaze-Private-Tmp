import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import json
import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import helpermethods  # noqa: E402

SUBJECTS = [1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26]
SESSIONS = [1, 2, 3, 4, 5, 6]

LEFT_EYE_OUTER = 263
LEFT_EYE_INNER = 362
RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

BASE_OPTIONS = python.BaseOptions(model_asset_path='face_landmarker.task')
OPTIONS = vision.FaceLandmarkerOptions(
    base_options=BASE_OPTIONS,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    running_mode=vision.RunningMode.IMAGE
)
FACE_LANDMARK_DETECTOR = vision.FaceLandmarker.create_from_options(OPTIONS)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_face_crop(frame, pixel_points):
    if len(pixel_points) != 6:
        return None

    frame_height, frame_width = frame.shape[:2]
    center_x = sum(p[0] for p in pixel_points) / len(pixel_points)
    center_y = sum(p[1] for p in pixel_points) / len(pixel_points)

    max_dist = 0.0
    for i in range(len(pixel_points)):
        for j in range(i + 1, len(pixel_points)):
            dist = math.hypot(pixel_points[i][0] - pixel_points[j][0],
                              pixel_points[i][1] - pixel_points[j][1])
            max_dist = max(max_dist, dist)

    box_size = max(1, int(round(max_dist * 1.5)))
    half = box_size // 2
    top = int(round(center_y - half))
    bottom = top + box_size
    left = int(round(center_x - half))
    right = left + box_size

    if top < 0:
        bottom -= top
        top = 0
    if left < 0:
        right -= left
        left = 0
    if bottom > frame_height:
        top -= bottom - frame_height
        bottom = frame_height
    if right > frame_width:
        left -= right - frame_width
        right = frame_width

    top = max(0, top)
    left = max(0, left)
    bottom = min(frame_height, bottom)
    right = min(frame_width, right)

    if bottom <= top or right <= left:
        return None

    return frame[top:bottom, left:right]


def get_anchor_points(landmarks, frame_width, frame_height):
    indices = [
        LEFT_EYE_INNER,
        LEFT_EYE_OUTER,
        RIGHT_EYE_INNER,
        RIGHT_EYE_OUTER,
        MOUTH_LEFT,
        MOUTH_RIGHT,
    ]

    points = []
    for idx in indices:
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        points.append(
            (
                round(lm.x * frame_width),
                round(lm.y * frame_height),
            )
        )

    return points


def save_headaug_json(path: Path, landmarks):
    headaug_payload = {}
    for idx, landmark in enumerate(landmarks):
        headaug_payload[f"{idx}x"] = landmark.x
        headaug_payload[f"{idx}y"] = landmark.y

    with path.open('w') as f:
        json.dump(headaug_payload, f, indent=2)


def process_subject_session(subject: int, session: int):
    csv_path = Path(f"preprocessed/logs/p{subject}_s{session}_log_preprocessed.csv")
    frame_dir = Path(f"preprocessed/frames/p{subject}/s{session}")
    face_output_dir = Path(f"preprocessed/input_face/p{subject}/s{session}")
    headaug_output_dir = Path(f"preprocessed/input_headaug/p{subject}/s{session}")

    ensure_dir(face_output_dir)
    ensure_dir(headaug_output_dir)

    if not csv_path.exists():
        print(f"CSV not found for p{subject} s{session}, skipping.")
        return

    df = pd.read_csv(csv_path)

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"Processing p{subject} s{session}"):
        frame_name = str(int(row['frameNum'])).zfill(5)
        frame_path = frame_dir / f"{frame_name}.jpg"
        if not frame_path.exists():
            print(f"Frame not found: {frame_path}, skipping.")
            continue

        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Unable to read frame: {frame_path}, skipping.")
            continue

        rotated_frame = helpermethods.rotate_image_clockwise(frame)
        if rotated_frame is None:
            print(f"Rotation failed for frame {frame_path}, skipping.")
            continue

        frame_rgb = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )
        results = FACE_LANDMARK_DETECTOR.detect(mp_image)
        if not results.face_landmarks:
            print(f"No landmarks detected for {frame_name}, skipping.")
            continue

        landmarks = results.face_landmarks[0]
        num_landmarks = len(landmarks)
        print(f"Frame {frame_name} (p{subject}s{session}): detected {num_landmarks} landmarks")
        if num_landmarks != 478:
            print("Unexpected landmark count. Skipping this frame.")
            continue

        frame_height, frame_width = rotated_frame.shape[:2]
        anchor_points = get_anchor_points(landmarks, frame_width, frame_height)
        if len(anchor_points) != 6:
            print(f"Insufficient anchor points for frame {frame_name}, skipping.")
            continue

        face_crop = compute_face_crop(rotated_frame, anchor_points)
        if face_crop is None:
            print(f"Failed to compute face crop for frame {frame_name}, skipping.")
            continue

        face_output_path = face_output_dir / f"{frame_name}_face.png"
        resized_face = cv2.resize(face_crop, (500, 500), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(face_output_path), resized_face)

        headaug_output_path = headaug_output_dir / f"{frame_name}_headaug.json"
        save_headaug_json(headaug_output_path, landmarks)


def main():
    parser = argparse.ArgumentParser(description="Generate face crops and head augmentation data.")
    parser.add_argument('-s', nargs='*', type=int, default=None,
                        help='List of subjects to process (default: all).')
    args = parser.parse_args()

    subjects = args.s if args.s is not None else SUBJECTS

    for subject in subjects:
        for session in SESSIONS:
            process_subject_session(subject, session)


if __name__ == "__main__":
    main()

