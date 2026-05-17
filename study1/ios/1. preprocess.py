import os
import cv2
import numpy as np
import json
from tqdm import tqdm
import mediapipe as mp
from datetime import datetime
import argparse
from pathlib import Path

SUBJECTS = [1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26]

# eye blink filtering
EYE_CLOSED_THRESHOLD = 0.16

# MediaPipe options - declared once
face_landmarker_options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=mp.tasks.vision.RunningMode.VIDEO
)

# MP indices
L_IRIS = 468
R_IRIS = 473

L_LEFT_CORNER = 33
L_RIGHT_CORNER = 133
L_TOP_MIDDLE = 159
L_BOTTOM_MIDDLE = 145

R_LEFT_CORNER = 362
R_RIGHT_CORNER = 263
R_TOP_MIDDLE = 386
R_BOTTOM_MIDDLE = 374


def get_eye_aspect_ratio(landmarks, image_width, image_height, is_left):
    """
    Calculate eye aspect ratio using 4 landmarks: left corner, right corner, top middle, bottom middle
    Eye Aspect Ratio (EAR) = vertical_distance / horizontal_distance
    """
    # Extract points
    if is_left:
        left_corner = landmarks[L_LEFT_CORNER]
        right_corner = landmarks[L_RIGHT_CORNER]
        top_middle = landmarks[L_TOP_MIDDLE]
        bottom_middle = landmarks[L_BOTTOM_MIDDLE]
    else:
        left_corner = landmarks[R_LEFT_CORNER]
        right_corner = landmarks[R_RIGHT_CORNER]
        top_middle = landmarks[R_TOP_MIDDLE]
        bottom_middle = landmarks[R_BOTTOM_MIDDLE]
    
    # Calculate vertical distance (between top and bottom middle)
    vertical_dist = abs(top_middle.y - bottom_middle.y) * image_height
    
    # Calculate horizontal distance (between left and right corners)
    horizontal_dist = abs(left_corner.x - right_corner.x) * image_width
    
    # Calculate EAR
    if horizontal_dist == 0:
        return 0.0
    ear = vertical_dist / horizontal_dist
    return ear

def crop_eye(frame, landmarks, img_w, img_h, outer, upperouter, upperinner, inner, lowerinner, lowerouter):
    """
    Calculate eye crop rectangle based on Swift code logic and crop the eye from frame.
    Returns cropped eye image or None if out of bounds.
    """
    def to_px(lm, img_w, img_h):
        return (lm.x * img_w, lm.y * img_h)
    
    outer_px = to_px(landmarks[outer], img_w, img_h)
    upperouter_px = to_px(landmarks[upperouter], img_w, img_h)
    upperinner_px = to_px(landmarks[upperinner], img_w, img_h)
    inner_px = to_px(landmarks[inner], img_w, img_h)
    lowerinner_px = to_px(landmarks[lowerinner], img_w, img_h)
    lowerouter_px = to_px(landmarks[lowerouter], img_w, img_h)
    
    width = abs(outer_px[0] - inner_px[0]) * 1.5
    height = width * 0.5
    cx = (outer_px[0] + upperouter_px[0] + upperinner_px[0] + inner_px[0] + lowerinner_px[0] + lowerouter_px[0]) / 6.0
    cy = (outer_px[1] + upperouter_px[1] + upperinner_px[1] + inner_px[1] + lowerinner_px[1] + lowerouter_px[1]) / 6.0
    
    x = cx - width / 2
    y = cy - height / 2
    
    # Intersect with image bounds
    x = max(0, min(x, img_w))
    y = max(0, min(y, img_h))
    width = min(width, img_w - x)
    height = min(height, img_h - y)
    
    # Make integral (round to integers)
    x = int(round(x))
    y = int(round(y))
    width = int(round(width))
    height = int(round(height))
    
    # Check if valid
    if width <= 0 or height <= 0 or x >= img_w or y >= img_h:
        return None
    
    # Crop the eye from frame
    return frame[y:y+height, x:x+width]

def process_subject(subject):
    # Input directories
    frames_dir = Path(f"../study1_rawdata_processed/p{subject}/frames")
    json_dir = Path(f"../study1_rawdata_processed/p{subject}/json")
    screen_dir = Path(f"../study1_rawdata_processed/screen")
    
    # Output directories
    eyepatch_output_dir = Path(f"preprocessed_478/p{subject}/eyepatch")
    screen_output_dir = Path(f"preprocessed_478/p{subject}/screen")
    json_output_dir = Path(f"preprocessed_478/p{subject}/json")
    
    os.makedirs(eyepatch_output_dir, exist_ok=True)
    os.makedirs(screen_output_dir, exist_ok=True)
    os.makedirs(json_output_dir, exist_ok=True)
    
    # Get all frame files
    frame_files = sorted(frames_dir.glob("*.jpg"))
    
    if not frame_files:
        print(f"No frame files found in {frames_dir}")
        return
    
    skipped_closed_eyes = 0
    skipped_frames = []
    
    # Track current session, participant, and trial to reset detector when they change
    current_subject = None
    current_session = None
    current_trial = None
    faceLandmarkDetector = None
    
    for frame_path in tqdm(frame_files, desc=f"Processing subject {subject}"):
        frame_name = frame_path.stem  # e.g., "00001" from "00001.jpg"
        
        # Read corresponding JSON file
        json_path = json_dir / f"{frame_name}.json"
        if not json_path.exists():
            print(f"!!! ERROR: JSON file not found: {json_path}, skipping frame {frame_name}")
            continue
        
        with open(json_path, 'r') as f:
            frame_data = json.load(f)
        
        # Extract data from JSON
        frame_subject = frame_data.get('subject')
        session = frame_data.get('session')
        trial = frame_data.get('trial')
        frameNum = int(frame_data.get('frameNum'))
        gt_x_px = frame_data.get('gt_x_px')
        gt_y_px = frame_data.get('gt_y_px')
        backgroundA = frame_data.get('backgroundA')
        backgroundB = frame_data.get('backgroundB')
        dissolve = frame_data.get('dissolve')
        
        # Reset faceLandmarkDetector if subject, session, or trial changed
        if current_subject != frame_subject or current_session != session or current_trial != trial:
            current_subject = frame_subject
            current_session = session
            current_trial = trial
            if faceLandmarkDetector is not None:    
                faceLandmarkDetector.close()
            faceLandmarkDetector = mp.tasks.vision.FaceLandmarker.create_from_options(face_landmarker_options)
            print(f"Reset faceLandmarkDetector for subject {frame_subject}, session {session}, trial {trial}")

        # Read frame
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"!!! ERROR: Could not read frame: {frame_path}, skipping: subject {frame_subject}, frame {frame_name}")
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = frame_rgb.shape[:2]
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Process with MediaPipe (pass frameNum as second parameter)
        results = faceLandmarkDetector.detect_for_video(mp_image, frameNum)
        if not results.face_landmarks:
            print(f"No face landmarks detected, skipping: subject {frame_subject}, frame {frame_name}")
            continue
        landmarks = results.face_landmarks[0]
        
        # Filter out blink instances
        l_ear = get_eye_aspect_ratio(landmarks, frame_width, frame_height, is_left=True)
        r_ear = get_eye_aspect_ratio(landmarks, frame_width, frame_height, is_left=False)
        l_eye_open = l_ear > EYE_CLOSED_THRESHOLD
        r_eye_open = r_ear > EYE_CLOSED_THRESHOLD
        if not (l_eye_open or r_eye_open):
            print(f"Skipping eyes closed frame: subject {frame_subject}, frame {frame_name} (L_EAR: {l_ear:.3f}, R_EAR: {r_ear:.3f})")
            skipped_closed_eyes += 1
            skipped_frames.append(frame_name)
            continue
        
        # Convert landmarks to pixel coordinates for rotation-based cropping
        def landmark_to_px(lm_idx):
            return (landmarks[lm_idx].x * frame_width, landmarks[lm_idx].y * frame_height)
        
        # Crop left eye with rotation
        # Find angle of the line from L_LEFT_CORNER to L_RIGHT_CORNER
        l_left_corner_px = landmark_to_px(L_LEFT_CORNER)
        l_right_corner_px = landmark_to_px(L_RIGHT_CORNER)
        l_angle = np.arctan2(l_right_corner_px[1] - l_left_corner_px[1], l_right_corner_px[0] - l_left_corner_px[0])
        
        # Center is L_IRIS
        l_iris_px = landmark_to_px(L_IRIS)
        l_center = (int(l_iris_px[0]), int(l_iris_px[1]))
        
        # Calculate scale based on eye width (distance between corners)
        l_eye_width = np.sqrt((l_right_corner_px[0] - l_left_corner_px[0])**2 + (l_right_corner_px[1] - l_left_corner_px[1])**2)
        l_scale = 250 / l_eye_width
        
        M = cv2.getRotationMatrix2D(l_center, int(l_angle * 180 / np.pi), l_scale)
        rotated_left_img = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        
        # Crop a square of size 300x300 centered at the L_IRIS
        # After rotation, the center coordinates remain the same
        l_crop_y_start = int(l_center[1] - 150)
        l_crop_y_end = int(l_center[1] + 150)
        l_crop_x_start = int(l_center[0] - 150)
        l_crop_x_end = int(l_center[0] + 150)
        
        # Check bounds
        if (l_crop_y_start < 0 or l_crop_y_end > rotated_left_img.shape[0] or 
            l_crop_x_start < 0 or l_crop_x_end > rotated_left_img.shape[1]):
            print(f"Warning: Left eye crop out of bounds, skipping: subject {frame_subject}, frame {frame_name}")
            continue
        
        cropped_left_img = rotated_left_img[l_crop_y_start:l_crop_y_end, l_crop_x_start:l_crop_x_end]
        
        # Crop right eye with rotation
        # Find angle of the line from R_LEFT_CORNER to R_RIGHT_CORNER
        r_left_corner_px = landmark_to_px(R_LEFT_CORNER)
        r_right_corner_px = landmark_to_px(R_RIGHT_CORNER)
        r_angle = np.arctan2(r_right_corner_px[1] - r_left_corner_px[1], r_right_corner_px[0] - r_left_corner_px[0])
        
        # Center is R_IRIS
        r_iris_px = landmark_to_px(R_IRIS)
        r_center = (int(r_iris_px[0]), int(r_iris_px[1]))
        
        # Calculate scale based on eye width
        r_eye_width = np.sqrt((r_right_corner_px[0] - r_left_corner_px[0])**2 + (r_right_corner_px[1] - r_left_corner_px[1])**2)
        r_scale = 250 / r_eye_width
        
        M = cv2.getRotationMatrix2D(r_center, int(r_angle * 180 / np.pi), r_scale)
        rotated_right_img = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        
        # Crop a square of size 300x300 centered at the R_IRIS
        # After rotation, the center coordinates remain the same
        r_crop_y_start = int(r_center[1] - 150)
        r_crop_y_end = int(r_center[1] + 150)
        r_crop_x_start = int(r_center[0] - 150)
        r_crop_x_end = int(r_center[0] + 150)
        
        # Check bounds
        if (r_crop_y_start < 0 or r_crop_y_end > rotated_right_img.shape[0] or 
            r_crop_x_start < 0 or r_crop_x_end > rotated_right_img.shape[1]):
            print(f"Warning: Right eye crop out of bounds, skipping: subject {frame_subject}, frame {frame_name}")
            continue
        
        cropped_right_img = rotated_right_img[r_crop_y_start:r_crop_y_end, r_crop_x_start:r_crop_x_end]
        
        # Concatenate left and right images to make 600x300 image (two 300x300 images side by side)
        combined_img = np.concatenate((cropped_left_img, cropped_right_img), axis=1)
        
        # Save the combined image
        combined_eye_path = eyepatch_output_dir / f"{frame_name}_combined.png"
        cv2.imwrite(str(combined_eye_path), combined_img)
        
        # Create and save dissolved screen image
        backgroundA_path = screen_dir / f"{backgroundA}.jpg"
        backgroundB_path = screen_dir / f"{backgroundB}.jpg"
        
        backgroundA_img = None
        backgroundB_img = None
        
        if not backgroundA_path.exists() or not backgroundB_path.exists():
            print(f"Warning: Could not find screen images {backgroundA_path} or {backgroundB_path}, skipping: subject {frame_subject}, frame {frame_name}")
        else:
            backgroundA_img = cv2.imread(str(backgroundA_path))
            backgroundB_img = cv2.imread(str(backgroundB_path))
            
            if backgroundA_img is None or backgroundB_img is None:
                print(f"Warning: Could not load screen images, skipping: subject {frame_subject}, frame {frame_name}")
            else:
                # dissolve value 0 is purely backgroundA, 1 is purely backgroundB
                dissolved_screen = cv2.addWeighted(backgroundA_img, 1.0 - dissolve, backgroundB_img, dissolve, 0)
                screen_output_path = screen_output_dir / f"{frame_name}.jpg"
                cv2.imwrite(str(screen_output_path), dissolved_screen)
        
        # Save all landmarks and ground truth data as json
        landmark_data = {}
        landmark_data["gt_x_px"] = gt_x_px
        landmark_data["gt_y_px"] = gt_y_px
        num_landmarks = len(landmarks)
        for i in range(num_landmarks):
            landmark_data[f"{i}x"] = landmarks[i].x
            landmark_data[f"{i}y"] = landmarks[i].y
                
        json_output_path = json_output_dir / f"{frame_name}.json"
        with open(json_output_path, 'w') as f:
            json.dump(landmark_data, f, indent=2)
    
    print(f"\nSubject {subject}: Processed {len(frame_files)} frames, skipped {skipped_closed_eyes} blink frames")

def main():
    parser = argparse.ArgumentParser(description='Process eye patch landmarks')
    parser.add_argument('-s', nargs='*', type=int, default=None,
                       help='List of subjects to process (e.g., -s 1 2 3). If not specified, uses default list.')
    args = parser.parse_args()
    
    # Use custom subjects list if provided, otherwise use default
    subjects_to_process = args.s if args.s is not None else SUBJECTS
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting eye patch processing...")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Total subjects to process: {subjects_to_process}")
    
    for subject in subjects_to_process:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing subject p{subject}")
        process_subject(subject)
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] All files processed!")

if __name__ == "__main__":
    main()
