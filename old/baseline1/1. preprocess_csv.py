# Saving .pt for millions of data points is not feasible (4.8MB * 1000000 = 4.8TB)
# Thus, save cropping cordinates of left_x, right_x, top_y, bottom_y for each eye in csv file.
# Also, save the eye landmark coordinates (0-1 normalized scale) and the gaze location (cm, relative to the camera center) in that csv file.
# And use that csv file to make the tensor from the raw image at train time.

import cv2
import mediapipe as mp
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import json
import time
from tqdm import tqdm

def get_unique_indices(connections):
    return list(set([idx for connection in connections for idx in connection]))

def save_debug_files(image_rgb, row_data, frame_num):
    """Save debug files including cropped eye images and landmark visualization."""
    # Create debug directory if it doesn't exist
    debug_dir = Path("debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Make a copy of the image for visualization
    debug_image = image_rgb.copy()
    h, w = debug_image.shape[:2]
    
    # Draw landmarks on the debug image
    # Left eye corners (red for inner, blue for outer)
    left_inner_x = int(row_data['left_eye_inner_x'] * w)
    left_inner_y = int(row_data['left_eye_inner_y'] * h)
    left_outer_x = int(row_data['left_eye_outer_x'] * w)
    left_outer_y = int(row_data['left_eye_outer_y'] * h)
    
    # Right eye corners (red for inner, blue for outer)
    right_inner_x = int(row_data['right_eye_inner_x'] * w)
    right_inner_y = int(row_data['right_eye_inner_y'] * h)
    right_outer_x = int(row_data['right_eye_outer_x'] * w)
    right_outer_y = int(row_data['right_eye_outer_y'] * h)
    
    # Draw circles at landmark positions (red for inner corners, blue for outer corners)
    cv2.circle(debug_image, (left_inner_x, left_inner_y), 3, (255, 0, 0), -1)  # Red
    cv2.circle(debug_image, (left_outer_x, left_outer_y), 3, (0, 0, 255), -1)  # Blue
    cv2.circle(debug_image, (right_inner_x, right_inner_y), 3, (255, 0, 0), -1)  # Red
    cv2.circle(debug_image, (right_outer_x, right_outer_y), 3, (0, 0, 255), -1)  # Blue
    
    # Save the debug image with landmarks
    cv2.imwrite(str(debug_dir / f"debug_frame_{frame_num:05d}.jpg"), cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
    
    # Crop and save left eye
    left_eye_img = image_rgb[
        row_data['left_eye_top_y']:row_data['left_eye_bottom_y'],
        row_data['left_eye_left_x']:row_data['left_eye_right_x']
    ]
    # Flip left eye horizontally for consistency
    left_eye_img = cv2.flip(left_eye_img, 1)
    
    # Crop and save right eye
    right_eye_img = image_rgb[
        row_data['right_eye_top_y']:row_data['right_eye_bottom_y'],
        row_data['right_eye_left_x']:row_data['right_eye_right_x']
    ]
    
    # Save eye patch images
    cv2.imwrite(str(debug_dir / f"debug_left_eye_{frame_num:05d}.jpg"), cv2.cvtColor(left_eye_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(debug_dir / f"debug_right_eye_{frame_num:05d}.jpg"), cv2.cvtColor(right_eye_img, cv2.COLOR_RGB2BGR))

        
def process_frame_for_csv(image_path, face_mesh, mp_face_mesh, frame_num, gaze_location):
    """Process a single frame and return a dictionary with the required data for CSV."""
    
    # Define eye landmarks indices
    LEFT_EYE = list(mp_face_mesh.FACEMESH_LEFT_EYE)
    RIGHT_EYE = list(mp_face_mesh.FACEMESH_RIGHT_EYE)

    # Define eye corner indices
    LEFT_EYE_INNER_CORNER = 362
    LEFT_EYE_OUTER_CORNER = 263
    RIGHT_EYE_INNER_CORNER = 133
    RIGHT_EYE_OUTER_CORNER = 33
    
    # Define vertical eye points (fixed indices)
    LEFT_EYE_UPPER = 386
    LEFT_EYE_LOWER = 374
    RIGHT_EYE_UPPER = 159
    RIGHT_EYE_LOWER = 145
    
    # Get all indices for each eye
    LEFT_EYE_INDICES = get_unique_indices(LEFT_EYE)
    RIGHT_EYE_INDICES = get_unique_indices(RIGHT_EYE)
    
    # Read the image in BGR
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load the image: {image_path}")
    
    # Convert BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    
    # Process the image with MediaPipe
    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        raise ValueError(f"No face detected in frame {frame_num}")

    face_landmarks = results.multi_face_landmarks[0]
    
    # Function to get eye coordinates
    def get_eye_coords(inner_corner_idx, outer_corner_idx, upper_idx, lower_idx, landmarks):
        inner_corner = landmarks.landmark[inner_corner_idx]
        outer_corner = landmarks.landmark[outer_corner_idx]
        upper_point = landmarks.landmark[upper_idx]
        lower_point = landmarks.landmark[lower_idx]
        
        # Calculate center and size
        center_x = (inner_corner.x + outer_corner.x) / 2
        center_y = (upper_point.y + lower_point.y) / 2
        eye_width = abs(outer_corner.x - inner_corner.x) * w
        patch_size = int(eye_width * 2)
        
        center_x_px = int(center_x * w)
        center_y_px = int(center_y * h)
        half_size = patch_size // 2
        
        return {
            'left_x': center_x_px - half_size,
            'right_x': center_x_px + half_size,
            'top_y': center_y_px - half_size,
            'bottom_y': center_y_px + half_size
        }
    
    # Get eye coordinates
    left_eye_coords = get_eye_coords(
        LEFT_EYE_INNER_CORNER, LEFT_EYE_OUTER_CORNER,
        LEFT_EYE_UPPER, LEFT_EYE_LOWER,
        face_landmarks
    )
    
    right_eye_coords = get_eye_coords(
        RIGHT_EYE_INNER_CORNER, RIGHT_EYE_OUTER_CORNER,
        RIGHT_EYE_UPPER, RIGHT_EYE_LOWER,
        face_landmarks
    )
    
    # Check if eye patches go beyond image boundaries
    if (left_eye_coords['left_x'] < 0 or left_eye_coords['right_x'] >= w or 
        left_eye_coords['top_y'] < 0 or left_eye_coords['bottom_y'] >= h or
        right_eye_coords['left_x'] < 0 or right_eye_coords['right_x'] >= w or
        right_eye_coords['top_y'] < 0 or right_eye_coords['bottom_y'] >= h):
        print("Face is detected but eye patch goes over the image border")
        raise ValueError(f"Frame {frame_num} has face detection but eye patch goes over the image border")
    
    # Initialize row data
    row_data = {
        'frame_num': f"{frame_num:05d}",
        'left_eye_left_x': left_eye_coords['left_x'],
        'left_eye_right_x': left_eye_coords['right_x'],
        'left_eye_top_y': left_eye_coords['top_y'],
        'left_eye_bottom_y': left_eye_coords['bottom_y'],
        'right_eye_left_x': right_eye_coords['left_x'],
        'right_eye_right_x': right_eye_coords['right_x'],
        'right_eye_top_y': right_eye_coords['top_y'],
        'right_eye_bottom_y': right_eye_coords['bottom_y']
    }
    
    # Add left eye landmarks
    lm = face_landmarks.landmark[LEFT_EYE_INNER_CORNER]
    row_data['left_eye_inner_x'] = lm.x
    row_data['left_eye_inner_y'] = lm.y
    
    lm = face_landmarks.landmark[LEFT_EYE_OUTER_CORNER]
    row_data['left_eye_outer_x'] = lm.x
    row_data['left_eye_outer_y'] = lm.y
    
    # Add right eye landmarks
    lm = face_landmarks.landmark[RIGHT_EYE_INNER_CORNER]
    row_data['right_eye_inner_x'] = lm.x
    row_data['right_eye_inner_y'] = lm.y
    
    lm = face_landmarks.landmark[RIGHT_EYE_OUTER_CORNER]
    row_data['right_eye_outer_x'] = lm.x
    row_data['right_eye_outer_y'] = lm.y
    
    # Add gaze coordinates
    row_data['gaze_x_cm'] = gaze_location[0]
    row_data['gaze_y_cm'] = gaze_location[1]
    
    
    # Save debug files
    # save_debug_files(image_rgb, row_data, frame_num)
    
    return row_data

def process_subject_directory(subject_dir, face_mesh, mp_face_mesh):
    """Process all frames in a subject directory and return a list of data rows."""
    try:
        # Read frames.json, dotInfo.json, and screen.json
        with open(subject_dir / "frames.json", 'r') as f:
            frames_list = json.load(f)
        
        with open(subject_dir / "dotInfo.json", 'r') as f:
            dot_info = json.load(f)
            
        with open(subject_dir / "screen.json", 'r') as f:
            screen_info = json.load(f)
            
        # Get indices where orientation is 1
        orientation_list = screen_info['Orientation']
        valid_indices = [i for i, orientation in enumerate(orientation_list) if orientation == 1]
        
        # Filter frames_list to only include frames with orientation 1
        valid_frames = [frames_list[i] for i in valid_indices]
        
        frames_dir = subject_dir / "frames"
        data_rows = []
        
        # Process each valid frame
        for frame_file_name, frame_idx in zip(valid_frames, valid_indices):
            frame_num = int(frame_file_name.split('.')[0])
            
            try:
                # Get gaze location from dotInfo.json using the original frame_idx
                gaze_location = [
                    dot_info['XCam'][frame_idx],
                    dot_info['YCam'][frame_idx]
                ]
                
                # Process frame
                frame_path = frames_dir / frame_file_name
                row_data = process_frame_for_csv(frame_path, face_mesh, mp_face_mesh, frame_num, gaze_location)
                row_data['subject'] = subject_dir.name
                data_rows.append(row_data)
                
            except Exception as e:
                print(f"Error processing frame {frame_num} in subject {subject_dir.name}: {str(e)}")
                continue
        
        print(f"Subject {subject_dir.name}: {len(valid_indices)} frames with orientation 1 out of {len(frames_list)} total frames")
        return data_rows
    
    except Exception as e:
        print(f"Error processing subject {subject_dir.name}: {str(e)}")
        return []

def main():
    gazecapture_dir = Path(r"../GazeCapture")
    output_file = "GazeCapture_csv_portrait.csv"
    
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    
    all_data = []
    
    # Get all subject directories
    subject_dirs = [d for d in gazecapture_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    subject_dirs.sort()
    
    print(f"Processing {len(subject_dirs)} subjects...")
    
    for subject_dir in tqdm(subject_dirs):
        # Break soon for debugging
        # if len(all_data) >= 100:
        #    print("Breaking after 5 subjects for debugging...")
        #    break
        try:
            subject_data = process_subject_directory(subject_dir, face_mesh, mp_face_mesh)
            all_data.extend(subject_data)
        except Exception as e:
            print(f"Error processing subject {subject_dir.name}: {str(e)}")
    
    # Create DataFrame and reorder columns to put 'subject' first
    df = pd.DataFrame(all_data)
    columns = ['subject'] + [col for col in df.columns if col != 'subject']
    df = df[columns]
    
    # Save to CSVoutput_dir
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    main()

# Print library versions
print(f"\nLibrary versions:")
print(f"OpenCV version: {cv2.__version__}")
print(f"MediaPipe version: {mp.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")
