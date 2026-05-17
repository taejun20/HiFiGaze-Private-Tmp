import cv2
import mediapipe as mp
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import json

def get_unique_indices(connections):
    return list(set([idx for connection in connections for idx in connection]))

def save_debug_files(left_eye_tensor, right_eye_tensor, left_landmarks, right_landmarks, gaze_location, frame_num, output_dir):
    # Convert tensors to numpy arrays
    left_eye_img = left_eye_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
    right_eye_img = right_eye_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
    
    # Convert from RGB to BGR for saving
    left_eye_img = cv2.cvtColor(left_eye_img, cv2.COLOR_RGB2BGR)
    right_eye_img = cv2.cvtColor(right_eye_img, cv2.COLOR_RGB2BGR)
    
    # Create debug directory if it doesn't exist
    debug_dir = output_dir
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Save eye patch images
    cv2.imwrite(str(debug_dir / f"debug_left_eye_{frame_num:05d}.jpg"), left_eye_img)
    cv2.imwrite(str(debug_dir / f"debug_right_eye_{frame_num:05d}.jpg"), right_eye_img)
    
    # Create JSON data
    debug_data = {
        'left_eye_landmarks': left_landmarks.tolist(),
        'right_eye_landmarks': right_landmarks.tolist(),
        'gaze_location_cm': gaze_location.tolist()
    }
    
    # Save JSON file
    with open(debug_dir / f"debug_data_{frame_num:05d}.json", 'w') as f:
        json.dump(debug_data, f, indent=4)

def process_frame(image_path, face_mesh, mp_face_mesh, output_dir, frame_num, gaze_location):
    # Initialize MediaPipe Face Mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Define eye landmarks indices
    LEFT_EYE = list(mp_face_mesh.FACEMESH_LEFT_EYE)
    RIGHT_EYE = list(mp_face_mesh.FACEMESH_RIGHT_EYE)
    
    # Define eye corner indices
    LEFT_EYE_INNER_CORNER = 362
    LEFT_EYE_OUTER_CORNER = 263
    RIGHT_EYE_INNER_CORNER = 133
    RIGHT_EYE_OUTER_CORNER = 33
    
    # Define vertical eye points
    LEFT_EYE_UPPER = 386
    LEFT_EYE_LOWER = 374
    RIGHT_EYE_UPPER = 159
    RIGHT_EYE_LOWER = 145
    
    # Get all indices for each eye
    LEFT_EYE_INDICES = get_unique_indices(LEFT_EYE)
    RIGHT_EYE_INDICES = get_unique_indices(RIGHT_EYE)
    
    # Read the image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Could not load the image")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        raise ValueError(f"No face detected in frame {frame_num}")

    face_landmarks = results.multi_face_landmarks[0]

    # Function to process eye patch and get landmarks
    def process_eye(inner_corner_idx, outer_corner_idx, upper_idx, lower_idx, landmarks, eye_indices, flip=False):
        h, w = image_rgb.shape[:2]  # Use RGB image for consistency
        
        # Get corner points
        inner_corner = landmarks.landmark[inner_corner_idx]
        outer_corner = landmarks.landmark[outer_corner_idx]
        upper_point = landmarks.landmark[upper_idx]
        lower_point = landmarks.landmark[lower_idx]
        
        # Calculate center point
        center_x = (inner_corner.x + outer_corner.x) / 2
        center_y = (upper_point.y + lower_point.y) / 2
        
        # Calculate patch size based on eye width
        eye_width = abs(outer_corner.x - inner_corner.x) * w
        patch_size = int(eye_width * 2)  # twice the eye width
        
        # Calculate patch coordinates
        center_x_px = int(center_x * w)
        center_y_px = int(center_y * h)
        half_size = patch_size // 2
        
        # Check if patch would hit image boundaries
        left = center_x_px - half_size
        right = center_x_px + half_size
        top = center_y_px - half_size
        bottom = center_y_px + half_size
        
        if left < 0 or right >= w or top < 0 or bottom >= h:
            raise ValueError(f"Frame {frame_num} has face detection but not enough eye region area (patch would hit image boundaries)")
        
        # Extract patch from RGB image
        patch = image_rgb[top:bottom, left:right]
        
        # Flip if needed
        if flip:
            patch = cv2.flip(patch, 1)
        
        # Resize to 896x896
        patch = cv2.resize(patch, (896, 896), interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor [C, H, W]
        patch_tensor = torch.from_numpy(patch).to(torch.uint8).permute(2, 0, 1)
        
        # Get landmarks
        landmarks_list = []
        for idx in eye_indices:
            lm = landmarks.landmark[idx]
            x = lm.x
            y = lm.y
            landmarks_list.append([x, y])
        
        landmarks_tensor = torch.tensor(landmarks_list, dtype=torch.float32)
        
        return patch_tensor, landmarks_tensor

    # Process left and right eyes
    left_eye_tensor, left_landmarks = process_eye(
        LEFT_EYE_INNER_CORNER, LEFT_EYE_OUTER_CORNER,
        LEFT_EYE_UPPER, LEFT_EYE_LOWER,
        face_landmarks, LEFT_EYE_INDICES,
        flip=True
    )
    
    right_eye_tensor, right_landmarks = process_eye(
        RIGHT_EYE_INNER_CORNER, RIGHT_EYE_OUTER_CORNER,
        RIGHT_EYE_UPPER, RIGHT_EYE_LOWER,
        face_landmarks, RIGHT_EYE_INDICES,
        flip=False
    )

    # Save data as .pt file
    data = {
        'left_eye_patch': left_eye_tensor,  # [3, 896, 896] (flipped horizontally)
        'right_eye_patch': right_eye_tensor,  # [3, 896, 896]
        'left_eye_landmarks': left_landmarks,  # [16, 2] in 0-1 normalized scale
        'right_eye_landmarks': right_landmarks,  # [16, 2] in 0-1 normalized scale
        'gaze_location_cm': torch.tensor(gaze_location, dtype=torch.float32)  # [2] in cm relative to the camera center
    }
    
    output_path = output_dir / f"data_{frame_num:05d}.pt"
    torch.save(data, output_path)
    
    # Save debug files
    save_debug_files(
        left_eye_tensor,
        right_eye_tensor,
        left_landmarks,
        right_landmarks,
        data['gaze_location_cm'],
        frame_num,
        output_dir / "debug"
    )

if __name__ == "__main__":
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    # Setup paths
    dataset_dir = Path("src/datasets/dataset_go_keep")
    frames_dir = dataset_dir / "frames"
    output_dir = dataset_dir / "processed_data"
    output_dir.mkdir(exist_ok=True)
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(exist_ok=True)
    
    # Read gaze labels
    labels_df = pd.read_csv(dataset_dir / "gt_labels.csv")
    
    # Process each frame
    for frame_file in sorted(frames_dir.glob("frame_*.png")):

       
        frame_num = int(frame_file.stem.split('_')[1])
        print(f"Processing frame {frame_num}")
        
        # Debug: Only process first 5 frames
        #if frame_num >= 5:
        #    break

        try:
            # Get gaze location from CSV
            frame_data = labels_df[labels_df['frame_number'] == frame_num].iloc[0]
            gaze_location = [frame_data['dot_x_cm'], frame_data['dot_y_cm']]
            
            # Process frame and save .pt file
            process_frame(frame_file, face_mesh, mp_face_mesh, output_dir, frame_num, gaze_location)
            
        except Exception as e:
            print(f"Error processing frame {frame_num}: {str(e)}")
            continue
    
    face_mesh.close()
    print("Processing complete!")

# Print library versions
print(f"\nLibrary versions:")
print(f"OpenCV version: {cv2.__version__}")
print(f"MediaPipe version: {mp.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")
