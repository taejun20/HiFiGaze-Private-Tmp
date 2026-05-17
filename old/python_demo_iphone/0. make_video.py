import cv2
import pandas as pd
import numpy as np
import os
from pathlib import Path

TYPE = "RGBSOE"

def create_gaze_video():
    # Read the CSV file with predictions
    csv_path = f"data/{TYPE}_eval_screen2_filtered.csv"
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} not found")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} predictions from {csv_path}")
    
    # Sort by frame_num to ensure correct order
    df = df.sort_values('frame_num').reset_index(drop=True)
    
    # Load screen background
    screen_path = "data/screen2.jpg"
    if not os.path.exists(screen_path):
        print(f"Error: Screen image {screen_path} not found")
        return
    
    screen_img = cv2.imread(screen_path)
    if screen_img is None:
        print(f"Error: Could not load screen image from {screen_path}")
        return
    
    # Define video parameters
    fps = 30
    frame_width = 1920  # Increased width to accommodate all components
    frame_height = 1080  # Total video height
    
    # Define component sizes
    rawframe_height = 800  # Set height first to fit in video frame
    rawframe_width = int(rawframe_height * 9 / 16)  # 16:9 aspect ratio (width:height)
    eye_width = 400
    eye_height = 200
    eyepatch_width = 400  # Same size as eye images
    eyepatch_height = 200
    heatmap_width = 320
    heatmap_height = 440
    screen_width = 400
    screen_height = int(screen_width * 14.4 / 7.1)  # Maintain 7.1:14.4 ratio
   
    # Calculate positions
    # Raw frame positioned at the leftmost location
    rawframe_x = 20
    rawframe_y = (frame_height - rawframe_height) // 2
    
    # Eye processing components - vertically stacked to the right of raw frame
    processing_x = rawframe_x + rawframe_width + 20
    processing_y = (frame_height - eye_height - eyepatch_height - heatmap_height - 40) // 2
    
    # Eye images (eyepatch_and_soe) - left eye on left, right eye on right
    left_eye_x = processing_x
    left_eye_y = processing_y
    
    right_eye_x = processing_x + eye_width + 20
    right_eye_y = processing_y
    
    # Eyepatch images (input_eyepatch) - left eyepatch on left, right eyepatch on right, below eyes
    left_eyepatch_x = processing_x
    left_eyepatch_y = processing_y + eye_height + 10
    
    right_eyepatch_x = processing_x + eyepatch_width + 20
    right_eyepatch_y = left_eyepatch_y
    
    # Heatmap images (input_heatmap) - left heatmap on left, right heatmap on right, below eyepatches
    left_heatmap_x = processing_x
    left_heatmap_y = left_eyepatch_y + eyepatch_height + 10
    
    right_heatmap_x = processing_x + heatmap_width + 20
    right_heatmap_y = left_heatmap_y
    
    # Screen positioned at the rightmost location
    screen_x = right_eyepatch_x + eyepatch_width + 20
    screen_y = (frame_height - screen_height) // 2
    
    # Resize screen to fit our defined size using area interpolation for better text clarity (avoiding cubic and lanczos)
    screen_resized = cv2.resize(screen_img, (screen_width, screen_height), interpolation=cv2.INTER_AREA)
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "gaze_prediction_video.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    print(f"Creating video with {len(df)} frames at {fps} FPS...")
    print(f"Output: {output_path}")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing frames"):
        frame_num = str(int(row['frame_num']))
        pred_x = row['filtered_pred_px_norm_x']
        pred_y = row['filtered_pred_px_norm_y']
        
        # Create blank frame
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # Load and place raw frame
        rawframe_path = f"data/test2/rawdata_corrected/{frame_num}.jpg"
        if os.path.exists(rawframe_path):
            rawframe_img = cv2.imread(rawframe_path)
            if rawframe_img is not None:
                rawframe_resized = cv2.resize(rawframe_img, (rawframe_width, rawframe_height))
                frame[rawframe_y:rawframe_y+rawframe_height, rawframe_x:rawframe_x+rawframe_width] = rawframe_resized
        
        # Load and place left eye image
        left_eye_path = f"data/test2/preprocessed_corrected/input_debug/eyepatch_and_soe/{frame_num}_eyepatch_soe_left.jpg"
        if os.path.exists(left_eye_path):
            left_eye_img = cv2.imread(left_eye_path)
            if left_eye_img is not None:
                left_eye_resized = cv2.resize(left_eye_img, (eye_width, eye_height))
                left_eye_resized = cv2.flip(left_eye_resized, 1)
                frame[left_eye_y:left_eye_y+eye_height, left_eye_x:left_eye_x+eye_width] = left_eye_resized
        
        # Load and place right eye image
        right_eye_path = f"data/test2/preprocessed_corrected/input_debug/eyepatch_and_soe/{frame_num}_eyepatch_soe_right.jpg"
        if os.path.exists(right_eye_path):
            right_eye_img = cv2.imread(right_eye_path)
            if right_eye_img is not None:
                right_eye_resized = cv2.resize(right_eye_img, (eye_width, eye_height))
                right_eye_resized = cv2.flip(right_eye_resized, 1)
                frame[right_eye_y:right_eye_y+eye_height, right_eye_x:right_eye_x+eye_width] = right_eye_resized
        
        # Load and place left eyepatch image
        left_eyepatch_path = f"data/test2/preprocessed_corrected/input_eyepatch/{frame_num}_left.png"
        if os.path.exists(left_eyepatch_path):
            left_eyepatch_img = cv2.imread(left_eyepatch_path)
            if left_eyepatch_img is not None:
                left_eyepatch_resized = cv2.resize(left_eyepatch_img, (eyepatch_width, eyepatch_height))
                frame[left_eyepatch_y:left_eyepatch_y+eyepatch_height, left_eyepatch_x:left_eyepatch_x+eyepatch_width] = left_eyepatch_resized
        
        # Load and place right eyepatch image
        right_eyepatch_path = f"data/test2/preprocessed_corrected/input_eyepatch/{frame_num}_right.png"
        if os.path.exists(right_eyepatch_path):
            right_eyepatch_img = cv2.imread(right_eyepatch_path)
            if right_eyepatch_img is not None:
                right_eyepatch_resized = cv2.resize(right_eyepatch_img, (eyepatch_width, eyepatch_height))
                right_eyepatch_resized = cv2.flip(right_eyepatch_resized, 1)
                frame[right_eyepatch_y:right_eyepatch_y+eyepatch_height, right_eyepatch_x:right_eyepatch_x+eyepatch_width] = right_eyepatch_resized
        
        # Load and place left heatmap image
        left_heatmap_path = f"data/test2/preprocessed_corrected/input_heatmap/{frame_num}_left.png"
        if os.path.exists(left_heatmap_path):
            left_heatmap_img = cv2.imread(left_heatmap_path)
            if left_heatmap_img is not None:
                left_heatmap_resized = cv2.resize(left_heatmap_img, (heatmap_width, heatmap_height))
                frame[left_heatmap_y:left_heatmap_y+heatmap_height, left_heatmap_x:left_heatmap_x+heatmap_width] = left_heatmap_resized
        
        # Load and place right heatmap image
        right_heatmap_path = f"data/test2/preprocessed_corrected/input_heatmap/{frame_num}_right.png"
        if os.path.exists(right_heatmap_path):
            right_heatmap_img = cv2.imread(right_heatmap_path)
            if right_heatmap_img is not None:
                right_heatmap_resized = cv2.resize(right_heatmap_img, (heatmap_width, heatmap_height))
                right_heatmap_resized = cv2.flip(right_heatmap_resized, 1)
                frame[right_heatmap_y:right_heatmap_y+heatmap_height, right_heatmap_x:right_heatmap_x+heatmap_width] = right_heatmap_resized
        
        # Place screen background
        frame[screen_y:screen_y+screen_height, screen_x:screen_x+screen_width] = screen_resized

        # pred_x = 0.215
        # pred_y = 0.8716
        
        # Calculate gaze point position on screen
        # Multiply by 1000 and normalize with max values 710 and 1440
        gaze_x_norm = (pred_x * 1000) / 430.0
        gaze_y_norm = (pred_y * 1000) / 871.6
        
        # Clamp to valid range [0, 1]
        gaze_x_norm = np.clip(gaze_x_norm, 0, 1)
        gaze_y_norm = np.clip(gaze_y_norm, 0, 1)
        
        
        # Convert to screen coordinates using no-margin dimensions
        gaze_x_screen = int(screen_x + gaze_x_norm * screen_width)
        gaze_y_screen = int(screen_y + gaze_y_norm * screen_height)
        
        # Draw gaze point (red dot with white border)
        cv2.circle(frame, (gaze_x_screen, gaze_y_screen), 8, (255, 255, 255), -1)  # White border
        cv2.circle(frame, (gaze_x_screen, gaze_y_screen), 6, (0, 0, 255), -1)  # Red center
        
        # Add frame number text
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add prediction coordinates text
        cv2.putText(frame, f"Gaze: ({pred_x:.3f}, {pred_y:.3f})", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add labels
        # cv2.putText(frame, "Raw Frame", (rawframe_x + 10, rawframe_y - 10), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # cv2.putText(frame, "Left Eye", (left_eye_x, left_eye_y - 10), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # cv2.putText(frame, "Right Eye", (right_eye_x, right_eye_y - 10), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # cv2.putText(frame, "Left Eyepatch", (left_eyepatch_x, left_eyepatch_y - 10), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # cv2.putText(frame, "Right Eyepatch", (right_eyepatch_x, right_eyepatch_y - 10), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # cv2.putText(frame, "Left Heatmap", (left_heatmap_x, left_heatmap_y - 10), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        # cv2.putText(frame, "Right Heatmap", (right_heatmap_x, right_heatmap_y - 10), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        # cv2.putText(frame, "Gaze Prediction", (screen_x, screen_y - 10), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write frame to video
        out.write(frame)
        out.write(frame)
    
    # Release video writer
    out.release()
    print(f"Video saved as: {output_path}")
    print(f"Video duration: {len(df) / fps:.2f} seconds")

if __name__ == "__main__":
    from tqdm import tqdm
    create_gaze_video()
