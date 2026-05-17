import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import json

def load_and_resize_image(image_path, target_size):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.resize(img, target_size)

def create_debug_video(subject, session, fps=30):
    csv_path = f"preprocessed/logs/p{subject}_s{session}_log_preprocessed.csv"
    soe_image_dir = f"preprocessed/grabcut_ver/debug/eyepatch_and_soe/p{subject}/s{session}"
    original_image_dir = f"preprocessed/grabcut_ver/input_eyepatch/p{subject}/s{session}"
    landmark_dir = f"preprocessed/grabcut_ver/input_landmark_and_gt/p{subject}/s{session}"
    template_dir = f"preprocessed/debug/template/p{subject}/s{session}"
    
    output_path = f"debug_soe_video/debug_soe_p{subject}_s{session}.mp4"
    os.makedirs("debug_soe_video", exist_ok=True)
    
    # Read evaluation results
    df = pd.read_csv(csv_path)
    
    # Video settings
    frame_width = 1450  # Total frame width
    frame_height = 600  # Total frame height
    eye_size = (440, 230)  # Size for each eye image
    template_size = (95, 125)  # Size for template image (width=95, height=125)
    
    # Phone frame dimensions with correct aspect ratio (7.1:14.4 ≈ 1:2.028)
    phone_frame_width = 200  # Base width
    phone_frame_height = int(phone_frame_width * (14.4 / 7.1))  # Height maintains aspect ratio
    
    # Calculate positions
    r_eye_pos = (50, (frame_height - eye_size[1]) // 2 + 100)  # Moved down by 100
    l_eye_pos = (50 + eye_size[0] + 50, (frame_height - eye_size[1]) // 2 + 100)  # Moved down by 100
    phone_frame_pos = (l_eye_pos[0] + eye_size[0] + 100, (frame_height - phone_frame_height) // 2)
    
    # Center the template horizontally between the two eyes
    template_x = (r_eye_pos[0] + l_eye_pos[0] + eye_size[0]) // 2 - template_size[0] // 2
    # Place template with enough space from top of frame
    template_pos = (template_x, 30)  # Fixed position from top
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize counters
    soe_count = 0
    non_soe_count = 0
    
    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating video"):
            # Create white frame
            frame = np.full((frame_height, frame_width, 3), 255, dtype=np.uint8)
            
            # Get file number from the .pt filename
            frame_name = str(int(row['frameNum'])).zfill(5)

            # Add frame number at the top
            frame_label = f"Frame: {frame_name} (P{subject} S{session})"
            cv2.putText(frame, frame_label,
                      (frame_width // 2 - 150, 30), cv2.FONT_HERSHEY_SIMPLEX,
                      1.0, (0, 0, 0), 2)

            # Check if SOE exists for this frame
            l_soe_path = os.path.join(soe_image_dir, f"{frame_name}_eyepatch_soe_left.jpg")
            r_soe_path = os.path.join(soe_image_dir, f"{frame_name}_eyepatch_soe_right.jpg")
            
            has_soe = os.path.exists(l_soe_path) and os.path.exists(r_soe_path)
            if has_soe:
                soe_count += 1
            else:
                non_soe_count += 1
            
            # Add SOE counter at top right
            total_frames = soe_count + non_soe_count
            soe_percentage = (soe_count / total_frames * 100) if total_frames > 0 else 0
            counter_text = [
                f"Total Frames: {total_frames}",
                f"SoE detect: {soe_count} ({soe_percentage:.1f}%)",
                f"No detect : {non_soe_count}"
            ]
            
            # Display counter text
            for i, text in enumerate(counter_text):
                cv2.putText(frame, text,
                          (frame_width - 300, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                          0.7, (0, 0, 0), 2)

            # Load and display template image
            template_path = os.path.join(template_dir, f"{frame_name}_used_template.png")
            if os.path.exists(template_path):
                template_img = load_and_resize_image(template_path, template_size)
                frame[template_pos[1]:template_pos[1]+template_size[1],
                     template_pos[0]:template_pos[0]+template_size[0]] = template_img
                
                # Add "Template" label above the image
                cv2.putText(frame, "Template:",
                          (template_pos[0] - 40, template_pos[1] - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            landmark_path = os.path.join(landmark_dir, f"{frame_name}_landmark_and_gt.json")
            l_eye_inner = None
            l_eye_outer = None
            r_eye_inner = None
            r_eye_outer = None

            with open(landmark_path, 'r') as f:
                landmark_data = json.loads(f.read())
                l_eye_inner = (landmark_data["l_eye_inner_corner_x"], 
                                landmark_data["l_eye_inner_corner_y"])
                l_eye_outer = (landmark_data["l_eye_outer_corner_x"],
                                landmark_data["l_eye_outer_corner_y"]) 
                r_eye_inner = (landmark_data["r_eye_inner_corner_x"],
                                 landmark_data["r_eye_inner_corner_y"])
                r_eye_outer = (landmark_data["r_eye_outer_corner_x"],
                                 landmark_data["r_eye_outer_corner_y"])
            
            # Load and place eye images
            # Check if eye patch images exist
            l_eye_path = f"{soe_image_dir}/{frame_name}_eyepatch_soe_left.jpg"
            if not os.path.exists(l_eye_path):
                l_eye_path = f"{original_image_dir}/{frame_name}_left.png"

            r_eye_path = f"{soe_image_dir}/{frame_name}_eyepatch_soe_right.jpg"
            if not os.path.exists(r_eye_path):
                r_eye_path = f"{original_image_dir}/{frame_name}_right.png"
            
            l_eye = load_and_resize_image(l_eye_path, eye_size)
            r_eye = load_and_resize_image(r_eye_path, eye_size)
            
            frame[l_eye_pos[1]:l_eye_pos[1]+eye_size[1], 
                  l_eye_pos[0]:l_eye_pos[0]+eye_size[0]] = l_eye
            frame[r_eye_pos[1]:r_eye_pos[1]+eye_size[1], 
                  r_eye_pos[0]:r_eye_pos[0]+eye_size[0]] = r_eye
            
            # Transform coordinates
            true_x = float(row['px_norm_x']) * (7.1/0.430)
            true_y = float(row['px_norm_y']) * (14.4/0.8716)

            # Convert cm coordinates to pixel coordinates within the frame
            true_x_px = int(phone_frame_pos[0] + (true_x * (430/7.1) * phone_frame_width / 430.0))
            true_y_px = int(phone_frame_pos[1] + (true_y * (871.6/14.4) * phone_frame_height / 871.6))
            
            # Draw true point (blue circle)
            cv2.circle(frame, (true_x_px, true_y_px), 5, (255, 0, 0), -1)
            
            # Draw phone frame and points
            cv2.rectangle(frame, (phone_frame_pos[0], phone_frame_pos[1]), (phone_frame_pos[0] + phone_frame_width, phone_frame_pos[1] + phone_frame_height), (0, 0, 0), 2)
            
            # Add text for coordinates
            text_y = phone_frame_pos[1] + phone_frame_height + 30
            cv2.putText(frame, f"GT: ({true_x:.2f}cm, {true_y:.2f}cm)", 
                      (phone_frame_pos[0], text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.6, (0, 0, 0), 1)
            
            # Add text for eye corner coordinates
            text_y = phone_frame_pos[1] + phone_frame_height + 60
            cv2.putText(frame, f"({r_eye_outer[0]:.4f}, {r_eye_outer[1]:.4f})",
                      (r_eye_pos[0] - 40, r_eye_pos[1] + eye_size[1] + 20), cv2.FONT_HERSHEY_SIMPLEX,
                      0.6, (0, 0, 0), 1)
            cv2.putText(frame, f"(right outer)",
                      (r_eye_pos[0] - 40 + 50, r_eye_pos[1] + eye_size[1] + 50), cv2.FONT_HERSHEY_SIMPLEX,
                      0.6, (0, 0, 0), 1)

            cv2.putText(frame, f"({r_eye_inner[0]:.4f}, {r_eye_inner[1]:.4f})", 
                      (r_eye_pos[0] + eye_size[0] // 2 + 40, r_eye_pos[1] + eye_size[1] + 20), cv2.FONT_HERSHEY_SIMPLEX,
                      0.6, (0, 0, 0), 1) 
            cv2.putText(frame, f"(right inner)",
                      (r_eye_pos[0] + eye_size[0] // 2 + 40 + 50, r_eye_pos[1] + eye_size[1] + 50), cv2.FONT_HERSHEY_SIMPLEX,
                      0.6, (0, 0, 0), 1)

            cv2.putText(frame, f"({l_eye_inner[0]:.4f}, {l_eye_inner[1]:.4f})", 
                      (l_eye_pos[0] - 40, l_eye_pos[1] + eye_size[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.6, (0, 0, 0), 1)
            cv2.putText(frame, f"(left inner)",
                      (l_eye_pos[0] - 40 + 50, l_eye_pos[1] + eye_size[1] + 50), cv2.FONT_HERSHEY_SIMPLEX,
                      0.6, (0, 0, 0), 1)
            cv2.putText(frame, f"({l_eye_outer[0]:.4f}, {l_eye_outer[1]:.4f})", 
                      (l_eye_pos[0] + eye_size[0] // 2 + 40, l_eye_pos[1] + eye_size[1] + 20), cv2.FONT_HERSHEY_SIMPLEX,
                      0.6, (0, 0, 0), 1)
            cv2.putText(frame, f"(left outer)",
                      (l_eye_pos[0] + eye_size[0] // 2 + 40 + 50, l_eye_pos[1] + eye_size[1] + 50), cv2.FONT_HERSHEY_SIMPLEX,
                      0.6, (0, 0, 0), 1)

            # Write the same frame four times
            for _ in range(4):
                out.write(frame)
    
    finally:
        out.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print(f"\nFinal Statistics for P{subject} S{session}:")
        print(f"Total Frames: {soe_count + non_soe_count}")
        print(f"SOE Frames: {soe_count} ({(soe_count / (soe_count + non_soe_count) * 100):.1f}%)")
        print(f"Non-SOE Frames: {non_soe_count}")

print("Creating debug video...")
for subject in range(1, 2):
    for session in range(1, 2):
        create_debug_video(subject, session)
        print(f"Video saved to debug_soe_video/debug_soe_p{subject}_s{session}.mp4")

print("Done!")