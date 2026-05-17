import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import json
import torch
import torchvision.transforms as transforms
from pathlib import Path
from model_architecture_rgbsoe import PretrainDataModelRGB_SOE

def load_and_resize_image(image_path, target_size):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.resize(img, target_size)

def load_model(checkpoint_path, device):
    """Load the trained RGBSOE model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    model = PretrainDataModelRGB_SOE()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def load_model_input(frame_name, eyepatch_dir, landmark_dir):
    """Load input data for the model (RGBSOE version - includes SOE cues)"""
    # Load left and right eye images
    l_eye_path = eyepatch_dir / f"{frame_name}_left.png"
    r_eye_path = eyepatch_dir / f"{frame_name}_right.png"
    json_path = landmark_dir / f"{frame_name}_landmark_and_gt.json"
    
    if not (l_eye_path.exists() and r_eye_path.exists() and json_path.exists()):
        return None, None, None, None, None
    
    # Load images
    l_eye_img = cv2.imread(str(l_eye_path))
    r_eye_img = cv2.imread(str(r_eye_path))
    
    # Convert BGR to RGB
    l_eye_img = cv2.cvtColor(l_eye_img, cv2.COLOR_BGR2RGB)
    r_eye_img = cv2.cvtColor(r_eye_img, cv2.COLOR_BGR2RGB)
    
    # Initialize transform for resizing images to 500x250 (width x height)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((250, 500)),  # (height, width)
        transforms.ToTensor()
    ])
    
    # Apply transforms
    l_eye = transform(l_eye_img).unsqueeze(0)  # [1, 3, 250, 500]
    r_eye = transform(r_eye_img).unsqueeze(0)  # [1, 3, 250, 500]
    
    # Load JSON data
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Extract SOE cues (first 8 values)
    soe_cues = torch.tensor([[
        json_data["l_irisbox_center_to_soe_center_x"],
        json_data["l_irisbox_center_to_soe_center_y"],
        json_data["l_soe_width"],
        json_data["l_soe_height"],
        json_data["r_irisbox_center_to_soe_center_x"],
        json_data["r_irisbox_center_to_soe_center_y"],
        json_data["r_soe_width"],
        json_data["r_soe_height"]
    ]], dtype=torch.float32)
    
    # Extract eye corner landmarks (8 values)
    eyecorner_lmk = torch.tensor([[
        json_data["l_eye_inner_corner_x"],
        json_data["l_eye_inner_corner_y"],
        json_data["l_eye_outer_corner_x"],
        json_data["l_eye_outer_corner_y"],
        json_data["r_eye_inner_corner_x"],
        json_data["r_eye_inner_corner_y"],
        json_data["r_eye_outer_corner_x"],
        json_data["r_eye_outer_corner_y"]
    ]], dtype=torch.float32)
    
    # Extract ground truth
    gt = torch.tensor([[
        json_data["px_norm_x"],
        json_data["px_norm_y"]
    ]], dtype=torch.float32)
    
    return l_eye, r_eye, soe_cues, eyecorner_lmk, gt

def create_debug_video(subject, session, model_checkpoint_path, fps=30):
    csv_path = f"preprocessed/logs/p{subject}_s{session}_log_preprocessed.csv"
    soe_image_dir = f"preprocessed/debug/eyepatch_and_soe/p{subject}/s{session}"
    original_image_dir = f"preprocessed/input_eyepatch/p{subject}/s{session}"
    landmark_dir = f"preprocessed/input_landmark_and_gt/p{subject}/s{session}"
    template_dir = f"preprocessed/debug/template/p{subject}/s{session}"
    
    output_path = f"debug_soe_video/pred_p{subject}_s{session}.mp4"
    os.makedirs("debug_soe_video", exist_ok=True)
    
    # Set device and load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {model_checkpoint_path}")
    model = load_model(model_checkpoint_path, device)
    print(f"Model loaded successfully on {device}")
    
    # Set up data paths
    eyepatch_dir = Path(original_image_dir)
    landmark_dir_path = Path(landmark_dir)
    
    # Read evaluation results
    df = pd.read_csv(csv_path)
    
    # Video settings
    frame_width = 1450  # Total frame width
    frame_height = 600  # Total frame height
    eye_size = (440, 220)  # Size for each eye image
    template_size = (95, 125)  # Size for template image (width=95, height=125)
    
    # Eye corner landmark frame (1:1.77 ratio)
    landmark_frame_size = (124, 220)  # Size for landmark visualization frame (width, height)
    
    # Phone frame dimensions with correct aspect ratio (7.1:14.4 ≈ 1:2.028)
    phone_frame_width = 200  # Base width
    phone_frame_height = int(phone_frame_width * (14.4 / 7.1))  # Height maintains aspect ratio
    
    # Calculate positions
    r_eye_pos = (50, (frame_height - eye_size[1]) // 2 + 100)  # Moved down by 100
    l_eye_pos = (50 + eye_size[0] + 50, (frame_height - eye_size[1]) // 2 + 100)  # Moved down by 100
    phone_frame_pos = (l_eye_pos[0] + eye_size[0] + 100, (frame_height - phone_frame_height) // 2)
    
    # Landmark frame position (moved right to avoid overlap)
    landmark_frame_x = (r_eye_pos[0] + l_eye_pos[0] + eye_size[0]) // 2 - landmark_frame_size[0] // 2 + 200  # Shifted 200px right
    landmark_frame_y = r_eye_pos[1] - landmark_frame_size[1] - 20  # 20px gap above eyes
    landmark_frame_pos = (landmark_frame_x, landmark_frame_y)
    
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
            
            # Add legend for GT (blue) and Prediction (red)
            legend_y_start = 70
            cv2.circle(frame, (50, legend_y_start), 5, (255, 0, 0), -1)  # Blue circle
            cv2.putText(frame, "Ground Truth", (65, legend_y_start + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            cv2.circle(frame, (200, legend_y_start), 5, (0, 0, 255), -1)  # Red circle  
            cv2.putText(frame, "Prediction", (215, legend_y_start + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            # Load JSON data and check if SOE exists for this frame
            json_file_path = os.path.join(landmark_dir, f"{frame_name}_landmark_and_gt.json")
            has_soe = False
            json_data = None
            
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as f:
                    json_data = json.load(f)
                    # Check if SOE detection was successful (not -999)
                    has_soe = json_data["l_irisbox_center_to_soe_center_y"] != -999
            
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

            # Use the JSON data we already loaded for SOE check
            if json_data is not None:
                landmark_data = json_data  # Already loaded above
                l_eye_inner = (landmark_data["l_eye_inner_corner_x"], 
                                landmark_data["l_eye_inner_corner_y"])
                l_eye_outer = (landmark_data["l_eye_outer_corner_x"],
                                landmark_data["l_eye_outer_corner_y"]) 
                r_eye_inner = (landmark_data["r_eye_inner_corner_x"],
                                 landmark_data["r_eye_inner_corner_y"])
                r_eye_outer = (landmark_data["r_eye_outer_corner_x"],
                                 landmark_data["r_eye_outer_corner_y"])
            else:
                # Skip this frame if JSON data is not available
                continue
            
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
            
            # Generate model prediction
            model_input = load_model_input(frame_name, eyepatch_dir, landmark_dir_path)
            if model_input[0] is not None:  # Check if data was loaded successfully
                l_eye_tensor, r_eye_tensor, soe_cues_tensor, eyecorner_lmk_tensor, gt_tensor = model_input
                
                # Move to device
                l_eye_tensor = l_eye_tensor.to(device)
                r_eye_tensor = r_eye_tensor.to(device)
                soe_cues_tensor = soe_cues_tensor.to(device)
                eyecorner_lmk_tensor = eyecorner_lmk_tensor.to(device)
                
                # Run inference
                with torch.no_grad():
                    pred = model(l_eye_tensor, r_eye_tensor, soe_cues_tensor, eyecorner_lmk_tensor)
                    pred = pred.cpu().numpy()[0]  # Get first (and only) prediction
                
                # Transform prediction coordinates to cm
                pred_x = pred[0] * (7.1/0.430)
                pred_y = pred[1] * (14.4/0.8716)
                
                # Convert cm coordinates to pixel coordinates within the frame
                pred_x_px = int(phone_frame_pos[0] + (pred_x * (430/7.1) * phone_frame_width / 430.0))
                pred_y_px = int(phone_frame_pos[1] + (pred_y * (871.6/14.4) * phone_frame_height / 871.6))
                
                # Draw prediction point (red circle)
                cv2.circle(frame, (pred_x_px, pred_y_px), 5, (0, 0, 255), -1)
                
                # Add prediction text
                pred_text_y = phone_frame_pos[1] + phone_frame_height + 55
                cv2.putText(frame, f"Pred: ({pred_x:.2f}cm, {pred_y:.2f}cm)", 
                          (phone_frame_pos[0], pred_text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.6, (0, 0, 0), 1)
                          
                # Calculate and display error
                error_cm = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
                error_text_y = phone_frame_pos[1] + phone_frame_height + 80
                cv2.putText(frame, f"Error: {error_cm:.2f}cm", 
                          (phone_frame_pos[0], error_text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.6, (0, 0, 0), 1)
            
            # Draw phone frame and points
            cv2.rectangle(frame, (phone_frame_pos[0], phone_frame_pos[1]), (phone_frame_pos[0] + phone_frame_width, phone_frame_pos[1] + phone_frame_height), (0, 0, 0), 2)
            
            # Draw landmark frame (1:1.77 ratio) above the eyes
            cv2.rectangle(frame, (landmark_frame_pos[0], landmark_frame_pos[1]), 
                         (landmark_frame_pos[0] + landmark_frame_size[0], landmark_frame_pos[1] + landmark_frame_size[1]), 
                         (0, 0, 0), 2)
            
            # Add landmark frame label
            cv2.putText(frame, "Eye Corners", 
                       (landmark_frame_pos[0], landmark_frame_pos[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Plot the four eye corner points within the landmark frame using normalized coordinates
            # Convert normalized coordinates (0-1) to pixel coordinates within the frame
            def plot_landmark_point(norm_x, norm_y, color):
                pixel_x = int(landmark_frame_pos[0] + norm_x * landmark_frame_size[0])
                pixel_y = int(landmark_frame_pos[1] + norm_y * landmark_frame_size[1])
                cv2.circle(frame, (pixel_x, pixel_y), 3, color, -1)
            
            # Plot four eye corner points (different colors for each)
            plot_landmark_point(r_eye_outer[0], r_eye_outer[1], (0, 0, 255))    # Red - Right Outer
            plot_landmark_point(r_eye_inner[0], r_eye_inner[1], (0, 165, 255)) # Orange - Right Inner  
            plot_landmark_point(l_eye_inner[0], l_eye_inner[1], (0, 255, 0))   # Green - Left Inner
            plot_landmark_point(l_eye_outer[0], l_eye_outer[1], (255, 0, 0))   # Blue - Left Outer
    
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

print("Creating debug video with model predictions...")

# Model checkpoint path
model_checkpoint_path = f"model_checkpoints/SOERGB_train_p1234567891121_eval_p10_Epoch27_EvalError3.6020.pt"

# Process subject 10, session 1 as requested
subject = 1

for session in range(1, 1):
    create_debug_video(subject, session, model_checkpoint_path)
    print(f"Video saved to debug_soe_video/debug_soe_predictions_p{subject}_s{session}.mp4")

print("Done!")