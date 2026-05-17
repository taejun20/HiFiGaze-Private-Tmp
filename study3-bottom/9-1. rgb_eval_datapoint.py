import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import cv2
import json
import torchvision.transforms as transforms
from GazeModelRGB import GazeModelRGB

# Constants
CHECKPOINT_PATH = "model_checkpoints/best/RGB_train_p1,2,7,9,10,11,13,14,15,16_val_p17,18_eval_p20,21,24,25,26_Epoch20_Point1_ValError1.47.pt"
SUBJECT = 25
SESSION = 3
FRAME_NUM = 434

# Image transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((250, 500)),  # (height, width)
    transforms.ToTensor()
])

def load_datapoint(subject, session, frame_num):
    """
    Load and preprocess a single datapoint
    Args:
        subject (int): Subject number
        session (int): Session number
        frame_num (str): Frame number
    Returns:
        dict: Dictionary containing preprocessed data
    """
    frame_num = str(frame_num).zfill(5)

    eyepatch_dir = Path(f"preprocessed/input_eyepatch/p{subject}/s{session}")
    landmark_dir = Path(f"preprocessed/input_landmark_and_gt/p{subject}/s{session}")
    
    # Get file paths
    left_eye_file = eyepatch_dir / f"{frame_num}_left.png"
    right_eye_file = eyepatch_dir / f"{frame_num}_right.png"
    json_file = landmark_dir / f"{frame_num}_landmark_and_gt.json"
    
    # Check if files exist
    if not all(f.exists() for f in [left_eye_file, right_eye_file, json_file]):
        raise FileNotFoundError(f"One or more required files not found for subject {subject}, session {session}, frame {frame_num}")
    
    # Load and preprocess images
    l_eye_img = cv2.imread(str(left_eye_file))
    r_eye_img = cv2.imread(str(right_eye_file))
    
    # Convert BGR to RGB
    l_eye_img = cv2.cvtColor(l_eye_img, cv2.COLOR_BGR2RGB)
    r_eye_img = cv2.cvtColor(r_eye_img, cv2.COLOR_BGR2RGB)
    
    # Apply transforms
    l_eye = transform(l_eye_img)
    r_eye = transform(r_eye_img)
    
    # Load JSON data
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    
    # Extract eye corner landmarks
    eyecorner_lmk = torch.tensor([
        json_data["l_eye_inner_corner_x"],
        json_data["l_eye_inner_corner_y"],
        json_data["l_eye_outer_corner_x"],
        json_data["l_eye_outer_corner_y"],
        json_data["r_eye_inner_corner_x"],
        json_data["r_eye_inner_corner_y"],
        json_data["r_eye_outer_corner_x"],
        json_data["r_eye_outer_corner_y"]
    ], dtype=torch.float32)
    
    # Extract ground truth
    gt = torch.tensor([
        json_data["px_norm_x"],
        json_data["px_norm_y"]
    ], dtype=torch.float32)
    
    return {
        'l_eye': l_eye.unsqueeze(0),  # Add batch dimension
        'r_eye': r_eye.unsqueeze(0),
        'eyecorner_lmk': eyecorner_lmk.unsqueeze(0),
        'gt': gt.unsqueeze(0)
    }

def init_model(device):
    """
    Initialize the model and load checkpoint
    Args:
        device: torch device to use
    Returns:
        model: Initialized and loaded model
    """
    model = GazeModelRGB()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def main(subject, session, frame_num):
    """
    Main function to perform inference on a single datapoint
    Args:
        subject (int): Subject number
        session (int): Session number
        frame_num (str): Frame number
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = init_model(device)

    try:
        # Load and preprocess data
        data = load_datapoint(subject, session, frame_num)
        
        # Move data to device
        l_eye = data['l_eye'].to(device)
        r_eye = data['r_eye'].to(device)
        eyecorner_lmk = data['eyecorner_lmk'].to(device)
        gt = data['gt'].to(device)

        # Perform inference
        with torch.no_grad():
            pred = model(l_eye, r_eye, eyecorner_lmk)

            # Scale predictions and ground truth to cm
            pred_cm = torch.stack([
                pred[:, 0] * (7.1000/0.430),
                pred[:, 1] * (14.3915/0.8716)
            ], dim=1)
            gt_cm = torch.stack([
                gt[:, 0] * (7.1000/0.430),
                gt[:, 1] * (14.3915/0.8716)
            ], dim=1)

            # Calculate Euclidean error
            error = torch.norm(pred_cm - gt_cm, dim=1).item()

            # Print results
            print(f"\nResults for Subject {subject}, Session {session}, Frame {frame_num}:")
            print(f"Predicted gaze point (cm): ({pred_cm[0,0]:.2f}, {pred_cm[0,1]:.2f})")
            print(f"Ground truth (cm): ({gt_cm[0,0]:.2f}, {gt_cm[0,1]:.2f})")
            print(f"Error: {error:.2f} cm")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main(SUBJECT, SESSION, FRAME_NUM)
