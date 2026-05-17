import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
from GazeModelRGB_IOS import GazeModelRGB_IOS
import warnings
warnings.filterwarnings('ignore', message=".*You are using `torch.load` with `weights_only=False`.*", category=FutureWarning)
import cv2
import json
import torchvision.transforms as transforms

# Configuration
ID = "00120"
MODEL = "RGB_IOS"
CHECKPOINT_PATH = "test/RGB_IOS_train_all_subjects_Epoch10_TrainLoss0.052.pt"

# Use CPU only
device = torch.device('cpu')

# Define file paths
datapoint_id = ID
l_eye_path = f"test/{datapoint_id}_left.png"
r_eye_path = f"test/{datapoint_id}_right.png"
json_path = f"test/{datapoint_id}_lms.json"
#template_path = f"{datapoint_id}.png"

# Initialize transform for resizing images to 500x250
eye_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((250, 500)),  # (height, width)
    transforms.ToTensor()
])

# Load left and right eye images
print("\nLoading images...")
l_eye_img = cv2.imread(l_eye_path)
r_eye_img = cv2.imread(r_eye_path)

# Convert BGR to RGB
l_eye_img = cv2.cvtColor(l_eye_img, cv2.COLOR_BGR2RGB)
r_eye_img = cv2.cvtColor(r_eye_img, cv2.COLOR_BGR2RGB)

# Apply transforms (resize and convert to tensor)
l_eye = eye_transform(l_eye_img)  # [3, 250, 500]
r_eye = eye_transform(r_eye_img)  # [3, 250, 500]

# Add batch dimension
l_eye = l_eye.unsqueeze(0)  # [1, 3, 250, 500]
r_eye = r_eye.unsqueeze(0)  # [1, 3, 250, 500]

# Load JSON data
print("Loading JSON data...")
with open(json_path, 'r') as f:
    json_data = json.load(f)

LEFT_EYE_OUTER = 263
LEFT_EYE_INNER = 362
RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133
FACE_LANDMARK_COUNT = 478

# Extract eye corner landmarks
eyecorner_lmk = torch.tensor([
    json_data[f"{LEFT_EYE_INNER}x"],
    json_data[f"{LEFT_EYE_INNER}y"],
    json_data[f"{LEFT_EYE_OUTER}x"],
    json_data[f"{LEFT_EYE_OUTER}y"],
    json_data[f"{RIGHT_EYE_INNER}x"],
    json_data[f"{RIGHT_EYE_INNER}y"],
    json_data[f"{RIGHT_EYE_OUTER}x"],
    json_data[f"{RIGHT_EYE_OUTER}y"]
], dtype=torch.float32)

# Extract full face landmarks (0x/0y ... 477x/477y)
face_lmk_values = []
for idx in range(FACE_LANDMARK_COUNT):
    face_lmk_values.extend([json_data[f"{idx}x"], json_data[f"{idx}y"]])
face_lmk = torch.tensor(face_lmk_values, dtype=torch.float32)

# Add batch dimension
eyecorner_lmk = eyecorner_lmk.unsqueeze(0)  # [1, 8]
face_lmk = face_lmk.unsqueeze(0)  # [1, 956]

# Extract ground truth for comparison
# gt = torch.tensor([
#     json_data["px_norm_x"],
#     json_data["px_norm_y"]
# ], dtype=torch.float32)

# Load model
print(f"\nLoading model from checkpoint: {CHECKPOINT_PATH}")
model = GazeModelRGB_IOS().to(device)

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        # If checkpoint is just the state dict
        model.load_state_dict(checkpoint)
        print("Loaded model state dict")
else:
    print(f"Warning: Checkpoint file not found: {CHECKPOINT_PATH}")
    print("Using randomly initialized model weights")

model.eval()

# Move tensors to device
l_eye = l_eye.to(device)
r_eye = r_eye.to(device)
eyecorner_lmk = eyecorner_lmk.to(device)
face_lmk = face_lmk.to(device)

# Run inference
#print(r_eye)
print("\nRunning inference...")
with torch.no_grad():
    pred = model(l_eye, r_eye, eyecorner_lmk, face_lmk)

# Convert prediction to numpy
pred_norm = pred[0].cpu().numpy()

# Print results
print("\n" + "="*60)
print("INFERENCE RESULTS")
print("="*60)
print(f"Datapoint ID: {datapoint_id}")
print(f"\nPrediction (1000 pixel):")
print(f"  X: {pred_norm[0]:.6f}")
print(f"  Y: {pred_norm[1]:.6f}")
print("="*60)