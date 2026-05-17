import json
import numpy as np
from pathlib import Path
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from GazeModelRGB_IOS import GazeModelRGB_IOS  # type: ignore
from GazeModelRGBT_IOS import GazeModelRGBT_IOS  # type: ignore

# Paths
IDX = 50
FRAMES_DIR = Path("test/Frames")
RGB_CHECKPOINT_PATH = "test/RGB_IOS_train_all_subjects_Epoch10_TrainLoss0.051.pt"
RGBT_CHECKPOINT_PATH = "test/RGBT_IOS_train_all_subjects_Epoch10_TrainLoss0.081.pt"

def load_inputs_for_index(idx: int):
    """Load one frame's inputs that were dumped on-device."""
    meta_path = FRAMES_DIR / f"{idx:05d}_meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    tensors = meta["tensors"]

    def load_tensor(name):
        info = tensors[name]
        filename = info["filename"]
        shape = info["shape"]  # e.g. [1, 3, 250, 500]
        arr = np.fromfile(FRAMES_DIR / filename, dtype=np.float32)
        arr = arr.reshape(shape)
        return arr

    left_eye = load_tensor("left_eye")
    right_eye = load_tensor("right_eye")
    eye_landmarks = load_tensor("eye_landmarks")
    face_landmarks = load_tensor("face_landmarks")
    template = load_tensor("template")

    # Raw frame path if you want for visualization
    raw_jpg_path = FRAMES_DIR / meta["raw_frame_jpg"]

    return left_eye, right_eye, eye_landmarks, face_landmarks, template, raw_jpg_path


# Example: load frame 0
left_eye_np, right_eye_np, eye_lm_np, face_lm_np, template_np, raw_jpg = load_inputs_for_index(
    IDX
)

# Convert to torch tensors
device = "cuda" if torch.cuda.is_available() else "cpu"
left_eye_t = torch.from_numpy(left_eye_np).to(device)
right_eye_t = torch.from_numpy(right_eye_np).to(device)
eye_lm_t = torch.from_numpy(eye_lm_np).to(device)
face_lm_t = torch.from_numpy(face_lm_np).to(device)
template_t = torch.from_numpy(template_np).to(device)

# Initialize models and load checkpoints
rgb_model = GazeModelRGB_IOS().to(device)
rgb_checkpoint = torch.load(RGB_CHECKPOINT_PATH, map_location=device, weights_only=False)
if "model_state_dict" in rgb_checkpoint:
    rgb_model.load_state_dict(rgb_checkpoint["model_state_dict"])
else:
    rgb_model.load_state_dict(rgb_checkpoint)

rgbt_model = GazeModelRGBT_IOS().to(device)
rgbt_checkpoint = torch.load(RGBT_CHECKPOINT_PATH, map_location=device, weights_only=False)
if "model_state_dict" in rgbt_checkpoint:
    rgbt_model.load_state_dict(rgbt_checkpoint["model_state_dict"])
else:
    rgbt_model.load_state_dict(rgbt_checkpoint)

rgb_model.eval()
rgbt_model.eval()

with torch.no_grad():
    rgb_gaze_pred = rgb_model(
        left_eye=left_eye_t,
        right_eye=right_eye_t,
        eye_landmarks=eye_lm_t,
        face_landmarks=face_lm_t,
    )
    rgb_gaze_x, rgb_gaze_y = rgb_gaze_pred[0].cpu().numpy().tolist()

    rgbt_gaze_pred = rgbt_model(
        left_eye=left_eye_t,
        right_eye=right_eye_t,
        template=template_t,
        eye_landmarks=eye_lm_t,
        face_landmarks=face_lm_t,
    )
    rgbt_gaze_x, rgbt_gaze_y = rgbt_gaze_pred[0].cpu().numpy().tolist()

print("PyTorch gaze (RGB):", rgb_gaze_x, rgb_gaze_y)
print("PyTorch gaze (RGBT):", rgbt_gaze_x, rgbt_gaze_y)
