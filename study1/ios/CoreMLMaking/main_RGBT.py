import torch
import coremltools as ct
from GazeModelRGBT_IOS import GazeModelRGBT_IOS

CHECKPOINT = "checkpoints/RGBT_IOS_train_all_subjects_Epoch20_TrainLoss0.060.pt"
OUTPUT = "GazeModelRGBT_e20.mlpackage"

def strip_module_prefix(state_dict):
    """
    If the checkpoint was trained with DataParallel, keys may start with 'module.'.
    This removes that prefix.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def load_state_dict(checkpoint_path, device="cpu"):
    """
    Loads a checkpoint that might be:
      - a raw state_dict
      - a dict containing 'state_dict' key
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    # If it's something like {"state_dict": ..., "epoch": ...}
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = strip_module_prefix(state_dict)

    return state_dict


device = "cpu"
model = GazeModelRGBT_IOS()
state_dict = load_state_dict(CHECKPOINT, device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ---- Dummy inputs ----
# left_eye: RGB image (width 500 x height 250) -> (batch, channels, height, width)
# right_eye: RGB image (width 500 x height 250) -> (batch, channels, height, width)
# eye_landmarks: 16 length vector -> (batch, 16)
left_eye = torch.randn(1, 3, 250, 500, device=device)
right_eye = torch.randn(1, 3, 250, 500, device=device)
template = torch.randn(1, 3, 101, 50, device=device)
eye_landmarks = torch.randn(1, 8, device=device)
face_landmarks = torch.randn(1, 956, device=device)

print(f"[INFO] Tracing model with inputs:")
print(f"  left_eye shape: {tuple(left_eye.shape)}")
print(f"  right_eye shape: {tuple(right_eye.shape)}")
print(f"  template shape: {tuple(template.shape)}")
print(f"  eye_landmarks shape: {tuple(eye_landmarks.shape)}")
print(f"  face_landmarks shape: {tuple(face_landmarks.shape)}")

with torch.no_grad():
    traced_model = torch.jit.trace(model, (left_eye, right_eye, template, eye_landmarks, face_landmarks))
    traced_model = torch.jit.freeze(traced_model)

# ---- CoreML conversion ----
convert_kwargs = {
    "inputs": [
        ct.TensorType(name="left_eye", shape=left_eye.shape),
        ct.TensorType(name="right_eye", shape=right_eye.shape),
        ct.TensorType(name="template", shape=template.shape),
        ct.TensorType(name="eye_landmarks", shape=eye_landmarks.shape),
        ct.TensorType(name="face_landmarks", shape=face_landmarks.shape)
    ],
    "outputs": [
        ct.TensorType(name="gaze")
    ],
    "convert_to": "mlprogram"
}

print("[INFO] Converting to CoreML...")
mlmodel = ct.convert(traced_model, **convert_kwargs)

# ---- Metadata ----
mlmodel.input_description["left_eye"] = (
    "left_eye (1, 3, 250, 500)."
)
mlmodel.input_description["right_eye"] = (
    "right_eye (1, 3, 250, 500)."
)
mlmodel.input_description["template"] = (
    "template (1, 3, 101, 50)."
)
mlmodel.input_description["eye_landmarks"] = (
    "eye_landmarks (1, 8)."
)
mlmodel.input_description["face_landmarks"] = (
    "face_landmarks (1, 956)."
)
mlmodel.output_description["gaze"] = (
    "gaze (x, y) (unit: 1000px, top-left corner as (0, 0))."
)
# ---- Save as .mlpackage ----
if not OUTPUT.endswith(".mlpackage"):
    output_path = OUTPUT + ".mlpackage"
else:
    output_path = OUTPUT

mlmodel.save(output_path)
print(f"[SUCCESS] Saved CoreML model to: {output_path}")

# Optional: print a quick summary
print("\n[INFO] Model spec summary:")
print(mlmodel)

