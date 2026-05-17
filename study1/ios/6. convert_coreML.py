import torch
import torch.nn as nn
import coremltools as ct

from model import HiFiGaze_RGBModel_v2


RGB_CHECKPOINT = "checkpoints/RGB_v2_lr3e-4_train2,6,7,8,9,10,12,14,16,17,18,19,20,21,22,24,25_val1,11,13,15,26_Epoch4_ValLoss1.4222.pt"


class HiFiGaze_RGB_CoreML(nn.Module):
    def __init__(self, checkpoint_path: str):
        super().__init__()

        self.model = HiFiGaze_RGBModel_v2()

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def forward(self, eye, eye_lmk):
        """
        eye:     [1, 3, 300, 600] float32, range [0,1]
        eye_lmk: [1, 8]           float32, normalized [0,1]
        """
        return self.model(eye, eye_lmk)


print("🔁 Loading model…")
coreml_model = HiFiGaze_RGB_CoreML(RGB_CHECKPOINT)
coreml_model.eval()

# Dummy inputs (used ONLY for tracing)
example_eye = torch.rand(1, 3, 300, 600, dtype=torch.float32)
example_eye_lmk = torch.rand(1, 8, dtype=torch.float32)

print("🧵 Tracing…")
traced = torch.jit.trace(
    coreml_model,
    (example_eye, example_eye_lmk),
)

print("🍏 Converting to CoreML…")
mlmodel = ct.convert(
    traced,
    compute_precision=ct.precision.FLOAT32,
    inputs=[
        ct.TensorType(
            name="eye",
            shape=example_eye.shape,
        ),
        ct.TensorType(
            name="eye_lmk",
            shape=example_eye_lmk.shape,
        ),
    ],
    outputs=[
        ct.TensorType(name="output"),
    ],
)

mlmodel.save("HiFiGaze_RGB.mlpackage")
print("✅ Saved HiFiGaze_RGB.mlpackage")
