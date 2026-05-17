import asyncio
import base64
import json
import struct
from io import BytesIO
import sys

import numpy as np
from PIL import Image
import torch

from GazeModelRGB_IOS import GazeModelRGB_IOS


def load_model(ckpt_path: str, device: str):
    model = GazeModelRGB_IOS().to(device)

    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False
    )

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


def decode_eye_tensor_raw(b64: str) -> torch.Tensor:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.float32)
    return torch.from_numpy(arr).view(1, 3, 250, 500)


async def handle_client(reader, writer, model, device):
    addr = writer.get_extra_info("peername")
    print(f"✅ client connected: {addr}", file=sys.stderr)

    try:
        while True:
            header = await reader.readexactly(4)
            (length,) = struct.unpack(">I", header)
            payload = await reader.readexactly(length)

            req = json.loads(payload.decode("utf-8"))
            #print("📥 received request:", req, file=sys.stderr)
            idx = req.get("index", -1)
            print(f"\n=========== FRAME {idx} ===========", file=sys.stderr)
            left = decode_eye_tensor_raw(req["left_eye_raw"]).to(device)
            right = decode_eye_tensor_raw(req["right_eye_raw"]).to(device)

          

            eye_landmarks = torch.tensor(
                req["eye_landmarks"], dtype=torch.float32, device=device
            ).view(1, 8)

            face_landmarks = torch.tensor(
                req["face_landmarks"], dtype=torch.float32, device=device
            ).view(1, -1)

            print(f"left: {left.shape} (dtype: {left.dtype}, min: {left.min().item()}, max: {left.max().item()})", file=sys.stderr)
            print(f"right: {right.shape} (dtype: {right.dtype}, min: {right.min().item()}, max: {right.max().item()})", file=sys.stderr)
            print(f"eye_landmarks: {eye_landmarks.shape} (dtype: {eye_landmarks.dtype}, min: {eye_landmarks.min().item()}, max: {eye_landmarks.max().item()})", file=sys.stderr)
            print(f"face_landmarks: {face_landmarks.shape} (dtype: {face_landmarks.dtype}, min: {face_landmarks.min().item()}, max: {face_landmarks.max().item()})", file=sys.stderr)

            try:
                with torch.no_grad():
                    pred = model(left, right, eye_landmarks, face_landmarks)
                    pred = pred.detach().cpu()
                    x = float(pred[0, 0].item())
                    y = float(pred[0, 1].item())
                resp = {"x": x, "y": y}

            except Exception as e:
                print("❌ INFERENCE ERROR:", repr(e), file=sys.stderr)
                resp = {"error": str(e), "x": 0.0, "y": 0.0}

            print("📤 sending response:", resp, file=sys.stderr)
            resp_bytes = json.dumps(resp).encode("utf-8")
            writer.write(struct.pack(">I", len(resp_bytes)))
            writer.write(resp_bytes)
            await writer.drain()

    except asyncio.IncompleteReadError:
        print(f"⚠️ client disconnected: {addr}", file=sys.stderr)

    finally:
        writer.close()
        await writer.wait_closed()


async def main(host, port, ckpt, device):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt, device)
    print(f"🚀 model loaded on {device}", file=sys.stderr)

    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, model, device),
        host,
        port,
    )
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9999)
    p.add_argument("--ckpt", default="GazeModelRGB_e10.pt")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    asyncio.run(main(args.host, args.port, args.ckpt, args.device))
