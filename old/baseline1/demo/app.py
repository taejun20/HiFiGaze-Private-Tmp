from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import time
import threading
import json
import argparse
import torch
import mediapipe as mp
from camera import Camera
from utils import *
from model_architecture import iTrackerCNN

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
cap = None

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = iTrackerCNN().to(device)
checkpoint = torch.load("eye_tracking_model_final.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define eye landmarks indices
LEFT_EYE_INNER_CORNER = 362
LEFT_EYE_OUTER_CORNER = 263
RIGHT_EYE_INNER_CORNER = 133
RIGHT_EYE_OUTER_CORNER = 33
LEFT_EYE_UPPER = 386
LEFT_EYE_LOWER = 374
RIGHT_EYE_UPPER = 159
RIGHT_EYE_LOWER = 145

# Initialize dot position and movement parameters
dot_x = 100  # Start from a visible position
dot_y = 100
direction_x = 1
direction_y = 1
dot_speed = 2  # Reduced speed for smoother movement
dot_radius = 10
screen_width = 980  # Default values
screen_height = 1684

def get_eye_coords(inner_corner_idx, outer_corner_idx, upper_idx, lower_idx, landmarks, w, h):
    inner_corner = landmarks.landmark[inner_corner_idx]
    outer_corner = landmarks.landmark[outer_corner_idx]
    upper_point = landmarks.landmark[upper_idx]
    lower_point = landmarks.landmark[lower_idx]
    
    # Calculate center and size
    center_x = (inner_corner.x + outer_corner.x) / 2
    center_y = (upper_point.y + lower_point.y) / 2
    eye_width = abs(outer_corner.x - inner_corner.x) * w
    patch_size = int(eye_width * 2)
    
    center_x_px = int(center_x * w)
    center_y_px = int(center_y * h)
    half_size = patch_size // 2
    
    return {
        'left_x': center_x_px - half_size,
        'right_x': center_x_px + half_size,
        'top_y': center_y_px - half_size,
        'bottom_y': center_y_px + half_size
    }

def convert_gaze_to_screen_coords(gaze_x, gaze_y, screen_width, screen_height):
    """Convert gaze coordinates to screen coordinates.
    gaze_x: -3.65 to 3.65 maps to screen_x: 0 to 420
    gaze_y: -3.3 to -17 maps to screen_y: 0 to 863
    """
    # Convert x coordinate
    # Map from [-3.65, 3.65] to [0, 420]
    x_range = 3.65 - (-3.65)  # 7.3 cm total width
    normalized_x = (gaze_x - (-3.65)) / x_range  # 0 to 1
    dot_x = normalized_x * 420
    
    # Convert y coordinate
    # Map from [-3.3, -17] to [0, 863]
    y_range = -3.3 - (-17)  # 13.7 cm total height
    normalized_y = (gaze_y - (-3.3)) / y_range  # 0 to 1
    dot_y = normalized_y * 863
    
    return int(dot_x), int(dot_y)

def process_eye_image(img, crop_coord, is_left_eye=True):
    """Process eye image to match training data format."""

    # Check if crop coordinates are outside image boundaries
    h, w = img.shape[:2]
    if (crop_coord['top_y'] < 0 or 
        crop_coord['bottom_y'] > h or
        crop_coord['left_x'] < 0 or 
        crop_coord['right_x'] > w):
        return None

    # Crop
    img = img[crop_coord['top_y']:crop_coord['bottom_y'], 
              crop_coord['left_x']:crop_coord['right_x']]
    
    # Flip left eye horizontally
    if is_left_eye:
        img = cv2.flip(img, 1)
    
    # Resize and convert to tensor
    img = cv2.resize(img, (128, 128))
    tensor = torch.from_numpy(img).to(torch.float32).permute(2, 0, 1)  # [3, 128, 128]

    # Calculate mean and std for each RGB channel separately
    r_mean = tensor[0].mean()
    g_mean = tensor[1].mean()
    b_mean = tensor[2].mean()
    
    r_std = tensor[0].std()
    g_std = tensor[1].std()
    b_std = tensor[2].std()
    
    # Normalize each channel separately
    tensor[0] = (tensor[0] - r_mean) / r_std
    tensor[1] = (tensor[1] - g_mean) / g_std
    tensor[2] = (tensor[2] - b_mean) / b_std
            
    return tensor

def camera_capture():
    while True:
        success, frame = cap.read()
        if not success:
            continue

        start_time = time.time()
        
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]        
        
        # Process the image with MediaPipe
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Get eye coordinates
            left_eye_coords = get_eye_coords(
                LEFT_EYE_INNER_CORNER, LEFT_EYE_OUTER_CORNER,
                LEFT_EYE_UPPER, LEFT_EYE_LOWER,
                face_landmarks, w, h
            )
            
            right_eye_coords = get_eye_coords(
                RIGHT_EYE_INNER_CORNER, RIGHT_EYE_OUTER_CORNER,
                RIGHT_EYE_UPPER, RIGHT_EYE_LOWER,
                face_landmarks, w, h
            )
            
            # Process eye images
            left_eye_tensor = process_eye_image(frame_rgb, left_eye_coords, is_left_eye=True)
            right_eye_tensor = process_eye_image(frame_rgb, right_eye_coords, is_left_eye=False)
            
            if left_eye_tensor is None or right_eye_tensor is None:
                continue

            # Get eye landmarks
            eye_landmarks = torch.tensor([
                face_landmarks.landmark[LEFT_EYE_INNER_CORNER].x,
                face_landmarks.landmark[LEFT_EYE_INNER_CORNER].y,
                face_landmarks.landmark[LEFT_EYE_OUTER_CORNER].x,
                face_landmarks.landmark[LEFT_EYE_OUTER_CORNER].y,
                face_landmarks.landmark[RIGHT_EYE_INNER_CORNER].x,
                face_landmarks.landmark[RIGHT_EYE_INNER_CORNER].y,
                face_landmarks.landmark[RIGHT_EYE_OUTER_CORNER].x,
                face_landmarks.landmark[RIGHT_EYE_OUTER_CORNER].y,
            ], dtype=torch.float32)
            
            # Prepare input tensors
            left_eye_tensor = left_eye_tensor.unsqueeze(0).to(device)  # Add batch dimension
            right_eye_tensor = right_eye_tensor.unsqueeze(0).to(device)
            eye_landmarks = eye_landmarks.unsqueeze(0).to(device)
            
            # Run inference
            with torch.no_grad():
                gaze_prediction = model(left_eye_tensor, right_eye_tensor, eye_landmarks)
                gaze_coords = gaze_prediction.cpu().numpy()[0]
                
                # Convert gaze coordinates to screen coordinates
                dot_x, dot_y = convert_gaze_to_screen_coords(gaze_coords[0], gaze_coords[1], screen_width, screen_height)
                
                # Emit the dot position through SocketIO
                socketio.emit('dot_position', {'x': dot_x, 'y': dot_y})
                print(f"Gaze prediction: {gaze_coords}, Screen position: ({dot_x}, {dot_y})")

            # Save the frame as jpg with timestamp
            time_str = time.strftime('%Y-%m-%d_%H_%M_%S')
            file_name = f"frame_{time_str}.jpg"
            cv2.imwrite(file_name, frame)
                
        end_time = time.time()
        # print(f"Loop took {end_time - start_time:.3f} seconds")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('dimensions')
def handle_dimensions(data):
    global screen_width, screen_height
    screen_width = data['width']
    screen_height = data['height']
    # Reset dot position to center when dimensions change
    #global dot_x, dot_y
    #dot_x = screen_width / 2
    #dot_y = screen_height / 2
    print(f"Screen dimensions: {screen_width}x{screen_height}")
    print(f"Dot coordinates: ({dot_x}, {dot_y})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tuning-file', type=str, required=False, help="tuning file path")
    parser.add_argument('-i', '--index', type=int, required=False, default=0, help='set camera index')
    parser.add_argument('-v', '--VideoCaptureAPI', type=int, required=False, default=0, choices=range(0, len(selector_list)), help=VideoCaptureAPIs)

    args = parser.parse_args()
    index = args.index
    selector = selector_list[args.VideoCaptureAPI]

    index = 0

    cap = Camera(index, selector)
    cap.set_width(1280)
    cap.set_height(720)
    cap.set_fps(0)
    cap.open()

    if not cap.isOpened():
        print("Can't open camera")
        exit()

    # Start camera capture in a separate thread
    camera_thread = threading.Thread(target=camera_capture)
    camera_thread.daemon = True
    camera_thread.start()
    
    # Run the SocketIO server
    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)