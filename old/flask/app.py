import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../arducam/')))

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import time
import threading
import json
import argparse
from camera import Camera
from isp import arducam108mp_isp
from utils import *

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
cap = None

# Initialize dot position and movement parameters
dot_x = 100  # Start from a visible position
dot_y = 100
direction_x = 1
direction_y = 1
dot_speed = 2  # Reduced speed for smoother movement
dot_radius = 10
screen_width = 980  # Default values
screen_height = 1684

def camera_capture():
    while True:
        success, frame = cap.read()
        if not success:
            continue

        start_time = time.time()
        
        frame = arducam108mp_isp(frame.reshape(9000, 12000), ccm=False, ccm_list=[])
        
        # Save the frame as jpg with timestamp
        time_str = time.strftime('%Y-%m-%d_%H_%M_%S')
        file_name = f"frame_{time_str}.jpg"
        cv2.imwrite(file_name, frame)

        end_time = time.time()
        print(f"Loop took {end_time - start_time:.3f} seconds")

        time.sleep(0.01)  # Small delay to control capture rate

def dot_movement():
    global dot_x, dot_y, direction_x, direction_y, screen_width, screen_height
    
    while True:
        # Update dot position
        dot_x += dot_speed * direction_x
        dot_y += dot_speed * direction_y
        
        # Bounce off the edges with padding
        if dot_x <= dot_radius:
            dot_x = dot_radius
            direction_x = 1
        elif dot_x >= screen_width - dot_radius:
            dot_x = screen_width - dot_radius
            direction_x = -1
            
        if dot_y <= dot_radius:
            dot_y = dot_radius
            direction_y = 1
        elif dot_y >= screen_height - dot_radius:
            dot_y = screen_height - dot_radius
            direction_y = -1
        
        # Print dot position
        #print(f"Dot position: x={dot_x}, y={dot_y}")
        
        # Emit the dot position through SocketIO
        socketio.emit('dot_position', {'x': int(dot_x), 'y': int(dot_y)})
        
        # Add a small delay to control the movement speed
        time.sleep(0.01)

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

    cap = Camera(index, selector)
    cap.set_width(6000)
    cap.set_height(9000)
    cap.set_fps(0)
    cap.open()

    if not cap.isOpened():
        print("Can't open camera")
        exit()

    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    # Start camera capture in a separate thread
    camera_thread = threading.Thread(target=camera_capture)
    camera_thread.daemon = True
    camera_thread.start()
    
    # Start dot movement in a separate thread
    dot_thread = threading.Thread(target=dot_movement)
    dot_thread.daemon = True
    dot_thread.start()
    
    # Run the SocketIO server
    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)