from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import time
import threading
import argparse
import random
from camera import Camera
from utils import *
from isp import arducam108mp_isp
from camera_to_eye_distance_estimation import estimate_eye_distance
import os
import glob
import csv

# variables manually set by experimenter
subject = "p1"
current_session_num = 1

current_trial_num = 1   # 1-based index
current_frame_num = 0    # 0-based index

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
cap = None

# Screen dimensions
screen_width = 980  # (px)
screen_height = 1990  # (px)

# Grid settings
horizontal_pos_count = 2
vertical_pos_count = 2
dot_pos_list = []

# State variables
waiting_for_start = True
dot_start_time = None
waiting_for_response = False
arrow_direction = None
response_correct = None
program_start_time = None
trial_start_frame = None
camera_capture_called = False   # Boolean used to make it called once in the loop
do_camera_capture = False       # Boolean used to execute the camera capture in the thread
capture_started_time = None
do_arrow_display = False

# Timing constants
CAPTURE_START_TIME = 0.5  # Start recording 0.8 seconds after dot appears
ARROW_DURATION = 0.05     # Show arrow for 0.1 seconds

# Load all screenshot filenames
screenshot_files = glob.glob('static/screenshot-10k-1170x2372/*.jpg')
screenshot_files = [os.path.basename(f) for f in screenshot_files]
random.shuffle(screenshot_files)

current_background_index = 0
current_background_filename = screenshot_files[current_background_index]

# Initialize experiment data list
experiment_data = []

def get_next_background():
    global current_background_index, current_background_filename
    current_background_index += 1
    current_background_filename = screenshot_files[current_background_index]

def generate_positions():
    dot_pos_list.clear()
    for i in range(horizontal_pos_count):
        for j in range(vertical_pos_count):
            pos_x = float(screen_width) / (horizontal_pos_count + 1) * (i + 1)
            pos_y = float(screen_height) / (vertical_pos_count + 1) * (j + 1)
            dot_pos_list.append((round(pos_x), round(pos_y)))
    random.shuffle(dot_pos_list)

def update_dot_position(x, y):
    global dot_x, dot_y
    dot_x = x
    dot_y = y
    socketio.emit('dot_position', {'x': dot_x, 'y': dot_y})

def emit_show_response_buttons():
    socketio.emit('experiment_state', {'state': 'show_response_buttons'})

def write_experiment_data():
    with open('exp_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subject', 'session', 'trial', 'px_x', 'px_y', 'cm_x', 'cm_y', 'frame_num', 'background', 'time'])
        writer.writerows(experiment_data)

def camera_capture():
    global do_camera_capture, capture_started_time, do_arrow_display, waiting_for_response, current_frame_num
    while True:
        success, frame = cap.read() # to clear the early frame
        if not success:
            continue
             
        if do_camera_capture:            
            until_cap_read_time = time.time()
            print(f"until cap.read() took {until_cap_read_time - capture_started_time:.3f} seconds")

            # Log experiment data before capture
            px_x, px_y = dot_pos_list[current_trial_num-1]
            cm_x = (7.1/980) * px_x - 3.55
            cm_y = (14.4/1990) * px_y - 15.9
            experiment_data.append([subject, current_session_num, current_trial_num, px_x, px_y, cm_x, cm_y, current_frame_num, current_background_filename, time.time() - program_start_time])

            success, frame = cap.read() # to clear the early frame
            if not success:
                continue
            
            cap_read_time = time.time()
            print(f"cap.read() took {cap_read_time - until_cap_read_time:.3f} seconds")
            do_arrow_display = True

            frame = arducam108mp_isp(frame.reshape(9000, 12000), False, [])

            isp_time = time.time()
            print(f"ISP took {isp_time - cap_read_time:.3f} seconds")

            file_name = f"frames/{current_frame_num:05d}.jpg"
            cv2.imwrite(file_name, frame)
            current_frame_num += 1

            save_time = time.time()
            print(f"Image saving took {save_time - isp_time:.3f} seconds")
         
            do_camera_capture = False   
            # Emit show_response_buttons state after camera capture is complete
            emit_show_response_buttons()

            end_time = time.time()
            print(f"Entire took {end_time - capture_started_time:.3f} seconds\n")       
        else:
            frame = arducam108mp_isp(frame.reshape(9000, 12000), False, [])
            distance = estimate_eye_distance(frame)
            if distance is not None and distance < 45 and distance >= 15:        
                # Linear interpolation between known distance-focus pairs
                # Map distance to focus value using linear interpolation
                # Known points: (30cm -> 460), (40cm -> 430)
                # Linear interpolation between these two points
                val = 460 - (distance - 30) * (460 - 430) / (40 - 30)
                val = int(val)  # Ensure integer value
                cap.set_focus(val)
                # print(f"Autofocus: Eye-Cam Dist: {distance:.1f} cm, focus: {val}")
 

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    global current_trial_num, waiting_for_start, dot_start_time, waiting_for_response, arrow_direction, response_correct, program_start_time, trial_start_frame, do_camera_capture, camera_capture_called
    print('Client connected')

    # Grid settings
    current_trial_num = 1
    generate_positions()

    # State variables
    waiting_for_start = True
    dot_start_time = None
    waiting_for_response = False
    arrow_direction = None
    response_correct = None
    program_start_time = None
    trial_start_frame = None
    do_camera_capture = False
    camera_capture_called = False
    experiment_data.clear()
    socketio.emit('update_background', {'filename': current_background_filename})       

    print(f"background file: {current_background_filename}\n")
    socketio.emit('experiment_state', {'state': 'start'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_experiment')
def handle_start_experiment():
    global waiting_for_start, dot_start_time, program_start_time, do_camera_capture    
    program_start_time = time.time()
    current_x, current_y = dot_pos_list[current_trial_num-1]
    update_dot_position(current_x, current_y)
    waiting_for_start = False
    dot_start_time = time.time()
    socketio.emit('experiment_state', {'state': 'show_dot'})

@socketio.on('response')
def handle_response(data):
    global waiting_for_response, response_correct, current_trial_num, dot_start_time, do_camera_capture, camera_capture_called
    waiting_for_response = False
    response_correct = (data['direction'] == arrow_direction)
    if response_correct:
        if current_trial_num < len(dot_pos_list):
            current_trial_num += 1
            current_x, current_y = dot_pos_list[current_trial_num-1]
            update_dot_position(current_x, current_y)
            do_camera_capture = False
            camera_capture_called = False
            dot_start_time = time.time()
            socketio.emit('experiment_state', {'state': 'show_dot'})
        else:
            # Write experiment data to CSV when experiment is complete
            write_experiment_data()
            socketio.emit('experiment_state', {'state': 'complete'})
    else:
        socketio.emit('experiment_state', {'state': 'wrong_response'})

@socketio.on('redo_trial')
def handle_redo():
    global dot_start_time, do_camera_capture, waiting_for_response, camera_capture_called, current_frame_num
    current_x, current_y = dot_pos_list[current_trial_num-1]
    update_dot_position(current_x, current_y)
    waiting_for_response = False
    do_camera_capture = False
    camera_capture_called = False
    
    # Decrement frame number
    current_frame_num -= 1
    
    # Remove the experiment data row for current trial
    experiment_data[:] = [row for row in experiment_data if row[2] != current_trial_num]
    
    # Delete the frame file
    frame_filename = f"{current_frame_num:05d}.jpg"
    try:
        os.remove(frame_filename)
        print(f"Deleted frame file: {frame_filename}")
    except FileNotFoundError:
        print(f"Frame file not found: {frame_filename}")
    
    dot_start_time = time.time()
    socketio.emit('experiment_state', {'state': 'show_dot'})

def check_trial_state():
    global arrow_direction, waiting_for_response, do_camera_capture, capture_started_time, do_arrow_display, camera_capture_called
    while True:
        if not waiting_for_start:    # after 'Start' button is touched at first
            current_time = time.time()
            dot_elapsed = current_time - dot_start_time
            if not waiting_for_response:    # not during the response of the Left / Right button
                if dot_elapsed >= CAPTURE_START_TIME and not camera_capture_called:
                    camera_capture_called = True
                    print(f"### Camera capture triggered ###")
                    # Start camera capture in a separate thread
                    # capture_thread = threading.Thread(target=camera_capture)
                    # capture_thread.start()
                    capture_started_time = time.time()
                    do_camera_capture = True
                
                if do_arrow_display:
                    arrow_direction = 'left' if random.random() < 0.5 else 'right'
                    waiting_for_response = True
                    do_arrow_display = False
                       # Get next screenshot filename and emit it
                    get_next_background()
                    print(f"background file: {current_background_filename}")
                    socketio.emit('update_background', {'filename': current_background_filename})
                    socketio.emit('experiment_state', {'state': 'show_arrow', 'direction': arrow_direction})




@socketio.on('dimensions')
def handle_dimensions(data):
    print(f"Screen dimensions: {data['width']}x{data['height']} (px)")

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
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    if not cap.isOpened():
        print("Can't open camera")
        exit()

    # Start continuous autofocus in a separate thread
    camera_capture_thread = threading.Thread(target=camera_capture)
    camera_capture_thread.daemon = True
    camera_capture_thread.start()
    
    # Start trial state checker in a separate thread
    state_thread = threading.Thread(target=check_trial_state)
    state_thread.daemon = True
    state_thread.start()
    
    # Run the SocketIO server
    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)