import cv2
import numpy as np
import os
import time
import csv
import random
import threading
import queue
from pathlib import Path
from threading import Condition

# Get the absolute path of the script's directory
SCRIPT_DIR = Path(__file__).parent.absolute()

# Background image settings
BACKGROUND_IMAGES = [
    "background_laptop_1.png",
    "background_laptop_2.png",
    "background_laptop_3.png",
    "background_laptop_4.png"
]
current_background_index = 0

def load_background():
    return cv2.imread(str(SCRIPT_DIR / "images" / BACKGROUND_IMAGES[current_background_index]))
   
def generate_positions():
    dot_pos_list.clear()
    for i in range(horizontal_pos_count):
        for j in range(vertical_pos_count):
            pos_x = float(screen_width) / (horizontal_pos_count + 1) * (i + 1)
            pos_y = float(screen_height) / (vertical_pos_count + 1) * (j + 1)
            dot_pos_list.append((pos_x, pos_y))
    random.shuffle(dot_pos_list)


def save_frames_worker():
    while True:
        item = frame_queue.get()
        if item is None:  # Shutdown signal
            break
        
        frame_number, webcam_frame = item
        frame_path = recording_dir / f"frame_{frame_number:05d}.png"
        cv2.imwrite(str(frame_path), webcam_frame)
        frame_queue.task_done()

# Add before the webcam_capture_worker function
capture_condition = Condition()
last_ui_update_time = 0

# Add after other global variables
data_rows = []

def webcam_capture_worker():
    global frame_number, last_ui_update_time
    while True:
        with capture_condition:
            # Wait for signal from UI thread
            capture_condition.wait()
            
            if capture_event.is_set():  # Check if we should stop
                break
            
            capture_start = time.time()
            ret, frame = cap.read()
            if ret:
                try:
                    frame_queue.put_nowait((frame_number, frame.copy()))
                    
                    # Save position data to list
                    elapsed_time = time.time() - program_start_time
                    data_rows.append([
                        frame_number,
                        (current_x * (32.68 / float(3420)) - 16.34),
                        (current_y * (-20.45 / float(2138)) - 0.69),
                        current_x,
                        current_y,
                        current_pos_index,
                        current_background_index,
                        elapsed_time
                    ])
                    
                    # Increment frame counter
                    frame_number += 1
                except queue.Full:
                    print("Warning: Frame queue full, skipping frame save")


def delete_recent_frames():
    global frame_number
    if trial_start_frame is not None:
        # Delete frames
        for frame_num in range(trial_start_frame, frame_number):
            frame_path = recording_dir / f"frame_{frame_num:05d}.png"
            if frame_path.exists():
                frame_path.unlink()

        # Reset frame number
        frame_number = trial_start_frame
        
        # Remove rows from data list
        while len(data_rows) > trial_start_frame:
            data_rows.pop()
        print(f"Last valid frame: {trial_start_frame}")

# Add timing constants at the top with other constants
CAPTURE_START_DELAY = 1.0  # Start recording 1 second after dot appears
DOT_DISPLAY_TIME = 1.5    # Show dot for 1.5 seconds total
ARROW_DURATION = 0.05     # Show arrow for 0.05 seconds

# Screen and dot properties
screen_width = 3420  # New width to match desired resolution
screen_height = 2138  # Height remains the same
dot_radius = 12  # unit: pixels
dot_color = (0, 0, 255)  # BGR: Red
dot_border_color = (255, 255, 255)  # BGR: White
dot_border_thickness = 5  # unit: pixels
move_speed = 10.0  # increased for higher resolution
arrow_duration = 0.05  # 50ms


# Grid settings
horizontal_pos_count = 4
vertical_pos_count = 4
dot_pos_list = []
current_pos_index = 0

# State variables
waiting_for_response = False
program_start_time = None
dot_start_time = None
ready_for_next = False
waiting_for_start = True
trial_start_frame = None
waiting_for_background_start = False  # New state for background transition

# Load initial background
current_background = load_background()

generate_positions()    

# Initialize dot position
current_x, current_y = dot_pos_list[0]

# Setup recording
recording_dir = SCRIPT_DIR / "recording"
recording_dir.mkdir(parents=True, exist_ok=True)

# CSV setup
csv_path = recording_dir / "gt_labels.csv"

# Frame saving queue and thread
frame_queue = queue.Queue(maxsize=1000)
save_thread = threading.Thread(target=save_frames_worker)
save_thread.daemon = True
save_thread.start()

# Frame counter
frame_number = 0


# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam")
    exit(0)

# Set webcam properties
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)

# Create fullscreen window
cv2.namedWindow('exp', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('exp', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Event for signaling threads to stop
capture_event = threading.Event()

# Start webcam capture thread
capture_thread = threading.Thread(target=webcam_capture_worker)
capture_thread.daemon = True
capture_thread.start()

program_start_time = time.time()
print("Experiment started. Press ESC to exit.")
print("Press UP ARROW to start dot movement.")
print(f"Recording to: {recording_dir.absolute()}")

# Function to draw dot at current position
def draw_dot(canvas, x, y):
    cv2.circle(canvas, (int(x), int(y)), 
              dot_radius + dot_border_thickness, dot_border_color, -1)
    cv2.circle(canvas, (int(x), int(y)), 
              dot_radius, dot_color, -1)

# Function to draw arrow at position
def draw_arrow(canvas, x, y, direction):
    arrow_center = (int(x), int(y))
    arrow_length = 80
    arrow_color = (0, 0, 0)  # Black
    
    if direction == 'left':
        arrow_start = (arrow_center[0] + arrow_length//2, arrow_center[1])
        arrow_end = (arrow_center[0] - arrow_length//2, arrow_center[1])
    else:
        arrow_start = (arrow_center[0] - arrow_length//2, arrow_center[1])
        arrow_end = (arrow_center[0] + arrow_length//2, arrow_center[1])
    
    # Draw white border arrow first
    cv2.arrowedLine(canvas, arrow_start, arrow_end, (255, 255, 255), 8, tipLength=0.5)
    # Draw black arrow on top
    cv2.arrowedLine(canvas, arrow_start, arrow_end, arrow_color, 4, tipLength=0.5)

# Main loop
try:
    while True:
        # Get current position
        current_x, current_y = dot_pos_list[current_pos_index]
        canvas = current_background.copy()

        # Wait for initial up arrow press
        if waiting_for_start:
            cv2.imshow('exp', canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == 0:  # Up arrow
                waiting_for_start = False
                dot_start_time = None
            continue

        # Wait for background start
        if waiting_for_background_start:
            cv2.imshow('exp', canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == 0:  # Up arrow
                waiting_for_background_start = False
                dot_start_time = None
                trial_start_frame = frame_number
            continue
        
        # Handle response waiting state
        if waiting_for_response:
            cv2.imshow('exp', canvas)
            key = cv2.waitKey(1) & 0xFF  # Changed from waitKey(0) to waitKey(1)
            if key == 27:  # ESC
                break
            if key in [2, 3]:  # Left or Right arrow
                response_correct = (
                    (key == 2 and arrow_direction == 'left') or 
                    (key == 3 and arrow_direction == 'right')
                )
                waiting_for_response = False
                if response_correct:
                    ready_for_next = True
                else:
                    delete_recent_frames()
                    dot_start_time = None
            continue
        
        # Handle waiting for next trial
        if ready_for_next:
            cv2.imshow('exp', canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == 0:  # Up arrow
                if current_pos_index == len(dot_pos_list) - 1:
                    # Last position in current background
                    current_pos_index = 0
                    current_background_index += 1
                    
                    if current_background_index >= len(BACKGROUND_IMAGES):
                        print("Experiment complete! All backgrounds visited.")
                        break
                    else:
                        # Load new background
                        current_background = load_background()
                        print(f"Moving to background {current_background_index + 1}")
                        print("Press UP ARROW to start with new background")
                        waiting_for_background_start = True
                else:
                    current_pos_index += 1
                    dot_start_time = None
                    trial_start_frame = frame_number
                
                ready_for_next = False
            continue

        # Main trial display
        current_time = time.time()
        if dot_start_time is None:
            dot_start_time = current_time

        dot_elapsed = current_time - dot_start_time
        
        # Show dot until it's time for arrow
        if dot_elapsed < DOT_DISPLAY_TIME:
            draw_dot(canvas, current_x, current_y)
            cv2.imshow('exp', canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            # Handle recording timing
            if CAPTURE_START_DELAY <= dot_elapsed < DOT_DISPLAY_TIME:
                with capture_condition:
                    last_ui_update_time = time.time()
                    capture_condition.notify()
        
        # Show arrow when dot display time is up
        elif dot_elapsed >= DOT_DISPLAY_TIME:
            arrow_direction = 'left' if random.random() < 0.5 else 'right'
            draw_arrow(canvas, current_x, current_y, arrow_direction)
            
            cv2.imshow('exp', canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            time.sleep(ARROW_DURATION)
            
            # Clear screen after arrow
            cv2.imshow('exp', current_background.copy())
            cv2.waitKey(1)
            
            waiting_for_response = True

except KeyboardInterrupt:
    print("\nStopped by user")

finally:
    # Signal capture thread to stop and wake it up
    capture_event.set()
    with capture_condition:
        capture_condition.notify()
    capture_thread.join()
    
    # Write all data to CSV at once
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame_number', 'dot_x_cm', 'dot_y_cm', 'dot_x_pixel', 'dot_y_pixel', 'dot_pos_index', 'background_index', 'elapsed_time'])
        csv_writer.writerows(data_rows)
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Stop the saving thread
    frame_queue.put(None)
    save_thread.join()
    
    print(f"Recording complete. {frame_number} frames saved.")
    print(f"Frames saved to: {recording_dir.absolute()}")
    print(f"Data saved to: {csv_path.absolute()}")
