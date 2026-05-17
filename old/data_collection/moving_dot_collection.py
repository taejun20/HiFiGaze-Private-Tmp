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

def delete_recent_frames():
    global frame_number
    if movement_start_frame is not None:
        # Delete frames
        for frame_num in range(movement_start_frame, frame_number):
            frame_path = recording_dir / f"frame_{frame_num:05d}.png"
            if frame_path.exists():
                frame_path.unlink()

        # Reset frame number
        frame_number = movement_start_frame
        
        # Remove rows from data list
        while len(data_rows) > movement_start_frame:
            data_rows.pop()
        print(f"Last valid frame: {movement_start_frame}")

def webcam_capture_worker():
    global frame_number, last_ui_update_time
    while True:
        with capture_condition:
            # Wait for signal from UI thread
            capture_condition.wait()
            
            if capture_event.is_set():  # Check if we should stop
                break
            
            capture_start = time.time()
            #print(f"Capture delay from UI: {(capture_start - last_ui_update_time)*1000:.2f}ms")
            ret, frame = cap.read()
            if ret:  # Only save frames during movement
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
is_moving = False
show_arrow = False
arrow_start_time = 0
arrow_direction = None  # 'left' or 'right'
waiting_for_response = False
program_start_time = None
movement_start_frame = None

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

try:
    while True:
        # Use current background instead of loading every time
        canvas = current_background.copy()

        # Calculate delta time for smooth movement
        current_time = time.time()

        # Update dot position if moving
        if is_moving:
            dx = target_x - current_x
            dy = target_y - current_y
            distance = np.sqrt(dx*dx + dy*dy)

            if distance < move_speed:
                # Reached target
                current_x = target_x
                current_y = target_y
                is_moving = False
                
                # Show random arrow
                show_arrow = True
                arrow_direction = 'left' if random.random() < 0.5 else 'right'
                arrow_start_time = time.time()
            else:
                # Move toward target with delta time for smooth movement
                current_x += (dx / distance) * move_speed
                current_y += (dy / distance) * move_speed

        # Draw dot with border
        cv2.circle(canvas, (int(current_x), int(current_y)), 
                    dot_radius + dot_border_thickness, dot_border_color, -1)  # White border
        cv2.circle(canvas, (int(current_x), int(current_y)), 
                    dot_radius, dot_color, -1)  # Red dot

        # Draw arrow if needed
        if show_arrow:
            current_time = time.time()
            if current_time - arrow_start_time > arrow_duration:
                show_arrow = False
                waiting_for_response = True
            else:
                # Draw arrow
                arrow_center = (int(current_x), int(current_y))
                arrow_length = 80  # Doubled from 40
                arrow_color = (0, 0, 0)  # Black
                if arrow_direction == 'left':
                    arrow_start = (arrow_center[0] + arrow_length//2, arrow_center[1])
                    arrow_end = (arrow_center[0] - arrow_length//2, arrow_center[1])
                else:  # right
                    arrow_start = (arrow_center[0] - arrow_length//2, arrow_center[1])
                    arrow_end = (arrow_center[0] + arrow_length//2, arrow_center[1])
                
                # Draw white border arrow first
                cv2.arrowedLine(canvas, arrow_start, arrow_end, (255, 255, 255), 8, tipLength=0.5)
                # Draw black arrow on top
                cv2.arrowedLine(canvas, arrow_start, arrow_end, arrow_color, 4, tipLength=0.5)  # Thickness increased from 2 to 4

        # Display exp
        cv2.imshow('exp', canvas)

        # Handle user input
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            # Signal capture thread to stop and wake it up
            capture_event.set()
            with capture_condition:
                capture_condition.notify()
            break

        if waiting_for_response:
            response_correct = None
            if key == 2:  # Left arrow
                response_correct = (arrow_direction == 'left')
            elif key == 3:  # Right arrow
                response_correct = (arrow_direction == 'right')

            if response_correct is not None:
                waiting_for_response = False
                if response_correct:
                    current_pos_index += 1
                else:
                    current_x, current_y = dot_pos_list[current_pos_index]
                    delete_recent_frames()
        elif key == 0:  # Up arrow
            if current_pos_index <= len(dot_pos_list) - 1:
                is_moving = True
                if current_pos_index == len(dot_pos_list) - 1:
                    target_x, target_y = dot_pos_list[0]
                else:
                    target_x, target_y = dot_pos_list[current_pos_index + 1]
                movement_start_frame = frame_number
            else:
                # Cycle to next background and reset positions
                current_background_index = current_background_index + 1
                if current_background_index == len(BACKGROUND_IMAGES):
                    print("Experiment complete! All backgrounds visited.")
                    break
                else:
                    print(f"Moving to background {current_background_index + 1}")
                    # Load new background here
                    current_background = load_background()
                    current_pos_index = 0
                    generate_positions()  # Shuffle positions for new background
                    current_x, current_y = dot_pos_list[0]
                    target_x, target_y = current_x, current_y
                    print("Press UP ARROW to start with new background")
        
        # Signal webcam capture thread that UI has updated
        if is_moving:
            with capture_condition:
                last_ui_update_time = time.time()
                capture_condition.notify()

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
        # Write header
        csv_writer.writerow(['frame_number', 'dot_x_cm', 'dot_y_cm', 'dot_x_pixel', 'dot_y_pixel', 'dot_pos_index', 'background_index', 'elapsed_time'])
        # Write all rows
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
