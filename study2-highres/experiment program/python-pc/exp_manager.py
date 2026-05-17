import socket
import threading
import time
import cv2
import numpy as np
from camera import Camera
from isp import arducam108mp_isp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import math
import base64
import os
import keyboard

# Create FaceLandmarker with iris support
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    running_mode=vision.RunningMode.IMAGE
)

faceLandmarkDetector = vision.FaceLandmarker.create_from_options(options)

LEFT_IRIS_INNER = 476
LEFT_IRIS_OUTER = 474
LEFT_IRIS_TOP = 475
LEFT_IRIS_BOTTOM = 477
LEFT_IRIS_CENTER = 473

RIGHT_IRIS_INNER = 469
RIGHT_IRIS_OUTER = 471
RIGHT_IRIS_TOP = 470
RIGHT_IRIS_BOTTOM = 472
RIGHT_IRIS_CENTER = 468

EYE_CROP_SCALE_FACTOR = 2.8     # used in crop_eye(), the eye patch crop range: iris width * EYE_CROP_SCALE_FACTOR

selector_list = [
    cv2.CAP_ANY,
    cv2.CAP_MSMF,
    cv2.CAP_DSHOW,
    cv2.CAP_V4L2
]

class ExperimentManager:
    def __init__(self, host='0.0.0.0', port=12345):
        # Key state tracking
        self.c_key_pressed = False
        self.r_key_pressed = False
        
        # Camera setup
        self.cap = Camera(0, selector_list[1])  # index=0, selector=0
        self.cap.set_width(6000)
        self.cap.set_height(9000)
        self.cap.set_fps(0)
        self.cap.open()        
        if not self.cap.isOpened():
            print("Failed to open camera")
            return False
            
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0) 
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 0) 
        self.cap.set(cv2.CAP_PROP_GAIN,1000)

        # exp
        self.setting_focus = False
        self.current_focus = 210  # Initialize focus value
        self.trial_started = False
        self.should_capture_next = False
        self.frame_to_save = None

        self.frame_lock = threading.Lock()

        self.thread_running = True
        self.capturing = True

        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        # TCP server setup
        print(f"Starting TCP server on {host}:{port} ...")
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        self.client_conn, self.client_addr = self.server_socket.accept()
        self.client_conn.settimeout(1.0)  # Timeout in seconds
        print(f"TCP server started and connected to {self.client_addr}")

        # Start TCP listener thread
        self.tcp_thread = threading.Thread(target=self._tcp_listener)
        self.tcp_thread.daemon = True
        self.tcp_thread.start()
        
        print(f"ExperimentManager initialized and listening on {host}:{port}")
    
    def _capture_loop(self):
        while self.thread_running:
            if self.capturing:
                start = time.time()
                #print("### capture start ###")
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    time.sleep(0.1)  # Prevent tight loop on failure
                    continue
                else:
                    print(f"cap.read() end: {time.time() - start}")
                if not ret:
                    print("Failed to capture frame")
                    time.sleep(0.1)  # Prevent tight loop on failure
                    continue
                
                if self.setting_focus:
                    # Process and send the frame
                    processed_frame = arducam108mp_isp(frame.reshape(9000, 12000))
                    self.send_eyes(processed_frame)
                else:
                    if self.should_capture_next:
                        self.should_capture_next = False
                        self.trial_started = True
                        continue

                    if self.trial_started:
                        processed_frame = arducam108mp_isp(frame.reshape(9000, 12000))
                        self.frame_to_save = processed_frame
                        ret = self.send_eyes(self.frame_to_save)     
                        if ret:
                            self.trial_started = False
                            self.client_conn.sendall("text:trialfin\n".encode('utf-8'))
                            print(f"processed_frame end: {time.time() - start}")
          
            
            time.sleep(0.001)   # 1 ms    
    
    def _tcp_listener(self):
        buffer = ""
        while self.thread_running:
            try:
                data = self.client_conn.recv(1024)
                if not data:
                    continue
                
                # Add new data to buffer
                buffer += data.decode('utf-8')

                # Only process if we have complete messages (containing \n)
                if '\n' in buffer:
                    messages = buffer.split('\n')
                    buffer = messages[-1]  # Keep the last part that might be incomplete
                    
                    # Process all complete messages
                    for command in messages[:-1]:
                        command = command.strip()
                        if not command:  # Skip empty messages
                            continue
                        
                        print(f"Received command: {command}")                
                        if command == "setfocus":
                            self.setting_focus = True
                        elif command == "setfocusfin":
                            self.setting_focus = False
                        elif command.startswith("dist:"):
                            if self.setting_focus:
                                # Parse left and right distances from format "l293r298"
                                dist_str = command.split(":")[1]
                                left_dist = int(dist_str.split('l')[1].split('r')[0])
                                right_dist = int(dist_str.split('r')[1])
                                
                                # Average the distances
                                avg_dist_mm = (left_dist + right_dist) / 2.0
                                avg_dist_cm = avg_dist_mm / 10.0
                                
                                # Calculate focus value using formula
                                x = avg_dist_cm
                                focus_val = int(-0.0102*(x**3) + 1.2661*(x**2) - 55.0424*x + 1228.0000)
                                
                                # Set the focus
                                self.current_focus = focus_val
                                self.cap.set_focus(focus_val)
                                print(f"### autofocus: left: {left_dist/10.0:.1f} cm, right: {right_dist/10.0:.1f} cm, avg: {avg_dist_cm:.1f} cm, Focus set to: {focus_val}")
                                self.client_conn.sendall(f"text:focus{focus_val}\n".encode('utf-8'))
                        elif command == "start":  # trial started
                            self.should_capture_next = True
                        elif command.startswith("trialreviewfin:"):
                            # Parse the command format
                            parts = command.split(":")[1].strip()
                            
                            # Check if it's a calibration trial (p1scalsitt15 or p1scalstandt15) or regular session (p1s1t15)
                            if "scalsit" in parts:
                                # Handle calibration sitting format
                                p_num = parts.split('p')[1].split('scalsit')[0]  # Extract participant number
                                t_num = parts.split('scalsitt')[1]  # Extract trial number after 'scalsitt'
                                directory = f"rawdata/p{p_num}/calsit"
                            elif "scalstand" in parts:
                                # Handle calibration standing format
                                p_num = parts.split('p')[1].split('scalstand')[0]  # Extract participant number
                                t_num = parts.split('scalstandt')[1]  # Extract trial number after 'scalstandt'
                                directory = f"rawdata/p{p_num}/calstand"
                            else:
                                # Handle regular session format
                                p_num = parts.split('p')[1].split('s')[0]  # Extract participant number
                                s_num = parts.split('s')[1].split('t')[0]  # Extract session number
                                t_num = parts.split('t')[1]  # Extract trial number
                                directory = f"rawdata/p{p_num}/s{s_num}"
                            
                            # Create directory and construct filename
                            os.makedirs(directory, exist_ok=True)
                            filename = f"{directory}/t{t_num}.jpg"
                            cv2.imwrite(filename, self.frame_to_save)
                            print(f"Frame saved to {filename}")
                        
            except socket.timeout:
                continue  # 🔹 Normal behavior: no data received, loop again
            except Exception as e:
                print(f"Error in TCP listener: {e}")
    
    def cleanup(self):
        self.thread_running = False
        self.capture_thread.join()
        self.tcp_thread.join()
        self.cap.release()
        self.server_socket.close()

    def encode_image_to_base64(self, img):
        _, buffer = cv2.imencode('.jpg', img)
        return base64.b64encode(buffer).decode('utf-8')
    
    def send_eyes(self, frame):
        # Use the rotated image for all subsequent processing
        image_height, image_width = frame.shape[:2]
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Process with MediaPipe
        results = faceLandmarkDetector.detect(mp_image)        
        if not results.face_landmarks:
            print(f"! send_eyes() failed: No face landmarks detected")
            return False
        else:
            landmarks = results.face_landmarks[0]

            left_eyecrop_center = (int(landmarks[LEFT_IRIS_CENTER].x * image_width), int(landmarks[LEFT_IRIS_CENTER].y * image_height))
            left_eyecrop_width = EYE_CROP_SCALE_FACTOR * int(math.sqrt(((landmarks[LEFT_IRIS_OUTER].x - landmarks[LEFT_IRIS_INNER].x) * image_width) ** 2 + ((landmarks[LEFT_IRIS_OUTER].y - landmarks[LEFT_IRIS_INNER].y) * image_height) ** 2))
            left_eyecrop_width_half = int(left_eyecrop_width / 2.0)
            
            left_eyecrop_top = left_eyecrop_center[1] - int(left_eyecrop_width_half / 2.0)
            left_eyecrop_bottom = left_eyecrop_center[1] + int(left_eyecrop_width_half / 2.0)
            left_eyecrop_left = left_eyecrop_center[0] - int(left_eyecrop_width_half)
            left_eyecrop_right = left_eyecrop_center[0] + int(left_eyecrop_width_half)

            # Check if any boundary is out of the image frame
            if left_eyecrop_top < 0 or left_eyecrop_bottom > image_height or left_eyecrop_left < 0 or left_eyecrop_right > image_width:
                print(f"! send_eyes() failed: Left eye crop boundary is out of image frame")
                return False
            else:
                right_eyecrop_center = (int(landmarks[RIGHT_IRIS_CENTER].x * image_width), int(landmarks[RIGHT_IRIS_CENTER].y * image_height))
                right_eyecrop_width = EYE_CROP_SCALE_FACTOR * int(math.sqrt(((landmarks[RIGHT_IRIS_OUTER].x - landmarks[RIGHT_IRIS_INNER].x) * image_width) ** 2 + ((landmarks[RIGHT_IRIS_OUTER].y - landmarks[RIGHT_IRIS_INNER].y) * image_height) ** 2))
                right_eyecrop_width_half = int(right_eyecrop_width / 2.0)
                
                right_eyecrop_top = right_eyecrop_center[1] - int(right_eyecrop_width_half / 2.0)
                right_eyecrop_bottom = right_eyecrop_center[1] + int(right_eyecrop_width_half / 2.0)
                right_eyecrop_left = right_eyecrop_center[0] - int(right_eyecrop_width_half)
                right_eyecrop_right = right_eyecrop_center[0] + int(right_eyecrop_width_half)

                if right_eyecrop_top < 0 or right_eyecrop_bottom > image_height or right_eyecrop_left < 0 or right_eyecrop_right > image_width:
                    print(f"! send_eyes() failed: Right eye crop boundary is out of image frame")
                    return False
                else:
                    left_eye_cropped = frame[left_eyecrop_top:left_eyecrop_bottom, left_eyecrop_left:left_eyecrop_right]
                    right_eye_cropped = frame[right_eyecrop_top:right_eyecrop_bottom, right_eyecrop_left:right_eyecrop_right]

                    #print(f"type(left_eye_cropped): {type(left_eye_cropped)}")
                    #print(f"type(right_eye_cropped): {type(right_eye_cropped)}")

                    try:
                        left_encoded = self.encode_image_to_base64(left_eye_cropped)
                        right_encoded = self.encode_image_to_base64(right_eye_cropped)
                        response_message = f"image:{left_encoded}|{right_encoded}\n"    # \n is for the end of the message

                        #print(f"send_eyes(): size of response message: {len(response_message)} bytes")
                        self.client_conn.sendall(response_message.encode('utf-8'))
                        return True
                    except Exception as e:
                        print(f"❌ Error sending eye images: {e}") 
                        return False           


if __name__ == "__main__":
    # Create and start the experiment manager
    manager = ExperimentManager()

    try:
        while True:
            # Create a small black window with just the focus value
            info_window = np.zeros((600, 600, 3), dtype=np.uint8)  # Small black window
            cv2.putText(info_window, f"Focus: {manager.current_focus}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(info_window, "UP/DOWN to adjust", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Show the small window
            cv2.imshow("Focus Adjustment", info_window)
            
            # Check for key press using keyboard library
            cv2.waitKey(1)  # Still need this to update the window
            if keyboard.is_pressed('up'):
                print(f"up pressed")
                if manager.setting_focus:
                    manager.current_focus = min(1023, manager.current_focus + 5)  # Max focus value is 1023
                    manager.cap.set_focus(manager.current_focus)
                    manager.client_conn.sendall(f"text:focus{manager.current_focus}\n".encode('utf-8'))
                    print(f"Focus increased to: {manager.current_focus}")
                    time.sleep(0.1)  # Add small delay to prevent too rapid adjustments
            elif keyboard.is_pressed('down'):
                if manager.setting_focus:
                    manager.current_focus = max(0, manager.current_focus - 5)  # Min focus value is 0
                    manager.cap.set_focus(manager.current_focus)
                    manager.client_conn.sendall(f"text:focus{manager.current_focus}\n".encode('utf-8'))
                    print(f"Focus decreased to: {manager.current_focus}")
                    time.sleep(0.1)  # Add small delay to prevent too rapid adjustments
            
            
            if keyboard.is_pressed('c'):
                if not manager.c_key_pressed:  # Only trigger if key wasn't pressed before
                    print(f"c pressed")
                    if not manager.trial_started:
                        manager.client_conn.sendall(f"text:complete\n".encode('utf-8'))
                        print(f"trial completed")
                    manager.c_key_pressed = True
            elif not keyboard.is_pressed('c'):
                manager.c_key_pressed = False  # Reset the state when key is released
            
            if keyboard.is_pressed('r'):
                if not manager.r_key_pressed:  # Only trigger if key wasn't pressed before
                    if not manager.trial_started:
                        manager.client_conn.sendall(f"text:redo\n".encode('utf-8'))
                        print(f"trial redo")
                    manager.r_key_pressed = True
            elif not keyboard.is_pressed('r'):
                manager.r_key_pressed = False  # Reset the state when key is released
            
            time.sleep(0.01)  # Small delay when not in focus mode
                
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Shutting down...")
        manager.cleanup()
    except Exception as e:
        print(f"Unhandled exception: {e}")
        manager.cleanup()