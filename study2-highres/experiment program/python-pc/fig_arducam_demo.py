import cv2
import argparse

import numpy as np
from camera import Camera
from isp import arducam108mp_isp
from utils import *
import json
from rich import print
from camera_to_eye_distance_estimation import estimate_eye_distance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-W', '--width', type=int, required=False, default=1280, help='set camera image width')
    parser.add_argument('-H', '--height', type=int, required=False, default=720, help='set camera image height')
    parser.add_argument('-F', '--Focus', action='store_true', required=False, help='Add focus control on the display interface')
    parser.add_argument('-i', '--index', type=int, required=False, default=0, help='set camera index')

    args = parser.parse_args()
    width = args.width
    height = args.height
    index = args.index
    focus = args.Focus
    selector = selector_list[1] # 0: cv2.CAP_ANY, 1: cv2.CAP_MSMF, 2: cv2.CAP_DSHOW, 3: cv2.CAP_V4L2

    cap = Camera(index, selector)
    cap.set_width(width)
    cap.set_height(height)
    cap.set_fps(0)
    cap.open()
    
    if not cap.isOpened():
        print("Can't open camera")
        exit()

    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    


    if focus:
        cv2.createTrackbar('Focus', 'video', 187, 1023, cap.set_focus)

    if width == 6000 and height == 9000:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0) 
    cap.set(cv2.CAP_PROP_EXPOSURE, 0) 
    cap.set(cv2.CAP_PROP_GAIN,1000)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("cap.read() failed")
        
        if width == 6000 and height == 9000:
            frame = arducam108mp_isp(frame.reshape(9000, 12000))

        # display_fps(frame)
        cv2.imshow("video", frame)
        
        time_str = time.strftime('%Y-%m-%d') + time.strftime('_%H_%M_%S')
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):
            output_path = f"{width}x{height}_{time_str}.jpg"
            cv2.imwrite(f"{output_path}", frame)
            print(f"save success, file name: {output_path}")
        elif key == ord("a"):
            cap.set_width(6000)
            cap.set_height(9000)
            cap.reStart()
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

            if ret:
                frame = arducam108mp_isp(frame.reshape(9000, 12000))
                file_name = f"108MP_{time_str}.jpg"
                result = cv2.imwrite(file_name, frame)
                print(f"save success, file name: {file_name}, result: {result}")
            else:
                print("none frame, save failed")

            cap.set_width(width)
            cap.set_height(height)
            cap.reStart()

    cap.release()

    cv2.destroyAllWindows()
