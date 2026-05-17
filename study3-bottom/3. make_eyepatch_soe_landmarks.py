import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import cv2
import numpy as np
import json
import os
import math
from tqdm import tqdm
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import matplotlib.pyplot as plt
import time
from datetime import datetime
import argparse
from scipy import optimize
import helpermethods

DEBUG_SAVE_MODE = True

SUBJECTS = [1,2,3,4,5,6,7,8,9,10]
SESSIONS = [1,2,3,4,5,6]

# MediaPipe indices
LEFT_EYE_OUTER = 263
LEFT_EYELID_UPPEROUTER = 386
LEFT_EYELID_UPPERINNER = 385
LEFT_EYE_INNER = 362
LEFT_EYELID_LOWERINNER = 374
LEFT_EYELID_LOWEROUTER = 373

RIGHT_EYE_OUTER = 33
RIGHT_EYELID_UPPEROUTER = 159
RIGHT_EYELID_UPPERINNER = 158
RIGHT_EYE_INNER = 133
RIGHT_EYELID_LOWERINNER = 145
RIGHT_EYELID_LOWEROUTER = 144

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
IRIS_CROP_SCALE_FACTOR = 0.6     # used in crop_iris(), the iris patch crop range (range of template matching image)
EYE_CLOSED_THRESHOLD = 0.20

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

 
def get_eye_aspect_ratio(landmarks, image_width, image_height, is_left):
    """
    Calculate eye aspect ratio using 6 landmarks
    Eye Aspect Ratio (EAR) = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    # Extract points
    if is_left:
        p1 = landmarks[LEFT_EYE_OUTER]
        p2 = landmarks[LEFT_EYELID_UPPEROUTER]
        p3 = landmarks[LEFT_EYELID_UPPERINNER]
        p4 = landmarks[LEFT_EYE_INNER]
        p5 = landmarks[LEFT_EYELID_LOWERINNER]
        p6 = landmarks[LEFT_EYELID_LOWEROUTER]
    else:
        p1 = landmarks[RIGHT_EYE_OUTER]
        p2 = landmarks[RIGHT_EYELID_UPPEROUTER]
        p3 = landmarks[RIGHT_EYELID_UPPERINNER]
        p4 = landmarks[RIGHT_EYE_INNER]
        p5 = landmarks[RIGHT_EYELID_LOWERINNER]
        p6 = landmarks[RIGHT_EYELID_LOWEROUTER]
    
    # Calculate vertical distances
    v1 = abs(p2.y - p6.y) * image_height  # outer vertical
    v2 = abs(p3.y - p5.y) * image_height  # inner vertical
    
    # Calculate horizontal distance
    h = abs(p1.x - p4.x) * image_width
    
    # Calculate EAR
    ear = (v1 + v2) / (2.0 * h)
    return ear

def segment_iris(frame, landmarks, subject, session, frame_name, irisbox_center_frame, irisbox_width, irisbox_height, debug_iris_segment_dir, is_left):
    if is_left:
        iris_inner_mp = landmarks[LEFT_IRIS_INNER] 
        iris_outer_mp = landmarks[LEFT_IRIS_OUTER]
    else:
        iris_inner_mp = landmarks[RIGHT_IRIS_INNER]
        iris_outer_mp = landmarks[RIGHT_IRIS_OUTER]
    frame_height, frame_width = frame.shape[:2]
        
    # Crop eye patch for iris segmentation
    eyepatch_width = round(irisbox_width * 2.0)
    eyepatch_height = round(irisbox_width * 1.5)     
    eyepatch_left = irisbox_center_frame[0] - eyepatch_width // 2
    eyepatch_right = irisbox_center_frame[0] + eyepatch_width // 2
    eyepatch_top = irisbox_center_frame[1] - eyepatch_height // 2
    eyepatch_bottom = irisbox_center_frame[1] + eyepatch_height // 2
    if eyepatch_left < 0 or eyepatch_right > frame_width or eyepatch_top < 0 or eyepatch_bottom > frame_height:
        print(f" Warning: Eye patch for iris segmentation gets out of the frame, skipping: subject: {subject}, session: {session}, frame: {frame_name}, is_left: {is_left}")
        return None, None, None
    eyepatch = frame[eyepatch_top:eyepatch_bottom, eyepatch_left:eyepatch_right]
    
    # GrabCut Preparation
    initial_center_eyepatch = (irisbox_center_frame[0] - eyepatch_left, irisbox_center_frame[1] - eyepatch_top) # iris center in eyepatch coordinates    
    r1 = round((irisbox_height / 2.0) * 0.8)  # Definite foreground
    r2 = round(irisbox_width / 2.0)  # Probable foreground  
    r3 = round(irisbox_width / 2.0 * 1.3)  # Probable background    
    mask = np.zeros(eyepatch.shape[:2], dtype=np.uint8)    # Create initial mask for GrabCut, initialize with definite background: 0

    # Initialize mask with concentric circles & initialize Gaussian Mixture Models
    cv2.circle(mask, initial_center_eyepatch, r3, cv2.GC_PR_BGD, -1)  # Probable background: 1 (GC_PR_BGD)
    cv2.circle(mask, initial_center_eyepatch, r2, cv2.GC_PR_FGD, -1)  # Probable foreground: 2 (GC_PR_FGD)
    cv2.circle(mask, initial_center_eyepatch, r1, cv2.GC_FGD, -1)     # Definite foreground: 3 (GC_FGD)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # Run GrabCut algorithm on the eye patch
    try:
        cv2.grabCut(eyepatch, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    except Exception as e:
        print(f" Warning: GrabCut failed, Exception: {e}, skipping: subject: {subject}, session: {session}, frame: {frame_name}, is_left: {is_left}")
        return None, None, None
    
    # Create final mask with definite background and probable background
    final_mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')
    
    # Find contours and select the largest one (assumed to be the iris)
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print(" Warning: No contours found in iris segmentation, skipping: subject: {subject}, session: {session}, frame: {frame_name}, is_left: {is_left}")
        return None, None, None
    iris_contour_points = max(contours, key=cv2.contourArea)

    # Find the points at the actual iris contour in the side by filtering out the ones in the upper and lower eyelids (to fit the iris circle)
    iris_inner_to_outer_vector = np.array([(iris_outer_mp.x - iris_inner_mp.x) * frame_width, (iris_outer_mp.y - iris_inner_mp.y) * frame_height])
    if np.linalg.norm(iris_inner_to_outer_vector) > 0:
        iris_inner_to_outer_vector = iris_inner_to_outer_vector / np.linalg.norm(iris_inner_to_outer_vector)
    else:
        iris_inner_to_outer_vector = np.array([1.0, 0.0])  # Default to horizontal if landmarks are identical
    
    ## Filter points based on (1) distance from the irisbox center and (2) local direction
    filtered_points = []
    num_points = len(iris_contour_points)

    # (1) distance from the irisbox center: define exclusion zones
    inner_filtering_range = round((irisbox_width / 2.0) * 0.7)   # Inner exclusion zone (too close to the irisbox center)
    inner_min_x = initial_center_eyepatch[0] - inner_filtering_range
    inner_max_x = initial_center_eyepatch[0] + inner_filtering_range
    outer_filtering_range = round((irisbox_width / 2.0) * 1.2)  # Outer exclusion zone (too far from center)
    outer_min_x = initial_center_eyepatch[0] - outer_filtering_range
    outer_max_x = initial_center_eyepatch[0] + outer_filtering_range
    k = 2  # (2) local direction: number of previous and next points to consider
    for i in range(num_points):
        curr_point = iris_contour_points[i, 0]        
        if inner_min_x <= curr_point[0] <= inner_max_x:  # Skip points within inner exclusion zone (too close to the irisbox center)
            continue
        if curr_point[0] < outer_min_x or curr_point[0] > outer_max_x:  # Skip points outside outer exclusion zone (too far from the irisbox center)
            continue

        # Collect k previous & next points
        prev_points = []
        for j in range(1, k + 1):
            idx = (i - j) % num_points
            point = iris_contour_points[idx, 0]
            prev_points.append(point)
        next_points = []
        for j in range(1, k + 1):
            idx = (i + j) % num_points
            point = iris_contour_points[idx, 0]
            next_points.append(point)
        
        # Calculate local direction vector using k previous and k next points
        prev_vector = np.mean([curr_point - p for p in prev_points], axis=0)
        next_vector = np.mean([n - curr_point for n in next_points], axis=0)
        local_vector = (prev_vector + next_vector) / 2        
        if np.linalg.norm(local_vector) == 0:   # Skip if local vector is zero
            continue
        local_vector = local_vector / np.linalg.norm(local_vector)  # Normalize local vector        
        dot_product = np.dot(iris_inner_to_outer_vector, local_vector)  # Calculate angle between iris vector and local vector
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # Select points where angle is between pi/4 and 3pi/4 (45° to 135°)
        # This selects points where local contour direction is roughly perpendicular to iris direction
        if np.pi * 1/4 <= angle <= np.pi*3/4:
            filtered_points.append(curr_point.tolist())

    # Do circle fitting on the filtered contour points
    ransac_inliers = None
    grabcut_iris_center_eyepatch = None
    grabcut_iris_radius = None
    if len(filtered_points) < 10:
        # Not enough points for RANSAC, do fallback
        grabcut_iris_center_eyepatch = initial_center_eyepatch
        grabcut_iris_radius = round(irisbox_width / 2.0)
        print(f"Warning: Not enough points for RANSAC, using irisbox center and width, subject: {subject}, session: {session}, frame: {frame_name}, is_left: {is_left}")
    else:
        # Fit circle to selected points using RANSAC
        filtered_points_array = np.array(filtered_points, dtype=np.float32)
        grabcut_iris_center_eyepatch, grabcut_iris_radius, ransac_inliers = fit_circle_ransac(filtered_points_array)
        if grabcut_iris_center_eyepatch is None:
            # RANSAC failed, use minimum enclosing circle on selected points
            grabcut_iris_center_eyepatch = initial_center_eyepatch
            grabcut_iris_radius = round(irisbox_width / 2.0)    
            ransac_inliers = None
            print(f"Warning: RANSAC failed, using irisbox center and width, subject: {subject}, session: {session}, frame: {frame_name}, is_left: {is_left}")
        else:
            # Check if we have points on both sides of the initial center
            points_left = np.any(ransac_inliers[:, 0] < initial_center_eyepatch[0])
            points_right = np.any(ransac_inliers[:, 0] > initial_center_eyepatch[0])
            if not (points_left and points_right):
                # All points are on one side, do fallback
                grabcut_iris_center_eyepatch = initial_center_eyepatch
                grabcut_iris_radius = round(irisbox_width / 2.0)
                print(f"Warning: All filtered points are on the {'left' if points_left else 'right'} side of initial center, using irisbox center and width, subject: {subject}, session: {session}, frame: {frame_name}, is_left: {is_left}")
            
    # Also check if the final grabcut iris center is within the irisbox's x range    
    if grabcut_iris_center_eyepatch[0] < initial_center_eyepatch[0] - irisbox_width // 2 or grabcut_iris_center_eyepatch[0] > initial_center_eyepatch[0] + irisbox_width // 2:
        print(f"Warning: GrabCut fitted iris center x ({grabcut_iris_center_eyepatch[0]}) is outside the irisbox's x range ({initial_center_eyepatch[0] - irisbox_width // 2}, {initial_center_eyepatch[0] + irisbox_width // 2}), using irisbox center and width, subject: {subject}, session: {session}, frame: {frame_name}, is_left: {is_left}")
        grabcut_iris_center_eyepatch = initial_center_eyepatch
        grabcut_iris_radius = round(irisbox_width / 2.0)    

    # Check if the final grabcut iris center's y is within the irisbox's y range * 1.1
    allowing_range = round(irisbox_height * 1.1)
    if grabcut_iris_center_eyepatch[1] < initial_center_eyepatch[1] - allowing_range // 2 or grabcut_iris_center_eyepatch[1] > initial_center_eyepatch[1] + allowing_range // 2:
        print(f"Warning: GrabCut fitted iris center y ({grabcut_iris_center_eyepatch[1]}) is outside the irisbox's y range with margin ({initial_center_eyepatch[1] - allowing_range // 2}, {initial_center_eyepatch[1] + allowing_range // 2}), using irisbox center and width, subject: {subject}, session: {session}, frame: {frame_name}, is_left: {is_left}")
        grabcut_iris_center_eyepatch = initial_center_eyepatch
        grabcut_iris_radius = round(irisbox_width / 2.0)

    # Convert iris center back to full frame coordinates
    grabcut_iris_center_frame = (grabcut_iris_center_eyepatch[0] + eyepatch_left, grabcut_iris_center_eyepatch[1] + eyepatch_top)
   
    # Preparation for the later SoE filtering heuristics B:
    # calculate uppereyelid_to_center_y and pass it to find_soe()
    # first, calculate uppereyelid_grabcut_y and then subtracts with the center to finally derive uppereyelid_to_center_y
    uppereyelid_to_center_y_diff = None
    iris_contour_points_array = iris_contour_points.squeeze()  # Convert contour to array for easier processing    
    uppereyelid_grabcut_y = float('inf')
    for point in iris_contour_points_array:    # Find point directly above the iris center
        x, y = point
        if abs(x - initial_center_eyepatch[0]) <= 1:  # Allow 1 pixel tolerance
            if y < initial_center_eyepatch[1]:  # Only consider points above the initial center
                if y < uppereyelid_grabcut_y:
                    uppereyelid_grabcut_y = y
                    break
    
    # Calculate uppereyelid_to_center_y_diff using the final iris center
    if uppereyelid_grabcut_y is not float('inf'):
        uppereyelid_to_center_y_diff = abs(grabcut_iris_center_eyepatch[1] - uppereyelid_grabcut_y)
        # print(f"grabcut_iris_center_eyepatch[1]: {grabcut_iris_center_eyepatch[1]}, uppereyelid_grabcut_y: {uppereyelid_grabcut_y}, uppereyelid_to_center_y_diff: {uppereyelid_to_center_y_diff}")

    # Save the annotated eye patch for debug
    if DEBUG_SAVE_MODE:
        # Create visualization on the eye patch
        eyepatch_debug = eyepatch.copy()
        
        # Draw irisbox rectangle and its center in yellow
        irisbox_x1 = initial_center_eyepatch[0] - irisbox_width // 2
        irisbox_x2 = initial_center_eyepatch[0] + irisbox_width // 2
        irisbox_y1 = initial_center_eyepatch[1] - irisbox_height // 2
        irisbox_y2 = initial_center_eyepatch[1] + irisbox_height // 2
        cv2.rectangle(eyepatch_debug, (irisbox_x1, irisbox_y1), (irisbox_x2, irisbox_y2), (0, 255, 255), 1)  # Yellow box
        cv2.circle(eyepatch_debug, initial_center_eyepatch, 2, (0, 255, 255), -1)  # Yellow center

        # Draw r1, r2, r3 circles used for GrabCut initialization
        cv2.circle(eyepatch_debug, initial_center_eyepatch, r3, (128, 128, 128), 1)  # Gray
        cv2.circle(eyepatch_debug, initial_center_eyepatch, r2, (128, 128, 128), 1)  
        cv2.circle(eyepatch_debug, initial_center_eyepatch, r1, (128, 128, 128), 1) 
                
        # Draw full iris contour in green
        cv2.drawContours(eyepatch_debug, [iris_contour_points], -1, (0, 255, 0), 1)  # Green

        # Draw fitted circle in red (using the results from earlier circle fitting)
        cv2.circle(eyepatch_debug, grabcut_iris_center_eyepatch, grabcut_iris_radius, (0, 0, 255), 1)  # Red circle

        # Draw RANSAC inlier points in blue (only if RANSAC succeeded)
        if ransac_inliers is not None:  # Only draw points if RANSAC succeeded
            for point in ransac_inliers:
                cv2.circle(eyepatch_debug, (round(point[0]), round(point[1])), 1, (255, 0, 0), -1)  # Blue
        
        # Draw final iris center in red
        cv2.circle(eyepatch_debug, grabcut_iris_center_eyepatch, 3, (0, 0, 255), -1)  # Red
              
        # Save both the pure eye patch and the annotated eye patch
        eyeside_str = "left" if is_left else "right"
        pure_eyepatch_path = f"{debug_iris_segment_dir}/{frame_name}_{eyeside_str}_pure.jpg"
        annotated_eyepatch_path = f"{debug_iris_segment_dir}/{frame_name}_{eyeside_str}_annotated.jpg"
        cv2.imwrite(pure_eyepatch_path, eyepatch)        
        cv2.imwrite(annotated_eyepatch_path, eyepatch_debug)

    return grabcut_iris_center_frame, grabcut_iris_radius, uppereyelid_to_center_y_diff

def find_visible_iris_box(frame, landmarks, is_left):
    if is_left:
        iris_inner_mp = (landmarks[LEFT_IRIS_INNER].x, landmarks[LEFT_IRIS_INNER].y)
        iris_outer_mp = (landmarks[LEFT_IRIS_OUTER].x, landmarks[LEFT_IRIS_OUTER].y)
        iris_center_mp = (landmarks[LEFT_IRIS_CENTER].x, landmarks[LEFT_IRIS_CENTER].y)
        eyelid_upperouter_mp = (landmarks[LEFT_EYELID_UPPEROUTER].x, landmarks[LEFT_EYELID_UPPEROUTER].y)
        eyelid_lowerouter_mp = (landmarks[LEFT_EYELID_LOWEROUTER].x, landmarks[LEFT_EYELID_LOWEROUTER].y)
        eyelid_upperinner_mp = (landmarks[LEFT_EYELID_UPPERINNER].x, landmarks[LEFT_EYELID_UPPERINNER].y)
        eyelid_lowerinner_mp = (landmarks[LEFT_EYELID_LOWERINNER].x, landmarks[LEFT_EYELID_LOWERINNER].y)
    else:
        iris_inner_mp = (landmarks[RIGHT_IRIS_INNER].x, landmarks[RIGHT_IRIS_INNER].y)
        iris_outer_mp = (landmarks[RIGHT_IRIS_OUTER].x, landmarks[RIGHT_IRIS_OUTER].y)
        iris_center_mp = (landmarks[RIGHT_IRIS_CENTER].x, landmarks[RIGHT_IRIS_CENTER].y)
        eyelid_upperouter_mp = (landmarks[RIGHT_EYELID_UPPEROUTER].x, landmarks[RIGHT_EYELID_UPPEROUTER].y)
        eyelid_lowerouter_mp = (landmarks[RIGHT_EYELID_LOWEROUTER].x, landmarks[RIGHT_EYELID_LOWEROUTER].y)
        eyelid_upperinner_mp = (landmarks[RIGHT_EYELID_UPPERINNER].x, landmarks[RIGHT_EYELID_UPPERINNER].y)
        eyelid_lowerinner_mp = (landmarks[RIGHT_EYELID_LOWERINNER].x, landmarks[RIGHT_EYELID_LOWERINNER].y)
    
    frame_height, frame_width = frame.shape[:2]    
    initial_center_frame = (round(iris_center_mp[0] * frame_width), round(iris_center_mp[1] * frame_height))
    iris_width = round(math.sqrt(((iris_outer_mp[0] - iris_inner_mp[0]) * frame_width)**2 + ((iris_outer_mp[1] - iris_inner_mp[1]) * frame_height)**2))
    
    # Calculate vertical distance between eyelids
    v1 = abs(eyelid_upperouter_mp[1] - eyelid_lowerouter_mp[1]) * frame_height  # upper_outer to lower_outer
    v2 = abs(eyelid_upperinner_mp[1] - eyelid_lowerinner_mp[1]) * frame_height  # upper_inner to lower_inner
    v3 = abs(eyelid_upperouter_mp[1] - eyelid_lowerinner_mp[1]) * frame_height  # upper_outer to lower_inner 
    v4 = abs(eyelid_upperinner_mp[1] - eyelid_lowerouter_mp[1]) * frame_height  # upper_inner to lower_outer
    max_vertical_distance = round(max(v1, v2, v3, v4))
    
    # Set the width and height of the visible iris box
    irisbox_height = max_vertical_distance
    irisbox_width = round(iris_width * 0.92) # tighten to make sure the irisbox_width is not larger than the actual iris width

    # Find the darkest area by moving the box
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # Convert image to grayscale for brightness calculation
    best_score = float('inf')  # For darkest area, we want minimum average brightness
    irisbox_best_center = initial_center_frame    
    move_range = 10
    for dx in range(-move_range, move_range + 1):
        for dy in range(-move_range, move_range + 1):
            curr_center_x = initial_center_frame[0] + dx
            curr_center_y = initial_center_frame[1] + dy

            # Calculate rectangle boundaries
            rect_x1 = curr_center_x - irisbox_width // 2
            rect_x2 = curr_center_x + irisbox_width // 2
            rect_y1 = curr_center_y - irisbox_height // 2
            rect_y2 = curr_center_y + irisbox_height // 2

            # Extract rectangle patch and calculate average brightness
            try:
                rect_patch = gray_frame[rect_y1:rect_y2, rect_x1:rect_x2]
                avg_brightness = np.mean(rect_patch)
                
                # Lower brightness = darker area = better score
                if avg_brightness < best_score:
                    best_score = avg_brightness
                    irisbox_best_center = (curr_center_x, curr_center_y)
            except Exception as e:
                print(f"Warning: find_visible_iris_box(), Exception: {e}")
                continue

    return irisbox_best_center, irisbox_width, irisbox_height


def fit_circle_ransac(points, max_iterations=300, threshold=2.0, min_inliers=0.4):
    """
    Fit a circle to points using RANSAC algorithm with geometric least-squares refinement.
    
    Args:
        points: numpy array of shape (N, 2) containing 2D points
        max_iterations: maximum number of RANSAC iterations
        threshold: distance threshold for inlier classification
        min_inliers: minimum fraction of points that must be inliers
        
    Returns:
        tuple: ((center_x, center_y), radius, inlier_points) where inlier_points are the points used for final fitting
    """
    best_inliers_count = -1
    best_inliers_indices = None
    num_points = len(points)
    min_inlier_count = int(min_inliers * num_points)
    for _ in range(max_iterations):
        # Randomly sample 3 points
        sample_indices = np.random.choice(num_points, 3, replace=False)
        sample_points = points[sample_indices]
        try:
            x1, y1 = sample_points[0]
            x2, y2 = sample_points[1]
            x3, y3 = sample_points[2]
            
            A = np.array([
                [2*(x2-x1), 2*(y2-y1)],
                [2*(x3-x1), 2*(y3-y1)]
            ])
            b = np.array([
                [x2**2 + y2**2 - x1**2 - y1**2],
                [x3**2 + y3**2 - x1**2 - y1**2]
            ])
            
            if np.linalg.cond(A) > 1e15:
                continue
                
            center = np.linalg.solve(A, b).flatten()
            center_x, center_y = center
            radius = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
            
            distances = np.sqrt((points[:, 0] - center_x)**2 + (points[:, 1] - center_y)**2)
            inlier_indices = np.where(np.abs(distances - radius) < threshold)[0]
            inlier_count = len(inlier_indices)
            
            if inlier_count > best_inliers_count and inlier_count >= min_inlier_count:
                best_inliers_count = inlier_count
                best_inliers_indices = inlier_indices
                
        except np.linalg.LinAlgError:
            continue
    
    if best_inliers_indices is not None and len(best_inliers_indices) >= 3:
        best_inliers = points[best_inliers_indices]
        center, radius = fit_circle_geometric(best_inliers)   # Use the geometric fit for the final result
        return center, radius, best_inliers
    else:
        return None, None, None
    
def fit_circle_geometric(points):
    """
    Fits a circle to a set of 2D points using a non-linear geometric
    least-squares method. This is the most accurate method.
    
    Args:
        points: numpy array of shape (N, 2)
        
    Returns:
        tuple: (center_x, center_y, radius) as floats or (None, None, None)
    """
    if len(points) < 3:
        return None, None, None
    
    # Define the residuals function to be minimized
    def calc_residuals(params, points):
        center_x, center_y, radius = params
        distances = np.sqrt((points[:, 0] - center_x)**2 + (points[:, 1] - center_y)**2)
        return distances - radius

    # Initial guess for the parameters (center_x, center_y, radius)
    # A good guess is the mean of the points and the mean distance from that center
    x_mean, y_mean = np.mean(points, axis=0)
    initial_radius = np.mean(np.sqrt((points[:, 0] - x_mean)**2 + (points[:, 1] - y_mean)**2))
    initial_guess = [x_mean, y_mean, initial_radius]
    
    try:
        result = optimize.least_squares(calc_residuals, initial_guess, args=(points,))
        center_x, center_y, radius = result.x
        return (round(center_x), round(center_y)), round(radius)
    except Exception:
        return None, None
    

def find_soe(frame, landmarks,
                       subject, session, frame_name, backgroundA_id, backgroundB_id, dissolve, 
                       debug_heatmap_dir, debug_template_dir, debug_score_plot_dir, debug_eyepatch_and_soe_dir, 
                       template_output_dir,
                       irisbox_center_frame, irisbox_width, irisbox_height,
                       grabcut_iris_center_frame, uppereyelid_to_center_y_diff, is_left=None):   
    frame_height, frame_width = frame.shape[:2]
    left_or_right = "left" if is_left else "right"
    
    # make iriscrop, an area that template matching is to be performed on.
    iriscrop_width = round(irisbox_width * 0.40)
    iriscrop_height = round(irisbox_width * 0.55)        
    iriscrop_left = grabcut_iris_center_frame[0] - iriscrop_width // 2
    iriscrop_right = grabcut_iris_center_frame[0] + iriscrop_width // 2
    iriscrop_top = grabcut_iris_center_frame[1] - iriscrop_height // 2
    iriscrop_bottom = grabcut_iris_center_frame[1] + iriscrop_height // 2
    if iriscrop_left < 0 or iriscrop_right > frame_width or iriscrop_top < 0 or iriscrop_bottom > frame_height:
        print(f"Warning: iriscrop is out of frame bounds - SoE detection skip, subject: {subject}, session: {session}, frame_name: {frame_name}, is_left: {is_left}")
        return None, None
    iriscrop = frame[iriscrop_top:iriscrop_bottom, iriscrop_left:iriscrop_right]
        
    # Load background images and create the template to match using dissolve
    backgroundA_path = f"preprocessed/backgrounds/p{subject}/s{session}/{backgroundA_id}.jpg"
    backgroundB_path = f"preprocessed/backgrounds/p{subject}/s{session}/{backgroundB_id}.jpg"    
    backgroundA = cv2.imread(backgroundA_path)
    backgroundB = cv2.imread(backgroundB_path)    
    if backgroundA is None or backgroundB is None:
        print(f"Warning: Could not load background images for frame {frame_name}, subject: {subject}, session: {session}, frame_name: {frame_name}, is_left: {is_left}")
        return None, None     
    template = cv2.addWeighted(backgroundA, 1.0 - dissolve, backgroundB, dissolve, 0)
    template_width_start = round(irisbox_width / 10.0) - 1
    template_height_start = round(template_width_start * (24.0 / 14.0)) 

    # Perform template matching
    result = do_template_match(iriscrop, template, template_width_start, template_height_start, frame_name, debug_template_dir, template_output_dir, is_left)    
    heatmap = result['heatmap']
    heatmap_minmaxnormalized = result['heatmap_minmaxnormalized']
    template_size_without_padding = result['template_size_without_padding']
    best_score = result['best_score']
    best_x, best_y = result['best_position']    # top left corner of the template without padding in the iriscrop coordinates    
    template_type = result['template_type']
    template_padding = result['template_padding']

    iris_to_soe_center_x_norm = -999
    iris_to_soe_center_y_norm = -999
    soe_width_norm = -999
    soe_height_norm = -999
    valid_soe = False

    heatmap_h, heatmap_w = heatmap_minmaxnormalized.shape
    best_x_in_heatmap = best_x - template_padding  # top left corner of the template with padding in the iriscrop coordinates
    best_y_in_heatmap = best_y - template_padding  # top left corner of the template with padding in the iriscrop coordinates  
    
    ### Heuristics A: filter out invalid SoE detections - usually with dark background on dark iris.
    # Check the template matching scores at (best_x +-7, best_y), and ensure that the scores are greater than 2.0,
    # which are the most cases of the valid SoE. 
    # (When a correct bounding box is made at the SoE, the matching score increases quickly at the horizontal sides of the best match)
    # Also, filter out the template_type == 2, which is dark_nondetectable.
    if template_type == 0 or template_type == 1:  # bright or dark detectable
        valid_soe = True
        FILTER_SCORE_THRESHOLD = 0.20    # strong threshold that would even filter out some of correct SoE - for the sake of reliably rejecting false positivies (making the false positive rate lower is much more important than slightly increasing the true positive rate)
        if 0 <= best_x_in_heatmap - 7 < heatmap_w:
            if heatmap_minmaxnormalized[best_y_in_heatmap, best_x_in_heatmap - 7] < FILTER_SCORE_THRESHOLD:
                valid_soe = False
                print(f" Heuristics A: invalid SoE - frame_name: {frame_name}_{left_or_right}, bestmatch: ({best_x_in_heatmap}, {best_y_in_heatmap}), score at (best_x - 7, best_y): {heatmap_minmaxnormalized[best_y_in_heatmap, best_x_in_heatmap - 7]}")
        if valid_soe:
            if 0 <= best_x_in_heatmap + 7 < heatmap_w:
                if heatmap_minmaxnormalized[best_y_in_heatmap, best_x_in_heatmap + 7] < FILTER_SCORE_THRESHOLD:
                    valid_soe = False
                    print(f" Heuristics A: invalid SoE - frame_name: {frame_name}_{left_or_right}, bestmatch: ({best_x_in_heatmap}, {best_y_in_heatmap}), score at (best_x + 7, best_y): {heatmap_minmaxnormalized[best_y_in_heatmap, best_x_in_heatmap + 7]}")

    ### heuristics B:
    # If the upper eyelid is close to iris center, which indicates that the user is looking at the lower part of the screen,
    # (which logically makes the best match higher than the iris center)
    # if the best match is made low, then invalidate the SoE detection
    if valid_soe:                
        if uppereyelid_to_center_y_diff is not None:
            if uppereyelid_to_center_y_diff < irisbox_height * 0.30:  # might need to raise the threshold (toward rejecting false positives at the cost of losing some true positives)
                best_y_in_frame = best_y + template_size_without_padding[1] / 2.0 + iriscrop_top
                iriscenter_to_best_y = best_y_in_frame - grabcut_iris_center_frame[1]
                if iriscenter_to_best_y > irisbox_height * 0.15:  # might need to lower the threshold (toward rejecting false positives at the cost of losing some true positives)
                    valid_soe = False
                    print(f" Heuristics B: invalid SoE - frame_name: {frame_name}_{left_or_right}, uppereyelid_to_center_y_diff: {uppereyelid_to_center_y_diff}, iriscenter_to_best_y: {iriscenter_to_best_y}, irisbox_width: {irisbox_width}")

    # Finally, prepare the SoE values, which will be used for the additional signal for our eye tracking model.        
    if valid_soe:        
        soe_center_frame = (round(best_x + template_size_without_padding[0] / 2.0 + iriscrop_left), 
                            round(best_y + template_size_without_padding[1] / 2.0 + iriscrop_top))
        iris_to_soe_center_x_norm = (soe_center_frame[0] - grabcut_iris_center_frame[0]) / template_size_without_padding[0]
        iris_to_soe_center_y_norm = (soe_center_frame[1] - grabcut_iris_center_frame[1]) / template_size_without_padding[1]  # Using width for both to maintain aspect ratio
        #soe_width_norm = template_size_without_padding[0] / template_size_without_padding[0]
        #soe_height_norm = template_size_without_padding[1] / template_size_without_padding[1]  # Using width for both to maintain aspect ratio      
      
    ### debug: save the template matching heatmap
    if DEBUG_SAVE_MODE:
        heatmap_uint8 = (heatmap_minmaxnormalized * 255).astype(np.uint8)
        heatmap_color = cv2.cvtColor(heatmap_uint8, cv2.COLOR_GRAY2BGR)
        heatmap_color[best_y_in_heatmap, best_x_in_heatmap] = [0, 0, 255]  # Red (BGR)
        heatmap_path = f"{debug_heatmap_dir}/{frame_name}_heatmap_{left_or_right}_TemplateSize{template_size_without_padding}_BestAt({best_x_in_heatmap},{best_y_in_heatmap})_BestScore{best_score:.4f}.png"
        cv2.imwrite(heatmap_path, heatmap_color)
    
    ### debug: save score plots around the best match (X +-10, Y +- 10)
    if DEBUG_SAVE_MODE:
        if valid_soe:
            # Create score plot for X variation (Y fixed)
            x_scores = []
            x_positions = []
            for dx in range(-10, 11):
                x_pos = best_x_in_heatmap + dx
                if 0 <= x_pos < heatmap_w:
                    score = heatmap_minmaxnormalized[best_y_in_heatmap, x_pos]
                    x_scores.append(score)
                    x_positions.append(dx)  # Store relative position instead
            
            plt.figure(figsize=(10, 6))
            plt.plot(x_positions, x_scores, 'b-o', markersize=4)
            plt.axvline(x=0, color='r', linestyle='--', label=f'Best X')
            plt.xlabel('X Position Relative to Best Match')
            plt.ylabel('Template Match Score')
            plt.title(f'Score Variation along X axis (X={best_x_in_heatmap}, Y={best_y_in_heatmap})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.0)  # Set y-axis range from 0 to 1
            plt.xticks(range(min(x_positions), max(x_positions)+1, 1))  # Set x-axis ticks to increment by 1
            plt.yticks(np.arange(0, 1.0, 0.05))  # Set y-axis ticks to increment by 0.05
            x_plot_path = f"{debug_score_plot_dir}/{frame_name}_varyX_{left_or_right}.png"
            plt.savefig(x_plot_path, dpi=150, bbox_inches='tight')
            plt.close()

    ### debug: save the SoE annotated eye image
    # the size of the image is the same as the input eye patch image
    if DEBUG_SAVE_MODE:
        eyepatch_width = round(irisbox_width * EYE_CROP_SCALE_FACTOR)
        eyepatch_height = eyepatch_width // 2
        eyepatch_top = grabcut_iris_center_frame[1] - eyepatch_height // 2
        eyepatch_bottom = grabcut_iris_center_frame[1] + eyepatch_height // 2
        eyepatch_left = grabcut_iris_center_frame[0] - eyepatch_width // 2    
        eyepatch_right = grabcut_iris_center_frame[0] + eyepatch_width // 2
        if eyepatch_left < 0 or eyepatch_right > frame_width or eyepatch_top < 0 or eyepatch_bottom > frame_height:
            print(f"Warning: eyepatch is out of frame bounds - SoE detection skip, subject: {subject}, session: {session}, frame_name: {frame_name}, is_left: {is_left}")
            return None, None
        eyepatch_soe_debug = frame[eyepatch_top:eyepatch_bottom, eyepatch_left:eyepatch_right].copy()

        # Draw iris crop area in pink
        iriscrop_x1_in_vis = iriscrop_left - eyepatch_left
        iriscrop_x2_in_vis = iriscrop_right - eyepatch_left
        iriscrop_y1_in_vis = iriscrop_top - eyepatch_top
        iriscrop_y2_in_vis = iriscrop_bottom - eyepatch_top
        cv2.rectangle(eyepatch_soe_debug, (iriscrop_x1_in_vis, iriscrop_y1_in_vis), (iriscrop_x2_in_vis, iriscrop_y2_in_vis), (255, 0, 255), 1)  # Pink
        
        # Draw iris center from landmark in blue
        if is_left:
            iris_center = landmarks[LEFT_IRIS_CENTER]
        else:
            iris_center = landmarks[RIGHT_IRIS_CENTER]            
        mp_iris_center_eyepatch = (round(iris_center.x * frame_width) - eyepatch_left, round(iris_center.y * frame_height) - eyepatch_top)
        cv2.circle(eyepatch_soe_debug, mp_iris_center_eyepatch, 1, (255, 0, 0), -1)  # Blue center

        # Draw SoE in green, if the SoE detection is valid
        if valid_soe:
            soe_center_eyepatch = (soe_center_frame[0] - eyepatch_left, soe_center_frame[1] - eyepatch_top)
            soe_box_x1 = soe_center_eyepatch[0] - template_size_without_padding[0] // 2
            soe_box_x2 = soe_center_eyepatch[0] + template_size_without_padding[0] // 2
            soe_box_y1 = soe_center_eyepatch[1] - template_size_without_padding[1] // 2
            soe_box_y2 = soe_center_eyepatch[1] + template_size_without_padding[1] // 2            
            cv2.rectangle(eyepatch_soe_debug, (soe_box_x1, soe_box_y1), (soe_box_x2, soe_box_y2), (0, 255, 0), 1)  # Green
            cv2.circle(eyepatch_soe_debug, soe_center_eyepatch, 1, (0, 255, 0), -1)  # Green center

        # Draw fitted iris' center
        grabcut_iris_center_eyepatch = (grabcut_iris_center_frame[0] - eyepatch_left, grabcut_iris_center_frame[1] - eyepatch_top)
        cv2.circle(eyepatch_soe_debug, grabcut_iris_center_eyepatch, 1, (0, 255, 255), -1)  # Yellow center

        # Save the visualization
        debug_eyepatch_soe_path = f"{debug_eyepatch_and_soe_dir}/{frame_name}_eyepatch_soe_{left_or_right}.jpg"
        cv2.imwrite(debug_eyepatch_soe_path, eyepatch_soe_debug)
    
    #return (iris_to_soe_center_x_norm, iris_to_soe_center_y_norm), soe_width_norm, soe_height_norm
    return (iris_to_soe_center_x_norm, iris_to_soe_center_y_norm), heatmap_minmaxnormalized
    
def do_template_match(iriscrop, template, template_width_start, template_height_start, frame_name, debug_template_dir, template_output_dir, is_left):    
    # save the template for testing "rgbt" - RGB+template model
    kernel_size = 301
    sigma = kernel_size / 6
    template_blurred = helpermethods.separable_gaussian_blur(template, kernel_size, sigma)    
    template_small = cv2.resize(template_blurred, (50, 101), interpolation=cv2.INTER_LINEAR)
    template_path = f"{template_output_dir}/{frame_name}.png"
    cv2.imwrite(template_path, template_small)
 
    # Convert both the iriscrop and template to grayscale for template matching
    iriscrop = cv2.cvtColor(iriscrop, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)        
    template = cv2.flip(template, 1)

    # Apply separable Gaussian blur
    kernel_size = 301
    sigma = kernel_size / 6
    blurred_template = helpermethods.separable_gaussian_blur(template, kernel_size, sigma)

    bestmatch_score = float('inf')  # For TM_SQDIFF score, lower is better
    bestmatchscale_heatmap = None
    bestmatchscale_template = None
    bestmatchscale_template_size_without_padding = None
    bestmatch_position = None

    # Try several scales of template
    scales = [(w/1170.0, h/2372.0) for w in range(template_width_start, template_width_start + 5) for h in range(template_height_start, template_height_start + 5)]
    template_padding = (template_width_start + 5) // 2
    for scale in scales:
        # Resize template according to current scale & add black paddings around it
        curr_template = cv2.resize(blurred_template, (0, 0), fx=scale[0], fy=scale[1], interpolation=cv2.INTER_LINEAR)
        padded_template = np.zeros((curr_template.shape[0] + 2 * template_padding, 
                                    curr_template.shape[1] + 2 * template_padding), 
                                    dtype=curr_template.dtype)
        padded_template[template_padding:-template_padding, template_padding:-template_padding] = curr_template
        curr_template = padded_template
        template_h, template_w = curr_template.shape
                
        # Template matching & normalize the SQDiff values over template size
        result_heatmap = cv2.matchTemplate(iriscrop, curr_template, cv2.TM_SQDIFF)
        result_heatmap = result_heatmap / (template_h * template_w * 255 * 255)
        
        # Get all scores and their locations
        flat_heatmap = result_heatmap.flatten()
        all_indices = np.argsort(flat_heatmap)
        min_score_idx = all_indices[0]
        min_score = flat_heatmap[min_score_idx]

        # Update best match if this scale gives better score
        if min_score < bestmatch_score:
            bestmatch_score = min_score
            bestmatchscale_heatmap = result_heatmap.copy()
            bestmatchscale_template = curr_template.copy()
            bestmatchscale_template_size_without_padding = (template_w - 2 * template_padding, template_h - 2 * template_padding)
            
            # Convert best match location to (x,y) format
            y, x = np.unravel_index(min_score_idx, result_heatmap.shape)
            bestmatch_position = (x + template_padding, y + template_padding)   # top left corner of the template without padding in the iriscrop coordinates

            # minmax normalize
            bestmatchscale_heatmap_min = np.min(bestmatchscale_heatmap)
            bestmatchscale_heatmap_max = np.max(bestmatchscale_heatmap)
            bestmatchscale_heatmap_minmaxnormalized = (bestmatchscale_heatmap - bestmatchscale_heatmap_min) / (bestmatchscale_heatmap_max - bestmatchscale_heatmap_min)

    # Remove padding from template before classification
    template_no_padding = bestmatchscale_template[template_padding:-template_padding, template_padding:-template_padding]
    template_type = helpermethods.classify_patch(template_no_padding)    # 0: bright, 1: dark_detectable, 2: dark_nondetectable

    ### debug: Save the template used
    if DEBUG_SAVE_MODE:
        debug_template_path = f"{debug_template_dir}/{frame_name}_used_template.png"
        cv2.imwrite(debug_template_path, bestmatchscale_template)
        
    return {
        'heatmap': bestmatchscale_heatmap,
        'heatmap_minmaxnormalized': bestmatchscale_heatmap_minmaxnormalized,
        'template_size_without_padding': bestmatchscale_template_size_without_padding,
        'best_score': bestmatch_score,
        'best_position': bestmatch_position,    # top left corner of the template without padding in the iriscrop coordinates
        'template_type': template_type,
        'template_padding': template_padding
    }

def crop_eye(frame, subject, session, frame_name, iris_center_frame, irisbox_width, is_left):
    frame_height, frame_width = frame.shape[:2]    

    eyepatch_width = round(irisbox_width * EYE_CROP_SCALE_FACTOR)
    eyepatch_height = eyepatch_width // 2

    top = iris_center_frame[1] - eyepatch_height // 2
    bottom = iris_center_frame[1] + eyepatch_height // 2
    left = iris_center_frame[0] - eyepatch_width // 2
    right = iris_center_frame[0] + eyepatch_width // 2    
    if top < 0 or bottom > frame_height or left < 0 or right > frame_width:
        print(f" Warning: eye patch boundary is out of image frame, skipping: subject: {subject}, session: {session}, frame: {frame_name}, is_left: {is_left}")
        return None
    
    return frame[top:bottom, left:right]

def process_csv_file(subject, session):
    csv_file_path = f"preprocessed/logs/p{subject}_s{session}_log_preprocessed.csv"                
    frame_dir = f"preprocessed/frames/p{subject}/s{session}"    
        
    eyepatch_output_dir = f"preprocessed/input_eyepatch/p{subject}/s{session}"
    heatmap_output_dir = f"preprocessed/input_heatmap/p{subject}/s{session}"
    landmark_and_gt_output_dir = f"preprocessed/input_landmark_and_gt/p{subject}/s{session}"
    template_output_dir = f"preprocessed/input_template/p{subject}/s{session}"
    debug_heatmap_dir = f"preprocessed/input_debug/heatmap/p{subject}/s{session}"
    debug_template_dir = f"preprocessed/input_debug/template/p{subject}/s{session}" 
    debug_score_plot_dir = f"preprocessed/input_debug/score_plot/p{subject}/s{session}"
    debug_eyepatch_and_soe_dir = f"preprocessed/input_debug/eyepatch_and_soe/p{subject}/s{session}"
    debug_iris_segment_dir = f"preprocessed/input_debug/iris_segment/p{subject}/s{session}"

    os.makedirs(eyepatch_output_dir, exist_ok=True)
    os.makedirs(heatmap_output_dir, exist_ok=True)
    os.makedirs(landmark_and_gt_output_dir, exist_ok=True)
    os.makedirs(template_output_dir, exist_ok=True)
    os.makedirs(debug_heatmap_dir, exist_ok=True) 
    os.makedirs(debug_template_dir, exist_ok=True)
    os.makedirs(debug_score_plot_dir, exist_ok=True)
    os.makedirs(debug_eyepatch_and_soe_dir, exist_ok=True)
    os.makedirs(debug_iris_segment_dir, exist_ok=True)

    # Create directory for blink-filtered logs
    blinkfiltered_logs_dir = "preprocessed/logs_blinkfiltered"
    os.makedirs(blinkfiltered_logs_dir, exist_ok=True)
    blinkfiltered_csv_path = f"{blinkfiltered_logs_dir}/p{subject}_s{session}_log_blinkfiltered.csv"

    # Read CSV file
    df = pd.read_csv(csv_file_path)
    df_filtered = df.copy()  # Create a copy for filtering out blink instances

    # Initialize counters
    skipped_closed_eyes = 0
    skipped_frame_numbers = []  # Track skipped frame numbers
    
    # Initialize MediaPipe Face Mesh
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing subject {subject}, session {session}"):   
        # if index != 51:
        #     continue
        # else:
        #     print(f"!!! frame_name:{str(int(row['frameNum']))}")
        
        frame_name = str(int(row['frameNum'])).zfill(5)
        backgroundA_id = str(int(row['backgroundA']))
        backgroundB_id = str(int(row['backgroundB']))
        dissolve = float(row['dissolve'])
        px_norm_x = row['px_norm_x']
        px_norm_y = row['px_norm_y']
        
        # Construct image path
        frame_path = f"{frame_dir}/{frame_name}.jpg"
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"!!! ERROR: Could not read frame, skipping: subject {subject}, session {session}, frame {frame_name}")
            continue
        
        # Rotate the frame before processing
        rotated_frame = helpermethods.rotate_image_counterclockwise(frame)        
        if rotated_frame is None:
            print(f"!!! ERROR: Could not rotate frame, skipping: subject {subject}, session {session}, frame {frame_name}")
            continue
        
        # Use the rotated image for all subsequent processing
        frame = rotated_frame        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = frame_rgb.shape[:2]
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Process with MediaPipe
        results = faceLandmarkDetector.detect(mp_image)        
        if not results.face_landmarks:
            print(f"No face landmarks detected, skipping: subject {subject}, session {session}, frame {frame_name}")
            continue        
        landmarks = results.face_landmarks[0]
        
        ### filter out blink instances
        l_ear = get_eye_aspect_ratio(landmarks, frame_width, frame_height, is_left=True)
        r_ear = get_eye_aspect_ratio(landmarks, frame_width, frame_height, is_left=False)
        l_eye_open = l_ear > EYE_CLOSED_THRESHOLD
        r_eye_open = r_ear > EYE_CLOSED_THRESHOLD
        if not (l_eye_open or r_eye_open):
            print(f"Skipping eyes closed frame: subject {subject}, session {session}, frame {frame_name} (L_EAR: {l_ear:.3f}, R_EAR: {r_ear:.3f})")
            skipped_closed_eyes += 1
            skipped_frame_numbers.append(frame_name)
            # Remove this frame from filtered dataframe
            df_filtered = df_filtered[df_filtered['frameNum'] != int(frame_name)]
            continue       

        ### Find visible iris box, which are later used in segment_iris() and find_soe()
        l_irisbox_center_frame, l_irisbox_width, l_irisbox_height = find_visible_iris_box(frame, landmarks, is_left=True)
        r_irisbox_center_frame, r_irisbox_width, r_irisbox_height = find_visible_iris_box(frame, landmarks, is_left=False)
 
        ### Segment iris to derive accurate iris centers using GrabCut algorithm
        l_grabcut_iris_center_frame, l_grabcut_iris_radius, l_uppereyelid_to_center_y_diff = segment_iris(
            frame, landmarks, subject, session, frame_name, 
            l_irisbox_center_frame, l_irisbox_width, l_irisbox_height,
            debug_iris_segment_dir, is_left=True)
        r_grabcut_iris_center_frame, r_grabcut_iris_radius, r_uppereyelid_to_center_y_diff = segment_iris(
            frame, landmarks, subject, session, frame_name, 
            r_irisbox_center_frame, r_irisbox_width, r_irisbox_height,
            debug_iris_segment_dir, is_left=False)
        
        # Check if iris segmentation failed and use MediaPipe landmarks as fallback
        if l_grabcut_iris_center_frame is None:
            l_grabcut_iris_center_frame = (round(landmarks[LEFT_IRIS_CENTER].x * frame_width), round(landmarks[LEFT_IRIS_CENTER].y * frame_height))
            l_grabcut_iris_radius = round(math.sqrt(((landmarks[LEFT_IRIS_OUTER].x - landmarks[LEFT_IRIS_INNER].x) * frame_width)**2 + ((landmarks[LEFT_IRIS_OUTER].y - landmarks[LEFT_IRIS_INNER].y) * frame_height)**2)) // 2
            l_uppereyelid_to_center_y_diff = None  # No GrabCut result available
            print(f"Left iris segmentation failed, using MediaPipe-based iris center: subject {subject}, session {session}, frame {frame_name}")
            
        if r_grabcut_iris_center_frame is None:
            r_grabcut_iris_center_frame = (round(landmarks[RIGHT_IRIS_CENTER].x * frame_width), round(landmarks[RIGHT_IRIS_CENTER].y * frame_height))
            r_grabcut_iris_radius = round(math.sqrt(((landmarks[RIGHT_IRIS_OUTER].x - landmarks[RIGHT_IRIS_INNER].x) * frame_width)**2 + ((landmarks[RIGHT_IRIS_OUTER].y - landmarks[RIGHT_IRIS_INNER].y) * frame_height)**2)) // 2
            r_uppereyelid_to_center_y_diff = None  # No GrabCut result available
            print(f"Right iris segmentation failed, using MediaPipe-based iris center: subject {subject}, session {session}, frame {frame_name}")
        
        ### Find screen on eye using template matching with segmented iris centers
        l_iris_to_soe_center_norm, l_heatmap_minmaxnormalized = find_soe(
            frame, landmarks, subject, session, frame_name, 
            backgroundA_id, backgroundB_id, dissolve,
            debug_heatmap_dir, debug_template_dir, debug_score_plot_dir, debug_eyepatch_and_soe_dir, template_output_dir,
            l_irisbox_center_frame, l_irisbox_width, l_irisbox_height,
            l_grabcut_iris_center_frame, l_uppereyelid_to_center_y_diff, is_left=True,            
        )
        if l_heatmap_minmaxnormalized is not None:
            l_heatmap_uint8 = (l_heatmap_minmaxnormalized * 255).astype(np.uint8)
            # Flip left heatmap horizontally before saving
            l_heatmap_flipped = cv2.flip(l_heatmap_uint8, 1)  # 1 means horizontal flip
            l_heatmap_path = f"{heatmap_output_dir}/{frame_name.replace('.jpg', '')}_left.png"
            cv2.imwrite(l_heatmap_path, l_heatmap_flipped)
        else:
            print(f" Warning: No heatmap, skipping: subject {subject}, session {session}, frame {frame_name}, left")
            continue
        
        r_iris_to_soe_center_norm, r_heatmap_minmaxnormalized = find_soe(
            frame, landmarks, subject, session, frame_name, 
            backgroundA_id, backgroundB_id, dissolve,
            debug_heatmap_dir, debug_template_dir, debug_score_plot_dir, debug_eyepatch_and_soe_dir, template_output_dir,
            r_irisbox_center_frame, r_irisbox_width, r_irisbox_height,
            r_grabcut_iris_center_frame, r_uppereyelid_to_center_y_diff, is_left=False,            
        )
        if r_heatmap_minmaxnormalized is not None:
            r_heatmap_uint8 = (r_heatmap_minmaxnormalized * 255).astype(np.uint8)
            r_heatmap_path = f"{heatmap_output_dir}/{frame_name.replace('.jpg', '')}_right.png"
            cv2.imwrite(r_heatmap_path, r_heatmap_uint8)
        else:
            print(f" Warning: No heatmap, skipping: subject {subject}, session {session}, frame {frame_name}, right")
            continue

        ### do eye crop and save as PNG using segmented iris centers
        l_eye_crop = crop_eye(frame, subject, session, frame_name, l_grabcut_iris_center_frame, l_irisbox_width, is_left=True)        
        if l_eye_crop is not None:
            # Flip left eye horizontally before saving
            l_eye_flipped = cv2.flip(l_eye_crop, 1)  # 1 means horizontal flip
            l_eye_path = f"{eyepatch_output_dir}/{frame_name.replace('.jpg', '')}_left.png"
            cv2.imwrite(l_eye_path, l_eye_flipped)
        else:
            print(f" Warning: No eye crop, skipping: subject {subject}, session {session}, frame {frame_name}, left")
            continue

        r_eye_crop = crop_eye(frame, subject, session, frame_name, r_grabcut_iris_center_frame, r_irisbox_width, is_left=False)
        if r_eye_crop is not None:
            r_eye_path = f"{eyepatch_output_dir}/{frame_name.replace('.jpg', '')}_right.png"
            cv2.imwrite(r_eye_path, r_eye_crop)
        else:
            print(f" Warning: No eye crop, skipping: subject {subject}, session {session}, frame {frame_name}, right")
            continue
            
        ### Save landmarks and ground truth data as json        
        landmark_data = {
            "l_iris_to_soe_center_x_norm": l_iris_to_soe_center_norm[0],
            "l_iris_to_soe_center_y_norm": l_iris_to_soe_center_norm[1],
            "r_iris_to_soe_center_x_norm": r_iris_to_soe_center_norm[0],
            "r_iris_to_soe_center_y_norm": r_iris_to_soe_center_norm[1],
            "l_eye_inner_corner_x": landmarks[LEFT_EYE_INNER].x,
            "l_eye_inner_corner_y": landmarks[LEFT_EYE_INNER].y,
            "l_eye_outer_corner_x": landmarks[LEFT_EYE_OUTER].x,
            "l_eye_outer_corner_y": landmarks[LEFT_EYE_OUTER].y,
            "r_eye_inner_corner_x": landmarks[RIGHT_EYE_INNER].x,
            "r_eye_inner_corner_y": landmarks[RIGHT_EYE_INNER].y,
            "r_eye_outer_corner_x": landmarks[RIGHT_EYE_OUTER].x,
            "r_eye_outer_corner_y": landmarks[RIGHT_EYE_OUTER].y,
            "px_norm_x": px_norm_x,
            "px_norm_y": px_norm_y
        }

        json_path = f"{landmark_and_gt_output_dir}/{frame_name.replace('.jpg', '')}_landmark_and_gt.json"
        with open(json_path, 'w') as f:
            json.dump(landmark_data, f, indent=2)        

    ### Save the blink-filtered CSV
    print(f"\nSaving blink-filtered CSV with {len(df_filtered)} frames (removed {len(df) - len(df_filtered)} blink frames)")
    df_filtered.to_csv(blinkfiltered_csv_path, index=False)

def main():
    parser = argparse.ArgumentParser(description='Process eye patch SOE landmarks')
    parser.add_argument('-s', nargs='*', type=int, default=None, 
                       help='List of subjects to process (e.g., --subjects 1 2 3). If not specified, uses default list.')
    args = parser.parse_args()
    
    # Use custom subjects list if provided, otherwise use default
    subjects_to_process = args.s if args.s is not None else SUBJECTS
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting merged frame rotation and eyepatch SOE processing...")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Total subjects to process: {subjects_to_process}")
    
    for subject in subjects_to_process:
        for session in SESSIONS:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing subject p{subject}, session {session}")
            process_csv_file(subject, session)
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] All files processed!")

if __name__ == "__main__":
    main() 