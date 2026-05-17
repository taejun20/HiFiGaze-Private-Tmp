import cv2
import mediapipe as mp
import math

# Iris indices
LEFT_IRIS_IDX = [474, 475, 476, 477]  # Left iris ring
RIGHT_IRIS_IDX = [469, 470, 471, 472]  # Right iris ring
IRIS_CENTER_LEFT = 473
IRIS_CENTER_RIGHT = 468

# Estimated real-world iris diameter (in mm)
REAL_IRIS_DIAMETER_MM = 11.8

# Approximated camera focal length in pixels
# If you know horizontal FoV, you can compute: fx = w / (2 * tan(FoV / 2))
FOCAL_LENGTH_PX = 9000  # You can adjust this if known

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

def estimate_eye_distance(frame):
    """
    Estimate the distance between the camera and the eye in centimeters.
    
    Args:
        frame: RGB frame from cv2.VideoCapture.read()
        
    Returns:
        float: Estimated distance in centimeters, or None if no face detected
    """
    if frame is None:
        return None

    
    h, w, _ = frame.shape
    results = face_mesh.process(frame)

    if not results.multi_face_landmarks:
        return None

    face_landmarks = results.multi_face_landmarks[0]

    # Get iris landmarks (right eye)
    iris_points = []
    for idx in RIGHT_IRIS_IDX:
        lm = face_landmarks.landmark[idx]
        iris_points.append((int(lm.x * w), int(lm.y * h)))

    # Calculate iris diameter in pixels
    if len(iris_points) >= 2:
        # Pick two opposite points to estimate diameter
        d_px = math.dist(iris_points[0], iris_points[2])

        # Estimate distance
        distance_mm = (FOCAL_LENGTH_PX * REAL_IRIS_DIAMETER_MM) / d_px
        distance_cm = distance_mm / 10
        
        return distance_cm
    
    return None