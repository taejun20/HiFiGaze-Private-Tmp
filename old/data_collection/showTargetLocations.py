import cv2
import numpy as np
from pathlib import Path

# Get the absolute path of the script's directory
SCRIPT_DIR = Path(__file__).parent.absolute()

# Screen and dot properties
screen_width = 3420
screen_height = 2138
dot_radius = 12
dot_color = (0, 0, 255)  # BGR: Red
dot_border_color = (255, 255, 255)  # BGR: White
dot_border_thickness = 5

# Grid settings
horizontal_pos_count = 8
vertical_pos_count = 6
dot_pos_list = []

# Generate positions
def generate_positions():
    for i in range(horizontal_pos_count):
        for j in range(vertical_pos_count):
            pos_x = float(screen_width) / (horizontal_pos_count + 1) * (i + 1)
            pos_y = float(screen_height) / (vertical_pos_count + 1) * (j + 1)
            dot_pos_list.append((pos_x, pos_y))

# Load background
background = cv2.imread(str(SCRIPT_DIR / "images" / "background_laptop_1.png"))

# Create window
cv2.namedWindow('dots', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('dots', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Generate dot positions
generate_positions()

# Create display with all dots
canvas = background.copy()
for pos_x, pos_y in dot_pos_list:
    # Draw dot with border
    cv2.circle(canvas, (int(pos_x), int(pos_y)), 
               dot_radius + dot_border_thickness, dot_border_color, -1)  # White border
    cv2.circle(canvas, (int(pos_x), int(pos_y)), 
               dot_radius, dot_color, -1)  # Red dot

# Draw dot at bottom-right (3420, 2138)
cv2.circle(canvas, (3420, 2138), 
           dot_radius + dot_border_thickness, dot_border_color, -1)  # White border
cv2.circle(canvas, (3420, 2138), 
           dot_radius, dot_color, -1)  # Red dot

# Display the image
cv2.imshow('dots', canvas)

# Wait for ESC key to exit
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()
