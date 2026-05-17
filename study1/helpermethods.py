import cv2
import numpy as np

def rotate_image_clockwise(image):
    try:
        # Direct rotation in OpenCV (much more efficient than PIL)
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    except Exception as e:
        print(f"An error occurred while rotating image: {e}")
        return None

def rotate_image_counterclockwise(image):
    try:
        # Direct rotation in OpenCV (much more efficient than PIL)
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception as e:
        print(f"An error occurred while rotating image: {e}")
        return None

def separable_gaussian_blur(img, kernel_size, sigma):
    # Create 1D Gaussian kernel
    k = (kernel_size - 1) // 2
    x = np.linspace(-k, k, kernel_size)
    kernel_1d = np.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Apply horizontal blur
    kernel_h = kernel_1d.reshape(1, -1)
    blurred_h = cv2.filter2D(img, -1, kernel_h)
    
    # Apply vertical blur
    kernel_v = kernel_1d.reshape(-1, 1)
    blurred = cv2.filter2D(blurred_h, -1, kernel_v)
    
    return blurred

def classify_patch(image_gray):
    img = image_gray.astype(np.float32) / 255.0
    mean_val = np.mean(img)
    #print(f"mean_val: {mean_val}")

    h, w = img.shape
    if mean_val > 0.7:
        return 0    # bright   
    else:
        # Threshold to find bright regions
        bright_threshold = 0.7  # Adjust this threshold as needed
        bright_mask = (img > bright_threshold).astype(np.uint8)
        
        # Check for horizontal stripes (fully or partially fitting)
        thickness = 2
        for y in range(h):
            if y + thickness > h:
                continue
            stripe = bright_mask[y:y+thickness, :]
            #print(f"stripe: {stripe}")
            # Check if bright thick stripe
            if np.mean(np.mean(stripe, axis=1)) > 0.7:
                #print(f"Found horizontal stripe at y={y}, thickness={thickness}, mean: {np.mean(np.mean(stripe, axis=1))}")
                return 1    # dark_detectable

        # Check for vertical stripes (fully or partially fitting)
        for x in range(w):
            if x + thickness > w:
                continue
            stripe = bright_mask[:, x:x+thickness]
            # Check if any column is mostly bright (>50% pixels are bright)
            if np.mean(np.mean(stripe, axis=0)) > 0.7:
                #print(f"Found vertical stripe at x={x}, thickness={thickness}, mean: {np.mean(np.mean(stripe, axis=0))}")
                return 1    # dark_detectable
    return 2    # dark_nondetectable