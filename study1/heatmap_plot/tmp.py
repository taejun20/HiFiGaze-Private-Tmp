from PIL import Image, ImageFilter

# Image names to process
image_names = ["rgb.png", "rgbt.png", "rgbsoe.png"]

# Process each image
for image_name in image_names:
    input_path = image_name
    output_path = image_name.replace(".png", "_gaussian_blur.png")
    
    # Load image
    img = Image.open(input_path)
    
    # Apply Gaussian blur
    # radius controls blur strength (try 2–10)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=150))
    
    # Save result
    blurred.save(output_path)
    
    print(f"Saved blurred image to {output_path}")
