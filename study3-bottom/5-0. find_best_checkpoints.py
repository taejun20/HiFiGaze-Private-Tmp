import os
import re
import shutil
from pathlib import Path
from collections import defaultdict

# Source and destination directories
SRC_DIR = "model_checkpoints/250904_S3"
DST_DIR = "model_checkpoints/best/"

def extract_val_error(filename):
    """Extract validation error from checkpoint filename."""
    match = re.search(r'Error(\d+\.\d+)\.pt$', filename)
    if match:
        return float(match.group(1))
    return None

def get_model_base_name(filename):
    """Extract base name up to '_Epoch' from filename."""
    return filename.split('_Epoch')[0]

def main():
    # Create destination directory if it doesn't exist
    os.makedirs(DST_DIR, exist_ok=True)

    # Group files by their base names
    checkpoints = defaultdict(list)
    
    # List all .pt files in source directory
    for file in os.listdir(SRC_DIR):
        if file.endswith('.pt'):
            base_name = get_model_base_name(file)
            val_error = extract_val_error(file)
            if val_error is not None:
                checkpoints[base_name].append((val_error, file))

    # Store best checkpoints info
    best_checkpoints = []

    # Find best checkpoint for each base name
    for base_name, files in sorted(checkpoints.items()):  # Sort by base name for consistent output
        if files:
            # Sort by validation error (ascending) and get the best one
            best_error, best_file = min(files, key=lambda x: x[0])
            
            # Source and destination paths
            src_path = os.path.join(SRC_DIR, best_file)
            dst_path = os.path.join(DST_DIR, best_file)
            
            # Copy the file
            print(f"Copying {best_file} (Val Error: {best_error:.2f})")
            shutil.copy2(src_path, dst_path)

            # Store info for summary
            best_checkpoints.append((base_name, best_file, best_error))

if __name__ == "__main__":
    main()
