import pandas as pd
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

# Define the types to analyze
TYPES = ['RGB', 'RGBSOE', 'RGBT']

def read_eval_csv(type_name):
    """Read all evaluation CSV files for a given type from tmp/ directory and return combined DataFrame"""
    tmp_dir = "tmp/"
    if not os.path.exists(tmp_dir):
        print(f"Warning: {tmp_dir} directory not found")
        return None
    
    # Find all CSV files matching the type pattern: Eval_{TYPE}_*.csv
    pattern = os.path.join(tmp_dir, f"Eval_{type_name}_*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"Warning: No CSV files found for {type_name} in {tmp_dir}")
        return None
    
    print(f"Found {len(csv_files)} CSV files for {type_name}")
    
    # Read and combine all CSV files
    dataframes = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dataframes.append(df)
            print(f"  Loaded {len(df)} rows from {os.path.basename(csv_file)}")
        except Exception as e:
            print(f"  Warning: Could not read {csv_file}: {e}")
            continue
    
    if not dataframes:
        print(f"Warning: No valid CSV files could be read for {type_name}")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Total rows for {type_name}: {len(combined_df)}")
    
    return combined_df

def find_log_file(subject, session):
    """Find the corresponding log file for a subject"""
    log_path = f"../study1_rawdata_processed/p{subject}/log.csv"
    if not os.path.exists(log_path):
        print(f"Warning: {log_path} not found")
        return None
    return log_path

def find_background_images(backgroundA, backgroundB, subject):
    """Find background image files in ../study1_rawdata_processed/screen/ directory"""
    base_path = "../study1_rawdata_processed/screen"
    
    # Look for the background images directly in the screen directory
    backgroundA_path = os.path.join(base_path, f"{backgroundA}.jpg")
    backgroundB_path = os.path.join(base_path, f"{backgroundB}.jpg")
    
    # Check if files exist
    if not os.path.exists(backgroundA_path):
        backgroundA_path = None
    if not os.path.exists(backgroundB_path):
        backgroundB_path = None
    
    return backgroundA_path, backgroundB_path

# Global image cache to avoid reloading the same images
_image_cache = {}

def calculate_brightness(backgroundA_path, backgroundB_path, dissolve, image_cache=None):
    """Load background images, create dissolved image, and calculate brightness"""
    if image_cache is None:
        image_cache = _image_cache
    
    try:
        # Load background images (with caching)
        if backgroundA_path not in image_cache:
            img = cv2.imread(backgroundA_path)
            if img is None:
                print(f"Warning: Could not load background image: {backgroundA_path}")
                return None
            image_cache[backgroundA_path] = img
        backgroundA = image_cache[backgroundA_path]
        
        if backgroundB_path not in image_cache:
            img = cv2.imread(backgroundB_path)
            if img is None:
                print(f"Warning: Could not load background image: {backgroundB_path}")
                return None
            image_cache[backgroundB_path] = img
        backgroundB = image_cache[backgroundB_path]
        
        # Create dissolved background
        screen_dissolved = cv2.addWeighted(backgroundA, 1.0 - dissolve, backgroundB, dissolve, 0)
        
        # Calculate mean brightness (mean of all RGB values)
        mean_brightness = np.mean(screen_dissolved)
        
        return mean_brightness
        
    except Exception as e:
        print(f"Error calculating brightness: {e}")
        return None

def analyze_type(type_name):
    """Analyze a single type (RGB, RGBSOE, RGBT)"""
    print(f"\n=== Analyzing {type_name} ===")
    
    # Read evaluation CSV
    eval_df = read_eval_csv(type_name)
    if eval_df is None:
        return []
    
    results = []
    
    # Group by subject for analysis
    unique_subjects = eval_df['subject'].unique()
    for subject in tqdm(unique_subjects, desc=f"Processing {type_name} subjects"):
        print(f"\nProcessing subject {subject}...")
        
        subject_data = eval_df[eval_df['subject'] == subject]
        bright_losses = []
        dark_losses = []
        
        # Cache log files and images to avoid re-reading
        log_cache = {}
        image_cache = {}
        
        for _, row in tqdm(subject_data.iterrows(), total=len(subject_data), desc=f"  Subject {subject} rows", leave=False):
            subject_id = int(row['subject'])
            session = int(row['session'])
            frame_num = row['frame_num']
            loss = row['loss']
            
            # Find corresponding log file
            log_path = find_log_file(subject_id, session)
            if log_path is None:
                continue
            
            # Read log file and find the frame (with caching)
            if log_path not in log_cache:
                log_cache[log_path] = pd.read_csv(log_path)
            log_df = log_cache[log_path]
            
            frame_data = log_df[log_df['frameNum'] == frame_num]
            
            if frame_data.empty:
                print(f"Warning: Frame {frame_num} not found in {log_path}")
                continue
            
            # Extract background information
            backgroundA = str(int(frame_data.iloc[0]['backgroundA']))
            backgroundB = str(int(frame_data.iloc[0]['backgroundB']))
            dissolve = frame_data.iloc[0]['dissolve']

            # Find background images
            backgroundA_path, backgroundB_path = find_background_images(backgroundA, backgroundB, subject_id)
            
            if backgroundA_path is None or backgroundB_path is None:
                print(f"Warning: Background images not found for frame {frame_num}")
                continue
            
            # Calculate brightness
            brightness = calculate_brightness(backgroundA_path, backgroundB_path, dissolve, image_cache)
            
            if brightness is None:
                continue
            
            # Classify as bright or dark (threshold = 127)
            if brightness > 127:
                bright_losses.append(loss)
            else:
                dark_losses.append(loss)
        
        # Calculate mean losses for this subject
        if bright_losses or dark_losses:
            mean_bright_loss = np.mean(bright_losses) if bright_losses else np.nan
            mean_dark_loss = np.mean(dark_losses) if dark_losses else np.nan
            
            results.append({
                'subject': subject_id,
                'type': type_name,
                'loss_bright': mean_bright_loss,
                'loss_dark': mean_dark_loss
            })
            
            print(f"  Subject {subject_id}: {len(bright_losses)} bright frames, {len(dark_losses)} dark frames")
            print(f"    Mean bright loss: {mean_bright_loss:.4f}")
            print(f"    Mean dark loss: {mean_dark_loss:.4f}")
    
    return results

def main():
    """Main analysis function"""
    print("Starting brightness accuracy analysis...")
    
    all_results = []
    
    # Analyze each type
    for type_name in TYPES:
        type_results = analyze_type(type_name)
        all_results.extend(type_results)
    
    # Create results DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Transform to loss table format
        # Create pivot table with subject as index, type as columns for bright/dark
        loss_table = []
        
        # Get all unique subjects
        all_subjects = sorted(results_df['subject'].unique())
        
        for subject in all_subjects:
            subject_data = results_df[results_df['subject'] == subject]
            
            row = {'id': f'p{subject}'}
            
            # Add columns for each type and condition
            for type_name in TYPES:
                type_data = subject_data[subject_data['type'] == type_name]
                if not type_data.empty:
                    row[f'{type_name.lower()}_bright'] = type_data.iloc[0]['loss_bright']
                    row[f'{type_name.lower()}_dark'] = type_data.iloc[0]['loss_dark']
                else:
                    row[f'{type_name.lower()}_bright'] = np.nan
                    row[f'{type_name.lower()}_dark'] = np.nan
            
            loss_table.append(row)
        
        # Create DataFrame with proper column order
        loss_table_df = pd.DataFrame(loss_table)
        column_order = ['id'] + [f'{t.lower()}_bright' for t in TYPES] + [f'{t.lower()}_dark' for t in TYPES]
        loss_table_df = loss_table_df[column_order]
        
        # Save to CSV
        output_file = "brightness_accuracy_results.csv"
        loss_table_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        # Print summary statistics
        print("\n=== SUMMARY STATISTICS ===")
        
        # Overall means (using original results_df)
        overall_bright = results_df['loss_bright'].mean()
        overall_dark = results_df['loss_dark'].mean()
        
        print(f"Overall mean loss (bright backgrounds): {overall_bright:.4f}")
        print(f"Overall mean loss (dark backgrounds): {overall_dark:.4f}")
        
        # Per-type statistics
        print("\nPer-type statistics:")
        for type_name in TYPES:
            type_data = results_df[results_df['type'] == type_name]
            if not type_data.empty:
                type_bright = type_data['loss_bright'].mean()
                type_dark = type_data['loss_dark'].mean()
                print(f"{type_name}: Bright={type_bright:.4f}, Dark={type_dark:.4f}")
    
    else:
        print("No results to analyze")

if __name__ == "__main__":
    main()
