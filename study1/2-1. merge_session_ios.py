import pandas as pd
import os
import sys
from tqdm import tqdm
import time
from datetime import datetime
import argparse
import cv2
import json
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import helpermethods

#SUBJECTS = [25]

SUBJECTS = [1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26]
SESSION_PAIRS = [
    ("1p", "1s", "1"),  # (session1, session2, output_session)
    ("2p", "2s", "2"),
    ("3p", "3s", "3"),
    ("4p", "4s", "4"),
    ("5p", "5s", "5"),
    ("6p", "6s", "6"),
]

TEMPLATE_KERNEL_SIZE = 301
TEMPLATE_SIGMA = TEMPLATE_KERNEL_SIZE / 6
TEMPLATE_SIZE = (50, 101)  # (width, height)


def build_template(background_dir, background_a, background_b, dissolve, output_path):
    background_a_path = f"{background_dir}/{background_a}.jpg"
    background_b_path = f"{background_dir}/{background_b}.jpg"

    if not os.path.exists(background_a_path) or not os.path.exists(background_b_path):
        print(f"Warning: Background missing for template {output_path} ({background_a_path}, {background_b_path})")
        return

    img_a = cv2.imread(background_a_path)
    img_b = cv2.imread(background_b_path)

    if img_a is None or img_b is None:
        print(f"Warning: Failed to read background images for template {output_path}")
        return

    try:
        dissolve = float(dissolve)
    except (TypeError, ValueError):
        print(f"Warning: Invalid dissolve value '{dissolve}' for template {output_path}")
        return
    template = cv2.addWeighted(img_a, 1.0 - dissolve, img_b, dissolve, 0)

    template_blurred = helpermethods.separable_gaussian_blur(template, TEMPLATE_KERNEL_SIZE, TEMPLATE_SIGMA)
    template_small = cv2.resize(template_blurred, TEMPLATE_SIZE, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(output_path, template_small)


def process_and_copy_frames(df, subject, input_session, output_session, source_session_dir, target_session_dir, start_frame_num=1):
    # Create a copy of the dataframe
    df_updated = df.copy()
    current_frame_num = start_frame_num
    
    # Create mapping of old frame numbers to new ones
    frame_mapping = {}
    
    # Prepare frame metadata lookup
    frame_metadata = {}
    for _, row in df.iterrows():
        key = str(int(row['frameNum'])).zfill(5)
        if key not in frame_metadata:
            frame_metadata[key] = {
                "backgroundA": row.get('backgroundA'),
                "backgroundB": row.get('backgroundB'),
                "dissolve": row.get('dissolve')
            }

    # Get all frame numbers in order
    frame_numbers = sorted([int(x) for x in df['frameNum'].unique()])
    
    for old_num in tqdm(frame_numbers, desc="Processing frames"):
        # Map old frame number to new sequential number
        old_frame = str(old_num).zfill(5)
        new_frame = str(current_frame_num).zfill(5)
        frame_mapping[old_frame] = new_frame
        
        # Copy and rename eye patch PNGs (left/right)
        for eye_side in ["left", "right"]:
            source_file = f"{source_session_dir}/{old_frame}_{eye_side}.png"
            target_file = f"{target_session_dir}/{new_frame}_{eye_side}.png"
            
            if os.path.exists(source_file):
                image = cv2.imread(source_file)
                if image is None:
                    print(f"Warning: Failed to read image - {source_file}")
                else:
                    cv2.imwrite(target_file, image)
            else:
                print(f"Warning: Source frame not found - {source_file}")
        
        # Copy and rename full-frame JPG
        source_jpg = f"{source_session_dir}/{old_frame}.jpg"
        target_jpg = f"{target_session_dir}/{new_frame}.jpg"
        if os.path.exists(source_jpg):
            shutil.copy2(source_jpg, target_jpg)
        else:
            print(f"Warning: Source JPG not found - {source_jpg}")
        
        # Copy and rename landmarks JSON
        source_json = f"{source_session_dir}/{old_frame}_lms.json"
        target_json = f"{target_session_dir}/{new_frame}_lms.json"
        if os.path.exists(source_json):
            shutil.copy2(source_json, target_json)
        else:
            print(f"Warning: Source JSON not found - {source_json}")
        
        # Build template image
        meta = frame_metadata.get(old_frame, {})
        background_a = meta.get("backgroundA")
        background_b = meta.get("backgroundB")
        dissolve = meta.get("dissolve")

        if background_a is not None and background_b is not None and dissolve is not None:
            background_dir = f"study1-backgrounds/p{subject}/{input_session}"
            template_path = f"{target_session_dir}/{new_frame}_template.png"
            build_template(background_dir, background_a, background_b, dissolve, template_path)
        else:
            print(f"Warning: Missing background metadata for frame {old_frame} (subject {subject}, session {input_session})")

        current_frame_num += 1
    
    # Update frameNum in dataframe
    df_updated['frameNum'] = df_updated['frameNum'].apply(lambda x: frame_mapping[str(int(x)).zfill(5)])
        
    return df_updated, current_frame_num

def add_ground_truth_to_jsons(df, output_frame_dir):
    """
    Add px_norm_x and px_norm_y ground truth values to each landmarks JSON file.
    
    Args:
        df: Merged dataframe with frameNum, px_norm_x, px_norm_y fields
        output_frame_dir: Directory containing the {frameNum}_lms.json files
    """
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Adding ground truth to JSONs"):
        frame_num = str(int(row['frameNum'])).zfill(5)
        json_path = f"{output_frame_dir}/{frame_num}_lms.json"
        
        if not os.path.exists(json_path):
            print(f"Warning: JSON file not found - {json_path}")
            continue
        
        try:
            # Read existing JSON
            with open(json_path, 'r') as f:
                lms_data = json.load(f)
            
            # Add ground truth fields
            lms_data['px_norm_x'] = float(row['px_norm_x'])
            lms_data['px_norm_y'] = float(row['px_norm_y'])
            
            # Write back
            with open(json_path, 'w') as f:
                json.dump(lms_data, f, indent=2)
                
        except Exception as e:
            print(f"Error processing {json_path}: {e}")

def merge_sessions(subject, input_session1, input_session2, output_session):
    # Create output directories
    output_frame_dir = f"preprocessed/ios_merged/p{subject}/s{output_session}"
    os.makedirs(output_frame_dir, exist_ok=True)
    
    # Read input CSV files
    csv1_path = f"preprocessed/logs_corrected/p{subject}_{input_session1}_log_corrected.csv"
    csv2_path = f"preprocessed/logs_corrected/p{subject}_{input_session2}_log_corrected.csv"
    
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
        
    # Process first session
    source_frame_dir1 = f"preprocessed/ios/p{subject}/{input_session1}"
    df1_updated, next_frame_num = process_and_copy_frames(
        df1, subject, input_session1, output_session, source_frame_dir1, output_frame_dir, start_frame_num=1
    )
    
    # Process second session
    source_frame_dir2 = f"preprocessed/ios/p{subject}/{input_session2}"
    df2_updated, _ = process_and_copy_frames(
        df2, subject, input_session2, output_session, source_frame_dir2, output_frame_dir, start_frame_num=next_frame_num
    )
    
    # Update session numbers in both dataframes
    df1_updated['session'] = output_session
    df2_updated['session'] = output_session
    
    # Concatenate dataframes
    df_merged = pd.concat([df1_updated, df2_updated], ignore_index=True)
    
    # Save merged CSV    
    output_csv = f"preprocessed/logs_ios/p{subject}_s{output_session}_log_preprocessed.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_merged.to_csv(output_csv, index=False)
    
    # Add ground truth (px_norm_x, px_norm_y) to each landmarks JSON
    add_ground_truth_to_jsons(df_merged, output_frame_dir)
            
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Merge session data for specified subjects')
    parser.add_argument('-s', '--subjects', nargs='+', type=int, 
                       help='List of subject numbers to process')
    args = parser.parse_args()
    subjects = args.subjects if args.subjects else SUBJECTS
    
    # Create base output directory
    os.makedirs('preprocessed', exist_ok=True)

    total_start_time = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting entire merging process...")
    print(f"Processing subjects: {subjects}")

    for subject in subjects:
        for s1, s2, out_s in SESSION_PAIRS:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing p{subject}, sessions {s1} & {s2} -> merged session {out_s}")
            merge_sessions(subject, s1, s2, out_s)

    total_elapsed = time.time() - total_start_time
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Entire merging process completed in {total_elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
