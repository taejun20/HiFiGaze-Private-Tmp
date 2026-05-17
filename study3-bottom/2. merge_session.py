import pandas as pd
import os
import shutil
from tqdm import tqdm
import time
from datetime import datetime
import argparse

#SUBJECTS = [25]

SUBJECTS = [1,2,3,4,5,6,7,8,9,10]
SESSION_PAIRS = [
    ("1p", "1s", "1"),  # (session1, session2, output_session)
    ("2p", "2s", "2"),
    ("3p", "3s", "3"),
    ("4p", "4s", "4"),
    ("5p", "5s", "5"),
    ("6p", "6s", "6"),
]

def process_and_copy_frames(df, source_session_dir, target_session_dir, start_frame_num=1):    
    # Create a copy of the dataframe
    df_updated = df.copy()
    current_frame_num = start_frame_num
    
    # Create mapping of old frame numbers to new ones
    frame_mapping = {}
    
    # Get all frame numbers in order
    frame_numbers = sorted([int(x) for x in df['frameNum'].unique()])
    
    for old_num in tqdm(frame_numbers, desc="Processing frames"):
        # Map old frame number to new sequential number
        old_frame = str(old_num).zfill(5)
        new_frame = str(current_frame_num).zfill(5)
        frame_mapping[old_frame] = new_frame
        
        # Copy and rename frame file
        source_file = f"{source_session_dir}/{old_frame}.jpg"
        target_file = f"{target_session_dir}/{new_frame}.jpg"
        
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)
        else:
            print(f"Warning: Source frame not found - {source_file}")
        
        current_frame_num += 1
    
    # Update frameNum in dataframe
    df_updated['frameNum'] = df_updated['frameNum'].apply(lambda x: frame_mapping[str(int(x)).zfill(5)])
        
    return df_updated, current_frame_num

def copy_backgrounds(subject, input_session1, input_session2, output_session):    
    output_bg_dir = f"preprocessed/backgrounds/p{subject}/s{output_session}"
    os.makedirs(output_bg_dir, exist_ok=True)   
    
    # Copy backgrounds from both sessions
    for input_session in tqdm([input_session1, input_session2], desc="Copying backgrounds"):
        source_bg_dir = f"../../backgrounds-study3/p{subject}/{input_session}"
        
        if os.path.exists(source_bg_dir):
            # Copy all jpg files from source to target
            for file in os.listdir(source_bg_dir):
                if file.endswith('.jpg'):
                    source_file = os.path.join(source_bg_dir, file)
                    target_file = os.path.join(output_bg_dir, file)
                    shutil.copy2(source_file, target_file)
        else:
            print(f"Warning: Background directory not found - {source_bg_dir}")
    
def merge_sessions(subject, input_session1, input_session2, output_session):
    # Create output directories
    output_frame_dir = f"preprocessed/frames/p{subject}/s{output_session}"
    os.makedirs(output_frame_dir, exist_ok=True)
    
    # Read input CSV files
    csv1_path = f"preprocessed/logs_corrected/p{subject}_{input_session1}_log_corrected.csv"
    csv2_path = f"preprocessed/logs_corrected/p{subject}_{input_session2}_log_corrected.csv"
    
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
        
    # Process first session
    source_frame_dir1 = f"../../study3-rawdata/p{subject}/{input_session1}"
    df1_updated, next_frame_num = process_and_copy_frames(df1, source_frame_dir1, output_frame_dir, start_frame_num=1)
    
    # Process second session
    source_frame_dir2 = f"../../study3-rawdata/p{subject}/{input_session2}"
    df2_updated, _ = process_and_copy_frames(df2, source_frame_dir2, output_frame_dir, start_frame_num=next_frame_num)
    
    # Update session numbers in both dataframes
    df1_updated['session'] = output_session
    df2_updated['session'] = output_session
    
    # Concatenate dataframes
    df_merged = pd.concat([df1_updated, df2_updated], ignore_index=True)
    
    # Save merged CSV    
    output_csv = f"preprocessed/logs/p{subject}_s{output_session}_log_preprocessed.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_merged.to_csv(output_csv, index=False)
        
    # Copy background images
    copy_backgrounds(subject, input_session1, input_session2, output_session)
            
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
