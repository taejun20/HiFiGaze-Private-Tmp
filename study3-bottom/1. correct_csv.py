import pandas as pd
import os
import argparse
import time
from datetime import datetime
from tqdm import tqdm

#SUBJECTS = [25]

SUBJECTS = [1,2,3,4,5,6,7,8,9,10]
SESSIONS = ["1p","1s","2p","2s","3p","3s","4p","4s","5p","5s","6p","6s"]
RAWDATA_DIR = '../../study3-rawdata'

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_csv(input_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Get unique trials
    trials = df['trial'].unique()
    
    # Store indices to remove
    indices_to_remove = []
    
    # For each trial
    for trial in trials:
        # Get first row of the trial
        trial_start = df[df['trial'] == trial].iloc[0]
        start_timestamp = trial_start['VSyncStartTimeOfPreviousUpdate']
        
        # Get indices of rows within 0.5s from trial start
        trial_rows = df[
            (df['trial'] == trial) & 
            (df['VSyncStartTimeOfPreviousUpdate'] >= start_timestamp) & 
            (df['VSyncStartTimeOfPreviousUpdate'] < start_timestamp + 0.5)
        ].index
        
        indices_to_remove.extend(trial_rows)
    
    # Remove rows with frameNum == '-'
    invalid_frame_indices = df[df['frameNum'] == '-'].index
    indices_to_remove.extend(invalid_frame_indices)
    
    # Remove all identified rows
    df_cleaned = df.drop(indices_to_remove)
    
    # Create output directory and filename
    output_dir = f"preprocessed/logs_corrected/"
    os.makedirs(output_dir, exist_ok=True)   
    
    ensure_dir(output_dir)
    output_file = os.path.join(output_dir, os.path.basename(input_file).replace('_data.csv', '_log_corrected.csv'))
    
    # Save the cleaned dataframe
    df_cleaned.to_csv(output_file, index=False)
    #print(f"Processed {input_file} -> {output_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Correct CSV files for specified subjects')
    parser.add_argument('-s', '--subjects', nargs='*', type=int, default=None,
                       help='List of subjects to use for training (e.g., -t 1 2 3). If not specified, uses default TRAIN_SUBJECTS list.')
    args = parser.parse_args()

    # Use provided subjects or defaults
    subjects = args.subjects if args.subjects is not None else SUBJECTS

    total_start_time = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting CSV correction process...")

    # Process all CSV files in rawdata directory
    for subject in tqdm(subjects, desc="Processing subjects"):
        for session in SESSIONS:
            input_file = os.path.join(RAWDATA_DIR, f'p{subject}/p{subject}_{session}_data.csv')
            if os.path.exists(input_file):
                process_csv(input_file)
            else:
                print(f"Warning: File not found - {input_file}")

    total_elapsed = time.time() - total_start_time
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] CSV correction process completed in {total_elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
