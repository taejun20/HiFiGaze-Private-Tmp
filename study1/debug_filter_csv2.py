import pandas as pd
from pathlib import Path

# Constants
SUBJECTS = [1,2,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26]  # Array of subjects
SESSIONS = [1,2,3,4,5,6]   # Array of sessions
TIME_THRESHOLD = 0.2  # seconds

# Base path
base_path = Path("../study1-rawdata")

def process_csv(csv_path):
    """
    Process a CSV file to find frames with large time intervals within the same trial
    Args:
        csv_path: Path to the CSV file
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize variables
    prev_time = None
    prev_trial = None
    large_interval_frames = []
    
    # Process each row
    for _, row in df.iterrows():
        curr_time = float(row['frameEstimatedReadoutStartTime'])
        curr_trial = int(row['trial'])
        
        if curr_time == -1:
            continue

        # If this is a new trial or first row
        if curr_trial != prev_trial:
            prev_time = curr_time
            prev_trial = curr_trial
            continue

        frame_num = int(row['frameNum'])

        # Check time interval within same trial
        if prev_time is not None:
            time_diff = curr_time - prev_time
            #print(f"Time diff: {time_diff}")
            if time_diff > TIME_THRESHOLD:
                large_interval_frames.append(frame_num)
        
        # Update previous values
        prev_time = curr_time
        prev_trial = curr_trial
    
    return large_interval_frames

def process_subject_session(subject, session):
    """
    Process both primary and secondary CSV files for a given subject and session
    Args:
        subject (int): Subject number
        session (int): Session number
    """
    # Define CSV paths for this subject and session
    primary_csv = base_path / f"p{subject}" / f"p{subject}_{session}p_data.csv"
    secondary_csv = base_path / f"p{subject}" / f"p{subject}_{session}s_data.csv"
    
    # Process primary CSV
    print(f"\nProcessing primary CSV (p{subject}_{session}p_data.csv):")
    if primary_csv.exists():
        primary_frames = process_csv(primary_csv)
        if primary_frames:
            print("Frames with large time intervals:")
            for frame in primary_frames:
                print(f"Frame, subject: {subject}, session: {session}p, frame: {frame}")
        # else:
        #     print("No frames with large time intervals found.")
    else:
        print(f"Error: Primary CSV file not found at {primary_csv}")

    # Process secondary CSV
    print(f"\nProcessing secondary CSV (p{subject}_{session}s_data.csv):")
    if secondary_csv.exists():
        secondary_frames = process_csv(secondary_csv)
        if secondary_frames:
            print("Frames with large time intervals:")
            for frame in secondary_frames:
                print(f"Frame, subject: {subject}, session: {session}s, frame: {frame}")
        # else:
        #     print("No frames with large time intervals found.")
    else:
        print(f"Error: Secondary CSV file not found at {secondary_csv}")

def main():
    # Process each subject and session combination
    for subject in SUBJECTS:
        for session in SESSIONS:
            print(f"\n{'='*50}")
            print(f"Processing Subject {subject}, Session {session}")
            print(f"{'='*50}")
            process_subject_session(subject, session)

if __name__ == "__main__":
    main()