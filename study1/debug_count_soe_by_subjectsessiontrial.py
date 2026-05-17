import json
import os
import csv
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

SUBJECTS = [1,2,3,4,5,6,7,8,9,10,11]
SESSIONS = [1,2,3,4,5,6]

# Initialize results list
results = []

# Process subjects p1 to p11
for subject_num in tqdm(SUBJECTS, desc="Processing subjects"):
    subject = f"p{subject_num}"
    
    print(f"Processing {subject}...")
    
    for session_num in SESSIONS:
        session = f"s{session_num}"
        
        print(f"  Processing {session}...")
        
        # Construct log file path
        logs_path = "preprocessed/logs"
        log_file = f"{subject}_{session}_log_preprocessed.csv"
        log_path = os.path.join(logs_path, log_file)
        
        # Check if log file exists
        if not os.path.exists(log_path):
            print(f"    Warning: Log file not found: {log_file}")
            continue
            
        # Read the log file
        try:
            df = pd.read_csv(log_path)
        except Exception as e:
            print(f"    Error reading {log_file}: {e}")
            continue
        
        # Group by trial
        trials = df['trial'].unique()
        
        for trial in sorted(trials):
            trial_data = df[df['trial'] == trial]
            soe_count = 0
            non_soe_count = 0
            
            print(f"    Processing trial {trial} ({len(trial_data)} frames)...")
            
            # Process each frame in this trial
            for _, row in trial_data.iterrows():
                frame_num = str(int(row['frameNum'])).zfill(5)
                
                # Construct the landmark file path
                landmark_file = f"{frame_num}_landmark_and_gt.json"
                landmark_path = f"preprocessed/input_landmark_and_gt/{subject}/{session}/{landmark_file}"
                
                # Check if landmark file exists and read it
                if os.path.exists(landmark_path):
                    try:
                        with open(landmark_path, 'r') as f:
                            landmark_data = json.load(f)
                        
                        # Check if either eye has SoE data (not -999)
                        l_soe = landmark_data.get('l_irisbox_center_to_soe_center_x', -999)
                        r_soe = landmark_data.get('r_irisbox_center_to_soe_center_x', -999)
                        
                        # If either eye has valid SoE data (not -999), count as SoE
                        if l_soe != -999 or r_soe != -999:
                            soe_count += 1
                        else:
                            non_soe_count += 1
                            
                    except (json.JSONDecodeError, FileNotFoundError) as e:
                        print(f"      Error reading {landmark_file}: {e}")
                        continue
                else:
                    print(f"      Warning: Landmark file not found: {landmark_path}")
            
            # Store results for this trial
            results.append({
                'subject': subject,
                'session': session,
                'trial': trial,
                'soe_count': soe_count,
                'non_soe_count': non_soe_count
            })
            
            print(f"      Trial {trial}: SoE={soe_count}, non-SoE={non_soe_count}")

# Create CSV file
csv_filename = 'soe_count_by_trial.csv'
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['subject', 'session', 'trial', 'soe_count', 'non_soe_count']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write header
    writer.writeheader()
    
    # Write data
    for result in results:
        writer.writerow(result)

print(f"\nResults saved to {csv_filename}")
print(f"Total rows written: {len(results)}")

# Print summary statistics
print("\nSummary by subject:")
subject_summary = defaultdict(lambda: {'total_soe': 0, 'total_non_soe': 0, 'total_trials': 0})

for result in results:
    subject = result['subject']
    subject_summary[subject]['total_soe'] += result['soe_count']
    subject_summary[subject]['total_non_soe'] += result['non_soe_count']
    subject_summary[subject]['total_trials'] += 1

for subject in sorted(subject_summary.keys()):
    stats = subject_summary[subject]
    total_frames = stats['total_soe'] + stats['total_non_soe']
    print(f"{subject}: {stats['total_trials']} trials, {total_frames} frames total (SoE: {stats['total_soe']}, non-SoE: {stats['total_non_soe']})")