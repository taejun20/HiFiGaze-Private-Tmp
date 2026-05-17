import json
import os
import csv
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

SUBJECTS = [1,2,3,4,5,6,7,8,9,10,11]
SESSIONS = [1,2,3,4,5,6]

# Initialize results dictionary for aggregating by subject-trial combination
subject_trial_aggregated = defaultdict(lambda: {'soe_counts': [], 'non_soe_counts': [], 'total_sessions': 0})

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
            
            # Aggregate results by subject-trial combination (across sessions)
            key = (subject, trial)
            subject_trial_aggregated[key]['soe_counts'].append(soe_count)
            subject_trial_aggregated[key]['non_soe_counts'].append(non_soe_count)
            subject_trial_aggregated[key]['total_sessions'] += 1
            
            print(f"      Trial {trial}: SoE={soe_count}, non-SoE={non_soe_count}")

# Calculate aggregated statistics by subject-trial combination
aggregated_results = []
for (subject, trial), data in subject_trial_aggregated.items():
    total_sessions = data['total_sessions']
    total_soe = sum(data['soe_counts'])
    total_non_soe = sum(data['non_soe_counts'])
    avg_soe = total_soe / total_sessions if total_sessions > 0 else 0
    avg_non_soe = total_non_soe / total_sessions if total_sessions > 0 else 0
    
    aggregated_results.append({
        'subject': subject,
        'trial': trial,
        'total_sessions': total_sessions,
        'total_soe_count': total_soe,
        'total_non_soe_count': total_non_soe,
        'avg_soe_count': round(avg_soe, 2),
        'avg_non_soe_count': round(avg_non_soe, 2)
    })

# Sort results by subject and trial
aggregated_results.sort(key=lambda x: (x['subject'], x['trial']))

# Create CSV file with aggregated results
csv_filename = 'soe_count_by_subject_trial_aggregated.csv'
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['subject', 'trial', 'total_sessions', 'total_soe_count', 'total_non_soe_count', 'avg_soe_count', 'avg_non_soe_count']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write header
    writer.writeheader()
    
    # Write aggregated data
    for result in aggregated_results:
        writer.writerow(result)

print(f"\nAggregated results saved to {csv_filename}")
print(f"Total subject-trial combinations: {len(aggregated_results)}")

# Print summary statistics
print("\nSummary by subject-trial combination:")
for result in aggregated_results:
    subject = result['subject']
    trial = result['trial']
    sessions = result['total_sessions']
    total_soe = result['total_soe_count']
    total_non_soe = result['total_non_soe_count']
    avg_soe = result['avg_soe_count']
    avg_non_soe = result['avg_non_soe_count']
    total_frames = total_soe + total_non_soe
    
    print(f"{subject} Trial {trial}: {sessions} sessions, {total_frames} total frames")
    print(f"  Total: SoE={total_soe}, non-SoE={total_non_soe}")
    print(f"  Average per session: SoE={avg_soe}, non-SoE={avg_non_soe}")
    print()