import pandas as pd
import os
import re
from pathlib import Path

# Set variables
TYPE = ["RGB_FF", "RGB_HA", "RGB_FFHA"]
RUN = "251119_FullFaceHeadAugment"

# Fixed five-fold eval subject groups
EVAL_GROUPS = [
    ["21", "22", "24", "25", "26"],
    ["17", "18", "19", "20"],
    ["13", "14", "15", "16"],
    ["9", "10", "11", "12"],
    ["1", "2", "6", "7", "8"],
]

# Directory containing CSV files
CHECKPOINT_DIR = f"model_checkpoints/{RUN}"

def extract_eval_subjects_from_filename(filename):
    """
    Extract eval subjects from filename like:
    Eval_RGB_train_p20,21,22,24,25,26,1,2,6,7,8,9,10,11,12,13,14,15,16_val_p17,18_eval_p19_Epoch16_ValError1.4053.csv
    
    Returns list of subject numbers like ['19'] or ['1', '2', '6']
    """
    # Match eval_p followed by numbers
    match = re.search(r'eval_p([\d,]+)', filename)
    if match:
        subjects_str = match.group(1)
        # Split by comma and return list
        return [s.strip() for s in subjects_str.split(',')]
    return []

def extract_val_error_from_filename(filename):
    """
    Extract ValError value from filename like:
    Eval_RGB_train_..._Epoch16_ValError1.4053.csv
    
    Returns float value or None
    """
    # Match ValError followed by digits and dots
    match = re.search(r'ValError([\d.]+)', filename)
    if match:
        val_str = match.group(1)
        # Remove trailing periods (they might be captured from filename extension)
        val_str = val_str.rstrip('.')
        # Ensure it's a valid number string
        if val_str and (val_str.replace('.', '', 1).isdigit()):
            try:
                return float(val_str)
            except ValueError:
                return None
    return None

def find_best_csv_for_group(checkpoint_dir, type_name, eval_group):
    """
    Find CSV with lowest ValError for a TYPE and exact eval subject group.
    """
    target_set = set(eval_group)
    target_len = len(eval_group)
    matching_files = []
    
    for file in os.listdir(checkpoint_dir):
        if not (file.startswith(f"Eval_{type_name}_") and file.endswith('.csv')):
            continue
        subjects = extract_eval_subjects_from_filename(file)
        if len(subjects) != target_len:
            continue
        if set(subjects) == target_set:
            val_error = extract_val_error_from_filename(file)
            if val_error is not None:
                matching_files.append((file, val_error))
    
    if not matching_files:
        return None
    
    best_file = min(matching_files, key=lambda x: x[1])
    return os.path.join(checkpoint_dir, best_file[0])

def calculate_average_loss(csv_path):
    """
    Read CSV file and calculate average of 'loss' column.
    The CSV file already contains only the eval subjects, so we average all losses.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        Average loss value or None if error
    """
    try:
        df = pd.read_csv(csv_path)
        if 'loss' not in df.columns:
            print(f"Warning: 'loss' column not found in {csv_path}")
            return None
        return df['loss'].mean()
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def main():
    # Check if checkpoint directory exists
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"Error: Directory {CHECKPOINT_DIR} does not exist!")
        return
    
    print(f"Using predefined eval groups: {EVAL_GROUPS}")
    
    unique_subjects = sorted({int(s) for group in EVAL_GROUPS for s in group})
    subject_results = {subj: {'subject': f'p{subj}'} for subj in unique_subjects}
    
    for idx, eval_group in enumerate(EVAL_GROUPS, start=1):
        group_label = f"group{idx}_p{','.join(eval_group)}"
        for type_name in TYPE:
            best_csv = find_best_csv_for_group(CHECKPOINT_DIR, type_name, eval_group)
            if not best_csv:
                print(f"{group_label} | {type_name}: No matching CSV found")
                continue
            
            try:
                df = pd.read_csv(best_csv)
            except Exception as e:
                print(f"Failed to read {best_csv}: {e}")
                continue
            
            if 'subject' not in df.columns or 'loss' not in df.columns:
                print(f"Required columns missing in {best_csv}, skipping.")
                continue
            
            for subject in eval_group:
                subject_int = int(subject)
                subject_rows = df[df['subject'] == subject_int]
                if subject_rows.empty:
                    print(f"{group_label} | {type_name}: subject p{subject} not found in {os.path.basename(best_csv)}")
                    continue
                avg_loss = subject_rows['loss'].mean()
                subject_results[subject_int][type_name] = avg_loss
                print(f"{group_label} | {type_name} | p{subject}: {avg_loss:.6f} ({os.path.basename(best_csv)})")
    
    results = [subject_results[subj] for subj in unique_subjects]
    
    # Create DataFrame
    result_df = pd.DataFrame(results)
    
    # Save to CSV
    output_path = f"{RUN}_result_summary.csv"
    result_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\nSummary:")
    print(result_df.to_string(index=False))

if __name__ == "__main__":
    main()

