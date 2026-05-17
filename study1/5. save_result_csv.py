import pandas as pd
import os
import re
from pathlib import Path

# Set variables
TYPE = ["RGB", "RGBT", "RGBH", "RGBSOE", "RGBHSOE", "H", "HSOE", "SOE"]
RUN = "251111_NewVal3"

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

def find_best_csv_for_eval_subject(checkpoint_dir, type_name, eval_subject):
    """
    Find the CSV file with the lowest ValError for a specific TYPE and eval subject.
    
    Args:
        checkpoint_dir: Directory containing CSV files
        type_name: One of TYPE values (RGB, RGBSOE, RGBT)
        eval_subject: Subject number as string (e.g., '1', '19')
    
    Returns:
        Path to best CSV file or None
    """
    pattern = f"Eval_{type_name}_*_eval_p*{eval_subject}*_Epoch*_ValError*.csv"
    
    # Find all matching CSV files
    matching_files = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith(f"Eval_{type_name}_") and file.endswith('.csv'):
            # Check if this file is for the eval subject
            eval_subjects = extract_eval_subjects_from_filename(file)
            if eval_subject in eval_subjects:
                val_error = extract_val_error_from_filename(file)
                if val_error is not None:
                    matching_files.append((file, val_error))
    
    if not matching_files:
        return None
    
    # Find file with lowest ValError
    best_file = min(matching_files, key=lambda x: x[1])
    return os.path.join(checkpoint_dir, best_file[0])

def get_all_eval_subjects(checkpoint_dir, type_name):
    """
    Get all unique eval subjects from CSV filenames for a given TYPE.
    
    Returns:
        Sorted list of subject numbers as strings
    """
    eval_subjects = set()
    
    for file in os.listdir(checkpoint_dir):
        if file.startswith(f"Eval_{type_name}_") and file.endswith('.csv'):
            subjects = extract_eval_subjects_from_filename(file)
            eval_subjects.update(subjects)
    
    # Convert to list, sort numerically
    subject_list = sorted([int(s) for s in eval_subjects])
    return [str(s) for s in subject_list]

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
    
    # Get all eval subjects (use first TYPE to get subjects, they should be similar)
    all_eval_subjects = get_all_eval_subjects(CHECKPOINT_DIR, TYPE[0])
    print(f"Found eval subjects: {all_eval_subjects}")
    
    # Create result dictionary
    results = []
    
    # Process each eval subject
    for eval_subject in all_eval_subjects:
        row = {'subject': f'p{eval_subject}'}
        
        # For each TYPE, find best CSV and calculate average loss
        for type_name in TYPE:
            best_csv = find_best_csv_for_eval_subject(CHECKPOINT_DIR, type_name, eval_subject)
            
            if best_csv:
                avg_loss = calculate_average_loss(best_csv)
                row[type_name] = avg_loss
                print(f"  {type_name} - p{eval_subject}: {avg_loss:.6f} (from {os.path.basename(best_csv)})")
            else:
                row[type_name] = None
                print(f"  {type_name} - p{eval_subject}: No matching CSV found")
        
        results.append(row)
    
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

