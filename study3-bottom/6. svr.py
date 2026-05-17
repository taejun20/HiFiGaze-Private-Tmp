import numpy as np
from sklearn.svm import SVR
import pandas as pd
import os
from tqdm import tqdm
import argparse
import pickle

#SUBJECTS = [1]

TYPES = ["RGB", "RGBH", "RGBHSOE", "RGBSOE", "H", "HSOE", "SOE", "RGBT"]
SUBJECTS = [1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26]

def load_train_data(subject, type):
    """
    Load training data from RGB_CAL evaluation results
    """
    csv_path = f"model_checkpoints/best/1-1/{type}_CAL_eval_pall.csv"
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found!")
        return None
        
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Filter for the specific subject
    subject_df = df[df['subject'] == subject]
    
    if len(subject_df) == 0:
        print(f"No training data found for subject {subject}")
        return None
    
    # Extract features (predictions) and targets (ground truth)
    X = subject_df[['pred_x', 'pred_y']].values
    X = X / 14.4
    y = subject_df[['gt_x', 'gt_y']].values
    y = y / 14.4
    
    print(f"Loaded {len(X)} training samples for subject {subject}")
    return X, y

def load_eval_data(subject, type):
    """
    Load evaluation data from RGB evaluation results
    """
    csv_path = f"model_checkpoints/best/1-1/{type}_eval_pall.csv"
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found!")
        return None
        
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Filter for the specific subject
    subject_df = df[df['subject'] == subject]
    
    if len(subject_df) == 0:
        print(f"No evaluation data found for subject {subject}")
        return None
    
    # Extract features (predictions) and targets (ground truth)
    X = subject_df[['pred_x', 'pred_y']].values
    X = X / 14.4
    y = subject_df[['gt_x', 'gt_y']].values
    y = y / 14.4
    
    print(f"Loaded {len(X)} evaluation samples for subject {subject}")
    return X, y

def train_and_save(subject, type):
    """
    Train SVR models for x and y coordinates and save them
    """
    # Load training data
    train_data = load_train_data(subject, type)
    if train_data is None:
        return None
    X_train, y_train = train_data
    
    # Load evaluation data
    eval_data = load_eval_data(subject, type)
    if eval_data is None:
        return None
    X_eval, y_eval = eval_data
    
    # Create and train SVR models for x and y coordinates
    svr_x = SVR(kernel='rbf', C=20.0, gamma=0.06)
    svr_y = SVR(kernel='rbf', C=20.0, gamma=0.06)
    
    print(f"Training SVR model for subject {subject}...")
    print("Training X coordinate model...")
    svr_x.fit(X_train, y_train[:, 0])
    print("Training Y coordinate model...")
    svr_y.fit(X_train, y_train[:, 1])
    
    # Make predictions on evaluation data
    print("Making predictions...")
    pred_x_cm = svr_x.predict(X_eval) * 14.4
    pred_y_cm = svr_y.predict(X_eval) * 14.4
    predictions = np.column_stack((pred_x_cm, pred_y_cm))
    
    gt_cm = np.column_stack((
        y_eval[:, 0] * 14.4,
        y_eval[:, 1] * 14.4
    ))
    
    # Calculate error metrics
    errors = np.linalg.norm(predictions - gt_cm, axis=1)
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    std_error = np.std(errors)
    
    print(f"\nResults for subject {subject}:")
    print(f"Mean Error: {mean_error:.4f} cm")
    print(f"Median Error: {median_error:.4f} cm")
    print(f"Std Error: {std_error:.4f} cm")
    
    # Save results
    save_dir = "model_checkpoints/best/1-3"
    os.makedirs(save_dir, exist_ok=True)
    
    # Create DataFrame with results
    results_df = pd.DataFrame({
        'subject': [subject] * len(X_eval),
        'pred_x': pred_x_cm,
        'pred_y': pred_y_cm,
        'gt_x': y_eval[:, 0],
        'gt_y': y_eval[:, 1],
        'loss': errors,
    })
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description=f'Train SVR models using {TYPES} penultimate values')
    parser.add_argument('-s', '--subjects', nargs='*', type=int, default=None,
                       help='List of subjects to process (e.g., -s 1 2 3). If not specified, uses default SUBJECTS list.')
    args = parser.parse_args()
    
    # Use provided subjects or defaults
    subjects_to_process = args.subjects if args.subjects is not None else SUBJECTS
    
    print(f"Processing subjects: {subjects_to_process}")
    
    # Process each subject and collect results
    for type in TYPES:
        all_results = []
        for subject in subjects_to_process:
            print(f"\nProcessing subject {subject}")
            results_df = train_and_save(subject, type)
            if results_df is not None:
                all_results.append(results_df)
        
        if all_results:
            # Combine all results
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # Save aggregated results
            save_dir = "model_checkpoints/best/1-3"
            aggregated_path = os.path.join(save_dir, f"{type}_pall_SVR_eval.csv")
            combined_df.to_csv(aggregated_path, index=False)
            print(f"\nAggregated results saved to {aggregated_path}")

if __name__ == "__main__":
    main()