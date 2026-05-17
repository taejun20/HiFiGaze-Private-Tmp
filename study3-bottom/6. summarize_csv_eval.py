import pandas as pd
import numpy as np
import os

# CSV file paths
#TYPES = ["RGB", "RGBSOE", "RGBT"]
#TYPES = ["RGBT"]
TYPES = ["RGB", "RGBSOE"]


def calculate_mean_loss_by_subject(csv_path):
    """
    Calculate mean loss for each subject from CSV file
    
    Args:
        csv_path (str): Path to the input CSV file
    Returns:
        DataFrame with mean losses or None if error
    """
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found!")
        return None
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Check if required columns exist
        required_columns = ['subject', 'loss']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return None
        
        # Calculate mean loss for each subject
        subject_mean_loss = df.groupby('subject')['loss'].mean()
        
        # Find all unique subjects and sort them
        all_subjects = sorted(subject_mean_loss.index.tolist())
        
        # Create result with all subjects in order
        result_data = []
        for subject in all_subjects:
            result_data.append(subject_mean_loss[subject])
            
        return pd.Series(result_data, index=all_subjects)
        
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return None

def main():
    # Dictionary to store results for each type
    all_results = {}
    
    # Process each type
    for type_name in TYPES:
        print(f"\nProcessing {type_name}...")
        csv_path = f"model_checkpoints/best/{type_name}_eval_pall.csv"
        result = calculate_mean_loss_by_subject(csv_path)
        
        if result is not None:
            all_results[type_name] = result
    
    if all_results:
        # Create DataFrame with all results
        result_df = pd.DataFrame(all_results)
        
        # Save combined results
        output_path = 'model_checkpoints/best/all_types_mean_loss.csv'
        result_df.to_csv(output_path)
        print(f"\nCombined results saved to {output_path}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print("\nMean error per type:")
        print(result_df.mean())
        print("\nMedian error per type:")
        print(result_df.median())
        print("\nStd error per type:")
        print(result_df.std())

if __name__ == "__main__":
    main()