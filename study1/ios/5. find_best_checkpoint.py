import os
import re
import pandas as pd
from pathlib import Path

MODELS = ["RGB_v2_nolmk","RGB_v2_nolmk_lr3e-4"]
VAL_SUBJECTS = [1, 11, 13, 15, 26]
CHECKPOINT_DIR = "model_checkpoints/251215_IOS/"

def extract_val_loss_from_filename(filename):
    """Extract ValLoss value from filename like 'Eval_..._Epoch14_ValLoss2.4456.csv'"""
    match = re.search(r'_ValLoss([\d.]+)\.csv', filename)
    if match:
        return float(match.group(1))
    return None

def extract_epoch_from_filename(filename):
    """Extract epoch number from filename like 'Eval_..._Epoch14_ValLoss2.4456.csv'"""
    match = re.search(r'_Epoch(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

# Find best CSV for each model
best_csvs = {}
best_epochs = {}

for model in MODELS:
    pattern = f"Eval_{model}_train"
    checkpoint_path = Path(CHECKPOINT_DIR)
    
    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint directory {CHECKPOINT_DIR} does not exist")
        continue
    
    # Find all CSV files starting with the pattern
    csv_files = list(checkpoint_path.glob(f"{pattern}*.csv"))
    
    if not csv_files:
        print(f"Warning: No CSV files found for model {model}")
        continue
    
    # Find the one with lowest ValLoss
    best_file = None
    best_val_loss = float('inf')
    best_epoch = None
    
    for csv_file in csv_files:
        val_loss = extract_val_loss_from_filename(csv_file.name)
        epoch = extract_epoch_from_filename(csv_file.name)
        
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_file = csv_file
            best_epoch = epoch
    
    if best_file:
        best_csvs[model] = best_file
        best_epochs[model] = best_epoch
        print(f"Model {model}: Best checkpoint is {best_file.name} (Epoch {best_epoch}, ValLoss {best_val_loss:.4f})")
    else:
        print(f"Warning: Could not find valid checkpoint for model {model}")

# Calculate mean loss per subject for each model
results = {}
for model in MODELS:
    if model not in best_csvs:
        continue
    
    df = pd.read_csv(best_csvs[model])
    
    # Calculate mean loss per subject
    subject_losses = {}
    for subject in VAL_SUBJECTS:
        subject_data = df[df['subject'] == subject]
        if len(subject_data) > 0:
            mean_loss = subject_data['loss'].mean()
            subject_losses[subject] = mean_loss
        else:
            subject_losses[subject] = None
    
    results[model] = subject_losses

# Create result CSV
# Header: subject, MODELS[0], MODELS[1], ...
header = ['subject'] + MODELS

# First data row: epoch, epoch1, epoch2, ...
epoch_row = ['epoch'] + [str(best_epochs.get(model, '')) for model in MODELS]

# Subsequent rows: p1, mean_loss1, mean_loss2, ...
data_rows = [epoch_row]
for subject in VAL_SUBJECTS:
    row = [f"p{subject}"]
    for model in MODELS:
        if model in results and subject in results[model]:
            loss = results[model][subject]
            row.append(f"{loss:.4f}" if loss is not None else "")
        else:
            row.append("")
    data_rows.append(row)

# Save to CSV
result_df = pd.DataFrame(data_rows, columns=header)
result_df.to_csv("result.csv", index=False)

print(f"\nResult saved to result.csv")
print(result_df.to_string())

