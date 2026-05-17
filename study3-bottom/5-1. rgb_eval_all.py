import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import os
import json
import cv2
import torchvision.transforms as transforms
import pandas as pd
from tqdm import tqdm
import warnings
import glob
warnings.filterwarnings("ignore", message=".*You are using `torch.load` with `weights_only=False`.*", category=FutureWarning)
from GazeModelRGB import GazeModelRGB

# Constants
BEST_CHECKPOINTS_DIR = "model_checkpoints/best"
SESSIONS = [1,2,3,4,5,6]
BATCH_SIZE = 2

class GazePTDataset(Dataset):
    def __init__(self, subjects):
        self.data_samples = []
        for subject in subjects:
            for session in SESSIONS:
                eyepatch_dir = Path(f"preprocessed/input_eyepatch/p{subject}/s{session}")
                landmark_dir = Path(f"preprocessed/input_landmark_and_gt/p{subject}/s{session}")
                
                # Find all left eye PNG files
                left_files = sorted(eyepatch_dir.glob("*_left.png"))
                
                for left_eye_file in left_files:
                    # Get frame number from filename
                    frame_num = left_eye_file.stem.replace("_left", "")
                    right_eye_file = eyepatch_dir / f"{frame_num}_right.png"
                    json_file = landmark_dir / f"{frame_num}_landmark_and_gt.json"
                    
                    # Check if corresponding right eye and JSON files exist
                    if right_eye_file.exists() and json_file.exists():
                        self.data_samples.append({
                            'subject': subject,
                            'session': session,
                            'frame_num': frame_num,
                            'left_eye_path': left_eye_file,
                            'right_eye_path': right_eye_file,
                            'json_path': json_file
                        })
             
        assert len(self.data_samples) > 0, f"No data samples found for subjects {subjects}"
        print(f"Loaded {len(self.data_samples)} data samples from {len(subjects)} subjects")
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((250, 500)),  # (height, width)
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        
        # Load left and right eye images
        l_eye_img = cv2.imread(str(sample['left_eye_path']))
        r_eye_img = cv2.imread(str(sample['right_eye_path']))
        
        # Convert BGR to RGB
        l_eye_img = cv2.cvtColor(l_eye_img, cv2.COLOR_BGR2RGB)
        r_eye_img = cv2.cvtColor(r_eye_img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        l_eye = self.transform(l_eye_img)
        r_eye = self.transform(r_eye_img)
        
        # Load JSON data
        with open(sample['json_path'], 'r') as f:
            json_data = json.load(f)
        
        # Extract eye corner landmarks
        eyecorner_lmk = torch.tensor([
            json_data["l_eye_inner_corner_x"],
            json_data["l_eye_inner_corner_y"],
            json_data["l_eye_outer_corner_x"],
            json_data["l_eye_outer_corner_y"],
            json_data["r_eye_inner_corner_x"],
            json_data["r_eye_inner_corner_y"],
            json_data["r_eye_outer_corner_x"],
            json_data["r_eye_outer_corner_y"]
        ], dtype=torch.float32)
        
        # Extract ground truth
        gt = torch.tensor([
            json_data["px_norm_x"],
            json_data["px_norm_y"]
        ], dtype=torch.float32)
        
        return {
            'subject': sample['subject'],
            'session': sample['session'],
            'frame_num': sample['frame_num'],
            'l_eye': l_eye,
            'r_eye': r_eye,
            'eyecorner_lmk': eyecorner_lmk,
            'gt': gt
        }

def euclidean_loss(pred, target):
    return torch.norm(pred - target, dim=1)

def find_rgb_checkpoints():
    """Find all RGB checkpoints in the best directory"""
    pattern = os.path.join(BEST_CHECKPOINTS_DIR, "RGB_*.pt")
    checkpoints = glob.glob(pattern)
    return sorted(checkpoints)

def extract_eval_subjects_from_checkpoint(checkpoint_path):
    """Extract evaluation subjects from checkpoint filename"""
    filename = os.path.basename(checkpoint_path)
    if 'eval_p' in filename:
        eval_part = filename.split('eval_p')[1].split('_')[0]
        return [int(x) for x in eval_part.split(',')]
    else:
        print(f"!!! ERROR !!!: No eval_p in {filename}")
        return None

def evaluate_checkpoint(checkpoint_path, device):
    """Evaluate a single checkpoint"""
    print(f"\nEvaluating checkpoint: {os.path.basename(checkpoint_path)}")
    
    # Extract evaluation subjects from checkpoint filename
    eval_subjects = extract_eval_subjects_from_checkpoint(checkpoint_path)
    print(f"Evaluation subjects: {eval_subjects}")
    
    # Load model
    model = GazeModelRGB()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Create dataset and dataloader
    dataset = GazePTDataset(eval_subjects)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=6, pin_memory=True)

    # Lists to store results
    all_results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", position=1):
            # Move data to device
            l_eye = batch['l_eye'].to(device)
            r_eye = batch['r_eye'].to(device)
            eyecorner_lmk = batch['eyecorner_lmk'].to(device)
            gt = batch['gt'].to(device)

            # Forward pass
            pred = model(l_eye, r_eye, eyecorner_lmk)

            # Scale predictions and ground truth to cm
            pred_cm = torch.stack([
                pred[:, 0] * (7.1000/0.430),
                pred[:, 1] * (14.3915/0.8716)
            ], dim=1)
            gt_cm = torch.stack([
                gt[:, 0] * (7.1000/0.430),
                gt[:, 1] * (14.3915/0.8716)
            ], dim=1)

            # Calculate losses
            losses = euclidean_loss(pred_cm, gt_cm)

            # Store results
            for i in range(len(batch['subject'])):
                all_results.append({
                    'checkpoint': os.path.basename(checkpoint_path),
                    'subject': int(batch['subject'][i]),
                    'session': int(batch['session'][i]),
                    'frame_num': batch['frame_num'][i],
                    'pred_x': pred_cm[i, 0].item(),
                    'pred_y': pred_cm[i, 1].item(),
                    'gt_x': gt_cm[i, 0].item(),
                    'gt_y': gt_cm[i, 1].item(),
                    'loss': losses[i].item()
                })

    return all_results

def main():
    # Set device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Find all RGB checkpoints
    rgb_checkpoints = find_rgb_checkpoints()
    print(f"Found {len(rgb_checkpoints)} RGB checkpoints:")
    for cp in rgb_checkpoints:
        print(f"  - {os.path.basename(cp)}")

    if not rgb_checkpoints:
        print("No RGB checkpoints found!")
        return

    # Evaluate each checkpoint
    all_aggregated_results = []
    
    for checkpoint_path in rgb_checkpoints:
        try:
            results = evaluate_checkpoint(checkpoint_path, device)
            all_aggregated_results.extend(results)            
        except Exception as e:
            print(f"Error evaluating {checkpoint_path}: {e}")
            continue

    if not all_aggregated_results:
        print("No results to save!")
        return

    # Convert to DataFrame and sort by loss
    df = pd.DataFrame(all_aggregated_results)
    df = df.sort_values('loss', ascending=False)

    # Calculate and print overall statistics
    print(f"\n=== OVERALL STATISTICS ===")
    print(f"Total samples evaluated: {len(df)}")
    print(f"Overall mean loss: {df['loss'].mean():.2f} cm")    

    # Calculate and print mean loss per subject
    print("\nMean loss by subject:")
    subject_means = df.groupby('subject')['loss'].mean().sort_values()
    for subject, mean in subject_means.items():
        print(f"Subject {subject}: {mean:.2f} cm")

    # Save to CSV
    csv_path = f"{BEST_CHECKPOINTS_DIR}/RGB_eval_pall.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nAll results saved to {csv_path}")

if __name__ == "__main__":
    main()
