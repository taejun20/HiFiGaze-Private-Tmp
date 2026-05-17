import sys
import os
_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _ROOT)
sys.path.append(os.path.abspath(os.path.join(_ROOT, '..')))
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
import random
import wandb
from models.GazeModel_LMK import GazeModel_LMK
import warnings
warnings.filterwarnings('ignore', message=".*You are using `torch.load` with `weights_only=False`.*", category=FutureWarning)
import torch.serialization
import cv2
import json
import torchvision.transforms as transforms
import argparse

SUBJECT = [2,6,7,8,10,11,12,13,15,16,19,25] # 12 subjects. Excluding the 10 participants who have also participated in study 2.

WandB_PROJECT = "260507_highres_pretrain"
TYPE = "LMK"

# Make it reproducible
seed = 2026
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
model_init_generator = torch.Generator()
model_init_generator.manual_seed(seed)
def seed_worker(worker_id):    
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

# Hyper-parameters
BATCH_SIZE = 30
NUM_EPOCHS = 12
GLOBAL_LR = 3e-5
WEIGHT_DECAY = 1e-5

# Create model_checkpoints directory
os.makedirs("model_checkpoints", exist_ok=True)
CHECKPOINT_DIR = f"model_checkpoints/{WandB_PROJECT}"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# WandB init
WANDB_CONFIG = {
    'batch_size': BATCH_SIZE,
    'num_epochs': NUM_EPOCHS,
    'learning_rate': GLOBAL_LR,  # Single learning rate for all parameters
    'weight_decay': WEIGHT_DECAY,
}

# Mediapipe indices
LEFT_EYE_INNER = 362
LEFT_EYE_OUTER = 263
RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133

class GazePTDataset(torch.utils.data.Dataset):
    def __init__(self, subjects):
        self.data_samples = []        
        for subject in subjects:
            landmark_dir = Path(f"study1_rawdata_processed/p{subject}/preprocessed_landmark")
            metadata_dir = Path(f"study1_rawdata_processed/p{subject}/json")
            
            # Find all left eye PNG files
            metadata_files = sorted(metadata_dir.glob("*.json"))
            
            for metadata_file in metadata_files:
                # Get frame number from filename
                frame_num = metadata_file.stem.replace(".json", "")
                landmark_file = landmark_dir / f"{frame_num}.json"
                
                # Check if corresponding right eye and JSON files exist
                if landmark_file.exists():
                    self.data_samples.append({
                        'subject': subject,
                        'frame_num': frame_num,
                        'landmark_path': landmark_file,
                        'metadata_path': metadata_file,
                    })
             
        assert len(self.data_samples) > 0, f"No data samples found in preprocessed for subjects {subjects}"
        print(f"Loaded {len(self.data_samples)} data samples from {len(subjects)} subjects")
        
    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, idx):        
        sample = self.data_samples[idx]
        
        # Load landmark data
        with open(sample['landmark_path'], 'r') as f:
            landmark_data = json.load(f)

        lmk = torch.tensor([
            landmark_data[f"{LEFT_EYE_INNER}x"],
            landmark_data[f"{LEFT_EYE_INNER}y"],
            landmark_data[f"{LEFT_EYE_OUTER}x"],
            landmark_data[f"{LEFT_EYE_OUTER}y"],
            landmark_data[f"{RIGHT_EYE_INNER}x"],
            landmark_data[f"{RIGHT_EYE_INNER}y"],
            landmark_data[f"{RIGHT_EYE_OUTER}x"],
            landmark_data[f"{RIGHT_EYE_OUTER}y"]
        ], dtype=torch.float32)

        with open(sample['metadata_path'], 'r') as f:
            metadata_data = json.load(f)

        gt = torch.tensor([
            metadata_data["gt_x_px"] * 0.001,  # convert to unit of 1000 px (make the value between 0 and 1)
            metadata_data["gt_y_px"] * 0.001  # convert to unit of 1000 px (make the value between 0 and 1)
        ], dtype=torch.float32)

        return {
            'subject': sample['subject'],
            'frame_num': sample['frame_num'],
            'lmk': lmk,
            'gt': gt
        }

def build_optimizer(model):
    optimizer = optim.Adam(
        model.parameters(),
        lr=GLOBAL_LR,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY
    )
    return optimizer

def train(subjects, device):
    train_dataset = GazePTDataset(subjects)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=6, pin_memory=True,
                              worker_init_fn=seed_worker,
                              generator=model_init_generator)

    model = GazeModel_LMK().to(device)
    optimizer = build_optimizer(model)

    run_tag = f"train_p{','.join(map(str, subjects))}"
    run_name = f"{TYPE}_{run_tag}"
    wandb.init(
        project=WandB_PROJECT,
        name=run_name,
        config=WANDB_CONFIG,
        reinit=True
    )

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", position=0):
            lmk = batch['lmk'].to(device)
            gt = batch['gt'].to(device)

            optimizer.zero_grad()
            pred = model(lmk)
            pred_cm = torch.stack([
                pred[:, 0] * (7.1000/0.430),
                pred[:, 1] * (14.3915/0.8716)
            ], dim=1)
            gt_cm = torch.stack([
                gt[:, 0] * (7.1000/0.430),
                gt[:, 1] * (14.3915/0.8716)
            ], dim=1)
            loss = torch.norm(pred_cm - gt_cm, dim=1).mean()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_lr": optimizer.param_groups[0]["lr"],
        })

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train loss: {epoch_loss:.3f}")

        checkpoint_path = (
            f"{CHECKPOINT_DIR}/{TYPE}_{run_tag}_Epoch{epoch + 1}_TrainLoss{epoch_loss:.3f}.pt"
        )
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)

    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Train RGBLMK gaze prediction model on all subjects")
    parser.add_argument('-g', type=int, default=0,
                       help='GPU device to use (e.g., -g 0 for cuda:0, -g 1 for cuda:1). Default: 0')
    args = parser.parse_args()

    # Set device based on GPU argument
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.g}')
        torch.cuda.set_device(args.g)
    else:
        device = torch.device('cpu')
        print("Warning: CUDA not available, using CPU")
    
    print("CUDA available:", torch.cuda.is_available())
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print("Device name:", torch.cuda.get_device_name(device))
    print("Number of CPU cores:", os.cpu_count())

    print(f"\nTraining on all subjects (n={len(SUBJECT)}): {SUBJECT}")
    train(SUBJECT, device)


if __name__ == "__main__":
    main()