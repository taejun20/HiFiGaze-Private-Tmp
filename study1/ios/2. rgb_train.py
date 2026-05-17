import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random
import wandb
from model import HiFiGaze_RGBModel_v2_nolmk
from dataset import HiFiGaze_Dataset_v2
import warnings
warnings.filterwarnings('ignore', message=".*You are using `torch.load` with `weights_only=False`.*", category=FutureWarning)
import numpy as np
import argparse
import pandas as pd

ALL_SUBJECTS = [1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26]
VAL_SUBJECTS = [1,11,13,15,26]
TRAIN_SUBJECTS = [s for s in ALL_SUBJECTS if s not in VAL_SUBJECTS]

WandB_PROJECT = "251215_IOS"
TYPE = "RGB_v2_nolmk"
os.makedirs("model_checkpoints", exist_ok=True)
CHECKPOINT_DIR = f"model_checkpoints/{WandB_PROJECT}"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# make it reproducible
seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
data_generator = torch.Generator()
data_generator.manual_seed(seed)
def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Hyper-parameters
BATCH_SIZE = 64
NUM_EPOCHS = 20
GLOBAL_LR = 3e-5
STEP_SIZE_EPOCHS = 10
GAMMA = 0.1
FREEZE_EPOCHS = 2
WEIGHT_DECAY = 1e-5

# WandB init
WANDB_CONFIG = {
    'batch_size': BATCH_SIZE,
    'num_epochs': NUM_EPOCHS,
    'learning_rate': GLOBAL_LR,  # Single learning rate for all parameters
    'step_size': STEP_SIZE_EPOCHS,
    'gamma': GAMMA,
    'weight_decay': WEIGHT_DECAY,
    'freeze_epochs': FREEZE_EPOCHS
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

def train(device):
    train_dataset = HiFiGaze_Dataset_v2(TRAIN_SUBJECTS)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=data_generator
    )
    
    model = HiFiGaze_RGBModel_v2_nolmk().to(device)   
    optimizer = build_optimizer(model)
    
    # Initialize WandB run
    train_subjects_str = ','.join(map(str, TRAIN_SUBJECTS))
    val_subjects_str = ','.join(map(str, VAL_SUBJECTS))
    run_name = f"{TYPE}_train{train_subjects_str}_val{val_subjects_str}"
    wandb.init(
        project=WandB_PROJECT,
        name=run_name,
        config=WANDB_CONFIG,
        reinit=True
    )
    
    # Create validation dataset and loader
    val_dataset = HiFiGaze_Dataset_v2(VAL_SUBJECTS)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=data_generator
    )
    
    global_step = 0
    for epoch_idx, epoch in enumerate(range(NUM_EPOCHS)):
        # Training phase
        model.train()
        running_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", position=0)):
            eye = batch['eye'].to(device)
            gt = batch['gt'].to(device)
            
            optimizer.zero_grad()
            pred = model(eye)
            # Convert to cm space for loss calculation
            pred_cm = pred * 16.5116
            gt_cm = gt * 16.5116
            loss = torch.norm(pred_cm - gt_cm, dim=1).mean()  # mean of euclidean losses in cm
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            global_step += 1
        
        # Evaluation at the end of each epoch
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_csv_rows = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{NUM_EPOCHS}", position=0):
                eye = batch['eye'].to(device)
                gt = batch['gt'].to(device)
                
                pred = model(eye)
                # Convert to cm space for loss calculation
                pred_cm = pred * 16.5116
                gt_cm = gt * 16.5116
                loss = torch.norm(pred_cm - gt_cm, dim=1).mean()  # loss in cm
                val_running_loss += loss.item()
                
                # Compute per-sample losses for CSV
                losses = torch.norm(pred_cm - gt_cm, dim=1)
                pred_cm_np = pred_cm.detach().cpu().numpy()
                gt_cm_np = gt_cm.detach().cpu().numpy()
                losses_np = losses.detach().cpu().numpy()
                subjects_np = batch['subject'].numpy()
                frame_ids = batch['frame_id']
                
                for i in range(len(frame_ids)):
                    val_csv_rows.append({
                        'subject': int(subjects_np[i]),
                        'frame_id': frame_ids[i],
                        'pred_x': float(pred_cm_np[i, 0]),
                        'pred_y': float(pred_cm_np[i, 1]),
                        'gt_x': float(gt_cm_np[i, 0]),
                        'gt_y': float(gt_cm_np[i, 1]),
                        'loss': float(losses_np[i])
                    })
        
        avg_val_loss = val_running_loss / len(val_loader)
        
        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "training_loss": avg_train_loss,
            "validation_loss": avg_val_loss,
            'training_lr': optimizer.param_groups[0]['lr'],
        })
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train loss: {avg_train_loss:.4f}")
        print(f"  Val loss: {avg_val_loss:.4f}")

        # Save validation CSV
        val_df = pd.DataFrame(val_csv_rows).sort_values('loss', ascending=False)
        train_subjects_str = ','.join(map(str, TRAIN_SUBJECTS))
        val_subjects_str = ','.join(map(str, VAL_SUBJECTS))
        csv_path = f"{CHECKPOINT_DIR}/Eval_{TYPE}_train{train_subjects_str}_val{val_subjects_str}_Epoch{epoch + 1}_ValLoss{avg_val_loss:.4f}.csv"
        val_df.to_csv(csv_path, index=False)

        # Save checkpoint with validation loss and subject info
        checkpoint_path = f"{CHECKPOINT_DIR}/{TYPE}_train{train_subjects_str}_val{val_subjects_str}_Epoch{epoch + 1}_ValLoss{avg_val_loss:.4f}.pt"           
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
            
    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description='Train gaze prediction model with validation')
    parser.add_argument('-g', type=int, default=1,
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
    print(f"Training subjects: {TRAIN_SUBJECTS}")
    print(f"Validation subjects: {VAL_SUBJECTS}")

    train(device)

if __name__ == "__main__":
    main()