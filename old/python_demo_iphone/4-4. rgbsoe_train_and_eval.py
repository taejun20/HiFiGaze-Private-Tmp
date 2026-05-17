import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
import random
import wandb
from GazeModelRGBSOE import GazeModelRGBSOE
import warnings
warnings.filterwarnings('ignore', message=".*You are using `torch.load` with `weights_only=False`.*", category=FutureWarning)
import torch.serialization
import numpy as np
import cv2
import json
import torchvision.transforms as transforms
import argparse

SUBJECTS = [1,2,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26]
SESSIONS = [1,2,3,4,5,6]
TRAIN_SUBJECTS = [1,2,7,9,10,11,13,14,15,16]
VAL_SUBJECTS = [17,18]
EVAL_SUBJECTS = [20,21,24,25,26]

WandB_PROJECT = "250820"
TYPE = "RGBSOE"

# Make it reproducible
seed = 2025
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
BATCH_SIZE = 64
NUM_EPOCHS = 20
GLOBAL_LR = 3e-5
#BACKBONE_LR = 1e-4
#HEAD_LR = 1e-3
STEP_SIZE_EPOCHS = 10
GAMMA = 0.1
FREEZE_EPOCHS = 2
WEIGHT_DECAY = 1e-5

# Create model_checkpoints directory
os.makedirs("model_checkpoints", exist_ok=True)

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

class GazePTDataset(torch.utils.data.Dataset):
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
                            'left_eye_path': left_eye_file,
                            'right_eye_path': right_eye_file,
                            'json_path': json_file,
                        })
             
        assert len(self.data_samples) > 0, f"No data samples found in preprocessed for subjects {subjects}"
        print(f"Loaded {len(self.data_samples)} data samples from {len(subjects)} subjects")
        
        # Initialize transform for resizing images to 500x250
        self.eye_transform = transforms.Compose([
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
        
        # Apply transforms (resize and convert to tensor)
        l_eye = self.eye_transform(l_eye_img)  # [3, 250, 500]
        r_eye = self.eye_transform(r_eye_img)  # [3, 250, 500]

        # Load JSON data
        with open(sample['json_path'], 'r') as f:
            json_data = json.load(f)
        
        # Extract SOE cues (first 8 values)
        soe_cues = torch.tensor([
            json_data["l_iris_to_soe_center_x_norm"],
            json_data["l_iris_to_soe_center_y_norm"],
            json_data["r_iris_to_soe_center_x_norm"],
            json_data["r_iris_to_soe_center_y_norm"],
        ], dtype=torch.float32)
        
        # Extract eye corner landmarks (next 8 values)
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
        
        # Extract ground truth (last 2 values)
        gt = torch.tensor([
            json_data["px_norm_x"],
            json_data["px_norm_y"]
        ], dtype=torch.float32)
        
        return l_eye, r_eye, soe_cues, eyecorner_lmk, gt
      
def euclidean_loss(pred, target):
    return torch.norm(pred - target, dim=1).mean()

def evaluate_model(model, loader, device):
    model.eval()
    # total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for l_eye, r_eye, soe_cues, eyecorner_lmk, gaze in tqdm(loader, desc="Evaluating", position=1):
            l_eye, r_eye = l_eye.to(device), r_eye.to(device)
            soe_cues, eyecorner_lmk, gaze = soe_cues.to(device), eyecorner_lmk.to(device), gaze.to(device)
            
            pred = model(l_eye, r_eye, soe_cues, eyecorner_lmk)
            # Scale x and y coordinates for both pred and gaze
            pred_cm = torch.stack([
                pred[:, 0] * (7.1000/0.430),
                pred[:, 1] * (14.3915/0.8716)
            ], dim=1)
            gaze_cm = torch.stack([
                gaze[:, 0] * (7.1000/0.430), 
                gaze[:, 1] * (14.3915/0.8716)
            ], dim=1)

            #loss = euclidean_loss(pred_cm, gaze_cm)
            #total_loss += loss.item()
            
            all_preds.append(pred_cm.cpu().numpy())
            all_targets.append(gaze_cm.cpu().numpy())
    
    #avg_loss = total_loss / len(loader)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Calculate additional metrics
    errors = np.linalg.norm(all_preds - all_targets, axis=1)
    metrics = {
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'std_error': np.std(errors),
    }
    
    return metrics

def build_optimizer(model):
    # Previous version with separate learning rates for backbone and head
    # backbone, head = [], []
    # for name, param in model.named_parameters():
    #     if name.startswith('mobilenet_v4_backbone'):
    #         backbone.append(param)
    #     else:
    #         head.append(param)
    # optimizer = optim.Adam(
    #     [
    #         {'params': backbone, 'lr': GLOBAL_LR},
    #         {'params': head, 'lr': GLOBAL_LR}
    #     ],
    #     betas=(0.9, 0.999),
    #     eps=1e-8,
    #     weight_decay=WEIGHT_DECAY
    # )

    # New version with single learning rate for all parameters
    optimizer = optim.Adam(
        model.parameters(),
        lr=GLOBAL_LR,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY
    )
    return optimizer

def train_and_evaluate(train_subjects, val_subjects, eval_subjects, device):
    train_dataset = GazePTDataset(train_subjects)
    val_dataset = GazePTDataset(val_subjects)
    eval_dataset = GazePTDataset(eval_subjects)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            drop_last=True, num_workers=6, pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=model_init_generator)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=6, pin_memory=True,
                           worker_init_fn=seed_worker,
                           generator=model_init_generator)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=6, pin_memory=True,
                           worker_init_fn=seed_worker,
                           generator=model_init_generator)
    
    model = GazeModelRGBSOE(generator=model_init_generator).to(device)
   
    # Freeze backbone for first N epochs
    # for p in model.mobilenet_v4_backbone.parameters():
    #     p.requires_grad = False

    optimizer = build_optimizer(model)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE_EPOCHS, gamma=GAMMA)
    
    # Initialize WandB run
    run_name = f"{TYPE}_train_p{''.join(map(str,train_subjects))}_val_p{''.join(map(str,val_subjects))}_eval_p{''.join(map(str,eval_subjects))}"
    wandb.init(
        project=WandB_PROJECT,
        name=run_name,
        config=WANDB_CONFIG,
        reinit=True
    )
    
    global_step = 0
    best_val_error = float('inf')
    
    # Calculate evaluation intervals (2 times per epoch)
    eval_intervals_per_epoch = 2

    for epoch_idx, epoch in enumerate(range(NUM_EPOCHS)):
        # Unfreeze backbone after warm-up
        # if epoch == FREEZE_EPOCHS:
        #     for p in model.mobilenet_v4_backbone.parameters():
        #         p.requires_grad = True
        #     print(f"Backbone unfrozen at epoch {epoch + 1}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        epoch_losses = []  # Track losses for partial epoch reporting
        
        # Calculate when to do evaluations within the epoch
        total_batches = len(train_loader)
        eval_intervals = [total_batches // eval_intervals_per_epoch * (i + 1) for i in range(eval_intervals_per_epoch)]
        eval_intervals[-1] = total_batches  # Ensure last evaluation happens at end of epoch
        
        for batch_idx, (l_eye, r_eye, soe_cues, eyecorner_lmk, gaze) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", position=0)):
            l_eye, r_eye = l_eye.to(device), r_eye.to(device)
            soe_cues, eyecorner_lmk, gaze = soe_cues.to(device), eyecorner_lmk.to(device), gaze.to(device)
            
            optimizer.zero_grad()
            pred = model(l_eye, r_eye, soe_cues, eyecorner_lmk)
            loss = euclidean_loss(pred, gaze)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            epoch_losses.append(loss.item())
            global_step += 1
            
            if batch_idx % 10 == 0:
                wandb.log({
                    "training_batch_loss_every_10th": loss.item(),                    
                })
            
            # Check if we should do evaluation at this point
            if (batch_idx + 1) in eval_intervals:
                eval_point = eval_intervals.index(batch_idx + 1) + 1
                
                # Validation and Evaluation phase
                val_metrics = evaluate_model(model, val_loader, device)
                eval_metrics = evaluate_model(model, eval_loader, device)
                
                # Calculate partial epoch training loss
                partial_training_loss = running_loss / (batch_idx + 1)
                
                # Create sub-epoch identifier
                sub_epoch = epoch + (eval_point / eval_intervals_per_epoch)
                
                # Log metrics
                wandb.log({
                    "epoch": sub_epoch,
                    "training_partial_loss": partial_training_loss,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **{f"eval_{k}": v for k, v in eval_metrics.items()},
                    'training_lr': optimizer.param_groups[0]['lr'],
                    'eval_point': eval_point,
                    'total_eval_points': eval_intervals_per_epoch
                })
                
                print(f"Epoch {epoch+1}, Eval Point {eval_point}/{eval_intervals_per_epoch}:")
                print(f"  Train loss (partial): {partial_training_loss:.4f}")
                print(f"  Val loss: {val_metrics['mean_error']:.4f}")
                print(f"  Eval loss: {eval_metrics['mean_error']:.4f}")
        
                # Also save regular checkpoint
                checkpoint_dir = f"model_checkpoints/{WandB_PROJECT}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = f"{checkpoint_dir}/{TYPE}_train_p{','.join(map(str,train_subjects))}_val_p{','.join(map(str,val_subjects))}_eval_p{','.join(map(str,eval_subjects))}_Epoch{epoch + 1}_Point{eval_point}_ValError{val_metrics['mean_error']:.3f}.pt"           
                torch.save({
                    'epoch': epoch + 1,
                    'eval_point': eval_point,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'eval_metrics': eval_metrics,
                }, checkpoint_path)
                
                # Switch back to training mode
                model.train()

        # scheduler.step()
    
    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate RGB gaze prediction model')
    parser.add_argument('-t', '--train_subjects', nargs='*', type=int, default=None,
                       help='List of subjects to use for training (e.g., -t 1 2 3). If not specified, uses default TRAIN_SUBJECTS list.')
    parser.add_argument('-v', '--val_subjects', nargs='*', type=int, default=None,
                       help='List of subjects to use for validation (e.g., -v 4 5). If not specified, uses default VAL_SUBJECTS list.')
    parser.add_argument('-e', '--eval_subjects', nargs='*', type=int, default=None,
                       help='List of subjects to use for evaluation (e.g., -e 6 7 8). If not specified, uses default EVAL_SUBJECTS list.')
    parser.add_argument('-g', type=int, default=0,
                       help='GPU device to use (e.g., -g 0 for cuda:0, -g 1 for cuda:1). Default: 0')
    args = parser.parse_args()
    
    # Use custom subjects lists if provided, otherwise use defaults
    train_subjects_to_process = args.train_subjects if args.train_subjects is not None else TRAIN_SUBJECTS
    val_subjects_to_process = args.val_subjects if args.val_subjects is not None else VAL_SUBJECTS
    eval_subjects_to_process = args.eval_subjects if args.eval_subjects is not None else EVAL_SUBJECTS
    
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
    print(f"Training subjects: {train_subjects_to_process}")
    print(f"Validation subjects: {val_subjects_to_process}")
    print(f"Evaluation subjects: {eval_subjects_to_process}")

    train_and_evaluate(train_subjects_to_process, val_subjects_to_process, eval_subjects_to_process, device)

if __name__ == "__main__":
    main()