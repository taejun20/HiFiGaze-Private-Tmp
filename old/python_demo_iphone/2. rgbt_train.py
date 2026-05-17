import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
import random
import wandb
from GazeModelRGBT import GazeModelRGBT
import warnings
warnings.filterwarnings('ignore', message=".*You are using `torch.load` with `weights_only=False`.*", category=FutureWarning)
import torch.serialization
import numpy as np
import cv2
import json
import torchvision.transforms as transforms
import argparse

WandB_PROJECT = "250917_DEMO"
TYPE = "RGBT"

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

os.makedirs("checkpoints", exist_ok=True)

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
    def __init__(self):
        self.data_samples = []  
        eyepatch_dir = Path(f"data/train/preprocessed/input_eyepatch/")
        landmark_dir = Path(f"data/train/preprocessed/input_landmark_and_gt/")
        template_dir = Path(f"data/train/preprocessed/input_template/")

        # Find all left eye PNG files
        left_files = sorted(eyepatch_dir.glob("*_left.png"))
        
        for left_eye_file in left_files:
            # Get frame number from filename
            frame_num = left_eye_file.stem.replace("_left", "")
            right_eye_file = eyepatch_dir / f"{frame_num}_right.png"
            json_file = landmark_dir / f"{frame_num}_landmark_and_gt.json"
            template_file = template_dir / f"{frame_num}.png"

            # Check if corresponding right eye and JSON files exist
            if right_eye_file.exists() and json_file.exists() and template_file.exists():
                self.data_samples.append({
                    'left_eye_path': left_eye_file,
                    'right_eye_path': right_eye_file,
                    'json_path': json_file,
                    'template_path': template_file
                })

        assert len(self.data_samples) > 0, f"No data samples found in preprocessed"
        print(f"Loaded {len(self.data_samples)} data samples from train")
        
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

        template_img = cv2.imread(str(sample['template_path']))
        template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)
        template = torch.from_numpy(template_img).permute(2, 0, 1).float() / 255.0  # Convert to [3, H, W] tensor
        
        # Load JSON data
        with open(sample['json_path'], 'r') as f:
            json_data = json.load(f)
        
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
        
        return l_eye, r_eye, template, eyecorner_lmk, gt
      
def euclidean_loss(pred, target):
    return torch.norm(pred - target, dim=1).mean()

def evaluate_model(model, loader, device):
    model.eval()
    # total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for l_eye, r_eye, template, eyecorner_lmk, gaze in tqdm(loader, desc="Evaluating", position=1):
            l_eye, r_eye = l_eye.to(device), r_eye.to(device)
            template, eyecorner_lmk, gaze = template.to(device), eyecorner_lmk.to(device), gaze.to(device)
            
            pred = model(l_eye, r_eye, template, eyecorner_lmk)
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

def train_and_evaluate(device):
    train_dataset = GazePTDataset()
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            drop_last=True, num_workers=6, pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=model_init_generator)
    
    model = GazeModelRGBT(generator=model_init_generator).to(device)
    
    # Load pretrained checkpoint for transfer learning
    checkpoint_path = f"checkpoints/{TYPE}_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading pretrained model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model state dict loaded successfully")
        else:
            # If checkpoint only contains model state dict directly
            model.load_state_dict(checkpoint)
            print("Model state dict loaded successfully (direct format)")        
    else:
        print(f"Warning: Checkpoint file {checkpoint_path} not found. Training from scratch.")
   
    # Freeze backbone for first N epochs
    # for p in model.mobilenet_v4_backbone.parameters():
    #     p.requires_grad = False

    optimizer = build_optimizer(model)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE_EPOCHS, gamma=GAMMA)
    
    # Initialize WandB run
    run_name = f"{TYPE}"
    wandb.init(
        project=WandB_PROJECT,
        name=run_name,
        config=WANDB_CONFIG,
        reinit=True
    )
    
    global_step = 0
    best_val_error = float('inf')
    
    for epoch_idx, epoch in enumerate(range(NUM_EPOCHS)):
        # Unfreeze backbone after warm-up
        # if epoch == FREEZE_EPOCHS:
        #     for p in model.mobilenet_v4_backbone.parameters():
        #         p.requires_grad = True
        #     print(f"Backbone unfrozen at epoch {epoch + 1}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        for batch_idx, (l_eye, r_eye, template, eyecorner_lmk, gaze) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", position=0)):
            l_eye, r_eye = l_eye.to(device), r_eye.to(device)
            template, eyecorner_lmk, gaze = template.to(device), eyecorner_lmk.to(device), gaze.to(device)
            
            optimizer.zero_grad()
            pred = model(l_eye, r_eye, template, eyecorner_lmk)
            loss = euclidean_loss(pred, gaze)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            global_step += 1
            
            wandb.log({
                "batch_loss": loss.item(),                    
            }) 
        
        # Calculate epoch training loss
        epoch_training_loss = running_loss / len(train_loader)
                
        # Log metrics for the epoch
        wandb.log({
            "epoch": epoch + 1,
            "training_loss": epoch_training_loss,
            'training_lr': optimizer.param_groups[0]['lr']
        })
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train loss: {epoch_training_loss:.4f}")
        
        # Save checkpoint once per epoch
        checkpoint_dir = f"checkpoints/{WandB_PROJECT}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = f"{checkpoint_dir}/{TYPE}_Epoch{epoch + 1}_TrainError{epoch_training_loss:.3f}.pt"           
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")

        # scheduler.step()
    
    wandb.finish()

def main():    
    # Set device based on GPU argument
    if torch.cuda.is_available():
        device = torch.device(f'cuda:0')
        torch.cuda.set_device(0)
    else:
        device = torch.device('cpu')
        print("Warning: CUDA not available, using CPU")
    
    print("CUDA available:", torch.cuda.is_available())
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print("Device name:", torch.cuda.get_device_name(device))
    print("Number of CPU cores:", os.cpu_count())
 
    train_and_evaluate(device)

if __name__ == "__main__":
    main()