import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import random
import wandb
from GazeModelRGB_FFHA import GazeModelRGB_FFHA
import warnings
warnings.filterwarnings('ignore', message=".*You are using `torch.load` with `weights_only=False`.*", category=FutureWarning)
import numpy as np
import cv2
import json
import torchvision.transforms as transforms
import argparse
import pandas as pd
import torch.multiprocessing as mp

SUBJECTS = [1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26]
SESSIONS = [1,2,3,4,5,6]

WandB_PROJECT = "251119_FullFaceHeadAugment"
TYPE = "RGB_FFHA"
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
#BACKBONE_LR = 1e-4
#HEAD_LR = 1e-3
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

class GazePTDataset(torch.utils.data.Dataset):
    def __init__(self, subjects):
        self.data_samples = []        
        for subject in subjects:
            for session in SESSIONS:
                eyepatch_dir = Path(f"preprocessed/input_eyepatch/p{subject}/s{session}")
                landmark_dir = Path(f"preprocessed/input_landmark_and_gt/p{subject}/s{session}")
                face_dir = Path(f"preprocessed/input_face/p{subject}/s{session}")
                headaug_dir = Path(f"preprocessed/input_headaug/p{subject}/s{session}")

                # Find all left eye PNG files
                l_eye_paths = sorted(eyepatch_dir.glob("*_left.png"))
                for l_eye_path in l_eye_paths:
                    # Get frame number from filename
                    frame_num = l_eye_path.stem.replace("_left", "")
                    r_eye_path = eyepatch_dir / f"{frame_num}_right.png"
                    json_file = landmark_dir / f"{frame_num}_landmark_and_gt.json"
                    face_path = face_dir / f"{frame_num}_face.png"
                    headaug_path = headaug_dir / f"{frame_num}_headaug.json"

                    # Check if corresponding right eye and JSON files exist
                    if r_eye_path.exists() and json_file.exists():
                        self.data_samples.append({
                            'subject': subject,
                            'session': session,
                            'frame_num': frame_num,
                            'json_path': json_file,                           
                            'l_eye_path': l_eye_path,
                            'r_eye_path': r_eye_path,
                            'face_path': face_path,
                            'headaug_path': headaug_path
                        })
                    else:
                        raise ValueError(f"No corresponding right eye and JSON files found for eyepatch: {l_eye_path}_left.png")

        assert len(self.data_samples) > 0, f"No data samples found in preprocessed for subjects {subjects}"
        print(f"Loaded {len(self.data_samples)} data samples from {len(subjects)} subjects")
        
        # Initialize transform for resizing images to 500x250
        self.eye_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((250, 500)),  # (height, width)
            transforms.ToTensor()
        ])

        self.face_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((500, 500)),  # (height, width)
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, idx):        
        sample = self.data_samples[idx]

        face_img = cv2.imread(str(sample['face_path']))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face = self.eye_transform(face_img)  # [3, 500, 500]

        
        # Load left and right eye images
        l_eye_img = cv2.imread(str(sample['l_eye_path']))
        r_eye_img = cv2.imread(str(sample['r_eye_path']))
        
        # Convert BGR to RGB
        l_eye_img = cv2.cvtColor(l_eye_img, cv2.COLOR_BGR2RGB)
        r_eye_img = cv2.cvtColor(r_eye_img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms (resize and convert to tensor)
        l_eye = self.eye_transform(l_eye_img)  # [3, 250, 500]
        r_eye = self.eye_transform(r_eye_img)  # [3, 250, 500]

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
        
        # Load JSON data
        with open(sample['headaug_path'], 'r') as f:
            headaug_data = json.load(f)
        
        # Extract full head augmentation landmark vector (indices "1"..."476")
        headaug_lmk = torch.tensor(
            [headaug_data[f"{idx}{axis}"] for idx in range(478) for axis in ("x", "y")],
            dtype=torch.float32
        )

        # Extract ground truth
        gt = torch.tensor([
            json_data["px_norm_x"],
            json_data["px_norm_y"]
        ], dtype=torch.float32)
        
        return {
            'subject': sample['subject'],
            'session': sample['session'],
            'frame_num': sample['frame_num'],
            'eyecorner_lmk': eyecorner_lmk,
            'l_eye': l_eye,
            'r_eye': r_eye,
            'face': face,
            'headaug_lmk': headaug_lmk,
            'gt': gt
        }
      
def evaluate_model(model, loader, device, train_val_eval_str, epoch, dataset_type, val_error_4f_str):
    model.eval()
    all_preds = []
    all_targets = []
    csv_data = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating", position=1)):
            face = batch['face'].to(device)
            l_eye = batch['l_eye'].to(device)
            r_eye = batch['r_eye'].to(device)
            eyecorner_lmk = batch['eyecorner_lmk'].to(device)
            headaug_lmk = batch['headaug_lmk'].to(device)
            gt = batch['gt'].to(device)
            
            pred = model(face, l_eye, r_eye, eyecorner_lmk, headaug_lmk)
            # Scale x and y coordinates for both pred and gt
            pred_cm = torch.stack([
                pred[:, 0] * (7.1000/0.430),
                pred[:, 1] * (14.3915/0.8716)
            ], dim=1)
            gt_cm = torch.stack([
                gt[:, 0] * (7.1000/0.430), 
                gt[:, 1] * (14.3915/0.8716)
            ], dim=1)

            # Calculate losses
            losses = torch.norm(pred_cm - gt_cm, dim=1)     # euclidean loss
            
            # Store results for saving a csv file
            if dataset_type == "eval":
                for i in range(len(batch['subject'])):
                    csv_data.append({
                        'subject': int(batch['subject'][i]),
                        'session': int(batch['session'][i]),
                        'frame_num': batch['frame_num'][i],
                        'pred_x': pred_cm[i, 0].item(),
                        'pred_y': pred_cm[i, 1].item(),
                        'gt_x': gt_cm[i, 0].item(),
                        'gt_y': gt_cm[i, 1].item(),
                        'loss': losses[i].item()
                    })
            
            all_preds.append(pred_cm.cpu().numpy())
            all_targets.append(gt_cm.cpu().numpy())
    
    if dataset_type == "eval":
        df = pd.DataFrame(csv_data)
        df = df.sort_values('loss', ascending=False)
        # Save to CSV
        csv_path = f"{CHECKPOINT_DIR}/Eval_{TYPE}_{train_val_eval_str}_Epoch{epoch+1}_ValError{val_error_4f_str}.csv"
        df.to_csv(csv_path, index=False)

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
                            drop_last=True, num_workers=4, pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=data_generator)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True,
                           worker_init_fn=seed_worker)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True,
                           worker_init_fn=seed_worker)
    
    model = GazeModelRGB_FFHA().to(device)   
    optimizer = build_optimizer(model)
    
    # Initialize WandB run
    train_val_eval_str = f"train_p{','.join(map(str,train_subjects))}_val_p{','.join(map(str,val_subjects))}_eval_p{','.join(map(str,eval_subjects))}"
    run_name = f"{TYPE}_{train_val_eval_str}"
    wandb.init(
        project=WandB_PROJECT,
        name=run_name,
        config=WANDB_CONFIG,
        reinit=True
    )
    
    global_step = 0
    for epoch_idx, epoch in enumerate(range(NUM_EPOCHS)):
        # Training phase
        model.train()
        running_loss = 0.0       
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", position=0)):
            face = batch['face'].to(device)
            l_eye = batch['l_eye'].to(device)
            r_eye = batch['r_eye'].to(device)
            eyecorner_lmk = batch['eyecorner_lmk'].to(device)
            headaug_lmk = batch['headaug_lmk'].to(device)
            gt = batch['gt'].to(device)
            
            optimizer.zero_grad()
            pred = model(face, l_eye, r_eye, eyecorner_lmk, headaug_lmk)
            loss = torch.norm(pred - gt, dim=1).mean()  # mean of euclidean losses
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            global_step += 1
            
            if batch_idx % 10 == 0:
                wandb.log({
                    "training_loss_every_10th_batch": loss.item(),                    
                })
        
        # Evaluation at the end of each epoch
        avg_train_loss = running_loss / len(train_loader)
        val_metrics = evaluate_model(model, val_loader, device, train_val_eval_str, epoch, "val", None)
        eval_metrics = evaluate_model(model, eval_loader, device, train_val_eval_str, epoch, "eval", f"{val_metrics['mean_error']:.4f}")
        
        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "training_loss": avg_train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"eval_{k}": v for k, v in eval_metrics.items()},
            'training_lr': optimizer.param_groups[0]['lr'],
        })
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train loss: {avg_train_loss:.4f}")
        print(f"  Val loss: {val_metrics['mean_error']:.4f}")
        print(f"  Eval loss: {eval_metrics['mean_error']:.4f}")
    
        # Save checkpoint
        checkpoint_path = f"{CHECKPOINT_DIR}/{TYPE}_{train_val_eval_str}_Epoch{epoch + 1}_ValError{val_metrics['mean_error']:.3f}.pt"           
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
            
    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate gaze prediction model')
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
    train_subjects_to_process = args.train_subjects
    val_subjects_to_process = args.val_subjects
    eval_subjects_to_process = args.eval_subjects
    assert train_subjects_to_process is not None and val_subjects_to_process is not None and eval_subjects_to_process is not None, "Please provide train, val, and eval subjects"
    
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