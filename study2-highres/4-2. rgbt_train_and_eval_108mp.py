import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import random
import wandb
from models.GazeModel_RGBT import GazeModel_RGBT
import warnings
warnings.filterwarnings('ignore', message=".*You are using `torch.load` with `weights_only=False`.*", category=FutureWarning)
import cv2
import json
import argparse
import pandas as pd

#RESOLUTIONS = ["108mp","4k","sd","qqvga","qvga","54mp","8k","6k","2k","fhd","hd"]
RESOLUTIONS = ["qqvga","qvga","sd","hd","fhd","2k","4k","6k","8k","54mp","108mp"]

# Resume: skip until eval participant 3; for p3 skip qqvga..2k, start at 4k; later evals run full RESOLUTIONS.
RESUME_EVAL_SUBJECT = 3
RESUME_FIRST_RES_FOR_P3 = "4k"

SUBJECT = [1,2,3,4,5,6,7,8,9,10]

WandB_PROJECT = "260512_108MP"
TYPE = "RGBT"

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
BATCH_SIZE = 10
NUM_EPOCHS = 15
GLOBAL_LR = 3e-5
WEIGHT_DECAY = 1e-5

# Create model_checkpoints directory
os.makedirs("model_checkpoints", exist_ok=True)
CHECKPOINT_DIR = f"model_checkpoints/{WandB_PROJECT}"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Initial weights to load before training
INIT_WEIGHTS_PATH = Path(
    "model_checkpoints/pretrained/RGBT_train_p2,6,7,8,10,11,12,13,15,16,19,25_Epoch12_TrainLoss0.639.pt"
)

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

# GT / pred in model space are 1000px-normalized; convert to cm for logging/metrics only.
PIXEL_X_TO_CM = 7.1000 / 0.430
PIXEL_Y_TO_CM = 14.3915 / 0.8716


def _mean_l2_error_cm(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean Euclidean error in cm. pred, gt in 1000px units."""
    pred_cm = torch.stack([pred[:, 0] * PIXEL_X_TO_CM, pred[:, 1] * PIXEL_Y_TO_CM], dim=1)
    gt_cm = torch.stack([gt[:, 0] * PIXEL_X_TO_CM, gt[:, 1] * PIXEL_Y_TO_CM], dim=1)
    return torch.norm(pred_cm - gt_cm, dim=1).mean()

def separable_gaussian_blur(img, kernel_size, sigma):
    k = (kernel_size - 1) // 2
    x = np.linspace(-k, k, kernel_size)
    kernel_1d = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    kernel_h = kernel_1d.reshape(1, -1)
    blurred_h = cv2.filter2D(img, -1, kernel_h)

    kernel_v = kernel_1d.reshape(-1, 1)
    blurred = cv2.filter2D(blurred_h, -1, kernel_v)
    return blurred

class GazePTDataset(torch.utils.data.Dataset):
    def __init__(self, subjects, res: str):
        self.data_samples = []        
        for subject in subjects:
            eyepatch_dir = Path(f"study2_rawdata_processed/p{subject}/preprocessed/{res}_frames/eyecrop")
            landmark_dir = Path(f"study2_rawdata_processed/p{subject}/preprocessed/{res}_frames/landmark")
            metadata_dir = Path(f"study2_rawdata_processed/p{subject}/json")
            
            # Find all left eye PNG files
            metadata_files = sorted(metadata_dir.glob("*.json"))
            
            for metadata_file in metadata_files:
                # Get frame number from filename
                frame_num = metadata_file.stem.replace(".json", "")
                l_eye_file = eyepatch_dir / f"{frame_num}_l.png"
                r_eye_file = eyepatch_dir / f"{frame_num}_r.png"
                landmark_file = landmark_dir / f"{frame_num}.json"
                
                # Check if corresponding right eye and JSON files exist
                if l_eye_file.exists() and r_eye_file.exists() and landmark_file.exists():
                    self.data_samples.append({
                        'subject': subject,
                        'frame_num': frame_num,
                        'l_eye_path': l_eye_file,
                        'r_eye_path': r_eye_file,
                        'landmark_path': landmark_file,
                        'metadata_path': metadata_file,
                    })
             
        assert len(self.data_samples) > 0, f"No data samples found in preprocessed for subjects {subjects}"
        print(f"Loaded {len(self.data_samples)} data samples from {len(subjects)} subjects")

    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, idx):        
        sample = self.data_samples[idx]
        
        # Load left and right eye images
        l_eye_img = cv2.imread(str(sample['l_eye_path']))
        r_eye_img = cv2.imread(str(sample['r_eye_path']))
        
        # Convert BGR to RGB
        l_eye_img = cv2.cvtColor(l_eye_img, cv2.COLOR_BGR2RGB)
        r_eye_img = cv2.cvtColor(r_eye_img, cv2.COLOR_BGR2RGB)
        
       # don't horizontally flip the left eye. The preprocessed left eye crop is already flipped.
        l_eye_img = cv2.resize(l_eye_img, (1400, 700), interpolation=cv2.INTER_LINEAR)  # (w,h) -> 1400x700
        r_eye_img = cv2.resize(r_eye_img, (1400, 700), interpolation=cv2.INTER_LINEAR)

        l_eye = torch.from_numpy(l_eye_img).permute(2, 0, 1).contiguous().float().mul_(1.0 / 255.0)
        r_eye = torch.from_numpy(r_eye_img).permute(2, 0, 1).contiguous().float().mul_(1.0 / 255.0)

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

        screen_dir = Path(f"study2_rawdata_processed/screen/")
        background_id = str(int(metadata_data["background"]))
        background_path = screen_dir / f"{background_id}.jpg"
        background = cv2.imread(str(background_path))
        if background is None:
            raise FileNotFoundError(
                f"Could not load background: {background_path}"
            )

        # Match template generation from preprocessing: blend -> blur -> downsample.
        template = separable_gaussian_blur(background, kernel_size=51, sigma=5)
        template = cv2.resize(template, (60, 120), interpolation=cv2.INTER_LINEAR)  # (width, height)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)     
        template = torch.from_numpy(template).permute(2, 0, 1).contiguous().float().mul_(1.0 / 255.0)

        gt = torch.tensor([
            float(metadata_data["gt_x_px"]) * 0.001,  # convert to unit of 1000 px (make the value between 0 and 1)
            float(metadata_data["gt_y_px"]) * 0.001  # convert to unit of 1000 px (make the value between 0 and 1)
        ], dtype=torch.float32)
        
        return {
            'subject': sample['subject'],
            'frame_num': sample['frame_num'],
            'l_eye': l_eye,
            'r_eye': r_eye,
            'template': template,
            'lmk': lmk,
            'gt': gt
        }
      
def evaluate_model(model, loader, device, run_tag, epoch, dataset_type):
    model.eval()
    all_preds = []
    all_targets = []
    all_preds_raw = []
    all_targets_raw = []
    csv_data = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", position=1):
            l_eye = batch['l_eye'].to(device)
            r_eye = batch['r_eye'].to(device)
            template = batch['template'].to(device)
            lmk = batch['lmk'].to(device)
            gt = batch['gt'].to(device)
            
            pred = model(l_eye, r_eye, template, lmk)
            pred_raw = pred
            gt_raw = gt
            # Scale x and y coordinates for both pred and gt
            pred_cm = torch.stack([
                pred[:, 0] * PIXEL_X_TO_CM,
                pred[:, 1] * PIXEL_Y_TO_CM
            ], dim=1)
            gt_cm = torch.stack([
                gt[:, 0] * PIXEL_X_TO_CM,
                gt[:, 1] * PIXEL_Y_TO_CM
            ], dim=1)

            losses_cm = torch.norm(pred_cm - gt_cm, dim=1)
            losses_raw = torch.norm(pred_raw - gt_raw, dim=1)

            if dataset_type == "eval":
                for i in range(len(batch['subject'])):
                    csv_data.append({
                        'subject': int(batch['subject'][i]),
                        'frame_num': batch['frame_num'][i],
                        'pred_x': pred_raw[i, 0].item(),
                        'pred_y': pred_raw[i, 1].item(),
                        'gt_x': gt_raw[i, 0].item(),
                        'gt_y': gt_raw[i, 1].item(),    
                        'loss': losses_raw[i].item(),
                        'pred_x_cm': pred_cm[i, 0].item(),
                        'pred_y_cm': pred_cm[i, 1].item(),
                        'gt_x_cm': gt_cm[i, 0].item(),
                        'gt_y_cm': gt_cm[i, 1].item(),
                        'loss_cm': losses_cm[i].item(),
                    })
            
            all_preds.append(pred_cm.cpu().numpy())
            all_targets.append(gt_cm.cpu().numpy())
            all_preds_raw.append(pred_raw.cpu().numpy())
            all_targets_raw.append(gt_raw.cpu().numpy())

    if dataset_type == "eval":
        df = pd.DataFrame(csv_data)
        df = df.sort_values('loss_cm', ascending=False)
        csv_path = f"{CHECKPOINT_DIR}/Eval_{TYPE}_{run_tag}_Epoch{epoch+1}.csv"
        df.to_csv(csv_path, index=False)
    
    #avg_loss = total_loss / len(loader)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_preds_raw = np.concatenate(all_preds_raw)
    all_targets_raw = np.concatenate(all_targets_raw)
    
    # Calculate additional metrics
    errors_cm = np.linalg.norm(all_preds - all_targets, axis=1)
    metrics = {
        'mean_loss_cm': np.mean(errors_cm),
        'std_loss_cm': np.std(errors_cm),
    }
    
    return metrics

def build_optimizer(model):
    optimizer = optim.Adam(
        model.parameters(),
        lr=GLOBAL_LR,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY
    )
    return optimizer

def train_and_evaluate(train_subjects, val_subjects, eval_subjects, device, res: str):
    train_dataset = GazePTDataset(train_subjects, res)
    val_dataset = GazePTDataset(val_subjects, res)
    eval_dataset = GazePTDataset(eval_subjects, res)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=6,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=model_init_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=model_init_generator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=model_init_generator,
    )

    model = GazeModel_RGBT().to(device)
    if INIT_WEIGHTS_PATH.exists():
        ckpt = torch.load(str(INIT_WEIGHTS_PATH), map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded init weights from: {INIT_WEIGHTS_PATH}")
    else:
        raise FileNotFoundError(f"Init weights not found: {INIT_WEIGHTS_PATH}")
    optimizer = build_optimizer(model)

    run_tag = (
        f"train_p{','.join(map(str,train_subjects))}_"
        f"val_p{','.join(map(str,val_subjects))}_"
        f"eval_p{','.join(map(str,eval_subjects))}"
    )

    run_name = f"{TYPE}_{run_tag}"
    wandb.init(
        project=WandB_PROJECT,
        name=run_name,
        config=WANDB_CONFIG,
        reinit=True,
    )

    for epoch_idx, epoch in enumerate(range(NUM_EPOCHS)):
        model.train()
        running_mean_cm = 0.0

        for batch_idx, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", position=0)
        ):
            l_eye = batch['l_eye'].to(device)
            r_eye = batch['r_eye'].to(device)
            template = batch['template'].to(device)
            lmk = batch['lmk'].to(device)
            gt = batch['gt'].to(device)
            optimizer.zero_grad()
            pred = model(l_eye, r_eye, template, lmk)
            # Train in 1000px-normalized space; cm only for logging (same as 1-1. rgb_pretrain.py).
            loss = torch.norm(pred - gt, dim=1).mean()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                running_mean_cm += _mean_l2_error_cm(pred, gt).item()

        n_batches = len(train_loader)
        epoch_loss_cm = running_mean_cm / n_batches
        val_metrics = evaluate_model(model, val_loader, device, run_tag, epoch, "val")
        eval_metrics = evaluate_model(model, eval_loader, device, run_tag, epoch, "eval")

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss_cm": epoch_loss_cm,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"eval_{k}": v for k, v in eval_metrics.items()},
            }
        )

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train loss (cm): {epoch_loss_cm:.3f}")
        print(f"  Val loss: {val_metrics['mean_loss_cm']:.3f}")
        print(f"  Eval loss: {eval_metrics['mean_loss_cm']:.3f}")

        checkpoint_path = (
            f"{CHECKPOINT_DIR}/{TYPE}_{run_tag}_"
            f"Epoch{epoch + 1}_"
            f"ValError{val_metrics['mean_loss_cm']:.3f}_"
            f"EvalError{eval_metrics['mean_loss_cm']:.3f}.pt"
        )
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )

    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate RGBT gaze prediction model')
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

    # Rolling split: 7 train / 2 val / 1 eval
    num_subjects = len(SUBJECT)
    passed_first_eval_p10 = False
    for iteration in range(num_subjects):
        train_indices = [(iteration + offset) % num_subjects for offset in range(7)]
        val_indices = [(iteration + offset) % num_subjects for offset in range(7, 9)]
        eval_indices = [(iteration + 9) % num_subjects]

        train_subjects = [SUBJECT[i] for i in train_indices]
        val_subjects = [SUBJECT[i] for i in val_indices]
        eval_subjects = [SUBJECT[i] for i in eval_indices]

        skip_eval_p10 = eval_subjects[0] == 10 and not passed_first_eval_p10
        if skip_eval_p10:
            passed_first_eval_p10 = True

        print(f"\n=== ITERATION {iteration + 1}/{num_subjects} ===")
        print(f"Evaluation subjects (n=1): {eval_subjects}")
        print(f"Validation subjects: {val_subjects}")
        print(f"Training subjects: {train_subjects}")
        if skip_eval_p10:
            print("Passing: skip this iteration (first fold with eval participant 10).")
            continue

        if eval_subjects[0] < RESUME_EVAL_SUBJECT:
            print(
                f"Skipping: resume starts at eval p{RESUME_EVAL_SUBJECT}, "
                f"{RESUME_FIRST_RES_FOR_P3} (skipping eval p{eval_subjects[0]})."
            )
            continue

        # Run all resolutions for this participant split
        for res in RESOLUTIONS:
            if (
                eval_subjects[0] == RESUME_EVAL_SUBJECT
                and RESOLUTIONS.index(res) < RESOLUTIONS.index(RESUME_FIRST_RES_FOR_P3)
            ):
                print(
                    f"Skipping resolution {res} for eval p{RESUME_EVAL_SUBJECT} "
                    f"(already done; resume at {RESUME_FIRST_RES_FOR_P3})."
                )
                continue
            global TYPE
            TYPE = f"RGBT_{res}"
            print(f"\n==================== RESOLUTION: {res} (TYPE={TYPE}) ====================")
            train_and_evaluate(train_subjects, val_subjects, eval_subjects, device, res)

        print(f"Completed iteration {iteration + 1}")
    

if __name__ == "__main__":
    main()
