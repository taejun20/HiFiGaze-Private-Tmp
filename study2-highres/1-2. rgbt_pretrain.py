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

SAVE_EYE_DEBUG = True # Save debug eye crops only for the first batch of each epoch.

SUBJECT = [2,6,7,8,10,11,12,13,15,16,19,25] # 12 subjects. Excluding the 10 participants who have also participated in study 2.

WandB_PROJECT = "260507_highres_pretrain"
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
BATCH_SIZE = 32
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

# GT unit to cm. (GT: 1000 px unit)
PIXEL_X_TO_CM = 7.1000 / 0.430
PIXEL_Y_TO_CM = 14.3915 / 0.8716

def _mean_l2_error_cm(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean Euclidean error in cm (no grad needed for logging). pred, gt were originally in 1000px units."""
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
    def __init__(self, subjects):
        self.data_samples = []
        for subject in subjects:
            eyepatch_dir = Path(f"study1_rawdata_processed/p{subject}/preprocessed_eyecrop")
            landmark_dir = Path(f"study1_rawdata_processed/p{subject}/preprocessed_landmark")
            metadata_dir = Path(f"study1_rawdata_processed/p{subject}/json")
            
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
        
        l_eye_img = cv2.cvtColor(l_eye_img, cv2.COLOR_BGR2RGB)
        r_eye_img = cv2.cvtColor(r_eye_img, cv2.COLOR_BGR2RGB)

        l_eye_img = cv2.resize(l_eye_img, (1400, 700), interpolation=cv2.INTER_LINEAR)  # (width, height): 1400x700
        r_eye_img = cv2.resize(r_eye_img, (1400, 700), interpolation=cv2.INTER_LINEAR)

        l_eye = torch.from_numpy(l_eye_img).permute(2, 0, 1).contiguous().float().mul_(1.0 / 255.0)
        r_eye = torch.from_numpy(r_eye_img).permute(2, 0, 1).contiguous().float().mul_(1.0 / 255.0)       

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

        screen_dir = Path(f"study1_rawdata_processed/screen/")
        background_a_id = str(int(metadata_data["backgroundA"]))
        background_b_id = str(int(metadata_data["backgroundB"]))
        dissolve = float(metadata_data["dissolve"])

        background_a_path = screen_dir / f"{background_a_id}.jpg"
        background_b_path = screen_dir / f"{background_b_id}.jpg"
      
        background_a = cv2.imread(str(background_a_path))
        background_b = cv2.imread(str(background_b_path))
        if background_a is None or background_b is None:
            raise FileNotFoundError(
                f"Could not load backgrounds: {background_a_path}, {background_b_path}"
            )

        # Match template generation from preprocessing: blend -> blur -> downsample.
        template = cv2.addWeighted(background_a, 1.0 - dissolve, background_b, dissolve, 0)
        template = separable_gaussian_blur(template, kernel_size=300, sigma=50)
        template = cv2.resize(template, (60, 120), interpolation=cv2.INTER_LINEAR)  # (width, height)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
        template = torch.from_numpy(template).permute(2, 0, 1).contiguous().float().mul_(1.0 / 255.0)

        gt = torch.tensor([
            float(metadata_data["gt_x_px"]) * 0.001,  # convert to unit of 1000 px to normalize the value in 0-1 range.
            float(metadata_data["gt_y_px"]) * 0.001,  # convert to unit of 1000 px to normalize the value in 0-1 range.
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
                              drop_last=True, num_workers=4, pin_memory=True,
                              worker_init_fn=seed_worker,
                              generator=model_init_generator)

    model = GazeModel_RGBT().to(device)
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
        running_mean_cm = 0.0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", position=0)):
            l_eye = batch['l_eye'].to(device)
            r_eye = batch['r_eye'].to(device)
            template = batch['template'].to(device)
            lmk = batch['lmk'].to(device)
            gt = batch['gt'].to(device)

            if SAVE_EYE_DEBUG and batch_idx == 0:
                eye_root = Path("debug_rgbt_eyes") / f"epoch_{epoch + 1}"
                for i in range(l_eye.shape[0]):
                    subj = int(batch["subject"][i])
                    frame = batch["frame_num"][i]
                    subdir = eye_root / f"p{subj}"
                    subdir.mkdir(parents=True, exist_ok=True)
                    out_l = subdir / f"{frame}_eye_l.png"
                    out_r = subdir / f"{frame}_eye_r.png"
                    if not out_l.exists():
                        img_l = (
                            l_eye[i].detach().cpu().clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
                        )
                        cv2.imwrite(str(out_l), cv2.cvtColor(img_l, cv2.COLOR_RGB2BGR))
                    if not out_r.exists():
                        img_r = (
                            r_eye[i].detach().cpu().clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
                        )
                        cv2.imwrite(str(out_r), cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR))

            optimizer.zero_grad()
            pred = model(l_eye, r_eye, template, lmk)
            loss = torch.norm(pred - gt, dim=1).mean()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                running_mean_cm += _mean_l2_error_cm(pred, gt).item()

        n_batches = len(train_loader)
        epoch_loss_cm = running_mean_cm / n_batches

        wandb.log({
            "epoch": epoch + 1,
            "train_loss_cm": epoch_loss_cm,
        })

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train loss (cm): {epoch_loss_cm:.3f}")

        checkpoint_path = (
            f"{CHECKPOINT_DIR}/{TYPE}_{run_tag}_Epoch{epoch + 1}_TrainLoss{epoch_loss_cm:.3f}.pt"
        )
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)

    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Train RGB gaze prediction model on all subjects")
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