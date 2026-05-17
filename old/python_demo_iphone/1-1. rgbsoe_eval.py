import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
import random
from GazeModelRGBSOE import GazeModelRGBSOE
import warnings
warnings.filterwarnings('ignore', message=".*You are using `torch.load` with `weights_only=False`.*", category=FutureWarning)
import torch.serialization
import numpy as np
import cv2
import json
import torchvision.transforms as transforms
import pandas as pd

TYPE = "RGBSOE"
#CHECKPOINT_PATH = "checkpoints/250917_DEMO/RGBSOE_Epoch17_TrainError0.033.pt"
CHECKPOINT_PATH = "checkpoints/RGBSOE_checkpoint.pt"

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

class GazePTDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data_samples = []  
        eyepatch_dir = Path(f"data/test2/preprocessed_corrected/input_eyepatch/")
        landmark_dir = Path(f"data/test2/preprocessed_corrected/input_landmark_and_gt/")
        original_landmark_dir = Path(f"data/test2/preprocessed/input_landmark_and_gt/")
    
        # Find all left eye PNG files
        left_files = sorted(eyepatch_dir.glob("*_left.png"))
        
        for left_eye_file in left_files:
            # Get frame number from filename
            frame_num = left_eye_file.stem.replace("_left", "")
            right_eye_file = eyepatch_dir / f"{frame_num}_right.png"
            json_file = landmark_dir / f"{frame_num}_landmark_and_gt.json"
            original_json_file = original_landmark_dir / f"{frame_num}_landmark_and_gt.json"

            # Check if corresponding right eye and JSON files exist
            if right_eye_file.exists() and json_file.exists():
                self.data_samples.append({
                    'left_eye_path': left_eye_file,
                    'right_eye_path': right_eye_file,
                    'json_path': json_file,
                    'original_json_path': original_json_file,
                })

        assert len(self.data_samples) > 0, f"No data samples found in preprocessed"
        print(f"Loaded {len(self.data_samples)} data samples from test")
        
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
        
        with open(sample['original_json_path'], 'r') as f:
            original_json_data = json.load(f)

        soe_cues = torch.tensor([
            json_data["l_iris_to_soe_center_x_norm"],
            json_data["l_iris_to_soe_center_y_norm"],
            json_data["r_iris_to_soe_center_x_norm"],
            json_data["r_iris_to_soe_center_y_norm"],
        ], dtype=torch.float32)
        
        # Extract eye corner landmarks (next 8 values)
        eyecorner_lmk = torch.tensor([
            original_json_data["l_eye_inner_corner_x"],
            original_json_data["l_eye_inner_corner_y"],
            original_json_data["l_eye_outer_corner_x"],
            original_json_data["l_eye_outer_corner_y"],
            original_json_data["r_eye_inner_corner_x"],
            original_json_data["r_eye_inner_corner_y"],
            original_json_data["r_eye_outer_corner_x"],
            original_json_data["r_eye_outer_corner_y"]
        ], dtype=torch.float32)

        # Get frame number from the sample
        frame_num = sample['left_eye_path'].stem.replace("_left", "")

        return l_eye, r_eye, soe_cues, eyecorner_lmk, frame_num
      
def euclidean_loss(pred, target):
    return torch.norm(pred - target, dim=1).mean()

def evaluate_model(model, loader, device):
    model.eval()
    results = []
    
    with torch.no_grad():
        for l_eye, r_eye, soe_cues, eyecorner_lmk, frame_num in tqdm(loader, desc="Evaluating", position=1):
            l_eye, r_eye = l_eye.to(device), r_eye.to(device)
            soe_cues, eyecorner_lmk = soe_cues.to(device), eyecorner_lmk.to(device)
            
            pred = model(l_eye, r_eye, soe_cues, eyecorner_lmk)
            
            # Convert predictions to numpy
            pred_norm = pred.cpu().numpy()  # Normalized predictions
            
            # Scale to cm coordinates
            pred_cm = np.stack([
                pred_norm[:, 0] * (7.1000/0.430),
                pred_norm[:, 1] * (14.3915/0.8716)
            ], axis=1)
            
            # Store results for each sample in the batch
            for i in range(len(frame_num)):
                results.append({
                    'frame_num': frame_num[i],
                    'pred_px_norm_x': pred_norm[i, 0],
                    'pred_px_norm_y': pred_norm[i, 1],
                    'pred_cm_x': pred_cm[i, 0],
                    'pred_cm_y': pred_cm[i, 1]
                })
    
    return results

def evaluate_and_save_results(device):
    test_dataset = GazePTDataset()
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            drop_last=False, num_workers=4, pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=model_init_generator)
    
    model = GazeModelRGBSOE(generator=model_init_generator).to(device)
    
    # Load pretrained checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading pretrained model from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        
        # Load model state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model state dict loaded successfully")
        else:
            # If checkpoint only contains model state dict directly
            model.load_state_dict(checkpoint)
            print("Model state dict loaded successfully (direct format)")        
    else:
        print(f"Error: Checkpoint file {CHECKPOINT_PATH} not found.")
        return
    
    # Run evaluation
    print("Starting evaluation...")
    results = evaluate_model(model, test_loader, device)
    
    # Convert results to DataFrame and save as CSV
    df = pd.DataFrame(results)
    
    # Sort by frame_num (small to large)
    df['frame_num'] = df['frame_num'].astype(int)
    df = df.sort_values('frame_num').reset_index(drop=True)
    
    csv_filename = f"{TYPE}_eval2.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")
    print(f"Total samples evaluated: {len(results)}")
    
    # Print some statistics
    print(f"\nPrediction statistics:")
    print(f"  Normalized X range: [{df['pred_px_norm_x'].min():.4f}, {df['pred_px_norm_x'].max():.4f}]")
    print(f"  Normalized Y range: [{df['pred_px_norm_y'].min():.4f}, {df['pred_px_norm_y'].max():.4f}]")
    print(f"  CM X range: [{df['pred_cm_x'].min():.2f}, {df['pred_cm_x'].max():.2f}]")
    print(f"  CM Y range: [{df['pred_cm_y'].min():.2f}, {df['pred_cm_y'].max():.2f}]")

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
 
    evaluate_and_save_results(device)

if __name__ == "__main__":
    main()