import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import json
import os
import sys
import wandb
from model_architecture import iTrackerCNN

DATASET_DIR =  "GazeCapture"

def save_debug_files(img_rgb, left_crop_coord, right_crop_coord, eye_landmarks_tensor, gaze_label_tensor, subject, frame_num):
    """Save debug files including cropped eye images and landmark/gaze data."""
    # Create base name for files
    base_name = f"{subject}_{frame_num}"
    
    # Create debug directory if it doesn't exist
    debug_dir = Path.cwd() / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Crop and save left eye
    left_eye = img_rgb[
        left_crop_coord[2]:left_crop_coord[3],
        left_crop_coord[0]:left_crop_coord[1]
    ]
    left_eye = cv2.flip(left_eye, 1)  # Flip horizontally
    cv2.imwrite(str(debug_dir / f"{base_name}_left_crop.jpg"), 
                cv2.cvtColor(left_eye, cv2.COLOR_RGB2BGR))
    
    # Crop and save right eye
    right_eye = img_rgb[
        right_crop_coord[2]:right_crop_coord[3],
        right_crop_coord[0]:right_crop_coord[1]
    ]
    cv2.imwrite(str(debug_dir / f"{base_name}_right_crop.jpg"), 
                cv2.cvtColor(right_eye, cv2.COLOR_RGB2BGR))
    
    # Save landmarks and gaze data as JSON
    debug_data = {
        'left_crop_coord': left_crop_coord.tolist(),
        'right_crop_coord': right_crop_coord.tolist(),
        'eye_landmarks_tensor': eye_landmarks_tensor.tolist(),
        'gaze_label_tensor': gaze_label_tensor.tolist()
    }
    
    with open(debug_dir / f"{base_name}_info.json", 'w') as f:
        json.dump(debug_data, f, indent=2)

def process_row(row):
    """Process a single row from the CSV file."""
    try:
        # Get subject and frame info
        subject = f"{int(row['subject']):05d}"
        frame_num = f"{int(row['frame_num']):05d}"
        
        # Create image path
        frame_path = f"{DATASET_DIR}/{subject}/frames/{frame_num}.jpg"

        # Get crop coordinates
        left_crop_coord = np.array([
            int(row['left_eye_left_x']),
            int(row['left_eye_right_x']),
            int(row['left_eye_top_y']),
            int(row['left_eye_bottom_y'])
        ], dtype=np.int32)
        
        right_crop_coord = np.array([
            int(row['right_eye_left_x']),
            int(row['right_eye_right_x']),
            int(row['right_eye_top_y']),
            int(row['right_eye_bottom_y'])
        ], dtype=np.int32)
                
        eye_landmarks_tensor = torch.tensor([float(row['left_eye_inner_x']), 
                                             float(row['left_eye_inner_y']),
                                             float(row['left_eye_outer_x']),
                                             float(row['left_eye_outer_y']),
                                             float(row['right_eye_inner_x']),
                                             float(row['right_eye_inner_y']),
                                             float(row['right_eye_outer_x']),
                                             float(row['right_eye_outer_y']),
                                             ], dtype=torch.float32) # [8]
                    
        # GT gaze position to tensor [2]
        gaze_label_tensor = torch.tensor([float(row['gaze_x_cm']), float(row['gaze_y_cm'])], dtype=torch.float32)

        # Debug: Save cropped eye images and a json file for landmarks and GT gaze label for debugging
        # img = cv2.imread(frame_path)
        # if img is not None:
        #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     save_debug_files(
        #         img_rgb, 
        #         left_crop_coord, 
        #         right_crop_coord, 
        #         eye_landmarks_tensor, 
        #         gaze_label_tensor,
        #         subject,
        #         frame_num
        #     )
        
        # Return tuple of processed data
        return (frame_path, left_crop_coord, right_crop_coord, eye_landmarks_tensor, gaze_label_tensor)
        
    except Exception as e:
        print(f"Error processing row for subject {int(row['subject'])}, frame {int(row['frame_num'])}: {str(e)}")
        return None

def prepare_dataset():
    # Set paths
    base_dir = Path.cwd()  # Assuming running from project root
    csv_path = base_dir / "GazeCapture_csv_portrait.csv"    
    
    print("Reading CSV file...")
    df = pd.read_csv(csv_path) 
    
    # Process each row
    data_list = []
    print("Processing rows...")
    for _, row in tqdm(df.iterrows(), desc="Reading GazeCapture_csv_portrait.csv", total=len(df)):
        processed_data = process_row(row)
        if processed_data is not None:
            data_list.append(processed_data)
        
    print(f"gaze_data.csv: Processed {len(data_list)} valid samples out of {len(df)} total rows")
    return data_list

class EyeTrackingDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        """
        Args:
            data_list: List of tuples containing:
                - left_eye_path: str (path to image)
                - left_crop_coord: np.array[4] (left, right, top, bottom)
                - right_crop_coord: np.array[4] (left, right, top, bottom)
                - eye_landmarks: tensor[8]  # (x1,y1,x2,y2,...,x8,y8)
                - gaze: tensor[2]
        """
        self.data = data_list   # the csv file data (string path to the image, eye cropping coordinates and other tensors)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        frame_path, left_crop_coord, right_crop_coord, eye_landmarks_tensor, gaze_label_tensor = self.data[idx]
        
        left_eye = self.process_eye_image(frame_path, left_crop_coord, is_left_eye=True)    # [3, 128, 128]
        right_eye = self.process_eye_image(frame_path, right_crop_coord, is_left_eye=False)  # [3, 128, 128]
        
        return left_eye, right_eye, eye_landmarks_tensor, gaze_label_tensor  # [3, 128, 128], [3, 128, 128], [8], [2]

    def process_eye_image(self, frame_path, crop_coord, is_left_eye=True):
        """Load and process eye image."""
        img = cv2.imread(frame_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Crop
        img = img[crop_coord[2]:crop_coord[3], crop_coord[0]:crop_coord[1]]
        
        # Flip left eye horizontally
        if is_left_eye:
            img = cv2.flip(img, 1)
            
        # Resize and convert to tensor
        img = cv2.resize(img, (128, 128))
        tensor = torch.from_numpy(img).to(torch.float32).permute(2, 0, 1)  # [3, 128, 128]

        # Calculate mean and std for each RGB channel separately
        r_mean = tensor[0].mean()
        g_mean = tensor[1].mean()
        b_mean = tensor[2].mean()
        
        r_std = tensor[0].std()
        g_std = tensor[1].std()
        b_std = tensor[2].std()
        
        # Normalize each channel separately
        tensor[0] = (tensor[0] - r_mean) / r_std
        tensor[1] = (tensor[1] - g_mean) / g_std
        tensor[2] = (tensor[2] - b_mean) / b_std
                
        return tensor
    
   
# Euclidean loss
def euclidean_loss(pred, target):
    return torch.norm(pred - target, dim=1).mean()

def adjust_learning_rate(optimizer, epoch, initial_lr=0.016, decay_rate=0.64):
    """Staircase decay of learning rate every epoch"""
    lr = initial_lr * (decay_rate ** int(epoch))  # int() makes it staircase
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train():
    # Hyperparameters
    batch_size = 256 # should be 256 though
    num_epochs = 10
    initial_lr = 0.016
    decay_rate = 0.64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    

    # Initialize wandb
    wandb.init(
        project="eye-tracking",
        config={
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "initial_lr": initial_lr,
            "decay_rate": decay_rate,
            "architecture": "iTrackerCNN",
            "dataset": "GazeCapture"
        }
    )
    
    # Prepare and load dataset
    print("Preparing dataset...")
    data_list = prepare_dataset()   # gaze_data.csv (The string path to the image, eye cropping coordinates and other tensors)
    dataset = EyeTrackingDataset(data_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=12, pin_memory=True)
    print(f"Dataset loaded with {len(dataset)} samples")

    # Model, optimizer, loss
    model = iTrackerCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, betas=(0.9, 0.999), eps=1e-7)
    
    # Set model to training mode
    model.train()

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "CPU")
    print("Number of CPU cores:", os.cpu_count())

    # Training loop with tqdm for epochs
    for epoch in tqdm(range(num_epochs), desc="Training epochs", position=0):
        epoch_loss = 0.0
        current_lr = adjust_learning_rate(optimizer, epoch, initial_lr, decay_rate)
        print(f"\nEpoch {epoch+1}/{num_epochs}, Learning Rate: {current_lr:.6f}")

        # Inner loop with tqdm for batches        
        for batch_idx, batch in enumerate(tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", position=1, leave=False)):
            left_eye_tensor, right_eye_tensor, eye_landmarks_tensor, gaze_label_tensor = [b.to(device) for b in batch] 
            # left_eye_tensor: [b, 3, 128, 128]
            # right_eye_tensor: [b, 3, 128, 128]
            # eye_landmarks_tensor: [b, 8]
            # gaze_label_tensor: [b, 2]

            if epoch == 0 and batch_idx == 0:
                print(f"Model is on device: {next(model.parameters()).device}")
                print(f"Left eye tensor device: {left_eye_tensor.device}")

            optimizer.zero_grad()   # clear the gradients

            pred = model(left_eye_tensor, right_eye_tensor, eye_landmarks_tensor)  # forward pass
            loss = euclidean_loss(pred, gaze_label_tensor)  # compute the loss
            loss.backward()  # backpropagation
            optimizer.step()  # update the weights
            epoch_loss += loss.item()
            
            # Log batch metrics
            if batch_idx % 10 == 0:  # Log every 10 batches
                wandb.log({
                    "batch_loss": loss.item(),
                    "learning_rate": current_lr,
                    "epoch": epoch + 1,
                    "batch": batch_idx
                })
            
        # Print average loss for the epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")

        # Log epoch metrics
        wandb.log({
            "epoch_loss": avg_epoch_loss,
            "epoch": epoch + 1,
            "learning_rate": current_lr
        })

        # Save model checkpoint
        checkpoint_path = f"eye_tracking_model_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }, checkpoint_path)
        
        # Log model checkpoint to wandb
        # wandb.save(checkpoint_path)

    # Save final model
    final_model_path = "eye_tracking_model_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_epoch_loss,
    }, final_model_path)
    # wandb.save(final_model_path)
    
    # Close wandb run
    wandb.finish()
    print("\nTraining completed!")

if __name__ == "__main__":
    train()