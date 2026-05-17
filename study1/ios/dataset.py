import torch
from pathlib import Path
import cv2
import json
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

def separable_gaussian_blur(img, kernel_size, sigma):
    # Create 1D Gaussian kernel
    k = (kernel_size - 1) // 2
    x = np.linspace(-k, k, kernel_size)
    kernel_1d = np.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Apply horizontal blur
    kernel_h = kernel_1d.reshape(1, -1)
    blurred_h = cv2.filter2D(img, -1, kernel_h)
    
    # Apply vertical blur
    kernel_v = kernel_1d.reshape(-1, 1)
    blurred = cv2.filter2D(blurred_h, -1, kernel_v)
    
    return blurred

class HiFiGaze_Dataset_v2(torch.utils.data.Dataset):
    def __init__(self, subjects, FILTERING_THRESHOLD=4.5):
        # Load eval.csv if it exists to filter out high-loss samples
        self.FILTERING_THRESHOLD = FILTERING_THRESHOLD
        eval_loss_lookup = {}
        eval_csv_path = Path("preprocessed/eval.csv")
        if eval_csv_path.exists():
            try:
                eval_df = pd.read_csv(eval_csv_path)
                for _, row in eval_df.iterrows():
                    subj = int(row['subject'])
                    # Ensure frame_id is an int and then zero-pad to 5 digits (e.g., 03199)
                    frame_id = f"{int(row['frame_id']):05d}"
                    loss = float(row['loss'])
                    eval_loss_lookup[(subj, frame_id)] = loss
                print(f"Loaded {len(eval_loss_lookup)} entries from eval.csv")
            except Exception as e:
                print(f"Warning: Could not load eval.csv: {e}")
        else:
            print("Warning: eval.csv not found, all samples will be included")
        
        self.data_samples = []        
        skipped_count = 0
        for subject in subjects:
            eyepatch_dir = Path(f"preprocessed/p{subject}/eyepatch")
            json_dir = Path(f"preprocessed/p{subject}/json")
            screen_dir = Path(f"preprocessed/p{subject}/screen")

            # Find all combined eye PNG files
            combined_eye_paths = sorted(eyepatch_dir.glob("*_combined.png"))
            for combined_eye_path in combined_eye_paths:
                # Get frame ID from filename (e.g., "04077_combined.png" -> "04077")
                frame_id = combined_eye_path.stem.replace("_combined", "")
                json_path = json_dir / f"{frame_id}.json"
                screen_path = screen_dir / f"{frame_id}.jpg"

                # Check if corresponding JSON file exists
                if json_path.exists():
                    # Check if this sample should be skipped based on eval.csv
                    key = (subject, frame_id)
                    if key in eval_loss_lookup:
                        loss = eval_loss_lookup[key]
                        if loss >= self.FILTERING_THRESHOLD:
                            skipped_count += 1
                            continue  # Skip this sample
                    
                    self.data_samples.append({
                        'subject': subject,
                        'frame_id': frame_id,
                        'json_path': json_path,
                        'eye_path': combined_eye_path,
                        'screen_path': screen_path,
                    })
                else:
                    print(f"No JSON File: {json_path}")

        assert len(self.data_samples) > 0, f"No data samples found in preprocessed for subjects {subjects}"
        print(f"Loaded {len(self.data_samples)} data samples from {len(subjects)} subjects")
        
        print(f"Skipped {skipped_count} samples with loss >= {self.FILTERING_THRESHOLD} (outliers) from the full dataset")
        
        # No eye_transform needed for v2 (images are already preprocessed)
        self.screen_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),  # (H, W)
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, idx):        
        sample = self.data_samples[idx]
        
        # Load combined eye image
        eye_img = cv2.imread(str(sample['eye_path']))
        
        # Convert BGR to RGB
        eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor (no transform needed, image is already preprocessed)
        eye = torch.from_numpy(eye_img).permute(2, 0, 1).float() / 255.0  # [H, W, C] -> [C, H, W] and normalize

        screen_img = cv2.imread(str(sample['screen_path']))
        screen_img = cv2.cvtColor(screen_img, cv2.COLOR_BGR2RGB)
        kernel_size = 301
        sigma = kernel_size / 6
        screen_img = separable_gaussian_blur(screen_img, kernel_size, sigma)
        
        # Save blurred screen image
        # screen_blur_dir = Path(f"preprocessed/p{sample['subject']}/screen_blur")
        # screen_blur_dir.mkdir(parents=True, exist_ok=True)
        # screen_blur_path = screen_blur_dir / f"{sample['frame_id']}.jpg"
        # screen_img_bgr = cv2.cvtColor(screen_img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(str(screen_blur_path), screen_img_bgr)
        
        screen = self.screen_transform(screen_img)  # [3, H, W]

        # Load JSON data
        with open(sample['json_path'], 'r') as f:
            json_data = json.load(f)
        
        L_LEFT_CORNER = 33
        L_RIGHT_CORNER = 133
        R_LEFT_CORNER = 362
        R_RIGHT_CORNER = 263

        # Extract eye corner landmarks
        eyecorner_lmk = torch.tensor([
            json_data[f"{R_LEFT_CORNER}x"],
            json_data[f"{R_LEFT_CORNER}y"],
            json_data[f"{R_RIGHT_CORNER}x"],
            json_data[f"{R_RIGHT_CORNER}y"],
            json_data[f"{L_RIGHT_CORNER}x"],
            json_data[f"{L_RIGHT_CORNER}y"],
            json_data[f"{L_LEFT_CORNER}x"],
            json_data[f"{L_LEFT_CORNER}y"]
        ], dtype=torch.float32)

        # Extract ground truth
        gt = torch.tensor([
            json_data["gt_x_px"] / 1000.0,
            json_data["gt_y_px"] / 1000.0
        ], dtype=torch.float32)
        
        return {
            'subject': sample['subject'],
            'frame_id': sample['frame_id'],
            'eyecorner_lmk': eyecorner_lmk,
            'eye': eye,
            'screen': screen, 
            'gt': gt
        }

class HiFiGaze_Dataset_OutlierFiltered(torch.utils.data.Dataset):
    def __init__(self, subjects, FILTERING_THRESHOLD=4.5):
        # Load eval.csv if it exists to filter out high-loss samples
        self.FILTERING_THRESHOLD = FILTERING_THRESHOLD
        eval_loss_lookup = {}
        eval_csv_path = Path("preprocessed/eval.csv")
        if eval_csv_path.exists():
            try:
                eval_df = pd.read_csv(eval_csv_path)
                for _, row in eval_df.iterrows():
                    subj = int(row['subject'])
                    # Ensure frame_id is an int and then zero-pad to 5 digits (e.g., 03199)
                    frame_id = f"{int(row['frame_id']):05d}"
                    loss = float(row['loss'])
                    eval_loss_lookup[(subj, frame_id)] = loss
                print(f"Loaded {len(eval_loss_lookup)} entries from eval.csv")
            except Exception as e:
                print(f"Warning: Could not load eval.csv: {e}")
        else:
            print("Warning: eval.csv not found, all samples will be included")
        
        self.data_samples = []        
        skipped_count = 0
        for subject in subjects:
            eyepatch_dir = Path(f"preprocessed/p{subject}/eyepatch")
            json_dir = Path(f"preprocessed/p{subject}/json")
            screen_dir = Path(f"preprocessed/p{subject}/screen")

            # Find all left eye PNG files
            l_eye_paths = sorted(eyepatch_dir.glob("*_left.png"))
            for l_eye_path in l_eye_paths:
                # Get frame ID from filename (e.g., "04077_left.png" -> "04077")
                frame_id = l_eye_path.stem.replace("_left", "")
                r_eye_path = eyepatch_dir / f"{frame_id}_right.png"
                json_path = json_dir / f"{frame_id}.json"
                screen_path = screen_dir / f"{frame_id}.jpg"

                # Check if corresponding right eye and JSON files exist
                if r_eye_path.exists():
                    if json_path.exists():
                        # Check if this sample should be skipped based on eval.csv
                        key = (subject, frame_id)
                        if key in eval_loss_lookup:
                            loss = eval_loss_lookup[key]
                            if loss >= self.FILTERING_THRESHOLD:
                                skipped_count += 1
                                continue  # Skip this sample
                        
                        self.data_samples.append({
                            'subject': subject,
                            'frame_id': frame_id,
                            'json_path': json_path,
                            'l_eye_path': l_eye_path,
                            'r_eye_path': r_eye_path,
                            'screen_path': screen_path,
                        })
                    else:
                        print(f"No JSON File: {json_path}")
                else:
                    print(f"No right eye: r_eye_path: {r_eye_path}")

        assert len(self.data_samples) > 0, f"No data samples found in preprocessed for subjects {subjects}"
        print(f"Loaded {len(self.data_samples)} data samples from {len(subjects)} subjects")
        
        print(f"Skipped {skipped_count} samples with loss >= {self.FILTERING_THRESHOLD} (outliers) from the full dataset")
        
        # Initialize transform for resizing images to 500x250
        self.eye_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((250, 500)),  # (height, width)
            transforms.ToTensor()
        ])

        self.screen_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),  # (H, W)
            transforms.ToTensor(),
        ])

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
        
        # Apply transforms (resize and convert to tensor)
        l_eye = self.eye_transform(l_eye_img)  # [3, 250, 500]
        r_eye = self.eye_transform(r_eye_img)  # [3, 250, 500]

        screen_img = cv2.imread(str(sample['screen_path']))
        screen_img = cv2.cvtColor(screen_img, cv2.COLOR_BGR2RGB)
        kernel_size = 301
        sigma = kernel_size / 6
        screen_img = separable_gaussian_blur(screen_img, kernel_size, sigma)
        
        # Save blurred screen image
        screen_blur_dir = Path(f"preprocessed/p{sample['subject']}/screen_blur")
        screen_blur_dir.mkdir(parents=True, exist_ok=True)
        screen_blur_path = screen_blur_dir / f"{sample['frame_id']}.jpg"
        screen_img_bgr = cv2.cvtColor(screen_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(screen_blur_path), screen_img_bgr)
        
        screen = self.screen_transform(screen_img)  # [3, H, W]

        # Load JSON data
        with open(sample['json_path'], 'r') as f:
            json_data = json.load(f)
        
        # MediaPipe indices
        LEFT_EYE_OUTER = 263
        LEFT_EYE_INNER = 362
        RIGHT_EYE_OUTER = 33
        RIGHT_EYE_INNER = 133

        # Extract eye corner landmarks
        eyecorner_lmk = torch.tensor([
            json_data[f"{LEFT_EYE_INNER}x"],
            json_data[f"{LEFT_EYE_INNER}y"],
            json_data[f"{LEFT_EYE_OUTER}x"],
            json_data[f"{LEFT_EYE_OUTER}y"],
            json_data[f"{RIGHT_EYE_INNER}x"],
            json_data[f"{RIGHT_EYE_INNER}y"],
            json_data[f"{RIGHT_EYE_OUTER}x"],
            json_data[f"{RIGHT_EYE_OUTER}y"]
        ], dtype=torch.float32)

        # Extract ground truth
        gt = torch.tensor([
            json_data["gt_x_px"] / 1000.0,
            json_data["gt_y_px"] / 1000.0
        ], dtype=torch.float32)
        
        return {
            'subject': sample['subject'],
            'frame_id': sample['frame_id'],
            'eyecorner_lmk': eyecorner_lmk,
            'l_eye': l_eye,
            'r_eye': r_eye,
            'screen': screen, 
            'gt': gt
        }

class HiFiGaze_Dataset_NoOutlierFiltered(torch.utils.data.Dataset):
    def __init__(self, subjects):
        self.data_samples = []        
        for subject in subjects:
            eyepatch_dir = Path(f"preprocessed/p{subject}/eyepatch")
            json_dir = Path(f"preprocessed/p{subject}/json")
            screen_dir = Path(f"preprocessed/p{subject}/screen")

            # Find all left eye PNG files
            l_eye_paths = sorted(eyepatch_dir.glob("*_left.png"))
            for l_eye_path in l_eye_paths:
                # Get frame ID from filename (e.g., "04077_left.png" -> "04077")
                frame_id = l_eye_path.stem.replace("_left", "")
                r_eye_path = eyepatch_dir / f"{frame_id}_right.png"
                json_path = json_dir / f"{frame_id}.json"
                screen_path = screen_dir / f"{frame_id}.jpg"

                # Check if corresponding right eye and JSON files exist
                if r_eye_path.exists():
                    if json_path.exists():
                        self.data_samples.append({
                            'subject': subject,
                            'frame_id': frame_id,
                            'json_path': json_path,
                            'l_eye_path': l_eye_path,
                            'r_eye_path': r_eye_path,
                            'screen_path': screen_path,
                        })
                    else:
                        print(f"No JSON File: {json_path}")
                else:
                    print(f"No right eye: r_eye_path: {r_eye_path}")

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
        
        # MediaPipe indices
        LEFT_EYE_OUTER = 263
        LEFT_EYE_INNER = 362
        RIGHT_EYE_OUTER = 33
        RIGHT_EYE_INNER = 133

        # Extract eye corner landmarks
        eyecorner_lmk = torch.tensor([
            json_data[f"{LEFT_EYE_INNER}x"],
            json_data[f"{LEFT_EYE_INNER}y"],
            json_data[f"{LEFT_EYE_OUTER}x"],
            json_data[f"{LEFT_EYE_OUTER}y"],
            json_data[f"{RIGHT_EYE_INNER}x"],
            json_data[f"{RIGHT_EYE_INNER}y"],
            json_data[f"{RIGHT_EYE_OUTER}x"],
            json_data[f"{RIGHT_EYE_OUTER}y"]
        ], dtype=torch.float32)

        # Extract ground truth
        gt = torch.tensor([
            json_data["gt_x_px"] / 1000.0,
            json_data["gt_y_px"] / 1000.0
        ], dtype=torch.float32)
        
        return {
            'subject': sample['subject'],
            'frame_id': sample['frame_id'],
            'eyecorner_lmk': eyecorner_lmk,
            'l_eye': l_eye,
            'r_eye': r_eye,
            'gt': gt
        }