import torch
import torch.nn as nn   
import torch.nn.functional as F
import timm

class GazeModelRGBH_CAL(nn.Module):
    def __init__(self):
        super().__init__()
       
        self.mobilenet_backbone = timm.create_model(
            'mobilenetv4_conv_small.e1200_r224_in1k',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
            global_pool='avg'  
        )

        # Input size is 40x55, 1 channel.
        self.heatmap_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: 20x27
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: 10x13
            nn.Flatten(),
            nn.Linear(32 * 10 * 13, 16)  # Output size of 16 features
        )
 
        # FC layers for eye landmarks
        self.fc1_landmarks = nn.Linear(8, 128)
        self.fc2_landmarks = nn.Linear(128, 16)
        self.fc3_landmarks = nn.Linear(16, 16)

        # MLP head
        self.fc4 = nn.Linear(1280 + 1280 + 16 + 16 + 16, 8)  # Added 2 for valid flags
        self.fc5 = nn.Linear(8, 4)
        self.fc6 = nn.Linear(4, 2)

    def forward(self, left_eye, right_eye, left_heatmap, right_heatmap, eye_landmarks):
        left_eye_tensor = self.mobilenet_backbone(left_eye)
        right_eye_tensor = self.mobilenet_backbone(right_eye)

        left_heatmap_tensor = self.heatmap_cnn(left_heatmap)
        right_heatmap_tensor = self.heatmap_cnn(right_heatmap)

        # Process landmarks
        eye_landmarks_tensor = F.relu(self.fc1_landmarks(eye_landmarks))
        eye_landmarks_tensor = F.relu(self.fc2_landmarks(eye_landmarks_tensor))
        eye_landmarks_tensor = F.relu(self.fc3_landmarks(eye_landmarks_tensor)) 

        # Concatenate all features including valid flags
        concat = torch.cat([
            left_eye_tensor,      # 1280
            right_eye_tensor,     # 1280
            left_heatmap_tensor,  # 16
            right_heatmap_tensor, # 16
            eye_landmarks_tensor  # 16
        ], dim=1)

        x = F.dropout(F.relu(self.fc4(concat)), p=0.12, training=self.training)
        penultimate = self.fc5(x)
        x = self.fc6(F.relu(penultimate))        
        return penultimate, x