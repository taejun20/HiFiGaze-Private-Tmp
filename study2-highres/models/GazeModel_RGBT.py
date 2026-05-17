import torch
import torch.nn as nn   
import torch.nn.functional as F
import timm

class GazeModel_RGBT(nn.Module):
    def __init__(self):
        super().__init__()
       
        self.mobilenet_backbone = timm.create_model(
            'mobilenetv4_conv_small.e1200_r224_in1k',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
            global_pool='avg'  
        )

        # Template input size is 60x120 (width x height), 3-channel PNG tensor.
        self.template_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: 60x30 (H x W)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: 30x15 (H x W)
            nn.Flatten(),
            nn.Linear(32 * (120 // 4) * (60 // 4), 256),
            nn.ReLU(inplace=True)
        )
 
        # FC layers for eye landmarks
        self.fc1_landmarks = nn.Linear(8, 128)
        self.fc2_landmarks = nn.Linear(128, 16)
        self.fc3_landmarks = nn.Linear(16, 16)

        # MLP head
        self.fc4 = nn.Linear(1280 * 2 + 256 + 16, 128)
        self.fc5 = nn.Linear(128, 2)

    def forward(self, left_eye, right_eye, template, eye_landmarks):
        left_eye_tensor = self.mobilenet_backbone(left_eye)
        right_eye_tensor = self.mobilenet_backbone(right_eye)
        template_tensor = self.template_cnn(template)

        # Process landmarks
        eye_landmarks_tensor = F.relu(self.fc1_landmarks(eye_landmarks))
        eye_landmarks_tensor = F.relu(self.fc2_landmarks(eye_landmarks_tensor))
        eye_landmarks_tensor = F.relu(self.fc3_landmarks(eye_landmarks_tensor)) 

        # Concatenate all features
        concat = torch.cat([
            left_eye_tensor,      # 1280
            right_eye_tensor,     # 1280
            template_tensor,      # 256
            eye_landmarks_tensor, # 16
        ], dim=1)

        x = F.dropout(F.relu(self.fc4(concat)), p=0.12, training=self.training)
        x = self.fc5(x)        
        return x