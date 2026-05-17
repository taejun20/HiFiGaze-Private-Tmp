import torch
import torch.nn as nn   
import torch.nn.functional as F
import timm

class GazeModelRGBT_IOS(nn.Module):
    def __init__(self):
        super().__init__()
     
        self.mobilenet_backbone = timm.create_model(
            'mobilenetv4_conv_small.e1200_r224_in1k',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
            global_pool='avg'  
        )

        # Input size is 40x55, 1 channel.
        self.template_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: 20x27
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: 10x13
            nn.Flatten(),
            nn.Linear(32 * (50//4) * (101//4), 256),  # Output size of 256 features
            nn.ReLU(inplace=True)
        )
    
        # FC layers for eye landmarks
        self.fc1_landmarks = nn.Linear(8, 128)
        self.fc2_landmarks = nn.Linear(128, 16)
        self.fc3_landmarks = nn.Linear(16, 16)

        self.fc1_landmarks_headaug = nn.Linear(956, 128)
        self.fc2_landmarks_headaug = nn.Linear(128, 16)
        self.fc3_landmarks_headaug = nn.Linear(16, 16)

        # MLP head
        self.fc4 = nn.Linear(1280 + 1280 + 256 + 16 + 16, 8)
        self.fc5 = nn.Linear(8, 4)
        self.fc6 = nn.Linear(4, 2)

    def forward(self, left_eye, right_eye, template, eye_landmarks, face_landmarks):
        left_eye_tensor = self.mobilenet_backbone(left_eye)
        right_eye_tensor = self.mobilenet_backbone(right_eye)
        template_tensor = self.template_cnn(template)

        eye_landmarks_tensor = F.relu(self.fc1_landmarks(eye_landmarks))
        eye_landmarks_tensor = F.relu(self.fc2_landmarks(eye_landmarks_tensor))
        eye_landmarks_tensor = F.relu(self.fc3_landmarks(eye_landmarks_tensor)) 

        face_landmarks_tensor = F.relu(self.fc1_landmarks_headaug(face_landmarks))
        face_landmarks_tensor = F.relu(self.fc2_landmarks_headaug(face_landmarks_tensor))
        face_landmarks_tensor = F.relu(self.fc3_landmarks_headaug(face_landmarks_tensor))

        concat = torch.cat([
            left_eye_tensor,    # 1280
            right_eye_tensor,   # 1280
            template_tensor,      # 256
            eye_landmarks_tensor, # 16
            face_landmarks_tensor  # 16
        ], dim=1)

        x = F.dropout(F.relu(self.fc4(concat)), p=0.12, training=self.training)
        x = F.relu(self.fc5(x))
        x = self.fc6(x)        
        return x