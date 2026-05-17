import torch
import torch.nn as nn   
import torch.nn.functional as F
import timm

class GazeModelRGB_FF(nn.Module):
    def __init__(self):
        super().__init__()
     
        self.mobilenet_backbone = timm.create_model(
            'mobilenetv4_conv_small.e1200_r224_in1k',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
            global_pool='avg'  
        )

        self.mobilenet_backbone_face = timm.create_model(
            'mobilenetv4_conv_small.e1200_r224_in1k',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
            global_pool='avg'  
        )
                
        # FC layers for eye landmarks
        self.fc1_landmarks = nn.Linear(8, 128)
        self.fc2_landmarks = nn.Linear(128, 16)
        self.fc3_landmarks = nn.Linear(16, 16)

        # MLP head
        self.fc4 = nn.Linear(1280 + 1280 + 1280 + 16, 8)
        self.fc5 = nn.Linear(8, 4)
        self.fc6 = nn.Linear(4, 2)

    def forward(self, face, left_eye, right_eye, eye_landmarks):
        face_tensor = self.mobilenet_backbone_face(face)
        left_eye_tensor = self.mobilenet_backbone(left_eye)
        right_eye_tensor = self.mobilenet_backbone(right_eye)

        eye_landmarks_tensor = F.relu(self.fc1_landmarks(eye_landmarks))
        eye_landmarks_tensor = F.relu(self.fc2_landmarks(eye_landmarks_tensor))
        eye_landmarks_tensor = F.relu(self.fc3_landmarks(eye_landmarks_tensor)) 

        concat = torch.cat([
            face_tensor,    # 1280
            left_eye_tensor,    # 1280
            right_eye_tensor,   # 1280
            eye_landmarks_tensor # 16
        ], dim=1)

        x = F.dropout(F.relu(self.fc4(concat)), p=0.12, training=self.training)
        x = F.relu(self.fc5(x))
        x = self.fc6(x)        
        return x