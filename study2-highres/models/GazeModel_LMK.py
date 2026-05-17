import torch
import torch.nn as nn   
import torch.nn.functional as F
import timm

class GazeModel_LMK(nn.Module):
    def __init__(self):
        super().__init__()
         
        # FC layers for eye landmarks
        self.fc1_landmarks = nn.Linear(8, 128)
        self.fc2_landmarks = nn.Linear(128, 16)
        self.fc3_landmarks = nn.Linear(16, 16)

        # MLP head
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 4)
        self.fc6 = nn.Linear(4, 2)

    def forward(self, eye_landmarks):
        eye_landmarks_tensor = F.relu(self.fc1_landmarks(eye_landmarks))
        eye_landmarks_tensor = F.relu(self.fc2_landmarks(eye_landmarks_tensor))
        eye_landmarks_tensor = F.relu(self.fc3_landmarks(eye_landmarks_tensor)) 

        x = F.dropout(F.relu(self.fc4(eye_landmarks_tensor)), p=0.12, training=self.training)
        x = F.relu(self.fc5(x))
        x = self.fc6(x)        
        return x