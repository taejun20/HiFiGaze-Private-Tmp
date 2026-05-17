import torch
import torch.nn as nn   
import torch.nn.functional as F

class iTrackerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared CNN layers for both eye images
        self.conv1 = nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(128, momentum=0.1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # FC layers for eye landmarks (shared between left and right)
        self.fc1_landmarks = nn.Linear(8, 128)  # 8 = 2 * 2 * 2 (inner and outer corner landmarks for left and right eyes)
        self.fc2_landmarks = nn.Linear(128, 16)
        self.fc3_landmarks = nn.Linear(16, 16)
        
        self.fc4 = nn.Linear(128 * 8 * 8 * 2 + 16, 8)
        self.fc5 = nn.Linear(8, 4)
        self.fc6 = nn.Linear(4, 2)

    def process_eye_image(self, x):
        # Process through CNN layers
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout(x, p=0.02, training=self.training)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout(x, p=0.02, training=self.training)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return x 

    def process_eye_landmarks(self, landmarks):
        # Process through FC layers for landmarks
        x = F.relu(self.fc1_landmarks(landmarks))
        x = F.relu(self.fc2_landmarks(x))
        x = F.relu(self.fc3_landmarks(x))
        
        return x

    def forward(self, left_eye, right_eye, eye_landmarks):
        """
        Args:
            left_eye: tensor of shape (batch_size, 3, 128, 128)
            right_eye: tensor of shape (batch_size, 3, 128, 128)
            eye_landmarks: tensor of shape (batch_size, 8) - flattened x,y coordinates of four eye landmarks
        Returns:
            gaze_prediction: tensor of shape (batch_size, 2) - x,y gaze coordinates
        """
        # Process eye images through shared CNN
        left_cnn = self.process_eye_image(left_eye)
        right_cnn = self.process_eye_image(right_eye)
        
        # Process landmarks through shared FC layers
        eye_landmarks_linear = self.process_eye_landmarks(eye_landmarks)
        
        # Concatenate all features
        concat = torch.cat([left_cnn, right_cnn, eye_landmarks_linear], dim=1)
        
        # Final FC layers
        x = F.relu(self.fc4(concat))
        x = F.relu(self.fc5(x))
        gaze = self.fc6(x)  # No ReLU after final layer
        
        return gaze
