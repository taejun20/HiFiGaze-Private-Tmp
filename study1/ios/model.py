
import torch
import torch.nn as nn   
import torch.nn.functional as F
import timm

# v1: mobileNet, shared weights, 
class HiFiGaze_RGBModel_v1(nn.Module):
    def __init__(self):
        super().__init__()
     
        self.backbone = timm.create_model('mobilenetv4_conv_small.e1200_r224_in1k', pretrained=True, num_classes=0, global_pool='avg')
    
        # FC layers for eye landmarks
        self.fc1_landmarks = nn.Linear(8, 128)
        self.fc2_landmarks = nn.Linear(128, 16)
        self.fc3_landmarks = nn.Linear(16, 16)

        # MLP head
        self.fc4 = nn.Linear(1280 + 1280 + 16, 8)
        self.fc5 = nn.Linear(8, 4)
        self.fc6 = nn.Linear(4, 2)

    def forward(self, left_eye, right_eye, eye_landmarks):
        left_eye_tensor = self.backbone(left_eye)
        right_eye_tensor = self.backbone(right_eye)

        eye_landmarks_tensor = F.relu(self.fc1_landmarks(eye_landmarks))
        eye_landmarks_tensor = F.relu(self.fc2_landmarks(eye_landmarks_tensor))
        eye_landmarks_tensor = F.relu(self.fc3_landmarks(eye_landmarks_tensor)) 

        concat = torch.cat([
            left_eye_tensor,    # 1280
            right_eye_tensor,   # 1280
            eye_landmarks_tensor, # 16
        ], dim=1)

        x = F.dropout(F.relu(self.fc4(concat)), p=0.12, training=self.training)
        x = F.relu(self.fc5(x))
        x = self.fc6(x)        
        return x

# v2: mobileNet, no shared backbone (input is concatenated eye side by side)
class HiFiGaze_RGBModel_v2(nn.Module):
    def __init__(self):
        super().__init__()
     
        self.backbone = timm.create_model('mobilenetv4_conv_small.e1200_r224_in1k', pretrained=True, num_classes=0, global_pool='avg')
        self.mlp = nn.Sequential(
            nn.Linear(1280 + 16, 8),
            nn.ReLU(),
            nn.Dropout(p=0.12),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )
        # FC layers for eye landmarks
        self.mlp_lmk = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
        )

    def forward(self, eye, lmk):
        eye_features = self.backbone(eye)
        lmk_tensor = self.mlp_lmk(lmk)
        concat = torch.cat([
            eye_features,    # 1280
            lmk_tensor, # 16
        ], dim=1)

        return self.mlp(concat)

# v2_nolmk: v2 but no lmk
class HiFiGaze_RGBModel_v2_nolmk(nn.Module):
    def __init__(self):
        super().__init__()
     
        self.backbone = timm.create_model('mobilenetv4_conv_small.e1200_r224_in1k', pretrained=True, num_classes=0, global_pool='avg')
        self.mlp = nn.Sequential(
            nn.Linear(1280, 8),
            nn.ReLU(),
            nn.Dropout(p=0.12),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )

    def forward(self, eye):
        eye_features = self.backbone(eye)
        return self.mlp(eye_features)

# v3: FastViT, no shared backbone
class HiFiGaze_RGBModel_v3(nn.Module):
    def __init__(self):
        super().__init__()
     
        self.backbone = timm.create_model('fastvit_t8.apple_dist_in1k', pretrained=True, num_classes=0)
        self.mlp = nn.Sequential(
            nn.Linear(768 + 16, 8),
            nn.ReLU(),
            nn.Dropout(p=0.12),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )
        # FC layers for eye landmarks
        self.mlp_lmk = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
        )

    def forward(self, eye, lmk):
        eye_features = self.backbone(eye)
        lmk_tensor = self.mlp_lmk(lmk)
        concat = torch.cat([
            eye_features,    # 768
            lmk_tensor, # 16
        ], dim=1)

        return self.mlp(concat)



# RGBT START
# RGBT START
# RGBT START
# RGBT START
# RGBT START
# RGBT START

class HiFiGaze_RGBTModel_v1(nn.Module):     
    def __init__(self):
        super().__init__()
     
        self.backbone = timm.create_model(
            'mobilenetv4_conv_small.e1200_r224_in1k',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
            global_pool='avg'  
        )

        self.template_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # /2
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 🔑 size-invariant
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
        )
    
        # FC layers for eye landmarks
        self.fc1_landmarks = nn.Linear(8, 128)
        self.fc2_landmarks = nn.Linear(128, 16)
        self.fc3_landmarks = nn.Linear(16, 16)

        # MLP head
        self.fc4 = nn.Linear(1280 + 1280 + 256 + 16, 8)
        self.fc5 = nn.Linear(8, 4)
        self.fc6 = nn.Linear(4, 2)

    def forward(self, left_eye, right_eye, template, eye_landmarks):
        left_eye_tensor = self.backbone(left_eye)
        right_eye_tensor = self.backbone(right_eye)
        template_tensor = self.template_cnn(template)

        eye_landmarks_tensor = F.relu(self.fc1_landmarks(eye_landmarks))
        eye_landmarks_tensor = F.relu(self.fc2_landmarks(eye_landmarks_tensor))
        eye_landmarks_tensor = F.relu(self.fc3_landmarks(eye_landmarks_tensor)) 

        concat = torch.cat([
            left_eye_tensor,    # 1280
            right_eye_tensor,   # 1280
            template_tensor,    # 256
            eye_landmarks_tensor, # 16
        ], dim=1)

        x = F.dropout(F.relu(self.fc4(concat)), p=0.12, training=self.training)
        x = F.relu(self.fc5(x))
        x = self.fc6(x)        
        return x

# v2: mobileNet, no shared backbone (input is concatenated eye side by side)
class HiFiGaze_RGBTModel_v2(nn.Module):
    def __init__(self):
        super().__init__()
     
        self.backbone = timm.create_model('mobilenetv4_conv_small.e1200_r224_in1k', pretrained=True, num_classes=0, global_pool='avg')
        self.screen_backbone = timm.create_model('mobilenetv4_conv_small.e1200_r224_in1k', pretrained=True, num_classes=0, global_pool='avg')
        self.mlp = nn.Sequential(
            nn.Linear(1280*2 + 16, 8),
            nn.ReLU(),
            nn.Dropout(p=0.12),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )
        # FC layers for eye landmarks
        self.mlp_lmk = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
        )

    def forward(self, eye, screen, lmk):
        eye_features = self.backbone(eye)
        screen_features = self.screen_backbone(screen)
        lmk_tensor = self.mlp_lmk(lmk)
        concat = torch.cat([
            eye_features,    # 1280
            screen_features, # 1280
            lmk_tensor, # 16
        ], dim=1)

        return self.mlp(concat)