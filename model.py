import torch
import torch.nn as nn
import torch.nn.functional as F

# This file contains code to define the model that will be used to process images to predict robot arm trajectors

# Single image convolutional network class
class SingleImageCNN(nn.Module):
    """This CNN extracts features from a single image"""
    
    def __init__(self, image_channels, image_height, image_width, output_dim, dropout):
        super(SingleImageCNN, self).__init__()
        
        # Convolutional layers: Shape is [b,c,h,w]
        self.conv1 = nn.Conv2d(image_channels, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.AvgPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Fully connected layers. #After 3 pooling layers becomes 22 x 40 (Note: This may be too large.)
        self.fc1 = nn.Linear(16 * 22 * 40, output_dim)

    def forward(self, x):
        # Apply ReLU after EACH conv layer    # [b,c,h,w]
        x = self.pool(F.relu(self.conv1(x)))  # [b,3,180,320] -> [b,16,90,160]
        x = self.pool(F.relu(self.conv2(x)))  # [b,16,90,160] -> [b,32,45,80]
        x = self.pool(F.relu(self.conv3(x)))  # [b,32,45,80] -> [b,64,22,40]
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc1(x)  # Map to proper output dimension
        
        return x
    
    
class FeatureDecoder(nn.Module):
    """This MLP extracts decodes features into pose estimates"""

    def __init__(self, input_channels, output_dim, dropout):
        super(FeatureDecoder, self).__init__()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(input_channels, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        
        # Apply ReLU after FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # NO activation on pose prediction (regression task)
        x = self.fc3(x)  #

        return x
    
class SingleCameraPosePredictor(nn.Module):
    """This model predicts pose using a single camera angle"""
    
    def __init__(self, image_channels, image_height, image_width, pose_channels, embed_dim , dropout):
        super(SingleCameraPosePredictor, self).__init__()
        
        self.image_encoder1 = SingleImageCNN(image_channels, image_height, image_width, output_dim=embed_dim, dropout=dropout)

        self.pose_decoder = FeatureDecoder(input_channels=embed_dim, output_dim=pose_channels, dropout=dropout)
        
    def forward(self,x):
        x = self.image_encoder1(x)
        y = self.pose_decoder(x)
        return y
    
class MultiCameraPosePredictor(nn.Module):
    """This model predicts pose using three different camera angles"""

    def __init__(self, image_channels, image_height, image_width, pose_channels, embed_dim, dropout):
        super(MultiCameraPosePredictor, self).__init__()
        
        self.image_encoder1 = SingleImageCNN(image_channels, image_height, image_width, embed_dim, dropout)
        self.image_encoder2 = SingleImageCNN(image_channels, image_height, image_width, embed_dim, dropout)
        self.image_encoder3 = SingleImageCNN(image_channels, image_height, image_width, embed_dim, dropout)

        self.pose_decoder = FeatureDecoder(input_channels=3*embed_dim, output_dim=pose_channels, dropout=dropout)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch, 3, channels, height, width]
               where 3 is the number of cameras
        
        Returns:
            pose: Tensor of shape [batch, pose_channels]
        """
        # Split the stacked images into individual camera views
        # x shape: [batch, 3, channels, height, width]
        external1 = x[:, 0, :, :, :]  # [batch, channels, height, width]
        external2 = x[:, 1, :, :, :]  # [batch, channels, height, width]
        wrist = x[:, 2, :, :, :]      # [batch, channels, height, width]
        
        # Encode each camera view separately
        x1 = self.image_encoder1(external1)  # [batch, embed_dim]
        x2 = self.image_encoder2(external2)  # [batch, embed_dim]
        x3 = self.image_encoder3(wrist)      # [batch, embed_dim]
        
        # Concatenate features
        combined_features = torch.cat([x1, x2, x3], dim=1)  # [batch, 3*embed_dim]
        
        y = self.pose_decoder(combined_features)
        
        return y
