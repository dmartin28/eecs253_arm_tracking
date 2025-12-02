import torch
import torch.nn as nn

# This file contains code to define the model that will be used to process images to predict robot arm trajectors

# Single image convolutional network class
class SingleImageCNN(nn.Module):
    """This CNN extracts features from a single image"""
    
    def __init__(self, image_height=180, image_width=320, input_channels = 3, output_dim = 64, dropout=0.1):
        super(SingleImageCNN, self).__init__()
        
        # Convolutional layers: Shape is [b,c,h,w]
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1) # [b,3,180,320] -> [b,32,180,320]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # [b,32,180,320] -> [b,64,180,320]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # [b,64,180,320] -> [b,128,180,320]
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Fully connected layers. #After 2 pooling layers becomes 45 x 80 (Note: This may be too large.)
        self.fc1 = nn.Linear(128 * 45 * 80, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, output_dim)

