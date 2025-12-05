import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

class MultiCamPoseDataset(Dataset):
    """Dataset for loading images and pose targets from CSV"""
    
    def __init__(self, csv_path, image_dirs, pose_stats=None, 
                 normalization='standardize'):
        """
        Args:
            csv_path: Path to CSV file
            image_dir_path: Directory containing images
            pose_stats: Dict with 'mean', 'std', 'min', 'max' for normalization
            normalization: 'standardize', or None
        """
        self.data = pd.read_csv(csv_path)
        self.pose_stats = pose_stats
        self.normalization = normalization
        self.external1_dir = Path(image_dirs[0])
        self.external2_dir = Path(image_dirs[1])
        self.wrist_dir = Path(image_dirs[2])
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL Image to tensor [C, H, W] and scales to [0, 1]
        ])
        
        print(f"Loaded {len(self.data)} samples from {csv_path}")
        if pose_stats is not None:
            print(f"Using pose normalization: {normalization}")
        
    def __len__(self):
        return len(self.data)
    
    def normalize_pose(self, pose):
        """Normalize pose using pre-computed statistics"""
        if self.pose_stats is None or self.normalization is None:
            return pose
        
        if self.normalization == 'standardize':
            # Z-score normalization: (x - mean) / std
            mean = torch.tensor(self.pose_stats['mean'], dtype=torch.float32)
            std = torch.tensor(self.pose_stats['std'], dtype=torch.float32)
            return (pose - mean) / (std + 1e-8)  # Add epsilon to avoid division by zero
    
    def __getitem__(self, idx):
        # Get row from CSV
        row = self.data.iloc[idx]
        
        # Load image
        image_name = row['frame'] + '.jpg'
        external1_path = self.external1_dir / image_name
        external2_path = self.external2_dir / image_name
        wrist_path = self.wrist_dir / image_name
        
        try:
            external1_img = Image.open(external1_path).convert('RGB')
            external2_img = Image.open(external2_path).convert('RGB')
            wrist_img = Image.open(wrist_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {e}")
            raise
        
        # Convert image to tensor
        external1_img = self.transform(external1_img)
        external2_img = self.transform(external2_img)
        wrist_img = self.transform(wrist_img)
        # This creates a new dimension at index 0 for the camera index
        images_stacked = torch.stack([
            external1_img,  # Camera 0
            external2_img,  # Camera 1
            wrist_img       # Camera 2
        ], dim=0)
        
        # Get pose targets
        pose = torch.tensor(row.iloc[1:4].values.astype(np.float32))
        
        # Normalize pose
        pose = self.normalize_pose(pose)
        
        return images_stacked, pose