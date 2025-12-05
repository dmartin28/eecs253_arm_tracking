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

class SingleCamPoseDataset(Dataset):
    """Dataset for loading images and pose targets from CSV"""
    
    def __init__(self, csv_path, image_dir_path, pose_stats=None, 
                 normalization='standardize'):
        """
        Args:
            csv_path: Path to CSV file
            image_dir_path: Directory containing images
            pose_stats: Dict with 'mean', 'std', 'min', 'max' for normalization
            normalization: 'standardize', or None
        """
        self.data = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir_path)
        self.pose_stats = pose_stats
        self.normalization = normalization
        
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
        image_path = self.image_dir / image_name
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise
        
        # Convert image to tensor
        image = self.transform(image)
        
        # Get pose targets
        pose = torch.tensor(row.iloc[1:4].values.astype(np.float32))
        
        # Normalize pose
        pose = self.normalize_pose(pose)
        
        return image, pose