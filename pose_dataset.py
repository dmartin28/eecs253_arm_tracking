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

class PoseDataset(Dataset):
    """Dataset for loading images and pose targets from CSV"""
    
    def __init__(self, csv_path, predict_len, pose_stats=None, 
                 normalization='standardize'):
        """
        Args:
            csv_path: Path to CSV file
            predict_len: Number of past poses to use as input sequence
            pose_stats: Dict with 'mean', 'std', 'min', 'max' for normalization
            normalization: 'standardize', or None
        """
        self.data = pd.read_csv(csv_path)
        self.predict_len = predict_len
        self.pose_stats = pose_stats
        self.normalization = normalization
        
        # Calculate valid dataset length
        # We need predict_len poses as input + 1 pose as target
        self.valid_length = len(self.data) - predict_len
        
        if self.valid_length <= 0:
            raise ValueError(
                f"Dataset too small! Need at least {predict_len + 1} samples, "
                f"but got {len(self.data)}"
            )
        
        print(f"Loaded {len(self.data)} total samples from {csv_path}")
        print(f"Created {self.valid_length} sequences with sequence_length={predict_len}")
        print(f"Each sequence: {predict_len} input poses → 1 target pose")
        if pose_stats is not None:
            print(f"Using pose normalization: {normalization}")
        
    def __len__(self):
        return self.valid_length
    
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
        """
        Returns a sequence of poses and the next pose as target.
        
        Args:
            idx: Index of the sequence start
            
        Returns:
            input_sequence: Tensor of shape [predict_len, pose_dim]
                           Contains poses from idx to idx+predict_len-1
            target_pose: Tensor of shape [pose_dim]
                        The pose at idx+predict_len (next timestep)
        """
        # Get sequence of predict_len poses as input
        input_poses = []
        for i in range(self.predict_len):
            row = self.data.iloc[idx + i]
            pose = torch.tensor(row.iloc[1:4].values.astype(np.float32))
            pose = self.normalize_pose(pose)
            input_poses.append(pose)
        
        # Stack into sequence tensor: [predict_len, pose_dim]
        input_sequence = torch.stack(input_poses, dim=0)
        
        # Get target pose (the next timestep after the sequence)
        target_row = self.data.iloc[idx + self.predict_len]
        target_pose = torch.tensor(target_row.iloc[1:4].values.astype(np.float32))
        target_pose = self.normalize_pose(target_pose)
        
        return input_sequence, target_pose