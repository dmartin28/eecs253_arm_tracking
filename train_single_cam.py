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
from single_cam_pose_dataset import PoseDataset

# Import your model classes here
from model import SingleCameraPosePredictor 

# Will need to compute pose statistics to normalize the dataset
def compute_pose_statistics(csv_path, pose_columns=['x', 'y', 'z']):
    """
    Compute mean and std (or min/max) for each pose dimension
    """
    df = pd.read_csv(csv_path)
    
    pose_data = df[pose_columns].values  # Shape: (N, 3)
    
    # Compute statistics
    mean = pose_data.mean(axis=0)
    std = pose_data.std(axis=0)
    min_vals = pose_data.min(axis=0)
    max_vals = pose_data.max(axis=0)
    
    stats = {
        'mean': mean,
        'std': std,
        'min': min_vals,
        'max': max_vals
    }
    
    # print("\n✅ Pose Statistics (from training data):")
    # for dim, name in enumerate(pose_columns):
    #     print(f"  {name}: mean={mean[dim]:.4f}, std={std[dim]:.4f}, "
    #           f"min={min_vals[dim]:.4f}, max={max_vals[dim]:.4f}")
    
    return stats

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.6f}")
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            
            # Store for analysis
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    val_loss = running_loss / len(dataloader)
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate per-dimension errors
    errors = torch.abs(all_predictions - all_targets)
    mean_errors = errors.mean(dim=0)
    
    return val_loss, mean_errors, all_predictions, all_targets


def plot_training_curves(train_losses, val_losses, save_path='training_curves.png'):
    """Plot and save training curves"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved to {save_path}")


def main():
    # ========================================================================
    # CONFIGURATION - MODIFY THESE PATHS
    # ========================================================================
    
    # Dataset paths
    TRAIN_CSV = "C:/Users/david/Projects/eecs253_arm_tracking/episode_84/train_data.csv"  # CSV with columns: image_name, x, y, z
    VAL_CSV = "C:/Users/david/Projects/eecs253_arm_tracking/episode_84/test_data.csv"
    IMAGE_DIR = "C:/Users/david/Projects/eecs253_arm_tracking/episode_84/exterior_image_2_left" # Directory containing RGB images
    
    # Model configuration
    IMAGE_HEIGHT = 180
    IMAGE_WIDTH = 320
    IMAGE_CHANNELS = 3
    POSE_CHANNELS = 3  # x, y, z
    EMBED_DIM = 64
    
    # Training hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    
    # Other settings
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_DIR = './checkpoints'
    NUM_WORKERS = 4  # For DataLoader
    
    # ========================================================================
    # SETUP
    # ========================================================================
    
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"Using device: {DEVICE}")
    print(f"Training with {NUM_EPOCHS} epochs, batch size {BATCH_SIZE}")
    
    # Create datasets
    train_stats = compute_pose_statistics(TRAIN_CSV)
    val_stats = compute_pose_statistics(VAL_CSV)
    print("Train Stats: ")
    print(train_stats)
    print("Val Stats")
    print(val_stats)
    train_dataset = PoseDataset(TRAIN_CSV, IMAGE_DIR, pose_stats=train_stats)
    
    # Create validation dataset if path provided
    val_dataset = None
    if os.path.exists(VAL_CSV):
        val_dataset = PoseDataset(VAL_CSV, IMAGE_DIR)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    # Initialize model
    model = SingleCameraPosePredictor(
        image_channels=IMAGE_CHANNELS,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        pose_channels=POSE_CHANNELS,
        embed_dim=EMBED_DIM, 
    )
    model = model.to(DEVICE)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    print("=" * 70)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)
        print(f"Train Loss: {train_loss:.6f}")
        
        # Validate
        val_loss, mean_errors, _, _ = validate(model, val_loader, criterion, DEVICE)
        val_losses.append(val_loss)
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Mean Errors per dimension: {mean_errors.numpy()}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(SAVE_DIR, 'best_model.pth'))
            print(f"✓ Saved best model (val_loss: {val_loss:.6f})")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, os.path.join(SAVE_DIR, f'checkpoint_epoch_{epoch + 1}.pth'))
            print(f"✓ Saved checkpoint at epoch {epoch + 1}")
    
    # ========================================================================
    # SAVE FINAL MODEL AND PLOTS
    # ========================================================================
    
    # Save final model
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, os.path.join(SAVE_DIR, 'final_model.pth'))
    
    # Plot training curves
    if val_loader:
        plot_training_curves(train_losses, val_losses, 
                           os.path.join(SAVE_DIR, 'training_curves.png'))
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Models saved to: {SAVE_DIR}")
    if val_loader:
        print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()