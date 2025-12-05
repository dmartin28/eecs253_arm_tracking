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
from multi_cam_pose_dataset import MultiCamPoseDataset

# Import your model classes here
from model import MultiCameraPosePredictor

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
    
    print("\n📊 Pose Statistics:")
    for dim, name in enumerate(pose_columns):
        print(f"  {name}: mean={mean[dim]:.4f}, std={std[dim]:.4f}, "
              f"min={min_vals[dim]:.4f}, max={max_vals[dim]:.4f}")
    
    return stats


def train_epoch(model, dataloader, criterion, optimizer, device, epoch_num):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        # ✅ images shape: [batch, 3_cameras, channels, height, width]
        # ✅ targets shape: [batch, 3] (x, y, z)
        
        images = images.to(device)
        targets = targets.to(device)
        
        # Debug: Print shapes on first batch
        if batch_idx == 0 and epoch_num == 1:
            print(f"\n🔍 Debug - First batch shapes:")
            print(f"  Input images: {images.shape}")  # [batch, 3, C, H, W]
            print(f"  Target poses: {targets.shape}")  # [batch, 3]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass - model handles splitting internally
        outputs = model(images)
        
        if batch_idx == 0 and epoch_num == 1:
            print(f"  Output poses: {outputs.shape}")  # [batch, 3]
        
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
            # ✅ images shape: [batch, 3_cameras, channels, height, width]
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
    max_errors = errors.max(dim=0)[0]
    
    return val_loss, mean_errors, max_errors, all_predictions, all_targets


def plot_training_curves(train_losses, val_losses, save_path='training_curves.png'):
    """Plot and save training curves"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Training curves saved to {save_path}")


def plot_predictions(predictions, targets, pose_stats, save_path='predictions.png'):
    """Plot predictions vs targets for each dimension"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    dim_names = ['X', 'Y', 'Z']
    
    # Denormalize if stats are available
    if pose_stats is not None:
        mean = pose_stats['mean']
        std = pose_stats['std']
        predictions_denorm = predictions.numpy() * std + mean
        targets_denorm = targets.numpy() * std + mean
    else:
        predictions_denorm = predictions.numpy()
        targets_denorm = targets.numpy()
    
    for i, ax in enumerate(axes):
        ax.scatter(targets_denorm[:, i], predictions_denorm[:, i], alpha=0.5, s=10)
        
        # Perfect prediction line
        min_val = min(targets_denorm[:, i].min(), predictions_denorm[:, i].min())
        max_val = max(targets_denorm[:, i].max(), predictions_denorm[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        ax.set_xlabel(f'True {dim_names[i]}')
        ax.set_ylabel(f'Predicted {dim_names[i]}')
        ax.set_title(f'{dim_names[i]} Prediction')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Prediction plots saved to {save_path}")


def test_dataset_and_model(train_dataset, model, device):
    """Test that dataset and model work together"""
    print("\n" + "="*70)
    print("🧪 Testing Dataset and Model Integration")
    print("="*70)
    
    # Get one sample
    images, pose = train_dataset[0]
    print(f"\n📦 Single sample:")
    print(f"  Images shape: {images.shape}")  # Should be [3, C, H, W]
    print(f"  Pose shape: {pose.shape}")      # Should be [3]
    print(f"  Pose values: {pose}")
    
    # Test with batch
    test_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    images_batch, poses_batch = next(iter(test_loader))
    
    print(f"\n📦 Batch test:")
    print(f"  Images batch shape: {images_batch.shape}")  # Should be [4, 3, C, H, W]
    print(f"  Poses batch shape: {poses_batch.shape}")    # Should be [4, 3]
    
    # Test model forward pass
    model.eval()
    with torch.no_grad():
        images_batch = images_batch.to(device)
        output = model(images_batch)
        print(f"\n🤖 Model output:")
        print(f"  Output shape: {output.shape}")  # Should be [4, 3]
        print(f"  Sample output: {output[0]}")
    
    print("\n✅ Dataset and model are compatible!")
    print("="*70 + "\n")


def main():
    # ========================================================================
    # CONFIGURATION - MODIFY THESE PATHS
    # ========================================================================
    
    # Dataset paths
    TRAIN_CSV = "C:/Users/david/Projects/eecs253_arm_tracking/episode_84/train_data.csv"
    VAL_CSV = "C:/Users/david/Projects/eecs253_arm_tracking/episode_84/test_data.csv"
    
    # ✅ THREE DIFFERENT IMAGE DIRECTORIES (one for each camera)
    EXTERNAL1_DIR = "C:/Users/david/Projects/eecs253_arm_tracking/episode_84/exterior_image_1_left"
    EXTERNAL2_DIR = "C:/Users/david/Projects/eecs253_arm_tracking/episode_84/exterior_image_2_left"
    WRIST_DIR = "C:/Users/david/Projects/eecs253_arm_tracking/episode_84/wrist_image_left"
    
    # ✅ Pass as list of 3 directories
    IMAGE_DIRS = [EXTERNAL1_DIR, EXTERNAL2_DIR, WRIST_DIR]
    
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
    NUM_WORKERS = 0  # Set to 0 for debugging, increase for faster loading
    
    # ========================================================================
    # SETUP
    # ========================================================================
    
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print("\n" + "="*70)
    print("🚀 Multi-Camera Pose Prediction Training")
    print("="*70)
    print(f"\n⚙️  Configuration:")
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Embed dim: {EMBED_DIM}")
    print(f"\n📁 Data paths:")
    print(f"  Train CSV: {TRAIN_CSV}")
    print(f"  Val CSV: {VAL_CSV}")
    print(f"  Camera 1 (External 1): {EXTERNAL1_DIR}")
    print(f"  Camera 2 (External 2): {EXTERNAL2_DIR}")
    print(f"  Camera 3 (Wrist): {WRIST_DIR}")
    
    # ✅ Compute pose statistics from TRAINING data only
    print("\n" + "="*70)
    print("📊 Computing Training Pose Statistics")
    print("="*70)
    train_stats = compute_pose_statistics(TRAIN_CSV)
    
    # ✅ Create datasets - validation uses SAME stats as training
    print("\n" + "="*70)
    print("📦 Creating Datasets")
    print("="*70)
    
    train_dataset = MultiCamPoseDataset(
        csv_path=TRAIN_CSV,
        image_dirs=IMAGE_DIRS,  # ✅ List of 3 directories
        pose_stats=train_stats,
        normalization='standardize'
    )
    
    val_dataset = None
    if os.path.exists(VAL_CSV):
        val_dataset = MultiCamPoseDataset(
            csv_path=VAL_CSV,
            image_dirs=IMAGE_DIRS,  # ✅ Same directories
            pose_stats=train_stats,  # ✅ Use TRAINING stats for validation too!
            normalization='standardize'
        )
        print(f"✅ Validation dataset created: {len(val_dataset)} samples")
    
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
    
    # ✅ Initialize model
    print("\n" + "="*70)
    print("🤖 Initializing Model")
    print("="*70)
    
    model = MultiCameraPosePredictor(
        image_channels=IMAGE_CHANNELS,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        pose_channels=POSE_CHANNELS,
        embed_dim=EMBED_DIM,
    )
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # ✅ Test dataset and model compatibility
    test_dataset_and_model(train_dataset, model, DEVICE)
    
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
    
    print("\n" + "="*70)
    print("🏋️  Starting Training")
    print("="*70)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*70}")
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        print(f"{'='*70}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch + 1)
        train_losses.append(train_loss)
        print(f"\n📈 Train Loss: {train_loss:.6f}")
        
        # Validate
        if val_loader:
            val_loss, mean_errors, max_errors, predictions, targets = validate(
                model, val_loader, criterion, DEVICE
            )
            val_losses.append(val_loss)
            print(f"📉 Val Loss: {val_loss:.6f}")
            print(f"📊 Mean Absolute Errors (normalized):")
            print(f"   X: {mean_errors[0]:.6f}, Y: {mean_errors[1]:.6f}, Z: {mean_errors[2]:.6f}")
            print(f"📊 Max Absolute Errors (normalized):")
            print(f"   X: {max_errors[0]:.6f}, Y: {max_errors[1]:.6f}, Z: {max_errors[2]:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'pose_stats': train_stats,  # ✅ Save stats for inference
                }, os.path.join(SAVE_DIR, 'best_model.pth'))
                print(f"✅ Saved best model (val_loss: {val_loss:.6f})")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(SAVE_DIR, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss if val_loader else None,
                'pose_stats': train_stats,
            }, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    # ========================================================================
    # SAVE FINAL MODEL AND PLOTS
    # ========================================================================
    
    print("\n" + "="*70)
    print("💾 Saving Final Results")
    print("="*70)
    
    # Save final model
    final_model_path = os.path.join(SAVE_DIR, 'final_model.pth')
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'pose_stats': train_stats,
    }, final_model_path)
    print(f"✅ Final model saved: {final_model_path}")
    
    # Plot training curves
    if val_loader:
        plot_training_curves(
            train_losses, 
            val_losses, 
            os.path.join(SAVE_DIR, 'training_curves.png')
        )
        
        # Plot final predictions
        _, _, _, final_predictions, final_targets = validate(
            model, val_loader, criterion, DEVICE
        )
        plot_predictions(
            final_predictions,
            final_targets,
            train_stats,
            os.path.join(SAVE_DIR, 'final_predictions.png')
        )
    
    # Print summary
    print("\n" + "="*70)
    print("🎉 Training Complete!")
    print("="*70)
    print(f"📁 Results saved to: {SAVE_DIR}")
    print(f"📊 Final train loss: {train_losses[-1]:.6f}")
    if val_loader:
        print(f"📊 Final val loss: {val_losses[-1]:.6f}")
        print(f"🏆 Best val loss: {best_val_loss:.6f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()