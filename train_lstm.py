import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import your classes
from model import PoseLSTM
from pose_dataset import PoseDataset


def compute_pose_statistics(csv_path, pose_columns=['x', 'y', 'z']):
    """
    Compute mean and std for each pose dimension from training data only
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
    
    print("\nPose Statistics (from training data):")
    print("="*60)
    for dim, name in enumerate(['x', 'y', 'z']):
        print(f"  {name}: mean={mean[dim]:.4f}, std={std[dim]:.4f}, "
              f"min={min_vals[dim]:.4f}, max={max_vals[dim]:.4f}")
    print("="*60 + "\n")
    
    return stats


def denormalize_pose(normalized_pose, pose_stats, normalization='standardize'):
    """
    Convert normalized predictions back to original scale
    
    Args:
        normalized_pose: Tensor of shape (N, 3) - normalized poses
        pose_stats: Dict with normalization statistics
        normalization: Type of normalization used
    
    Returns:
        Denormalized pose in original scale
    """
    if normalization == 'standardize':
        mean = torch.tensor(pose_stats['mean'], dtype=torch.float32)
        std = torch.tensor(pose_stats['std'], dtype=torch.float32)
        return normalized_pose * std + mean
    else:
        return normalized_pose


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    for batch_idx, (sequences, targets) in enumerate(dataloader):
        # sequences: [batch, seq_len, pose_dim]
        # targets: [batch, pose_dim]
        sequences = sequences.to(device)
        targets = targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(sequences)
        
        # Calculate loss
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for sequences, targets in dataloader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # Forward pass
            predictions = model(sequences)
            
            # Calculate loss
            loss = criterion(predictions, targets)
            running_loss += loss.item()
            
            # Store for analysis
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    val_loss = running_loss / len(dataloader)
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate per-dimension errors (normalized space)
    errors = torch.abs(all_predictions - all_targets)
    mean_errors = errors.mean(dim=0)
    
    return val_loss, mean_errors, all_predictions, all_targets


def plot_training_curves(train_losses, val_losses, save_path='training_curves.png'):
    """Plot and save training curves"""
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Training curves saved to {save_path}")


def plot_predictions_vs_targets(predictions, targets, predict_len, pose_stats, 
                                save_path='predictions_vs_targets.png'):
    """
    Plot denormalized predictions vs targets with proper temporal alignment
    
    Args:
        predictions: Normalized predictions (N, 3)
        targets: Normalized targets (N, 3)
        predict_len: Sequence length used for prediction
        pose_stats: Statistics for denormalization
        save_path: Path to save plot
    """
    # Denormalize predictions and targets
    predictions_denorm = denormalize_pose(predictions, pose_stats, 'standardize')
    targets_denorm = denormalize_pose(targets, pose_stats, 'standardize')
    
    # Create frame indices
    # If sequence 0 (frames 0 to predict_len-1) predicts frame predict_len,
    # then sequence i predicts frame i + predict_len
    num_predictions = len(predictions)
    frame_indices = np.arange(predict_len, predict_len + num_predictions)
    
    # Create subplots for each dimension
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    dim_names = ['X', 'Y', 'Z']
    colors_true = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    colors_pred = ['#aec7e8', '#ffbb78', '#98df8a']  # Light versions
    
    for i, (ax, dim_name, color_true, color_pred) in enumerate(
        zip(axes, dim_names, colors_true, colors_pred)
    ):
        true_values = targets_denorm[:, i].numpy()
        pred_values = predictions_denorm[:, i].numpy()
        
        # Plot ground truth
        ax.plot(frame_indices, true_values, 
               label='Ground Truth', 
               color=color_true, 
               linewidth=2.5, 
               alpha=0.8,
               marker='o',
               markersize=3)
        
        # Plot predictions
        ax.plot(frame_indices, pred_values, 
               label='Prediction', 
               color=color_pred, 
               linewidth=2, 
               alpha=0.9,
               marker='x',
               markersize=4,
               linestyle='--')
        
        # Calculate metrics
        mae = np.abs(pred_values - true_values).mean()
        rmse = np.sqrt(((pred_values - true_values) ** 2).mean())
        
        ax.set_ylabel(f'{dim_name} Position', fontsize=13, fontweight='bold')
        ax.set_title(f'{dim_name} Dimension: Predictions vs Ground Truth\\n'
                    f'MAE: {mae:.4f}, RMSE: {rmse:.4f}', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add shaded error region
        ax.fill_between(frame_indices, 
                       pred_values - (pred_values - true_values), 
                       pred_values,
                       alpha=0.2, 
                       color='red',
                       label='_nolegend_')
    
    axes[-1].set_xlabel('Frame Index', fontsize=13, fontweight='bold')
    
    # Add overall title
    fig.suptitle(f'Temporal Alignment: Predictions vs Ground Truth\\n'
                f'(Sequences of length {predict_len} predict next timestep)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Predictions vs targets plot saved to {save_path}")


def plot_error_analysis(predictions, targets, predict_len, pose_stats,
                       save_path='error_analysis.png'):
    """
    Plot error analysis with histograms and error over time
    
    Args:
        predictions: Normalized predictions (N, 3)
        targets: Normalized targets (N, 3)
        predict_len: Sequence length
        pose_stats: Statistics for denormalization
        save_path: Path to save plot
    """
    # Denormalize
    predictions_denorm = denormalize_pose(predictions, pose_stats, 'standardize')
    targets_denorm = denormalize_pose(targets, pose_stats, 'standardize')
    
    # Calculate errors
    errors = (predictions_denorm - targets_denorm).numpy()
    abs_errors = np.abs(errors)
    
    # Create frame indices
    num_predictions = len(predictions)
    frame_indices = np.arange(predict_len, predict_len + num_predictions)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    dim_names = ['X', 'Y', 'Z']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (dim_name, color) in enumerate(zip(dim_names, colors)):
        # Error histogram
        ax_hist = fig.add_subplot(gs[i, 0])
        ax_hist.hist(errors[:, i], bins=50, color=color, alpha=0.7, edgecolor='black')
        ax_hist.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax_hist.axvline(errors[:, i].mean(), color='green', linestyle='--', 
                       linewidth=2, label=f'Mean: {errors[:, i].mean():.4f}')
        ax_hist.set_xlabel(f'{dim_name} Error', fontsize=11, fontweight='bold')
        ax_hist.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax_hist.set_title(f'{dim_name} Error Distribution', fontsize=12, fontweight='bold')
        ax_hist.legend(fontsize=9)
        ax_hist.grid(True, alpha=0.3, axis='y')
        
        # Absolute error over time
        ax_time = fig.add_subplot(gs[i, 1])
        ax_time.plot(frame_indices, abs_errors[:, i], color=color, alpha=0.6, linewidth=1.5)
        ax_time.axhline(abs_errors[:, i].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {abs_errors[:, i].mean():.4f}')
        ax_time.set_xlabel('Frame Index', fontsize=11, fontweight='bold')
        ax_time.set_ylabel(f'{dim_name} Absolute Error', fontsize=11, fontweight='bold')
        ax_time.set_title(f'{dim_name} Error Over Time', fontsize=12, fontweight='bold')
        ax_time.legend(fontsize=9)
        ax_time.grid(True, alpha=0.3)
        
        # Scatter: Prediction vs Target
        ax_scatter = fig.add_subplot(gs[i, 2])
        pred_vals = predictions_denorm[:, i].numpy()
        true_vals = targets_denorm[:, i].numpy()
        ax_scatter.scatter(true_vals, pred_vals, alpha=0.5, s=20, c=color, edgecolors='none')
        
        # Perfect prediction line
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax_scatter.plot([min_val, max_val], [min_val, max_val], 
                       'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate R²
        correlation = np.corrcoef(true_vals, pred_vals)[0, 1]
        r_squared = correlation ** 2
        
        ax_scatter.set_xlabel(f'True {dim_name}', fontsize=11, fontweight='bold')
        ax_scatter.set_ylabel(f'Predicted {dim_name}', fontsize=11, fontweight='bold')
        ax_scatter.set_title(f'{dim_name} Prediction Quality (R²={r_squared:.4f})', 
                           fontsize=12, fontweight='bold')
        ax_scatter.legend(fontsize=9)
        ax_scatter.grid(True, alpha=0.3)
        ax_scatter.set_aspect('equal', adjustable='box')
    
    fig.suptitle('Error Analysis: LSTM Pose Prediction', 
                fontsize=16, fontweight='bold')
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Error analysis plot saved to {save_path}")


def main():
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    # Dataset paths
    TRAIN_CSV = "C:/Users/david/Projects/eecs253_arm_tracking/episode_84/train_data.csv"
    VAL_CSV = "C:/Users/david/Projects/eecs253_arm_tracking/episode_84/test_data.csv"
    
    # Model configuration
    POSE_DIM = 3           # x, y, z
    PREDICT_LEN = 5        # Number of past timesteps to use
    HIDDEN_SIZE = 64       # LSTM hidden units
    DROPOUT = 0.2
    
    # Training hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    
    # Other settings
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_DIR = './checkpoints_lstm'
    NUM_WORKERS = 0
    
    # ========================================================================
    # SETUP
    # ========================================================================
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # ========================================================================
    # COMPUTE NORMALIZATION STATISTICS FROM TRAINING DATA
    # ========================================================================
    
    print("Computing pose normalization statistics from training data...")
    train_stats = compute_pose_statistics(TRAIN_CSV, pose_columns=['x', 'y', 'z'])
    
    # ========================================================================
    # CREATE DATASETS
    # ========================================================================
    
    print("Creating datasets...")
    train_dataset = PoseDataset(
        csv_path=TRAIN_CSV,
        predict_len=PREDICT_LEN,
        pose_stats=train_stats,
        normalization='standardize'
    )
    
    val_dataset = None
    if os.path.exists(VAL_CSV):
        val_dataset = PoseDataset(
            csv_path=VAL_CSV,
            predict_len=PREDICT_LEN,
            pose_stats=train_stats,  # Use TRAINING stats for validation!
            normalization='standardize'
        )
    
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
            shuffle=False,  # Important: Don't shuffle for temporal alignment!
            num_workers=NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    # ========================================================================
    # INITIALIZE MODEL
    # ========================================================================
    
    print("Initializing model...")
    model = PoseLSTM(
        pose_dim=POSE_DIM,
        predict_len=PREDICT_LEN,
        hidden_size=HIDDEN_SIZE,
        dropout=DROPOUT
    )
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model created:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}\n")
    
    # ========================================================================
    # LOSS AND OPTIMIZER
    # ========================================================================
    
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
    print("="*70)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")
        print("-"*70)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)
        print(f"Train Loss: {train_loss:.6f}")
        
        # Validate
        if val_loader:
            val_loss, mean_errors, predictions, targets = validate(
                model, val_loader, criterion, DEVICE
            )
            val_losses.append(val_loss)
            print(f"Val Loss:   {val_loss:.6f}")
            print(f"Mean Absolute Errors (normalized):")
            print(f"   X: {mean_errors[0]:.6f}, Y: {mean_errors[1]:.6f}, Z: {mean_errors[2]:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'pose_stats': train_stats,
                    'predict_len': PREDICT_LEN,
                    'hidden_size': HIDDEN_SIZE,
                }, os.path.join(SAVE_DIR, 'best_model_lstm.pth'))
                print(f"✅ Saved best model (val_loss: {val_loss:.6f})")
    
    # ========================================================================
    # FINAL VALIDATION AND VISUALIZATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("CREATING FINAL VISUALIZATIONS")
    print("="*70)
    
    # Run final validation to get predictions
    if val_loader:
        final_val_loss, final_mean_errors, final_predictions, final_targets = validate(
            model, val_loader, criterion, DEVICE
        )
        
        print(f"\nFinal Validation Results:")
        print(f"  Loss: {final_val_loss:.6f}")
        
        # Denormalize for metrics
        final_pred_denorm = denormalize_pose(final_predictions, train_stats, 'standardize')
        final_targ_denorm = denormalize_pose(final_targets, train_stats, 'standardize')
        
        errors_denorm = torch.abs(final_pred_denorm - final_targ_denorm)
        print(f"  Mean Absolute Errors (original scale):")
        print(f"    X: {errors_denorm[:, 0].mean():.6f}")
        print(f"    Y: {errors_denorm[:, 1].mean():.6f}")
        print(f"    Z: {errors_denorm[:, 2].mean():.6f}")
        print()
        
        # Plot predictions vs targets with temporal alignment
        plot_predictions_vs_targets(
            final_predictions,
            final_targets,
            PREDICT_LEN,
            train_stats,
            os.path.join(SAVE_DIR, 'predictions_vs_targets_temporal.png')
        )
        
        # Plot error analysis
        plot_error_analysis(
            final_predictions,
            final_targets,
            PREDICT_LEN,
            train_stats,
            os.path.join(SAVE_DIR, 'error_analysis.png')
        )
    
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
        'pose_stats': train_stats,
        'predict_len': PREDICT_LEN,
        'hidden_size': HIDDEN_SIZE,
    }, os.path.join(SAVE_DIR, 'final_model_lstm.pth'))
    
    # Plot training curves
    if val_loader:
        plot_training_curves(
            train_losses,
            val_losses,
            os.path.join(SAVE_DIR, 'training_curves_lstm.png')
        )
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Models saved to: {SAVE_DIR}")
    print(f"Final train loss: {train_losses[-1]:.6f}")
    if val_loader:
        print(f"Final val loss:   {val_losses[-1]:.6f}")
        print(f"Best val loss:    {best_val_loss:.6f}")
    print(f"\nGenerated plots:")
    print(f"  - training_curves_lstm.png")
    print(f"  - predictions_vs_targets_temporal.png")
    print(f"  - error_analysis.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()