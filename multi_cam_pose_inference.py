import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

# Import your classes
from multi_cam_pose_dataset import MultiCamPoseDataset
from model import MultiCameraPosePredictor


def denormalize_pose(normalized_pose, pose_stats):
    """Convert normalized predictions back to original scale"""
    mean = torch.tensor(pose_stats['mean'], dtype=torch.float32)
    std = torch.tensor(pose_stats['std'], dtype=torch.float32)
    return normalized_pose * std + mean


def run_inference(model, dataloader, device, pose_stats):
    """Run inference and return denormalized results"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    print("Running inference...")
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate and denormalize
    predictions_norm = torch.cat(all_predictions, dim=0)
    targets_norm = torch.cat(all_targets, dim=0)
    
    predictions = denormalize_pose(predictions_norm, pose_stats)
    targets = denormalize_pose(targets_norm, pose_stats)
    
    print(f"✅ Inference complete: {len(predictions)} samples\n")
    
    return predictions, targets


def plot_predictions(predictions, targets, save_path='predictions.png'):
    """Plot predicted and true values vs sample number"""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    dim_names = ['X', 'Y', 'Z']
    colors_pred = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    colors_true = ['#C92A2A', '#0B7285', '#1864AB']
    
    sample_indices = range(len(predictions))
    
    for i, (ax, dim_name) in enumerate(zip(axes, dim_names)):
        pred_values = predictions[:, i].numpy()
        true_values = targets[:, i].numpy()
        
        # Plot true and predicted values
        ax.plot(sample_indices, true_values, 
                label='Ground Truth', color=colors_true[i], 
                linewidth=2, alpha=0.7)
        ax.plot(sample_indices, pred_values, 
                label='Predicted', color=colors_pred[i], 
                linewidth=2, alpha=0.7, linestyle='--')
        
        ax.set_ylabel(f'{dim_name} Position', fontsize=12, fontweight='bold')
        ax.set_title(f'{dim_name} Dimension: Predictions vs Ground Truth', 
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Sample Number', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot saved to {save_path}")


def main():
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    MODEL_PATH = "./checkpoints/best_model.pth"
    VAL_CSV = "C:/Users/david/Projects/eecs253_arm_tracking/episode_84/test_data.csv"
    
    EXTERNAL1_DIR = "C:/Users/david/Projects/eecs253_arm_tracking/episode_84/exterior_image_1_left"
    EXTERNAL2_DIR = "C:/Users/david/Projects/eecs253_arm_tracking/episode_84/exterior_image_2_left"
    WRIST_DIR = "C:/Users/david/Projects/eecs253_arm_tracking/episode_84/wrist_image_left"
    IMAGE_DIRS = [EXTERNAL1_DIR, EXTERNAL2_DIR, WRIST_DIR]
    
    IMAGE_HEIGHT = 180
    IMAGE_WIDTH = 320
    IMAGE_CHANNELS = 3
    POSE_CHANNELS = 3
    EMBED_DIM = 64
    DROPOUT = 0.2
    
    BATCH_SIZE = 32
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    OUTPUT_DIR = './inference_results'
    
    # ========================================================================
    # LOAD MODEL
    # ========================================================================
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    pose_stats = checkpoint['pose_stats']
    
    model = MultiCameraPosePredictor(
        image_channels=IMAGE_CHANNELS,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        pose_channels=POSE_CHANNELS,
        embed_dim=EMBED_DIM,
        dropout=DROPOUT
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    print(f"✅ Model loaded from epoch {checkpoint['epoch']}\n")
    
    # ========================================================================
    # LOAD VALIDATION DATASET
    # ========================================================================
    
    print("Loading validation dataset...")
    val_dataset = MultiCamPoseDataset(
        csv_path=VAL_CSV,
        image_dirs=IMAGE_DIRS,
        pose_stats=pose_stats,
        normalization='standardize'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    print(f"✅ Loaded {len(val_dataset)} samples\n")
    
    # ========================================================================
    # RUN INFERENCE AND PLOT
    # ========================================================================
    
    predictions, targets = run_inference(model, val_loader, DEVICE, pose_stats)
    
    plot_predictions(
        predictions, 
        targets, 
        save_path=os.path.join(OUTPUT_DIR, 'predictions_vs_truth.png')
    )
    
    print(f"\n✅ Done! Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()