import os
import csv
from pathlib import Path

import tensorflow_datasets as tfds
from PIL import Image

# === Paths ===
DATA_DIR = "/home/david/projects/EECS253/droid_100/1.0.0"  # directory containing dataset_info.json, tfrecords
OUTPUT_ROOT = Path("episode_84")     # output folder for episode 84
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# === Build dataset from on-disk TFDS directory ===
builder = tfds.builder_from_directory(str(DATA_DIR))
print("Available splits:", builder.info.splits)

ds = builder.as_dataset(split="train", shuffle_files=False)
ds_np = tfds.as_numpy(ds)

# === Create subfolders for each camera ===
exterior1_dir = OUTPUT_ROOT / "exterior_image_1_left"
exterior2_dir = OUTPUT_ROOT / "exterior_image_2_left"
wrist_dir = OUTPUT_ROOT / "wrist_image_left"

exterior1_dir.mkdir(parents=True, exist_ok=True)
exterior2_dir.mkdir(parents=True, exist_ok=True)
wrist_dir.mkdir(parents=True, exist_ok=True)

# === Extract episode 84 (index 83 if 0-indexed, or 84 if 1-indexed) ===
EPISODE_INDEX = 84  # Change to 83 if your episodes are 0-indexed

print(f"\n=== Extracting Episode {EPISODE_INDEX} ===")

# Skip to the desired episode
for idx, ep in enumerate(ds_np):
    if idx == EPISODE_INDEX:
        print(f"Found episode {idx}")
        
        steps = ep["steps"]
        
        # === Prepare CSV file for cartesian positions ===
        csv_path = OUTPUT_ROOT / "cartesian_positions.csv"
        
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write header
            csv_writer.writerow(['frame', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'])
            
            # === Loop over steps (frames) ===
            for step_i, step in enumerate(steps):
                obs = step["observation"]
                
                # Extract cartesian position [x, y, z, roll, pitch, yaw]
                cartesian_pos = obs["cartesian_position"]
                x, y, z = cartesian_pos[0], cartesian_pos[1], cartesian_pos[2]
                roll, pitch, yaw = cartesian_pos[3], cartesian_pos[4], cartesian_pos[5]
                
                # Write to CSV
                csv_writer.writerow([step_i, x, y, z, roll, pitch, yaw])
                
                # === Save images from each camera ===
                
                # Exterior camera 1
                if "exterior_image_1_left" in obs:
                    frame1 = obs["exterior_image_1_left"]
                    img1 = Image.fromarray(frame1)
                    img1.save(exterior1_dir / f"frame_{step_i:04d}.jpg")
                
                # Exterior camera 2
                if "exterior_image_2_left" in obs:
                    frame2 = obs["exterior_image_2_left"]
                    img2 = Image.fromarray(frame2)
                    img2.save(exterior2_dir / f"frame_{step_i:04d}.jpg")
                
                # Wrist camera
                if "wrist_image_left" in obs:
                    frame_wrist = obs["wrist_image_left"]
                    img_wrist = Image.fromarray(frame_wrist)
                    img_wrist.save(wrist_dir / f"frame_{step_i:04d}.jpg")
                
                if step_i % 10 == 0:
                    print(f"  Processed frame {step_i}")
        
        print(f"\n✓ Saved {len(steps)} frames")
        print(f"✓ Cartesian positions saved to: {csv_path}")
        print(f"✓ Exterior camera 1 images saved to: {exterior1_dir}")
        print(f"✓ Exterior camera 2 images saved to: {exterior2_dir}")
        print(f"✓ Wrist camera images saved to: {wrist_dir}")
        
        break  # Stop after processing episode 84
    
    # Skip other episodes
    if idx > EPISODE_INDEX:
        break

print(f"\nDone! Episode {EPISODE_INDEX} data saved to {OUTPUT_ROOT.resolve()}")