import os
from pathlib import Path

import tensorflow_datasets as tfds
from PIL import Image

# === Paths ===
DATA_DIR = Path("droid_100/1.0.0")   # directory containing dataset_info.json, tfrecords
OUTPUT_ROOT = Path("frames_r2d2_faceblur_all")  # top-level output folder
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# === Build dataset from on-disk TFDS directory ===
builder = tfds.builder_from_directory(str(DATA_DIR))
print("Available splits:", builder.info.splits)

ds = builder.as_dataset(split="train", shuffle_files=False)
ds_np = tfds.as_numpy(ds)

def is_r2d2_faceblur_episode(episode):
    file_path = episode["episode_metadata"]["file_path"].decode("utf-8")
    return "r2d2" in file_path.lower()

# === Choose which camera(s) to extract ===
camera_keys = [
    # "wrist_image_left",
    "exterior_image_1_left"
    # "exterior_image_2_left"
]
# Remove cameras that may not exist in some episodes
camera_keys = [k for k in camera_keys]

# === Iterate over ALL episodes ===
episode_count = 0
img_count = 0

for idx, ep in enumerate(ds_np):
    if not is_r2d2_faceblur_episode(ep):
        continue

    # Create folder for THIS episode
    episode_dir = OUTPUT_ROOT / f"episode_{idx:05d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Saving Episode {idx} to {episode_dir} ===")

    steps = ep["steps"]

    # Loop over steps (frames)
    for step_i, step in enumerate(steps):
        obs = step["observation"]

        for cam in camera_keys:
            if cam not in obs:
                continue  # skip missing cams in this episode

            frame = obs[cam]  # uint8 RGB array

            img = Image.fromarray(frame)
            out_path = episode_dir / f"{cam}_frame_{step_i:04d}.jpg"
            img.save(out_path)
            img_count += 1

    print(f'Episode ', episode_count, ' Images: ', img_count)
    episode_count += 1
    img_count = 0
    

print(f"\nDone. Saved {episode_count} episodes into {OUTPUT_ROOT.resolve()}")