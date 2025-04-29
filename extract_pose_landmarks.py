import os
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

# Paths
dataset_path = "yoga_poses"
output_csv = "yoga_data.csv"

# MediaPipe pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# Output list
data = []

# Pose folders
pose_folders = os.listdir(dataset_path)

for pose_name in pose_folders:
    pose_folder_path = os.path.join(dataset_path, pose_name)
    if not os.path.isdir(pose_folder_path):
        continue

    for img_name in tqdm(os.listdir(pose_folder_path), desc=f"Processing {pose_name}"):
        img_path = os.path.join(pose_folder_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            if len(landmarks) == 99:  # 33 landmarks * 3
                landmarks.append(pose_name)
                data.append(landmarks)

# Convert to DataFrame
columns = [f"{coord}{i}" for i in range(33) for coord in ("x", "y", "z")] + ["label"]
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv(output_csv, index=False)
print(f"Saved {len(df)} entries to {output_csv}")
