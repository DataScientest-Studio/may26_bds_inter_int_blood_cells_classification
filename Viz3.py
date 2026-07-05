import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset_dir = r"G:\Liora\PBC_dataset_normal_DIB"
classes = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

print("Analyzing cell color profiles... This will take a few seconds.")

color_data = []
for cls in classes:
    class_path = os.path.join(dataset_dir, cls)
    if os.path.exists(class_path):
        # Sample 100 images per class for a fast, statistically sound look
        files = os.listdir(class_path)[:100]
        for file in files:
            img_path = os.path.join(class_path, file)
            img = cv2.imread(img_path)
            if img is not None:
                # OpenCV loads images as BGR, let's extract individual channel means
                mean_b = np.mean(img[:, :, 0])
                mean_g = np.mean(img[:, :, 1])
                mean_r = np.mean(img[:, :, 2])
                
                color_data.append({
                    'Class': cls,
                    'Red_Mean': mean_r,
                    'Green_Mean': mean_g,
                    'Blue_Mean': mean_b
                })

df = pd.DataFrame(color_data)

# Calculate the global average per class to look at the differences
summary = df.groupby('Class').mean()
print("\n--- Average RGB Values Per Cell Class ---")
print(summary)

# Plot a grouped bar chart comparing RGB channels across classes
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(classes))
width = 0.25

# Plot each channel with its matching color signature
ax.bar(x - width, summary['Red_Mean'], width, label='Red Channel', color='#ff4d4d', edgecolor='black')
ax.bar(x, summary['Green_Mean'], width, label='Green Channel', color='#4dff4d', edgecolor='black')
ax.bar(x + width, summary['Blue_Mean'], width, label='Blue Channel', color='#4d4dff', edgecolor='black')

# Labeling and styling
ax.set_title('Color Channel (RGB) Intensity Profiles Across Cell Classes', fontsize=14, fontweight='bold')
ax.set_xlabel('Blood Cell Classification Category', fontsize=12)
ax.set_ylabel('Average Pixel Intensity (0-255)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([cls.upper() for cls in classes], rotation=30)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
print("\nDisplaying color profile chart window...")
plt.show()