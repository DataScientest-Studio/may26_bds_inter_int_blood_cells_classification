import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset_dir = r"G:\Liora\PBC_dataset_normal_DIB"
classes = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

print("Analyzing cell structural texture and edge density... Please wait.")

texture_data = []
for cls in classes:
    class_path = os.path.join(dataset_dir, cls)
    if os.path.exists(class_path):
        # Sample 100 images per class for a solid statistical baseline
        files = os.listdir(class_path)[:100]
        for file in files:
            img_path = os.path.join(class_path, file)
            # Read image in grayscale since texture doesn't depend on color
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Calculate Laplacian variance (higher value = sharper, more complex edges)
                laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
                
                texture_data.append({
                    'Class': cls,
                    'Edge_Complexity': laplacian_var
                })

df = pd.DataFrame(texture_data)

# Print out statistical metrics (median edge score) for your assignment report
print("\n--- Median Structural Complexity Score Per Class ---")
print(df.groupby('Class')['Edge_Complexity'].median().sort_values(ascending=False))

# Prepare data structure for the Box Plot layout
plot_data = [df[df['Class'] == cls]['Edge_Complexity'].values for cls in classes]

# Build the Box Plot
plt.figure(figsize=(12, 6))
box = plt.boxplot(plot_data, label=[cls.upper() for cls in classes], patch_artist=True,
                  boxprops=dict(facecolor='lavender', color='darkslateblue'),
                  medianprops=dict(color='crimson', linewidth=2))

# Styling and Labels
plt.title('Cell Morphological Complexity & Edge Density Variance', fontsize=14, fontweight='bold')
plt.xlabel('Blood Cell Classification Category', fontsize=12)
plt.ylabel('Edge Density Score (Laplacian Variance)', fontsize=12)
plt.xticks(rotation=30)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
print("\nDisplaying structural texture chart window...")
plt.show()