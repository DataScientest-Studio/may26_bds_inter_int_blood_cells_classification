import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset_dir = r"G:\Liora\PBC_dataset_normal_DIB"
classes = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

print("Analyzing background pixel uniformity... Please wait.")

bg_data = []
for cls in classes:
    class_path = os.path.join(dataset_dir, cls)
    if os.path.exists(class_path):
        files = os.listdir(class_path)[:150]
        for file in files:
            img_path = os.path.join(class_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Sample a 10x10 patch from the top left corner (pure background)
                bg_patch = img[0:10, 0:10]
                bg_mean = np.mean(bg_patch)
                bg_data.append({'Class': cls, 'Background_Brightness': bg_mean})

df = pd.DataFrame(bg_data)

# Print verification statistics
print("\n--- Mean Background Brightness Stability ---")
print(df.groupby('Class')['Background_Brightness'].agg(['mean', 'std']))

# Plotting Violin Plots to show background distribution density
plt.figure(figsize=(12, 6))
violin_parts = plt.violinplot([df[df['Class'] == cls]['Background_Brightness'].values for cls in classes],
                             showmeans=True, showmedians=False)

# Styling details
plt.title('Background Environmental Uniformity Across Dataset Folders', fontsize=14, fontweight='bold')
plt.xticks(range(1, len(classes) + 1), [cls.upper() for cls in classes], rotation=30)
plt.xlabel('Blood Cell Classification Category', fontsize=12)
plt.ylabel('Corner Pixel Mean Intensity (0-255)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()