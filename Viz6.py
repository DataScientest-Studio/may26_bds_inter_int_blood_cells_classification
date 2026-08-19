import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

dataset_dir = r"G:\Liora\PBC_dataset_normal_DIB"
classes = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

plt.figure(figsize=(12, 6))
print("Generating collective dataset intensity histograms... Please wait.")

for cls in classes:
    class_path = os.path.join(dataset_dir, cls)
    if os.path.exists(class_path) and len(os.listdir(class_path)) > 0:
        sample_files = os.listdir(class_path)[:30]
        
        # Initialize as a flat 1D array of 256 elements
        combined_hist = np.zeros(256)
        
        for file in sample_files:
            img_path = os.path.join(class_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Calculate histogram and flatten it to 1D immediately
                hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
                combined_hist += hist
                
        # Normalize the curve across the sample count
        combined_hist /= len(sample_files)
        plt.plot(combined_hist, label=cls.upper(), alpha=0.8, linewidth=1.8)

plt.title('Pixel Intensity Frequency Distribution Profiles', fontsize=14, fontweight='bold')
plt.xlabel('Pixel Intensity Value (0 = Pure Black, 255 = Pure White)', fontsize=12)
plt.ylabel('Average Frequency Count per Image', fontsize=12)
plt.xlim([0, 255])
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(loc='upper left')

plt.tight_layout()
print("Displaying intensity curves...")
plt.show()