import os
import cv2
import matplotlib.pyplot as plt

# Main dataset directory
dataset_dir = r"G:\Liora\PBC_dataset_normal_DIB"

# Explicit singular lowercase names matching your folders
classes = [
    'basophil', 'eosinophil', 'erythroblast', 'ig', 
    'lymphocyte', 'monocyte', 'neutrophil', 'platelet'
]

# Set up the plotting window (2 rows, 4 columns)
fig, axes = plt.subplots(2, 4, figsize=(15, 8))
axes = axes.flatten()  # Flattens the 2D grid array into 1D for easy looping

for i, cls in enumerate(classes):
    class_path = os.path.join(dataset_dir, cls)
    
    if os.path.exists(class_path) and len(os.listdir(class_path)) > 0:
        # Get the filename of the first image in the folder
        first_img_name = os.listdir(class_path)[0]
        img_path = os.path.join(class_path, first_img_name)
        
        # Read the image using OpenCV
        img = cv2.imread(img_path)
        
        if img is not None:
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Display image on its corresponding subplot panel
            axes[i].imshow(img_rgb)
            axes[i].set_title(f"{cls.upper()}\n({img.shape[1]}x{img.shape[0]})", fontsize=12, fontweight='bold')
        else:
            axes[i].text(0.5, 0.5, 'Error Reading Image', ha='center', va='center')
    else:
        axes[i].text(0.5, 0.5, f'Folder "{cls}"\nnot found', ha='center', va='center', color='red')
    
    # Clean up the visual by hiding the unnecessary grid axes lines
    axes[i].axis('off')

plt.suptitle("Morphological Overview: Blood Cell Types Sample Gallery", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()