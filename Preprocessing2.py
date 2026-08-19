#Image Color Code Analysis
import os
from collections import Counter
from PIL import Image
import pandas as pd

# Dataset path
dataset_path = r"G:\Liora\PBC_dataset_normal_DIB"

# Store image modes
image_modes = []

for class_name in sorted(os.listdir(dataset_path)):

    class_folder = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_folder):
        continue

    for image_name in os.listdir(class_folder):

        image_path = os.path.join(class_folder, image_name)

        try:
            with Image.open(image_path) as img:
                image_modes.append(img.mode)

        except Exception:
            pass

# Count each mode
mode_counts = Counter(image_modes)

print("=" * 50)
print("IMAGE COLOR MODE ANALYSIS")
print("=" * 50)

for mode, count in mode_counts.items():
    print(f"{mode}: {count}")

#Pixel Intensity (Brightness) Analysis

import numpy as np
import matplotlib.pyplot as plt


dataset_path = r"G:\Liora\PBC_dataset_normal_DIB"

brightness = []

for class_name in sorted(os.listdir(dataset_path)):

    class_folder = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_folder):
        continue

    for image_name in os.listdir(class_folder):

        image_path = os.path.join(class_folder, image_name)

        try:
            with Image.open(image_path) as img:

                img = np.array(img)

                brightness.append(img.mean())

        except:
            pass

brightness = np.array(brightness)

print("=" * 50)
print("PIXEL INTENSITY ANALYSIS")
print("=" * 50)

print(f"Minimum Brightness : {brightness.min():.2f}")
print(f"Maximum Brightness : {brightness.max():.2f}")
print(f"Average Brightness : {brightness.mean():.2f}")
print(f"Standard Deviation : {brightness.std():.2f}")

plt.figure(figsize=(8,5))

plt.hist(brightness, bins=30)

plt.title("Distribution of Average Image Brightness")
plt.xlabel("Average Pixel Intensity")
plt.ylabel("Number of Images")

plt.grid(True)

plt.show()