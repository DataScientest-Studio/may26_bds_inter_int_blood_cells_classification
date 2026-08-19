import os
from PIL import Image
import pandas as pd

dataset_path = r"G:\Liora\PBC_dataset_normal_DIB"

image_sizes = []

for class_name in sorted(os.listdir(dataset_path)):

    class_folder = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_folder):
        continue

    for image_name in os.listdir(class_folder):

        image_path = os.path.join(class_folder, image_name)

        try:
            with Image.open(image_path) as img:

                width, height = img.size

                image_sizes.append({
                    "Class": class_name,
                    "Width": width,
                    "Height": height
                })

        except:
            pass
df_sizes = pd.DataFrame(image_sizes)

print(df_sizes.head())

print("=" * 50)
print("IMAGE SIZE STATISTICS")
print("=" * 50)

print(df_sizes[["Width", "Height"]].describe())

unique_sizes = df_sizes.groupby(["Width", "Height"]).size().reset_index(name="Count")

print(unique_sizes)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

plt.scatter(
    df_sizes["Width"],
    df_sizes["Height"],
    alpha=0.4
)

plt.title("Image Dimension Distribution")
plt.xlabel("Width (pixels)")
plt.ylabel("Height (pixels)")

plt.grid(True)

plt.show()

import matplotlib.pyplot as plt

size_counts = (
    df_sizes
    .groupby(["Width", "Height"])
    .size()
    .reset_index(name="Count")
)

size_counts["Image Size"] = (
    size_counts["Width"].astype(str) +
    " x " +
    size_counts["Height"].astype(str)
)

plt.figure(figsize=(10,5))

plt.bar(size_counts["Image Size"], size_counts["Count"])

plt.title("Distribution of Image Dimensions")
plt.xlabel("Image Size (Width × Height)")
plt.ylabel("Number of Images")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

