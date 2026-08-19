import os
from PIL import Image
dataset_path = r"G:\Liora\PBC_dataset_normal_DIB"

#Check every image
corrupted_images = []
total_images = 0
valid_images = 0

for class_name in sorted(os.listdir(dataset_path)):

    class_folder = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_folder):
        continue

    for image_name in os.listdir(class_folder):

        image_path = os.path.join(class_folder, image_name)

        total_images += 1

        try:
            with Image.open(image_path) as img:
                img.verify()        # Verify that the image is not corrupted

            valid_images += 1

        except Exception:
            corrupted_images.append(image_path)


print("=" * 50)
print("DATASET INTEGRITY REPORT")
print("=" * 50)

print(f"Total images      : {total_images}")
print(f"Valid images      : {valid_images}")
print(f"Corrupted images  : {len(corrupted_images)}")

if corrupted_images:

    print("\nCorrupted Files:")

    for file in corrupted_images:
        print(file)

else:
    print("\nNo corrupted images found.")

import os

corrupted_images = [
    r"G:\Liora\PBC_dataset_normal_DIB\neutrophil\.DS_169665.jpg"
]

for file in corrupted_images:
    try:
        os.remove(file)
        print(f"Deleted: {file}")
    except Exception as e:
        print(e)