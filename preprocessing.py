from pathlib import Path
import json
import hashlib

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


PROJECT_DIR = Path(r"C:\Users\Startklar\Downloads\ML\Sudhanshu")
DATASET_DIR = PROJECT_DIR / "data" / "raw" / "PBC_dataset_normal_DIB"

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FIGURES_DIR = PROJECT_DIR / "figures"
OUTPUT_DIR = PROJECT_DIR / "preprocessing_output"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RANDOM_STATE = 42
IMAGE_SIZE = (224, 224)

CLASS_NAME_MAP = {
    "ig": "immature_granulocytes"
}


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def is_image_file(path):
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return False

    parts = [p.lower() for p in path.parts]

    if "__macosx" in parts:
        return False

    if path.name.startswith("._"):
        return False

    return True


def get_label(image_path):
    relative_parts = image_path.relative_to(DATASET_DIR).parts

    if len(relative_parts) < 2:
        label = image_path.parent.name.lower()
    else:
        label = relative_parts[0].lower()

    label = label.replace(" ", "_")
    return CLASS_NAME_MAP.get(label, label)


def md5_hash(path):
    try:
        h = hashlib.md5()

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)

        return h.hexdigest()

    except Exception:
        return None


def read_image_info(path):
    image = cv2.imread(str(path))

    if image is None:
        return {
            "is_readable": False,
            "width": np.nan,
            "height": np.nan,
            "channels": np.nan,
            "file_size_kb": np.nan
        }

    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1

    return {
        "is_readable": True,
        "width": width,
        "height": height,
        "channels": channels,
        "file_size_kb": round(path.stat().st_size / 1024, 2)
    }


def build_manifest():
    print("Dataset path:", DATASET_DIR)

    image_paths = [
        p for p in DATASET_DIR.rglob("*")
        if p.is_file() and is_image_file(p)
    ]

    if len(image_paths) == 0:
        raise ValueError(f"No images found in dataset path: {DATASET_DIR}")

    records = []

    for path in image_paths:
        info = read_image_info(path)

        records.append({
            "filepath": str(path),
            "filename": path.name,
            "label": get_label(path),
            "extension": path.suffix.lower(),
            "file_hash": md5_hash(path),
            **info
        })

    df = pd.DataFrame(records)

    print("Total image files found:", len(df))
    print("Classes found:", sorted(df["label"].unique()))
    print("Unreadable images:", int((~df["is_readable"]).sum()))

    return df


def clean_manifest(df):
    duplicate_count = int(df[df["is_readable"]].duplicated(subset="file_hash", keep="first").sum())

    clean_df = df[df["is_readable"] == True].copy()
    clean_df = clean_df.drop_duplicates(subset="file_hash", keep="first").copy()

    print("Exact duplicate images:", duplicate_count)
    print("Clean images used:", len(clean_df))

    return clean_df


def create_splits(clean_df):
    train_df, temp_df = train_test_split(
        clean_df,
        test_size=0.30,
        stratify=clean_df["label"],
        random_state=RANDOM_STATE
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=RANDOM_STATE
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"

    print("Train images:", len(train_df))
    print("Validation images:", len(val_df))
    print("Test images:", len(test_df))

    return train_df, val_df, test_df


def create_label_map(clean_df):
    classes = sorted(clean_df["label"].unique())
    return {class_name: i for i, class_name in enumerate(classes)}


def create_class_weights(train_df, label_map):
    classes = np.array(sorted(train_df["label"].unique()))

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label"].values
    )

    return {
        str(label_map[class_name]): float(weight)
        for class_name, weight in zip(classes, weights)
    }


def save_split_figure(train_df, val_df, test_df):
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)

    table = pd.crosstab(combined["label"], combined["split"])
    table = table[["train", "validation", "test"]]

    ax = table.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Train, Validation and Test Distribution by Class")
    ax.set_xlabel("Blood Cell Class")
    ax.set_ylabel("Number of Images")

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "fig7_split_distribution.png", dpi=300)
    plt.savefig(OUTPUT_DIR / "fig7_split_distribution.png", dpi=300)
    plt.close()


def save_augmentation_figure(train_df):
    sample_path = Path(train_df.iloc[0]["filepath"])
    image = cv2.imread(str(sample_path))

    if image is None:
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)

    h, w = image.shape[:2]

    flipped = cv2.flip(image, 1)

    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, 20, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    contrast = cv2.convertScaleAbs(image, alpha=1.2, beta=20)

    images = [image, flipped, rotated, contrast]
    titles = ["Original", "Horizontal Flip", "Rotate +20", "Contrast + Brightness"]

    plt.figure(figsize=(12, 4))

    for i, (img, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(1, 4, i)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "fig8_augmentation_demo.png", dpi=300)
    plt.savefig(OUTPUT_DIR / "fig8_augmentation_demo.png", dpi=300)
    plt.close()


def save_outputs(raw_df, clean_df, train_df, val_df, test_df, label_map, class_weights):
    raw_df.to_csv(OUTPUT_DIR / "image_manifest_raw.csv", index=False)
    clean_df.to_csv(OUTPUT_DIR / "image_manifest_clean.csv", index=False)

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    save_json(label_map, PROCESSED_DIR / "label_map.json")
    save_json(class_weights, PROCESSED_DIR / "class_weights.json")

    save_json(label_map, OUTPUT_DIR / "label_map.json")
    save_json(class_weights, OUTPUT_DIR / "class_weights.json")

    pipeline_config = {
        "project_dir": str(PROJECT_DIR),
        "dataset_dir": str(DATASET_DIR),
        "image_size": IMAGE_SIZE,
        "random_state": RANDOM_STATE,
        "total_images_found": int(len(raw_df)),
        "unreadable_images": int((~raw_df["is_readable"]).sum()),
        "clean_images_used": int(len(clean_df)),
        "train_images": int(len(train_df)),
        "validation_images": int(len(val_df)),
        "test_images": int(len(test_df)),
        "classes": label_map
    }

    save_json(pipeline_config, PROCESSED_DIR / "pipeline_config.json")
    save_json(pipeline_config, OUTPUT_DIR / "pipeline_config.json")


def run_preprocessing_pipeline():
    print("Blood Cells Classification - Preprocessing Pipeline")

    raw_df = build_manifest()
    clean_df = clean_manifest(raw_df)

    train_df, val_df, test_df = create_splits(clean_df)

    label_map = create_label_map(clean_df)
    class_weights = create_class_weights(train_df, label_map)

    save_outputs(
        raw_df,
        clean_df,
        train_df,
        val_df,
        test_df,
        label_map,
        class_weights
    )

    save_split_figure(train_df, val_df, test_df)
    save_augmentation_figure(train_df)

    print("\nPreprocessing completed successfully.")
    print("Files saved in:")
    print(PROCESSED_DIR)
    print(FIGURES_DIR)
    print(OUTPUT_DIR)


if __name__ == "__main__":
    run_preprocessing_pipeline()
