"""
Preprocessing pipeline for the PBC blood-cell image-classification project.

This file is designed to fit the current EDA pipeline in main.py and the
image-size utilities in image_size_analysis.py.

Course coverage used:
- 150  : Computer Vision with OpenCV -> image reading, RGB conversion, resizing
- 151.1: Dense Neural Networks with Keras -> optional flattened image tensors
- 151.2: Convolutional Neural Networks with Keras -> CNN-ready image tensors
- 155  : TensorFlow -> tf.data input pipeline, batching, prefetching

Main outputs:
- preprocessing_output/clean_manifest.csv
- preprocessing_output/train.csv
- preprocessing_output/val.csv
- preprocessing_output/test.csv
- preprocessing_output/label_map.json
- preprocessing_output/class_weights.json
- preprocessing_output/preprocessing_config.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from config import DATASET_PATH


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

VALID_EXTENSIONS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
)

SEED: int = 42
IMAGE_SIZE: Tuple[int, int] = (224, 224)
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15
BATCH_SIZE: int = 32

OUTPUT_DIR = Path("preprocessing_output")

# Standard ImageNet values are useful if transfer learning is used later.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def _ensure_output_dir(output_dir: Path | str = OUTPUT_DIR) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _class_folders(dataset_path: Path | str) -> list[str]:
    """Return sorted class folders from the dataset root."""
    dataset_path = Path(dataset_path)
    return sorted(
        item.name
        for item in dataset_path.iterdir()
        if item.is_dir()
    )


def _is_image_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in VALID_EXTENSIONS


# ---------------------------------------------------------------------
# 1. Build image manifest
# ---------------------------------------------------------------------

def build_image_manifest(dataset_path: Path | str = DATASET_PATH) -> pd.DataFrame:
    """
    Scan class-wise image folders and build a manifest.

    Expected dataset structure:
        dataset_root/
            basophil/
                image_1.jpg
            eosinophil/
                image_2.jpg
            ...

    Returned columns:
        image_path, relative_path, file_name, class_name
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    rows = []

    for class_name in _class_folders(dataset_path):
        class_path = dataset_path / class_name

        for image_path in sorted(class_path.iterdir()):
            if not image_path.is_file() or not _is_image_file(image_path):
                continue

            rows.append(
                {
                    "image_path": str(image_path),
                    "relative_path": str(image_path.relative_to(dataset_path)),
                    "file_name": image_path.name,
                    "class_name": class_name,
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No images found in dataset path: {dataset_path}")

    return df


# ---------------------------------------------------------------------
# 2. OpenCV-based validation and size metadata
# ---------------------------------------------------------------------

def read_image_opencv(image_path: str | Path) -> np.ndarray:
    """
    Read an image with OpenCV and convert BGR to RGB.

    OpenCV loads images in BGR format by default. For Keras/TensorFlow
    visualization and modelling, RGB is usually easier to interpret.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def add_image_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add width, height, channels, aspect_ratio, total_pixels, and readable flag.
    """
    records = []

    for _, row in df.iterrows():
        image_path = row["image_path"]

        try:
            image = read_image_opencv(image_path)
            height, width, channels = image.shape
            readable = True
            error = ""
        except Exception as exc:
            height, width, channels = np.nan, np.nan, np.nan
            readable = False
            error = str(exc)

        record = row.to_dict()
        record.update(
            {
                "width": width,
                "height": height,
                "channels": channels,
                "aspect_ratio": width / height if readable and height else np.nan,
                "total_pixels": width * height if readable else np.nan,
                "readable": readable,
                "read_error": error,
            }
        )
        records.append(record)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# 3. Duplicate detection using file hash
# ---------------------------------------------------------------------

def compute_file_hash(image_path: str | Path, block_size: int = 65536) -> str:
    """Compute md5 hash for exact duplicate detection."""
    import hashlib

    md5 = hashlib.md5()
    with open(image_path, "rb") as file:
        for block in iter(lambda: file.read(block_size), b""):
            md5.update(block)
    return md5.hexdigest()


def add_hash_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add md5 file hash to each readable image."""
    df = df.copy()
    df["hash"] = df["image_path"].apply(compute_file_hash)
    return df


def clean_manifest(df: pd.DataFrame, remove_duplicates: bool = True) -> pd.DataFrame:
    """
    Remove unreadable images and optionally remove exact duplicates.
    """
    clean_df = df[df["readable"]].copy()

    if remove_duplicates:
        clean_df = clean_df.drop_duplicates(subset="hash", keep="first")

    clean_df = clean_df.reset_index(drop=True)
    return clean_df


# ---------------------------------------------------------------------
# 4. Label encoding and stratified train/validation/test split
# ---------------------------------------------------------------------

def create_label_map(df: pd.DataFrame) -> Dict[str, int]:
    """Create class-name to integer-label mapping."""
    classes = sorted(df["class_name"].unique())
    return {class_name: idx for idx, class_name in enumerate(classes)}


def add_label_index(df: pd.DataFrame, label_map: Dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    df["label_idx"] = df["class_name"].map(label_map).astype(int)
    return df


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/validation/test splits.
    """
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=seed,
        stratify=df["label_idx"],
    )

    relative_test_ratio = test_ratio / (val_ratio + test_ratio)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=temp_df["label_idx"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ---------------------------------------------------------------------
# 5. Class weights for imbalanced classes
# ---------------------------------------------------------------------

def calculate_class_weights(train_df: pd.DataFrame) -> Dict[int, float]:
    """
    Calculate balanced class weights for Keras model.fit(class_weight=...).
    """
    labels = np.sort(train_df["label_idx"].unique())

    weights = compute_class_weight(
        class_weight="balanced",
        classes=labels,
        y=train_df["label_idx"].values,
    )

    return {int(label): float(weight) for label, weight in zip(labels, weights)}


# ---------------------------------------------------------------------
# 6. OpenCV preprocessing for single image
# ---------------------------------------------------------------------

def preprocess_image_opencv(
    image_path: str | Path,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    normalize: bool = True,
) -> np.ndarray:
    """
    Read, resize, and normalize one image with OpenCV.

    Output shape for CNN:
        (height, width, 3)
    Output value range:
        0 to 1 when normalize=True
    """
    image = read_image_opencv(image_path)
    image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)

    if normalize:
        image = image.astype("float32") / 255.0

    return image


def preprocess_for_dense_network(image: np.ndarray) -> np.ndarray:
    """
    Flatten a preprocessed image for a Dense Neural Network.

    Example:
        CNN input:   (224, 224, 3)
        DNN input:   (150528,)
    """
    return image.reshape(-1).astype("float32")


# ---------------------------------------------------------------------
# 7. TensorFlow / Keras input pipeline
# ---------------------------------------------------------------------

def build_keras_augmentation_layer():
    """
    Keras augmentation layer for the training set only.
    """
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="training_augmentation",
    )


def build_tf_dataset(
    manifest_df: pd.DataFrame,
    dataset_path: Path | str = DATASET_PATH,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    training: bool = False,
    model_type: str = "cnn",
    use_imagenet_normalization: bool = False,
):
    """
    Build a TensorFlow tf.data pipeline from a manifest dataframe.

    model_type:
        "cnn"   -> returns image tensors with shape (batch, H, W, 3)
        "dense" -> returns flattened tensors with shape (batch, H*W*3)
    """
    import tensorflow as tf

    if model_type not in {"cnn", "dense"}:
        raise ValueError("model_type must be either 'cnn' or 'dense'")

    dataset_path = tf.constant(str(dataset_path))
    image_height, image_width = image_size

    paths = manifest_df["relative_path"].astype(str).values
    labels = manifest_df["label_idx"].astype("int32").values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(buffer_size=len(manifest_df), seed=SEED, reshuffle_each_iteration=True)

    augmentation_layer = build_keras_augmentation_layer() if training else None

    def load_and_preprocess(relative_path, label):
        full_path = tf.strings.join([dataset_path, relative_path], separator=os.sep)
        image_bytes = tf.io.read_file(full_path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, [image_height, image_width])
        image = tf.cast(image, tf.float32) / 255.0

        if training:
            image = augmentation_layer(image, training=True)

        if use_imagenet_normalization:
            mean = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
            std = tf.constant(IMAGENET_STD, dtype=tf.float32)
            image = (image - mean) / std

        if model_type == "dense":
            image = tf.reshape(image, [-1])

        return image, label

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# ---------------------------------------------------------------------
# 8. Save artifacts
# ---------------------------------------------------------------------

def _save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_preprocessing_artifacts(
    clean_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_map: Dict[str, int],
    class_weights: Dict[int, float],
    output_dir: Path | str = OUTPUT_DIR,
) -> None:
    """Save all preprocessing outputs for modelling."""
    output_dir = _ensure_output_dir(output_dir)

    manifest_columns = [
        "relative_path",
        "class_name",
        "label_idx",
        "width",
        "height",
        "aspect_ratio",
        "total_pixels",
        "hash",
    ]

    clean_df[manifest_columns].to_csv(output_dir / "clean_manifest.csv", index=False)
    train_df[["relative_path", "class_name", "label_idx"]].to_csv(output_dir / "train.csv", index=False)
    val_df[["relative_path", "class_name", "label_idx"]].to_csv(output_dir / "val.csv", index=False)
    test_df[["relative_path", "class_name", "label_idx"]].to_csv(output_dir / "test.csv", index=False)

    _save_json(label_map, output_dir / "label_map.json")
    _save_json(class_weights, output_dir / "class_weights.json")

    config = {
        "dataset_path": str(DATASET_PATH),
        "image_size": list(IMAGE_SIZE),
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "split": {
            "train": TRAIN_RATIO,
            "validation": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "normalization": {
            "default": "scale pixels to 0-1",
            "optional_transfer_learning": {
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
            },
        },
        "augmentation_training_only": [
            "RandomFlip(horizontal_and_vertical)",
            "RandomRotation(0.08)",
            "RandomZoom(0.10)",
            "RandomContrast(0.10)",
        ],
        "courses_used": [
            "150 - Computer Vision with OpenCV (EN)",
            "151.1 - Dense Neural Networks with Keras (EN)",
            "151.2 - Convolutional Neural Networks with Keras (EN)",
            "155 - Tensorflow (EN)",
        ],
    }

    _save_json(config, output_dir / "preprocessing_config.json")


# ---------------------------------------------------------------------
# 9. Full preprocessing pipeline
# ---------------------------------------------------------------------

def run_preprocessing_pipeline(
    dataset_path: Path | str = DATASET_PATH,
    output_dir: Path | str = OUTPUT_DIR,
    remove_duplicates: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the complete preprocessing pipeline.

    Returns:
        train_df, val_df, test_df
    """
    output_dir = _ensure_output_dir(output_dir)

    print("Building image manifest ...")
    df = build_image_manifest(dataset_path)

    print("Adding OpenCV image metadata ...")
    df = add_image_metadata(df)

    unreadable_count = int((~df["readable"]).sum())
    print(f"Unreadable images: {unreadable_count}")

    print("Adding hash column for duplicate detection ...")
    df = add_hash_column(df[df["readable"]].copy())

    duplicate_count = int(df["hash"].duplicated().sum())
    print(f"Exact duplicate images: {duplicate_count}")

    print("Cleaning manifest ...")
    clean_df = clean_manifest(df, remove_duplicates=remove_duplicates)

    label_map = create_label_map(clean_df)
    clean_df = add_label_index(clean_df, label_map)

    print("Creating stratified train/validation/test split ...")
    train_df, val_df, test_df = stratified_split(clean_df)

    class_weights = calculate_class_weights(train_df)

    save_preprocessing_artifacts(
        clean_df=clean_df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_map=label_map,
        class_weights=class_weights,
        output_dir=output_dir,
    )

    print("\nPreprocessing completed successfully.")
    print(f"Clean images used: {len(clean_df)}")
    print(f"Train images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print(f"Test images: {len(test_df)}")
    print(f"Artifacts saved in: {Path(output_dir).resolve()}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    run_preprocessing_pipeline()
