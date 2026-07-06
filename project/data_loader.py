import os
import glob
import pandas as pd

from config import DATASET_PATH, DATASET_CSV, IMAGE_EXTENSION


def load_dataset(dataset_path=DATASET_PATH):
    """
    Read image paths from dataset folders and create a DataFrame.

    Returns:
        pd.DataFrame: DataFrame with image_path and label columns.
    """

    data = []

    image_paths = glob.glob(
        os.path.join(dataset_path, "*", IMAGE_EXTENSION)
    )

    for image_path in image_paths:
        label = os.path.basename(os.path.dirname(image_path))
        data.append([image_path, label])

    df = pd.DataFrame(
        data,
        columns=["image_path", "label"]
    )

    return df


def save_dataframe(df, csv_path=DATASET_CSV):
    """
    Save DataFrame to CSV file.
    """

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8"
    )


def load_dataframe(csv_path=DATASET_CSV):
    """
    Load DataFrame from CSV file.
    """

    df = pd.read_csv(csv_path)

    return df