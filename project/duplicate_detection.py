import hashlib
import pandas as pd
from PIL import Image


def image_hash(image_path):
    """
    Generate a SHA-256 hash for an image.

    Parameters
    ----------
    image_path : str
        Path to the image.

    Returns
    -------
    str
        SHA-256 hash value.
    """

    image = Image.open(image_path).convert("RGB")
    return hashlib.sha256(image.tobytes()).hexdigest()


def add_hash_column(df):
    """
    Add a hash column to the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    df = df.copy()
    df["hash"] = df["image_path"].apply(image_hash)

    return df


def find_duplicates(df):
    """
    Find duplicate images.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    duplicates = df.loc[
        df.duplicated("hash", keep=False),
        ["image_path", "label", "hash"]
    ].sort_values("hash")

    return duplicates


def count_duplicate_images(df):
    """
    Count duplicate images.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    int
    """

    return df.duplicated("hash").sum()


def group_duplicates(duplicates):
    """
    Group duplicate images by label and hash.

    Parameters
    ----------
    duplicates : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    grouped = (
        duplicates
        .groupby(["label", "hash"])["image_path"]
        .apply(list)
        .reset_index()
    )

    return grouped


def create_hash_dictionary(grouped_df):
    """
    Create a dictionary:
        key   -> hash
        value -> image path

    Parameters
    ----------
    grouped_df : pandas.DataFrame

    Returns
    -------
    dict
    """

    return (
        grouped_df
        .drop_duplicates("hash")
        .set_index("hash")["image_path"]
        .to_dict()
    )


def duplicate_heatmap_data(duplicates):
    """
    Create the matrix required for the duplicate heatmap.

    Parameters
    ----------
    duplicates : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    heatmap = (
        duplicates
        .groupby(["label", "hash"])
        .size()
        .unstack(fill_value=0)
    )

    return heatmap