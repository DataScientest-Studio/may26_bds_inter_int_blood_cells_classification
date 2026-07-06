import numpy as np
from PIL import Image


def get_mean_rgb(image_path):
    """
    Calculate the mean RGB values of an image.

    Parameters
    ----------
    image_path : str
        Path to the image.

    Returns
    -------
    numpy.ndarray
        Mean values of red, green, and blue channels.
    """

    image = Image.open(image_path).convert("RGB")
    array = np.array(image)

    return array.mean(axis=(0, 1))


def extract_rgb_features(df):
    """
    Add RGB mean features to the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing image_path column.

    Returns
    -------
    pandas.DataFrame
        DataFrame with R_mean, G_mean, and B_mean columns.
    """

    df = df.copy()

    rgb_means = np.array(
        [get_mean_rgb(path) for path in df["image_path"]]
    )

    df[["R_mean", "G_mean", "B_mean"]] = rgb_means

    return df


def get_mean_gray(image_path):
    """
    Calculate the mean grayscale intensity of an image.

    Parameters
    ----------
    image_path : str
        Path to the image.

    Returns
    -------
    float
        Mean grayscale intensity.
    """

    image = Image.open(image_path).convert("L")
    array = np.array(image)

    return array.mean()


def extract_gray_features(df):
    """
    Add grayscale mean feature to the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing image_path column.

    Returns
    -------
    pandas.DataFrame
        DataFrame with Gray_mean column.
    """

    df = df.copy()

    df["Gray_mean"] = df["image_path"].apply(get_mean_gray)

    return df


def extract_all_features(df):
    """
    Extract all image features.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        DataFrame with RGB and grayscale features.
    """

    df = extract_rgb_features(df)
    df = extract_gray_features(df)

    return df