import pandas as pd


def dataset_overview(df):
    """
    Show basic information about the dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset DataFrame.
    """

    print("Dataset shape:")
    print(df.shape)

    print("\nDataset information:")
    print(df.info())

    print("\nFirst rows:")
    print(df.head())


def check_missing_values(df):
    """
    Check missing values in each column.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.Series
        Number of missing values per column.
    """

    return df.isnull().sum()


def check_duplicate_rows(df):
    """
    Check duplicated rows in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    int
        Number of duplicated rows.
    """

    return df.duplicated().sum()


def class_distribution(df):
    """
    Count the number of images per class.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.Series
        Number of images for each class.
    """

    return df["label"].value_counts()


def imbalance_ratio(df):
    """
    Calculate class imbalance ratio.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    float
        Ratio between the largest and smallest class.
    """

    counts = class_distribution(df)

    return counts.max() / counts.min()


def descriptive_statistics(df):
    """
    Generate descriptive statistics for RGB features.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Descriptive statistics grouped by label.
    """

    return df.groupby("label")[["R_mean", "G_mean", "B_mean"]].describe()


def grayscale_statistics(df):
    """
    Generate descriptive statistics for grayscale feature.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.Series
        Descriptive statistics for Gray_mean.
    """

    return df["Gray_mean"].describe()


def rgb_correlation(df):
    """
    Calculate correlation between RGB features.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Correlation matrix.
    """

    return df[["R_mean", "G_mean", "B_mean"]].corr()


def print_eda_summary(df):
    """
    Print the main EDA results in a readable format.

    Parameters
    ----------
    df : pandas.DataFrame
    """

    print("========== Dataset Overview ==========")
    dataset_overview(df)

    print("\n========== Missing Values ==========")
    print(check_missing_values(df))

    print("\n========== Duplicate Rows ==========")
    print(check_duplicate_rows(df))

    print("\n========== Class Distribution ==========")
    print(class_distribution(df))

    print("\n========== Imbalance Ratio ==========")
    print(imbalance_ratio(df))

    if {"R_mean", "G_mean", "B_mean"}.issubset(df.columns):
        print("\n========== RGB Descriptive Statistics ==========")
        print(descriptive_statistics(df))

        print("\n========== RGB Correlation ==========")
        print(rgb_correlation(df))

    if "Gray_mean" in df.columns:
        print("\n========== Grayscale Statistics ==========")
        print(grayscale_statistics(df))