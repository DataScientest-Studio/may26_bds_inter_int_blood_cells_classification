import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import cv2


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


def calculate_blur_score(df):
    """
    Calculate the blur score for every image in the dataset using the
    Variance of the Laplacian method.

    Each image is:
        1. Loaded from its file path.
        2. Converted to grayscale.
        3. Processed using the Laplacian operator.
        4. Assigned a blur score equal to the variance of the Laplacian.

    Images that cannot be loaded are assigned a value of None.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing an 'image_path' column with the paths to
        all images in the dataset.

    Returns
    -------
    pandas.Series
        A Series named 'blur_score' containing the blur score for each
        image. Higher values indicate sharper images, whereas lower
        values indicate blurrier images.
    """    
    blur_scores = []
    
    for image_path in df['image_path']:
        image = cv2.imread(image_path)
        
        if image is None:
            blur_scores.append(None)
            continue
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        blur_scores.append(cv2.Laplacian(gray_image, cv2.CV_64F).var())
        
    return pd.Series(
        blur_scores,
        name="blur_score"
    )


def print_eda_summary(df, blur_scores):
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
    
    print("\n========== Blur Infos ==========")
        
    print(blur_scores.describe())
    
    Q1 = blur_scores.quantile(0.25)
    Q3 = blur_scores.quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    print(f"Lower bound : {lower:.2f}")
    print(f"Upper bound : {upper:.2f}")
    
    
def pca(df):
    X = df[["R_mean", "G_mean", "B_mean"]]
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    df["PC1"] = X_pca[:, 0]
    df["PC2"] = X_pca[:, 1]
    print("variances of PC1 & PC2: ", pca.explained_variance_ratio_)  
      
    return df, pca