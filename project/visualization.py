import os
import glob

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from PIL import Image
from matplotlib.colors import ListedColormap

from config import DATASET_PATH, CLASSES


# =====================================================
# Sample Images
# =====================================================

def show_sample_images(dataset_path=DATASET_PATH, classes=CLASSES):
    """
    Display one sample image from each class.
    """

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for ax, cls in zip(axes.flat, classes):

        image_path = glob.glob(
            os.path.join(dataset_path, cls, "*.jpg")
        )[0]

        image = Image.open(image_path)

        ax.imshow(image)
        ax.set_title(cls.capitalize())
        ax.axis("off")

    plt.suptitle(
        "Figure 1. Representative microscopic images of the eight peripheral blood cell classes",
        fontsize=16,
        fontweight="bold"
    )
    plt.tight_layout()
    
    plt.show()


# =====================================================
# Class Distribution
# =====================================================

def plot_class_distribution(df):

    counts = (
        df["label"]
        .value_counts()
        .sort_values()
        .reset_index()
    )

    counts.columns = ["Class Name", "Number Images"]

    fig = px.bar(
        counts,
        x="Class Name",
        y="Number Images",
        text="Number Images",
        title="Blood Cell Class Distribution"
    )

    fig = px.bar(
        counts,
        x="Class Name",
        y="Number Images",
        text="Number Images",
        title="Figure 4. Distribution of image samples across the eight blood cell classes"
    )

    fig.update_traces(textposition="outside")
    fig.show()


# =====================================================
# RGB Statistics
# =====================================================

def plot_rgb_statistics(df):

    des = (
        df
        .groupby("label")[["R_mean", "G_mean", "B_mean"]]
        .describe()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=des.index,
            y=des["R_mean"]["mean"],
            name="Red"
        )
    )

    fig.add_trace(
        go.Bar(
            x=des.index,
            y=des["G_mean"]["mean"],
            name="Green"
        )
    )

    fig.add_trace(
        go.Bar(
            x=des.index,
            y=des["B_mean"]["mean"],
            name="Blue"
        )
    )

    fig.update_layout(
        title="Figure 5. Average RGB channel intensities for each blood cell class",
        barmode="group"
    )

    fig.show()


# =====================================================
# Overall RGB Intensity
# =====================================================

def plot_overall_rgb(df):

    des = (
        df
        .groupby("label")[["R_mean", "G_mean", "B_mean"]]
        .describe()
    )

    overall = (
        des["R_mean"]["mean"] +
        des["G_mean"]["mean"] +
        des["B_mean"]["mean"]
    ) / 3

    fig = px.bar(
        x=overall.index,
        y=overall.values,
        title="Figure 6. Average overall image intensity for each blood cell class"
    )

    fig.show()


# =====================================================
# Correlation
# =====================================================

def plot_rgb_correlation(df):

    corr = df[
        ["R_mean", "G_mean", "B_mean"]
    ].corr()

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.title(
        "Figure 10. Correlation matrix of extracted RGB features",
        fontsize=14,
        fontweight="bold"
    )
    plt.show()


# =====================================================
# RGB Histograms
# =====================================================

def plot_rgb_histograms(df):

    for channel in ["R_mean", "G_mean", "B_mean"]:

        px.histogram(
            df,
            x=channel,
            nbins=50,
            title=f"Figure 11. Distribution of {channel} values across the dataset"
            ).show()


# =====================================================
# RGB Violin
# =====================================================

def plot_rgb_violin(df):

    melted = df.melt(
        id_vars="label",
        value_vars=["R_mean", "G_mean", "B_mean"],
        var_name="Channel",
        value_name="Intensity"
    )

    fig = px.violin(
        melted,
        x="label",
        y="Intensity",
        color="Channel",
        box=True,
        points="all",
        title="Figure 12. Violin plots of RGB intensity distributions across blood cell classes"
    )

    fig.show()

# =====================================================
# RGB Boxplot
# =====================================================

def plot_rgb_boxplot(df):

    melted = df.melt(
        id_vars="label",
        value_vars=["R_mean", "G_mean", "B_mean"],
        var_name="Channel",
        value_name="Intensity"
    )

    fig = px.box(
        melted,
        x="Channel",
        y="Intensity",
        color="Channel",
        title="Figure 13. the distribution of extracted RGB mean intensity features across the dataset"
    )

    fig.show()

# =====================================================
# Gray Distribution
# =====================================================

def plot_gray_distribution(df):

    px.histogram(
        df,
        x="Gray_mean",
        nbins=50,
        title="Figure 8. Distribution of grayscale mean intensity"

    ).show()

    px.box(
        df,
        x="label",
        y="Gray_mean",
        title="Figure 9. Box plots of grayscale intensity across blood cell classes"

    ).show()

    px.violin(
        df,
        x="label",
        y="Gray_mean",
        box=True,
        points="all",
        title="Figure 15. Violin plots of grayscale intensity across blood cell classes"
    ).show()


# =====================================================
# Duplicate Heatmap
# =====================================================

def plot_duplicate_heatmap(heat):

    cmap = ListedColormap(
        ["white", "red", "blue"]
    )

    plt.figure(figsize=(8, 3))

    sns.heatmap(
        heat,
        cmap=cmap,
        linewidths=5,
        annot=True,
        cbar=False
    )

    plt.title(
    "Figure 3. Distribution of duplicate images across blood cell classes",
    fontsize=14,
    fontweight="bold"
    )

    plt.show()