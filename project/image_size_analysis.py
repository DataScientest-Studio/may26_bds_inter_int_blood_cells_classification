import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

from config import DATASET_PATH

# Main dataset folder path
#DATASET_PATH = "../PBC_dataset_normal_DIB"


def build_image_size_dataframe(dataset_path):
    """
    Build a DataFrame containing image path, class name,
    width, height, aspect ratio, and total pixels.
    """

    data = []

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    for class_name in os.listdir(dataset_path):

        class_path = os.path.join(dataset_path, class_name)

        # Skip files and keep only folders
        if not os.path.isdir(class_path):
            continue

        for file_name in os.listdir(class_path):

            # Keep only image files
            if not file_name.lower().endswith(valid_extensions):
                continue

            image_path = os.path.join(class_path, file_name)

            try:
                # Open image and extract width and height
                with Image.open(image_path) as img:
                    width, height = img.size

                data.append({
                    "image_path": image_path,
                    "class_name": class_name,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                    "aspect_ratio": width / height,
                    "total_pixels": width * height
                })

            except Exception as error:
                print(f"Cannot read image: {image_path}")
                print(error)

    df = pd.DataFrame(data)

    return df


def summarize_image_sizes(df):
    """
    Generate descriptive statistics for image sizes by class.
    """

    summary = (
        df
        .groupby("class_name")
        [["width", "height", "aspect_ratio", "total_pixels"]]
        .describe()
    )

    return summary


def check_unique_sizes(df):
    """
    Count the number of unique image sizes for each class.
    """

    unique_sizes = (
        df
        .groupby("class_name")
        .apply(lambda x: x[["width", "height"]].drop_duplicates().shape[0])
        .reset_index(name="unique_size_count")
    )

    return unique_sizes


def plot_image_size_scatter(df):
    """
    Plot width versus height for all images, colored by class.
    """

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x="width",
        y="height",
        hue="class_name"
    )

    plt.title("Image size distribution by class")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.show()


def plot_image_size_by_class(df):
    """
    Plot width versus height for each class using subplots.
    """

    classes = sorted(df["class_name"].unique())

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    axes = axes.flatten()

    for ax, class_name in zip(axes, classes):

        class_df = df[df["class_name"] == class_name]

        sns.scatterplot(
            data=class_df,
            x="width",
            y="height",
            ax=ax,
            s=40
        )

        ax.set_title(class_name.capitalize())
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        ax.grid(True)

    # Hide unused axes (if any)
    for ax in axes[len(classes):]:
        ax.set_visible(False)

    plt.suptitle(
        "Image Size Distribution by Blood Cell Class",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.show()



def plot_total_pixels_by_class(df):
    """
    Plot total pixel distribution for each class.
    """

    plt.figure(figsize=(12, 6))

    sns.boxplot(
        data=df,
        x="class_name",
        y="total_pixels"
    )

    plt.title("Total pixel distribution by class")
    plt.xlabel("Class")
    plt.ylabel("Total pixels")
    plt.xticks(rotation=45)
    plt.show()


def plot_aspect_ratio_by_class(df):
    """
    Plot aspect ratio distribution for each class.
    """

    plt.figure(figsize=(12, 6))

    sns.boxplot(
        data=df,
        x="class_name",
        y="aspect_ratio"
    )

    plt.title("Aspect ratio distribution by class")
    plt.xlabel("Class")
    plt.ylabel("Aspect ratio")
    plt.xticks(rotation=45)
    plt.show()


def main():

    df_sizes = build_image_size_dataframe(DATASET_PATH)

    print("DataFrame:")
    print(df_sizes.head())

    print("\nShape:")
    print(df_sizes.shape)

    print("\nSummary by class:")
    print(summarize_image_sizes(df_sizes))

    print("\nUnique image sizes by class:")
    print(check_unique_sizes(df_sizes))

    # Save image size analysis results
    df_sizes.to_csv(
        "image_size_analysis.csv",
        index=False,
        encoding="utf-8"
    )

    plot_image_size_scatter(df_sizes)

    plot_image_size_by_class(df_sizes)

    plot_total_pixels_by_class(df_sizes)

    plot_aspect_ratio_by_class(df_sizes)


if __name__ == "__main__":
    main()