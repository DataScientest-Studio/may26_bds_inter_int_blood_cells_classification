import os

from config import DATASET_CSV, DATASET_PATH

from data_loader import (
    load_dataset,
    load_dataframe,
    save_dataframe
)

from feature_extraction import extract_all_features

from duplicate_detection import (
    add_hash_column,
    count_duplicate_images,
    find_duplicates,
    group_duplicates,
    duplicate_heatmap_data
)

from eda import print_eda_summary

from image_size_analysis import (
    build_image_size_dataframe,
    summarize_image_sizes,
    check_unique_sizes,
    plot_image_size_scatter,
    plot_image_size_by_class,
    plot_total_pixels_by_class,
    plot_aspect_ratio_by_class
)

from visualization import (
    show_sample_images,
    plot_class_distribution,
    plot_rgb_statistics,
    plot_overall_rgb,
    plot_rgb_correlation,
    plot_rgb_histograms,
    plot_rgb_boxplot,
    plot_rgb_violin,
    plot_gray_distribution,
    plot_duplicate_heatmap
)


def main():
    """
    Run the full data exploration pipeline.
    """

    if os.path.exists(DATASET_CSV):

        print("Loading existing dataset.csv ...")
        df = load_dataframe()

        if "hash" not in df.columns:

            print("Hash column not found.")
            print("Creating hash column ...")

            df = add_hash_column(df)
            save_dataframe(df)

    else:

        print("dataset.csv not found.")
        print("Creating dataset from image folders ...")

        df = load_dataset()
        df = extract_all_features(df)
        df = add_hash_column(df)

        save_dataframe(df)

        print("dataset.csv created successfully.")




    # --------------------------------
    # Image Size Analysis
    # --------------------------------

    df_sizes = build_image_size_dataframe(DATASET_PATH)

    print("\nImage Size Summary:")
    print(summarize_image_sizes(df_sizes))

    print("\nUnique Image Sizes:")
    print(check_unique_sizes(df_sizes))

    df_sizes.to_csv(
        "image_size_analysis.csv",
        index=False
    )

    plot_image_size_by_class(df_sizes)

    plot_image_size_scatter(df_sizes)

    plot_total_pixels_by_class(df_sizes)

    plot_aspect_ratio_by_class(df_sizes)



    # --------------------------------
    # Duplicate Images
    # --------------------------------
    duplicate_count = count_duplicate_images(df)

    print("Number of duplicate images:")
    print(duplicate_count)

    duplicates = find_duplicates(df)
    grouped_duplicates = group_duplicates(duplicates)
    heat = duplicate_heatmap_data(duplicates)

    print_eda_summary(df)

    print("\nGrouped duplicate images:")
    print(grouped_duplicates)

    show_sample_images()

    plot_class_distribution(df)

    plot_rgb_statistics(df)

    plot_overall_rgb(df)

    plot_rgb_correlation(df)

    plot_rgb_histograms(df)

    plot_rgb_boxplot(df)

    plot_rgb_violin(df)

    plot_gray_distribution(df)

    plot_duplicate_heatmap(heat)


if __name__ == "__main__":
    main()