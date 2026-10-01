import ipywidgets as widgets
from ipywidgets import Layout, VBox, Button, Accordion, SelectMultiple, IntText
import scipy.stats as stats
import numpy as np
from multiprocessing import Pool
import pandas as pd
import os
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from matplotlib.gridspec import GridSpec
import requests
from scipy.stats import zscore
from scipy.stats import ks_2samp
from sklearn.preprocessing import MinMaxScaler
from tifffile import imwrite
from tqdm.notebook import tqdm
import imageio
from celltracks.BoxPlots_Statistics import *

# Function to plot selected variables per cluster with data saved to CSV and plots saved as PDF
def plot_selected_vars_per_cluster(button, Cluster, checkboxes_dict, df, base_folder, ):
    print("Plotting in progress...")

    # Get selected variables
    variables_to_plot = []
    for category, checkboxes in checkboxes_dict.items():
        if isinstance(checkboxes, dict):
            for subcategory, subcheckboxes in checkboxes.items():
                for checkbox in subcheckboxes:
                    if checkbox.value:
                        variables_to_plot.append(checkbox.description)
        else:
            for checkbox in checkboxes:
                if checkbox.value:
                    variables_to_plot.append(checkbox.description)

    n_plots = len(variables_to_plot)

    if n_plots == 0:
        print("No variables selected for plotting")
        return

    for var in variables_to_plot:
        # Extract data for the specific variable and cluster
        data_to_save = df[[Cluster, var]]

        # Save data for the plot to CSV
        data_to_save.to_csv(f"{base_folder}/{var}_data_by_Cluster.csv", index=False)

        plt.figure(figsize=(16, 10))

        # Plotting
        sns.boxplot(x=Cluster, y=var, data=df, color='lightgray')  # Boxplot by cluster
        sns.stripplot(x=Cluster, y=var, data=df, jitter=True, alpha=0.2)  # Individual data points

        plt.title(f"{var} by Cluster")
        plt.xlabel(Cluster)
        plt.ylabel(var)
        plt.xticks(rotation=90)
        plt.tight_layout()

        # Save the plot
        plt.savefig(f"{base_folder}/{var}_Boxplots_by_Cluster.pdf")
        plt.show()

        # Save the plot
        plt.savefig(f"{base_folder}/{var}_Boxplots_by_Cluster.pdf")
        plt.show()
        
def plot_selected_vars_cluster(button, checkboxes_dict, df, Conditions, Cluster, cluster_dropdown, Results_Folder, condition_selector, stat_method_selector):
    plt.clf()  # Clear the current figure before creating a new plot

    # Get selected variables
    variables_to_plot = []
    for category, checkboxes in checkboxes_dict.items():
        if isinstance(checkboxes, dict):
            for subcategory, subcheckboxes in checkboxes.items():
                for checkbox in subcheckboxes:
                    if checkbox.value:
                        variables_to_plot.append(checkbox.description)
        else:
            for checkbox in checkboxes:
                if checkbox.value:
                    variables_to_plot.append(checkbox.description)

    method = stat_method_selector.value

    if len(variables_to_plot) == 0:
        print("No variables selected for plotting")
        return

    # Get selected conditions
    selected_conditions = condition_selector.value
    if len(selected_conditions) == 0:
        print("No conditions selected for plotting, therefore all available conditions are selected by default")
        selected_conditions = df[Conditions].dropna().unique().tolist()

    selected_cluster = cluster_dropdown.value
    print(f"Plotting in progress for Cluster {selected_cluster}...")

    filtered_df = df[
        (df[Conditions].isin(selected_conditions))
        & (df[Cluster] == selected_cluster)
    ].copy()

    if filtered_df.empty:
        print(
            f"No rows are available for Cluster {selected_cluster} "
            "and the selected conditions."
        )
        return

    # Ensure result folders exist before writing output files.
    os.makedirs(os.path.join(Results_Folder, "pdf"), exist_ok=True)
    os.makedirs(os.path.join(Results_Folder, "csv"), exist_ok=True)

    # Create empty data structures for statistics
    effect_size_matrices = {}
    p_value_matrices = {}
    bonferroni_matrices = {}

    unique_conditions = filtered_df[Conditions].dropna().unique().tolist()
    n_iterations = 10000

    for var in variables_to_plot:
        if var not in filtered_df.columns:
            print(f"Skipping '{var}': column not found in the DataFrame.")
            continue

        # Work on a variable-specific copy. This prevents string/object values,
        # NaNs, or infinities in one metric from affecting other selected metrics.
        var_df = filtered_df.copy()
        raw_values = var_df[var].copy()

        # Convert numeric strings to real numbers. Non-numeric text becomes NaN.
        var_df[var] = pd.to_numeric(raw_values, errors="coerce")
        var_df[var] = var_df[var].replace([np.inf, -np.inf], np.nan)

        # Report values that were present but could not be interpreted as numbers.
        invalid_mask = raw_values.notna() & var_df[var].isna()
        if invalid_mask.any():
            examples = (
                raw_values.loc[invalid_mask]
                .astype(str)
                .drop_duplicates()
                .head(5)
                .tolist()
            )
            print(
                f"{var}: ignoring {int(invalid_mask.sum())} non-numeric "
                f"value(s). Examples: {examples}"
            )

        rows_before = len(var_df)
        var_df = var_df.dropna(subset=[var]).copy()
        rows_removed = rows_before - len(var_df)

        if rows_removed:
            print(
                f"{var}: removed {rows_removed} row(s) without a finite "
                "numeric value for this analysis."
            )

        if var_df.empty:
            print(
                f"Skipping '{var}': no valid numeric values are available "
                f"for Cluster {selected_cluster}."
            )
            continue

        # Keep the full condition layout in the exported matrices, but mark
        # comparisons that cannot be calculated as NaN rather than inventing 0/1.
        effect_size_matrices[var] = pd.DataFrame(
            np.nan,
            index=unique_conditions,
            columns=unique_conditions,
            dtype=float,
        )
        p_value_matrices[var] = pd.DataFrame(
            np.nan,
            index=unique_conditions,
            columns=unique_conditions,
            dtype=float,
        )
        bonferroni_matrices[var] = pd.DataFrame(
            np.nan,
            index=unique_conditions,
            columns=unique_conditions,
            dtype=float,
        )

        # Conventional diagonal values for comparisons of a condition with itself.
        for condition in unique_conditions:
            if (var_df[Conditions] == condition).any():
                effect_size_matrices[var].loc[condition, condition] = 0.0
                p_value_matrices[var].loc[condition, condition] = 1.0
                bonferroni_matrices[var].loc[condition, condition] = 1.0

        # Bonferroni should count only comparisons that can actually be tested.
        valid_conditions = [
            condition
            for condition in unique_conditions
            if var_df.loc[var_df[Conditions] == condition, var].notna().sum() >= 2
        ]
        num_comparisons = len(valid_conditions) * (len(valid_conditions) - 1) // 2

        for cond1, cond2 in itertools.combinations(unique_conditions, 2):
            group1 = var_df.loc[var_df[Conditions] == cond1, var]
            group2 = var_df.loc[var_df[Conditions] == cond2, var]

            # Two observations per group are required for a variance-based effect
            # size and for the inferential tests used here.
            if len(group1) < 2 or len(group2) < 2:
                print(
                    f"{var}: skipping {cond1} vs {cond2} because one or both "
                    "conditions have fewer than 2 valid observations."
                )
                continue

            effect_size = abs(cohen_d(group1, group2))

            if method == "t-test":
                p_value = perform_t_test(var_df, cond1, cond2, var)
            elif method == "randomization test":
                p_value = perform_randomization_test_parallel(
                    var_df,
                    cond1,
                    cond2,
                    var,
                    n_iterations=n_iterations,
                )
            else:
                raise ValueError(f"Unsupported statistical method: {method}")

            effect_size = float(effect_size) if pd.notna(effect_size) else np.nan
            p_value = float(p_value) if pd.notna(p_value) else np.nan

            effect_size_matrices[var].loc[cond1, cond2] = effect_size
            effect_size_matrices[var].loc[cond2, cond1] = effect_size
            p_value_matrices[var].loc[cond1, cond2] = p_value
            p_value_matrices[var].loc[cond2, cond1] = p_value

            if pd.notna(p_value) and num_comparisons > 0:
                bonferroni_corrected_p_value = min(
                    p_value * num_comparisons,
                    1.0,
                )
            else:
                bonferroni_corrected_p_value = np.nan

            bonferroni_matrices[var].loc[cond1, cond2] = bonferroni_corrected_p_value
            bonferroni_matrices[var].loc[cond2, cond1] = bonferroni_corrected_p_value

        # Save statistics to CSV.
        combined_df = pd.concat(
            [
                effect_size_matrices[var].rename(
                    columns=lambda x: f"{x} (Effect Size)"
                ),
                p_value_matrices[var].rename(
                    columns=lambda x: f"{x} ({method} P-Value)"
                ),
                bonferroni_matrices[var].rename(
                    columns=lambda x: f"{x} ({method} Bonferroni-corrected P-Value)"
                ),
            ],
            axis=1,
        )
        combined_df.to_csv(
            os.path.join(
                Results_Folder,
                "csv",
                f"Cluster_{selected_cluster}_{var}_statistics_combined.csv",
            )
        )

        # Save only the data that actually contributes to this cluster/condition plot.
        data_columns = [
            column
            for column in [Conditions, var, "Repeat", "File_name"]
            if column in var_df.columns
        ]
        data_for_var = var_df[data_columns].copy()
        data_for_var.to_csv(
            os.path.join(
                Results_Folder,
                "csv",
                f"Cluster_{selected_cluster}_{var}_boxplot_data.csv",
            ),
            index=False,
        )

        # Use only valid, filtered numeric values when setting the y-axis limits.
        # The original code used the 20th and 80th percentiles, so that behaviour
        # is retained here (this is a central-percentile range, not the standard IQR).
        finite_values = var_df[var]
        q_low = finite_values.quantile(0.2)
        q_high = finite_values.quantile(0.8)
        percentile_range = q_high - q_low

        multiplier = 10
        lower_bound = max(
            finite_values.min(),
            q_low - multiplier * percentile_range,
        )
        upper_bound = min(
            finite_values.max(),
            q_high + multiplier * percentile_range,
        )

        if np.isfinite(lower_bound) and np.isfinite(upper_bound):
            if np.isclose(lower_bound, upper_bound):
                padding = max(abs(lower_bound) * 0.05, 0.5)
                lower_bound -= padding
                upper_bound += padding

        pdf_path = os.path.join(
            Results_Folder,
            "pdf",
            f"Cluster_{selected_cluster}_{var}_Boxplots_and_Statistics.pdf",
        )

        with PdfPages(pdf_path) as pdf_pages:
            fig = plt.figure(figsize=(16, 10))
            gs = GridSpec(2, 3, height_ratios=[1.5, 1])
            ax_box = fig.add_subplot(gs[0, :])

            sns.boxplot(
                x=Conditions,
                y=var,
                data=var_df,
                ax=ax_box,
                color="lightgray",
            )
            sns.stripplot(
                x=Conditions,
                y=var,
                data=var_df,
                ax=ax_box,
                hue="Repeat" if "Repeat" in var_df.columns else None,
                dodge=True if "Repeat" in var_df.columns else False,
                jitter=True,
                alpha=0.2,
                palette="tab10" if "Repeat" in var_df.columns else None,
            )

            if np.isfinite(lower_bound) and np.isfinite(upper_bound):
                ax_box.set_ylim([lower_bound, upper_bound])

            ax_box.set_title(f"{var} for Cluster {selected_cluster}")
            ax_box.set_xlabel("Condition")
            ax_box.set_ylabel(var)
            ax_box.tick_params(axis="x", labelrotation=90)

            if "Repeat" in var_df.columns:
                ax_box.legend(
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    title="Repeat",
                )

            # Statistical analyses and heatmaps.
            ax_d = fig.add_subplot(gs[1, 0])
            sns.heatmap(
                effect_size_matrices[var],
                annot=True,
                cmap="viridis",
                cbar=True,
                square=True,
                ax=ax_d,
                vmax=1,
            )
            ax_d.set_title("Effect Size (Cohen's d)")

            ax_p = fig.add_subplot(gs[1, 1])
            plot_heatmap(ax_p, p_value_matrices[var], f"{method} p-value")

            ax_bonf = fig.add_subplot(gs[1, 2])
            plot_heatmap(
                ax_bonf,
                bonferroni_matrices[var],
                "Bonferroni-corrected p-value",
            )

            plt.tight_layout()
            pdf_pages.savefig(fig)
            plt.show()
            plt.close(fig)

def display_cluster_dropdown(df, Cluster):
    # Extract unique clusters
    unique_clusters = df[Cluster].unique()
    cluster_dropdown = widgets.Dropdown(
        options=unique_clusters,
        description='Select Cluster:',
        disabled=False,
    )
    #display(cluster_dropdown)
    return cluster_dropdown        


# Function to display an error message
def display_error_message(message):
    with error_output:
        print(message)

def overlay_square_on_frame(frame, x, y, square_size=50, border_width=3):
    """Overlay a red square on a single frame."""
    overlaid_frame = frame.copy()

    half_size = square_size // 2

    # Define the coordinates for the top-left and bottom-right corners of the square
    top_left_x = max(0, x - half_size)
    top_left_y = max(0, y - half_size)
    bottom_right_x = min(frame.shape[1] - 1, x + half_size)
    bottom_right_y = min(frame.shape[0] - 1, y + half_size)

    # Overlay the red border on the frame
    # Horizontal lines
    overlaid_frame[top_left_y:top_left_y+border_width, top_left_x:bottom_right_x] = np.max(frame)
    overlaid_frame[bottom_right_y-border_width:bottom_right_y, top_left_x:bottom_right_x] = np.max(frame)

    # Vertical lines
    overlaid_frame[top_left_y:bottom_right_y, top_left_x:top_left_x+border_width] = np.max(frame)
    overlaid_frame[top_left_y:bottom_right_y, bottom_right_x-border_width:bottom_right_x] = np.max(frame)

    return overlaid_frame


def percentile_normalize_and_convert_uint8(image_sequence, low_percentile=1, high_percentile=99):
    """
    Normalize the image sequence to 0-255 based on percentiles and convert to uint8.

    Parameters:
    - image_sequence: The sequence of images to be normalized.
    - low_percentile: Lower percentile value used for normalization.
    - high_percentile: Higher percentile value used for normalization.

    Returns:
    - Normalized image sequence in uint8 format.
    """
    # Compute the percentiles
    min_val = np.percentile(image_sequence, low_percentile)
    max_val = np.percentile(image_sequence, high_percentile)

    # Clip the values outside the percentiles and normalize
    normalized = 255 * (np.clip(image_sequence, min_val, max_val) - min_val) / (max_val - min_val)

    return normalized.astype(np.uint8)

# Function to find a TIFF file that matches the given filename in the directory or its subdirectories
def find_matching_tiff_file(directory, filename):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(filename) and (file.endswith('.tif') or file.endswith('.tiff')):
                return os.path.join(root, file)
    return None

def overlay_square_on_frame(frame, x, y, square_size=50, border_width=3):
    """Overlay a red square on a single frame."""
    overlaid_frame = frame.copy()

    half_size = square_size // 2

    # Define the coordinates for the top-left and bottom-right corners of the square
    top_left_x = max(0, x - half_size)
    top_left_y = max(0, y - half_size)
    bottom_right_x = min(frame.shape[1] - 1, x + half_size)
    bottom_right_y = min(frame.shape[0] - 1, y + half_size)

    # Overlay the red border on the frame
    # Horizontal lines
    overlaid_frame[top_left_y:top_left_y+border_width, top_left_x:bottom_right_x] = np.max(frame)
    overlaid_frame[bottom_right_y-border_width:bottom_right_y, top_left_x:bottom_right_x] = np.max(frame)

    # Vertical lines
    overlaid_frame[top_left_y:bottom_right_y, top_left_x:top_left_x+border_width] = np.max(frame)
    overlaid_frame[top_left_y:bottom_right_y, bottom_right_x-border_width:bottom_right_x] = np.max(frame)

    return overlaid_frame
