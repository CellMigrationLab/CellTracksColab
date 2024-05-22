
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
    
    n_plots = len(variables_to_plot)
    method = stat_method_selector.value

    if n_plots == 0:
        print("No variables selected for plotting")
        return

    # Get selected conditions
    selected_conditions = condition_selector.value
    n_selected_conditions = len(selected_conditions)

    if n_selected_conditions == 0:
        print("No conditions selected for plotting, therefore all available conditions are selected by default")        
        selected_conditions = df[Conditions].unique().tolist()
        
    n_selected_conditions = len(selected_conditions)
    
    selected_cluster = cluster_dropdown.value
    print(f"Plotting in progress for Cluster {selected_cluster}...")
    filtered_df = df[(df[Conditions].isin(selected_conditions)) & (df[Cluster] == selected_cluster)].copy()

    # Create empty data structures for statistics
    effect_size_matrices = {}
    p_value_matrices = {}
    bonferroni_matrices = {}

    unique_conditions = filtered_df[Conditions].unique().tolist()
    num_comparisons = len(unique_conditions) * (len(unique_conditions) - 1) // 2
    n_iterations = 10000

    for var in variables_to_plot:
        pdf_pages = PdfPages(f"{Results_Folder}/pdf/Cluster_{selected_cluster}_{var}_Boxplots_and_Statistics.pdf")
        effect_size_matrices[var] = pd.DataFrame(0, index=unique_conditions, columns=unique_conditions)
        p_value_matrices[var] = pd.DataFrame(1, index=unique_conditions, columns=unique_conditions)
        bonferroni_matrices[var] = pd.DataFrame(1, index=unique_conditions, columns=unique_conditions)

        for cond1, cond2 in itertools.combinations(unique_conditions, 2):
            group1 = filtered_df[filtered_df[Conditions] == cond1][var]
            group2 = filtered_df[filtered_df[Conditions] == cond2][var]

            effect_size = abs(cohen_d(group1, group2))

            if method == 't-test':
                p_value = perform_t_test(filtered_df, cond1, cond2, var)
            if method == 'randomization test':                
                p_value = perform_randomization_test_parallel(filtered_df, cond1, cond2, var, n_iterations=n_iterations)


            # Set and mirror effect sizes and p-values
            effect_size_matrices[var].loc[cond1, cond2] = effect_size_matrices[var].loc[cond2, cond1] = effect_size
            p_value_matrices[var].loc[cond1, cond2] = p_value_matrices[var].loc[cond2, cond1] = p_value
            bonferroni_corrected_p_value = min(p_value * num_comparisons, 1.0)
            bonferroni_matrices[var].loc[cond1, cond2] = bonferroni_matrices[var].loc[cond2, cond1] = bonferroni_corrected_p_value

        # Save to CSV
        combined_df = pd.concat([
            effect_size_matrices[var].rename(columns=lambda x: f"{x} (Effect Size)"),
            p_value_matrices[var].rename(columns=lambda x: f"{x} ({method} P-Value)"),
            bonferroni_matrices[var].rename(columns=lambda x: f"{x} ({method} Bonferroni-corrected P-Value)")
        ], axis=1)

        combined_df.to_csv(f"{Results_Folder}/csv/Cluster_{selected_cluster}_{var}_statistics_combined.csv")

    # Create a new figure
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, height_ratios=[1.5, 1])
        ax_box = fig.add_subplot(gs[0, :])
    # Extract the data for this variable
        data_for_var = df[[Conditions, var, 'Repeat', 'File_name' ]]
    # Save the data_for_var to a CSV for replotting
        data_for_var.to_csv(f"{Results_Folder}/csv/Cluster_{selected_cluster}_{var}_boxplot_data.csv", index=False)

    # Calculate the Interquartile Range (IQR) using the 25th and 75th percentiles
        Q1 = df[var].quantile(0.2)
        Q3 = df[var].quantile(0.8)
        IQR = Q3 - Q1

    # Define bounds for the outliers
        multiplier = 10
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

    # Plotting
        sns.boxplot(x=Conditions, y=var, data=filtered_df, ax=ax_box, color='lightgray')  # Boxplot
        sns.stripplot(x=Conditions, y=var, data=filtered_df, ax=ax_box, hue='Repeat', dodge=True, jitter=True, alpha=0.2)  # Individual data points
        ax_box.set_ylim([max(min(filtered_df[var]), lower_bound), min(max(filtered_df[var]), upper_bound)])
        ax_box.set_title(f"{var} for Cluster {selected_cluster}")
        ax_box.set_xlabel('Condition')
        ax_box.set_ylabel(var)
        tick_labels = ax_box.get_xticklabels()
        tick_locations = ax_box.get_xticks()
        ax_box.xaxis.set_major_locator(FixedLocator(tick_locations))
        ax_box.set_xticklabels(tick_labels, rotation=90)
        ax_box.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Repeat')

    # Statistical Analyses and Heatmaps

      # Effect Size heatmap
        ax_d = fig.add_subplot(gs[1, 0])
        sns.heatmap(effect_size_matrices[var].fillna(0), annot=True, cmap="viridis", cbar=True, square=True, ax=ax_d, vmax=1)
        ax_d.set_title(f"Effect Size (Cohen's d)")

      # p-value heatmap using the new function
        ax_p = fig.add_subplot(gs[1, 1])
        plot_heatmap(ax_p, p_value_matrices[var], f"{method} p-value")

      # Bonferroni corrected p-value heatmap using the new function
        ax_bonf = fig.add_subplot(gs[1, 2])
        plot_heatmap(ax_bonf, bonferroni_matrices[var], "Bonferroni-corrected p-value")

        plt.tight_layout()
        pdf_pages.savefig(fig)  
        pdf_pages.close()
        plt.show()

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


