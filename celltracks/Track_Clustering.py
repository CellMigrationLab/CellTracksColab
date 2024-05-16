
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
from skimage import io
from tifffile import imread
from skimage.measure import label, regionprops, find_contours
from scipy.ndimage import distance_transform_edt
from tqdm.notebook import tqdm
from scipy.ndimage import distance_transform_edt
from scipy.stats import linregress
from celltracks.BoxPlots_Statistics import *

def plot_clustering_coordinates(filename, merged_spots_df, analysis_option, Results_Folder, r_values_end, max_x, max_y, show_plots=True):
    """
    Plot the coordinates used for the spatial analysis for a given file.

    Parameters:
    - filename: str, the name of the file to be plotted.
    - merged_spots_df: pd.DataFrame, the dataframe containing the spot data.
    - analysis_option: str, the analysis option to select the point ('beginning', 'end', 'middle', 'average', 'median').
    - Results_Folder: str, the folder where results are saved.
    - r_values_end: float, the end value for r in Ripley's L function.
    - max_x: float, the maximum x dimension of the dataset.
    - max_y: float, the maximum y dimension of the dataset.
    - show_plots: bool, whether to display the plots or not.
    """
    if filename:
        # Filter the DataFrame based on the filename
        filtered_df = merged_spots_df[merged_spots_df['File_name'] == filename]

        plt.figure(figsize=(10, 8))
        for unique_id in filtered_df['Unique_ID'].unique():
            unique_df = filtered_df[filtered_df['Unique_ID'] == unique_id].sort_values(by='POSITION_T')

            # Find and mark the selected analysis point
            analysis_point = select_analysis_point(unique_df, analysis_option)

            # Check if the analysis point is valid and mark accordingly
            if not analysis_point.isna().any():
                if (analysis_point['POSITION_X'] <= r_values_end or
                    analysis_point['POSITION_X'] >= (max_x - r_values_end) or
                    analysis_point['POSITION_Y'] <= r_values_end or
                    analysis_point['POSITION_Y'] >= (max_y - r_values_end)):
                    plt.scatter(analysis_point['POSITION_X'], analysis_point['POSITION_Y'], color='blue', s=50, label=f'Edge Track {unique_id}')
                else:
                    plt.scatter(analysis_point['POSITION_X'], analysis_point['POSITION_Y'], color='red', s=50, label=f'Track {unique_id}')

        plt.xlabel('POSITION_X')
        plt.ylabel('POSITION_Y')
        plt.title(f'Analysis points for {filename} ({analysis_option} point)')
        plt.savefig(f"{Results_Folder}/Track_Clustering/Coordinates/Analysis_points_{filename}_{analysis_option}.pdf")

        if show_plots:
            plt.show()
        else:
            plt.close()
    else:
        print("No valid filename selected")
        
 def simulate_random_points(num_points, x_range, y_range):
    x_coords = np.random.uniform(x_range[0], x_range[1], num_points)
    y_coords = np.random.uniform(y_range[0], y_range[1], num_points)
    return np.column_stack((x_coords, y_coords))
    
    
    
# Define Ripley's K function
def ripley_k(points, r, area):
    n = len(points)
    d_matrix = distance_matrix(points, points)
    sum_indicator = np.sum(d_matrix < r) - n  # Subtract n to exclude self-pairs

    K_r = (area / (n ** 2)) * sum_indicator

    # Check if K_r is negative and print relevant information
    if K_r < 0:
        print("Negative K_r encountered!")
        print("Distance matrix:", d_matrix)
        print("Sum indicator:", sum_indicator)
        print("Area:", area, "Number of points:", n, "Distance threshold r:", r)

    return K_r


# Define Ripley's L function

def ripley_l(points, r, area):
    K_r = ripley_k(points, r, area)
    # Check if K_r has negative values
    if np.any(K_r < 0):
        print("Warning: Negative value encountered in K_r")

    L_r = np.sqrt(K_r / np.pi) - r
    return L_r

import pandas as pd

def get_dimensions(spot_df):
    """
    Get the maximal x, y, z dimensions from the spot dataframe, assuming minimum values are 0.

    Parameters:
    spot_df (pd.DataFrame): DataFrame containing the spot data with columns 'x', 'y', and 'z'.

    Returns:
    dict: A dictionary with the maximal dimensions for x, y, and z, with min values assumed to be 0.
    """
    max_x = spot_df['POSITION_X'].max()
    max_y = spot_df['POSITION_Y'].max()
    max_z = spot_df['POSITION_Z'].max()

    return {'min_x': 0, 'max_x': max_x, 'min_y': 0, 'max_y': max_y, 'min_z': 0, 'max_z': max_z}      
    
def select_analysis_point(track, analysis_option):
    if analysis_option == "beginning":
        point = track.iloc[0][['POSITION_X', 'POSITION_Y']]
    elif analysis_option == "end":
        point = track.iloc[-1][['POSITION_X', 'POSITION_Y']]
    elif analysis_option == "middle":
        middle_index = len(track) // 2
        point = track.iloc[middle_index][['POSITION_X', 'POSITION_Y']]
    elif analysis_option == "average":
        point = track[['POSITION_X', 'POSITION_Y']].mean()
    elif analysis_option == "median":
        point = track[['POSITION_X', 'POSITION_Y']].median()
    else:
        point = pd.Series([np.nan, np.nan], index=['POSITION_X', 'POSITION_Y'])

    return point

def plot_coordinates_Clustering(filename, analysis_option, base_folder):
    if filename:
        # Filter the DataFrame based on the selected filename
        filtered_df = merged_spots_df[merged_spots_df['File_name'] == filename]

        plt.figure(figsize=(10, 8))
        for unique_id in filtered_df['Unique_ID'].unique():
            unique_df = filtered_df[filtered_df['Unique_ID'] == unique_id].sort_values(by='POSITION_T')
            plt.plot(unique_df['POSITION_X'], unique_df['POSITION_Y'], marker='o', linestyle='-', markersize=2)

            # Find and mark the selected analysis point
            analysis_point = select_analysis_point(unique_df, analysis_option)
            if not analysis_point.isna().any():
                plt.scatter(analysis_point['POSITION_X'], analysis_point['POSITION_Y'], color='red', s=50)

        plt.xlabel('POSITION_X')
        plt.ylabel('POSITION_Y')
        plt.title(f'Coordinates for {filename} ({analysis_option} point)')
        plt.savefig(f"{base_folder}/Tracks_{filename}_{analysis_option}.pdf")
        plt.show()
    else:
        print("No valid filename selected")    
    
    
    
    
    
    
    
    
    
    
           
