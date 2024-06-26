
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

def check_zero_pixel_percentage(ROI_img):
    zero_pixel_count = np.sum(ROI_img == 0)
    total_pixel_count = ROI_img.size
    zero_pixel_percentage = (zero_pixel_count / total_pixel_count) * 100
    return zero_pixel_percentage


def check_and_correct_coordinates(df, file_name, image_dir, ROI_name, Pixel_calibration):
    """
    Checks and corrects the coordinates in the DataFrame for a given file to ensure they are within the bounds
    of the associated image or video.

    Parameters:
    df (DataFrame): DataFrame containing the spots' data.
    file_name (str): The name of the file to check and correct.
    image_dir (str): Directory where the images or videos are stored.
    ROI_name (str): Suffix or identifier for the image or video file name.

    Returns:
    DataFrame: Updated DataFrame with corrected coordinates.
    """
    # Load the image or video
    ROI_img_path = f"{image_dir}/{file_name}_{ROI_name}.tif"
    try:
        ROI_img = io.imread(ROI_img_path)
    except FileNotFoundError:
        print(f"Image or video for {file_name} not found.")
        return df
    # Check the percentage of zero pixels
    zero_percentage = check_zero_pixel_percentage(ROI_img)
    print(f"File {file_name}: Percentage of background pixels is {zero_percentage:.2f}%.")

    # Determine if it's an image or a video
    if ROI_img.ndim == 3:  # Video
        max_x, max_y = ROI_img.shape[2] - 1, ROI_img.shape[1] - 1
    elif ROI_img.ndim == 2:  # Image
        max_x, max_y = ROI_img.shape[1] - 1, ROI_img.shape[0] - 1
    else:
        print(f"Unsupported number of dimensions ({ROI_img.ndim}) in the file {file_name}.")
        return df

    # Apply pixel calibration
    max_x, max_y = max_x * Pixel_calibration, max_y * Pixel_calibration

    # Filter dataframe for the current file
    file_df = df[df['File_name'] == file_name]

    # Correct each coordinate with tqdm for progress
    for idx in tqdm(file_df.index, desc=f"Processing {file_name}"):
        x, y = int(df.at[idx, 'POSITION_X']), int(df.at[idx, 'POSITION_Y'])
        corrected_x = max(0, min(x, max_x))
        corrected_y = max(0, min(y, max_y))
        if corrected_x != x or corrected_y != y:
            print(f"Corrected coordinates for index {idx} from (x={x}, y={y}) to (x={corrected_x}, y={corrected_y})")
        df.at[idx, 'POSITION_X'] = corrected_x
        df.at[idx, 'POSITION_Y'] = corrected_y

    return df
    
   
def print_filenames_with_nan_distances(spots_df, ROI_name):
    nan_filenames = []

    grouped_spots = spots_df.groupby('Unique_ID')

    for unique_id, group in grouped_spots:
        if group[f'DistanceTo{ROI_name}'].isna().any():
            # Store filenames associated with NaN distances
            nan_filenames.extend(group['File_name'].unique())

    # Print unique filenames with NaN distances
    unique_nan_filenames = set(nan_filenames)
    print(f"Filenames with NaN distances: {unique_nan_filenames}")

def get_distances_and_metrics(track_df, spots_df, ROI_name):
    results = []

    grouped_spots = spots_df.groupby('Unique_ID')

    for _, track in tqdm(track_df.iterrows(), total=track_df.shape[0], desc="Processing Tracks"):
        unique_id = track['Unique_ID']

        if unique_id in grouped_spots.groups:
            track_spots = grouped_spots.get_group(unique_id)
            distances = track_spots[f'DistanceTo{ROI_name}']

            if distances.empty or distances.isna().all():
                max_distance = min_distance = start_distance = end_distance = median_distance = std_dev_distance = average_rate_of_change = percentage_change = direction_of_movement = slope = np.nan
            else:
                # Basic metrics
                max_distance = distances.max(skipna=True)
                min_distance = distances.min(skipna=True)
                start_distance = distances.iloc[0] if not distances.empty else np.nan
                end_distance = distances.iloc[-1] if not distances.empty else np.nan
                median_distance = distances.median(skipna=True)
                std_dev_distance = distances.std(skipna=True)

                # Advanced metrics
                direction_of_movement = np.nan if pd.isna(start_distance) or pd.isna(end_distance) else end_distance - start_distance
                average_rate_of_change = np.nan if len(distances) == 0 else direction_of_movement / len(distances)
                percentage_change = np.nan if start_distance == 0 else (direction_of_movement / start_distance * 100)

                # Linear regression to determine trend
                slope, _, _, _, _ = linregress(range(len(distances)), distances) if len(distances) > 1 else (np.nan,)*5

            results.append({
                'Unique_ID': unique_id,
                f'MaxDistance_{ROI_name}': max_distance,
                f'MinDistance_{ROI_name}': min_distance,
                f'StartDistance_{ROI_name}': start_distance,
                f'EndDistance_{ROI_name}': end_distance,
                f'MedianDistance_{ROI_name}': median_distance,
                f'StdDevDistance_{ROI_name}': std_dev_distance,
                f'DirectionMovement_{ROI_name}': direction_of_movement,
                f'AvgRateChange_{ROI_name}': average_rate_of_change,
                f'PercentageChange_{ROI_name}': percentage_change,
                f'TrendSlope_{ROI_name}': slope
            })

    return pd.DataFrame(results)
    
def plot_tracks_vs_distance(dataframe, ROI_name):
    sns.set(style="whitegrid")
    plt.figure(figsize=(20, 6))

    # Creating a histogram with a bin size of 10
    ax = sns.histplot(data=dataframe, x=f'MaxDistance_{ROI_name}', bins=range(0, int(dataframe[f'MaxDistance_{ROI_name}'].max()) + 10, 10), kde=False)

    plt.title(f'Number of Tracks vs Max Distance to {ROI_name}')
    plt.xlabel(f'Distance to {ROI_name}')
    plt.ylabel('Number of Tracks')

    # Set x-ticks and rotate the labels for better readability
    plt.xticks(range(0, int(dataframe[f'MaxDistance_{ROI_name}'].max()) + 10, 10), rotation=90, ha='right')

    plt.tight_layout()  # Adjust the layout to accommodate label sizes
    plt.show()    


def classify_tracks_by_distance(dataframe, distance_threshold, ROI_name):
    classification_column = f'Track_Classification_{ROI_name}'
    dataframe[classification_column] = dataframe.apply(
        lambda row: f'Close_{ROI_name}' if row[f'MaxDistance_{ROI_name}'] <= distance_threshold else f'Far_{ROI_name}', axis=1)
    dataframe[f'Track_Classification_Condition_{ROI_name}'] = dataframe['Condition']+'_'+dataframe[classification_column]
    return dataframe        
