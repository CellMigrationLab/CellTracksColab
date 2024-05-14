
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
    
def compute_distances_using_distance_transform(df, image_dir, ROI_name):
    """
    Compute distances to the nearest labeled pixel for each spot using the distance transform method.
    Automatically detects if the file is a single image or a video sequence and checks if the frame
    number corresponds to the actual number of frames in the video.

    Parameters:
    df (DataFrame): The dataframe containing the spots' data.
    image_dir (str): The directory where the ROI images or videos are stored.
    """
    for file_name in tqdm(df['File_name'].unique(), desc="Processing files"):
        # Paths to the label images or video
        ROI_img_path = f"{image_dir}/{file_name}_{ROI_name}.tif"

        try:
            ROI_img = io.imread(ROI_img_path)
            # Ensure the image dimensions are 3 or below
            if ROI_img.ndim > 3:
                raise ValueError(f"Image file {file_name} has more than 3 dimensions, which is not supported.")

            # Determine if the file is a video by checking if it has more than two dimensions
            is_video = ROI_img.ndim == 3

            # Verify that the 'FRAME' number in the dataframe does not exceed the number of frames in the video
            if is_video and 'FRAME' in df.columns:
                file_df = df[df['File_name'] == file_name]
            # Compute max_frame_num for the current file_name
                max_frame_num = file_df['FRAME'].max()
                num_frames = ROI_img.shape[0]
                if max_frame_num > num_frames:
                    print(f"Error: max_frame_num ({max_frame_num}) exceeds num_frames ({num_frames}) in file {file_name}.")
                    raise ValueError(f"DataFrame contains 'FRAME' numbers that exceed the number of frames in the video for file {file_name}.")
                for frame_idx in range(num_frames):
                    # Process each frame with matching spots
                    process_frame(ROI_img[frame_idx], df, file_name, frame_idx)
            else:
                # Process a single image
                process_frame(ROI_img, df, file_name)

        except FileNotFoundError:
            print(f"Error: Image for {file_name} not found. Skipping...")
            continue
        except ValueError as e:
            print(e)
            break

    return df

def process_frame(ROI_img, df, file_name, ROI_name, frame_idx=None):
    """
    Process a single frame or image and update the dataframe with distance values.

    Parameters:
    ROI_img (ndarray): The ROI image or a single frame from a video.
    df (DataFrame): The dataframe to update.
    file_name (str): The name of the file being processed.
    frame_idx (int, optional): The index of the frame in the video.
    """
    # Compute distance transform
    distance_transform_ROI = distance_transform_edt(ROI_img == 0) * Pixel_calibration

    # Filter dataframe for the current file and frame
    file_df = df[df['File_name'] == file_name]
    if frame_idx is not None:
        file_df = file_df[file_df['FRAME'] == frame_idx]

    for idx, row in tqdm(file_df.iterrows(), total=file_df.shape[0], desc=f"Processing coordinates for {file_name}", leave=False):
        y, x = int(row['POSITION_Y'] / Pixel_calibration), int(row['POSITION_X'] / Pixel_calibration)
                # Check if x and y are within the bounds of the image

        if 0 <= x < distance_transform_ROI.shape[1] and 0 <= y < distance_transform_ROI.shape[0]:
            df.loc[df.index[idx], f'DistanceTo{ROI_name}'] = distance_transform_ROI[y, x]
        else:
            print(f"Warning: Coordinates (x={x}, y={y}) out of bounds for {file_name}")    
    
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
                max_distance = min_distance = start_distance = end_distance = median_distance = average_rate_of_change = percentage_change = direction_of_movement = np.nan
            # Basic metrics

            else:
              max_distance = distances.max(skipna=True)
              min_distance = distances.min(skipna=True)
              start_distance = distances.iloc[0] if not distances.empty else np.nan
              end_distance = distances.iloc[-1] if not distances.empty else np.nan
              median_distance = distances.median(skipna=True)
              std_dev_distance = distances.std(skipna=True)

              # Advanced metrics
              direction_of_movement = end_distance - start_distance
              average_rate_of_change = direction_of_movement / len(distances) if len(distances) > 0 else np.nan
              percentage_change = (direction_of_movement / start_distance * 100) if start_distance != 0 else np.nan

              # Linear regression to determine trend
              slope, _, _, _, _ = linregress(range(len(distances)), distances) if not distances.empty else (np.nan,)*5

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
