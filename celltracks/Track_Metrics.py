import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from tqdm.notebook import tqdm
import ipywidgets as widgets
from IPython.display import display, clear_output
from scipy.spatial import ConvexHull

# Functions related to the track filtering section of the notebook

def save_filter_smoothing_params(file_path, smoothing_neighbors,
                                 duration_range, mean_speed_range, max_speed_range,
                                 min_speed_range, total_distance_range):
    params = {
        'Smoothing Neighbors': smoothing_neighbors,
        'Duration Range': str(duration_range),
        'Mean Speed Range': str(mean_speed_range),
        'Max Speed Range': str(max_speed_range),
        'Min Speed Range': str(min_speed_range),
        'Total Distance Range': str(total_distance_range)
    }
    params_df = pd.DataFrame([params])
    params_df.to_csv(file_path, index=False)


def calculate_track_metrics(group):
    group = group.sort_values('POSITION_T')
    positions = group[['POSITION_X', 'POSITION_Y', 'POSITION_Z']].values
    times = group['POSITION_T'].values

    # Track Duration
    duration = times[-1] - times[0]

    # Speeds calculation
    deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    time_diffs = np.diff(times)
    time_diffs[time_diffs == 0] = 1e-10  # To avoid division by zero
    speeds = deltas / time_diffs

    return pd.Series({
        'Track Duration': duration,
        'Mean Speed': speeds.mean(),
        'Median Speed': np.median(speeds),  # Calculate median speed
        'Max Speed': speeds.max(),
        'Min Speed': speeds.min(),
        'Speed Standard Deviation': speeds.std(),
        'Total Distance Traveled': deltas.sum()
    })


def smooth_track(track_data, smoothing_neighbors):
    # Skip smoothing if smoothing_neighbors is 1
    if smoothing_neighbors == 1:
        return track_data

    track_data = track_data.sort_values(by='POSITION_T')
    smoothed_track = track_data.copy()

    if len(track_data) >= smoothing_neighbors:
        smoothed_X = track_data['POSITION_X'].rolling(window=smoothing_neighbors, center=True).mean()
        smoothed_Y = track_data['POSITION_Y'].rolling(window=smoothing_neighbors, center=True).mean()
        smoothed_Z = track_data['POSITION_Z'].rolling(window=smoothing_neighbors, center=True).mean()

        smoothed_track['POSITION_X'] = smoothed_X
        smoothed_track['POSITION_Y'] = smoothed_Y
        smoothed_track['POSITION_Z'] = smoothed_Z

    return smoothed_track


def apply_metrics_filters(df, filters):
    # Filters should be a dict with metric name as key and (min, max) tuple as value
    filtered_df = df.copy()
    for metric, (min_val, max_val) in filters.items():
        filtered_df = filtered_df[(filtered_df[metric] >= min_val) & (filtered_df[metric] <= max_val)]
    return filtered_df

def optimized_filter_and_smooth_tracks(merged_spots_df, metric_filters, smoothing_neighbors, global_metrics_df):
    # Apply filters based on precomputed metrics in global_metrics_df
    filtered_metrics_df = apply_metrics_filters(global_metrics_df, metric_filters)
    valid_tracks = filtered_metrics_df.index

    # Filter the original dataframe to keep only the valid tracks
    filtered_df = merged_spots_df[merged_spots_df['Unique_ID'].isin(valid_tracks)]

    # Implement smoothing
    tqdm.pandas(desc="Processing Tracks")
    smoothed_df = filtered_df.groupby('Unique_ID', group_keys=False).progress_apply(lambda x: smooth_track(x, smoothing_neighbors)).reset_index(drop=True)

    # Print number of tracks kept
    num_kept_tracks = filtered_df['Unique_ID'].nunique()  
    total_tracks = global_metrics_df.shape[0]  # Assuming each unique track has an entry in global_metrics_df
    print(f"Number of Tracks Kept: {num_kept_tracks} (out of {total_tracks})")
    
    return smoothed_df, filtered_metrics_df


def create_metric_slider(description, metric_key, global_metrics_df, width=None, align_text='left'):

    slider_min = global_metrics_df[metric_key].min()
    slider_max = global_metrics_df[metric_key].max()
    readout_format = '.2f'  # Assuming metric values require 2 decimal places

    # Consider adding a check for valid metric_key existence in global_metrics_df
    # if desired (e.g., if metric_key might be user-defined)

    style = {'description_width': 'initial'}  # Set text alignment

    return widgets.FloatRangeSlider(
        value=[slider_min, slider_max],
        min=slider_min,
        max=slider_max,
        step=0.01,
        description=description,
        style=style,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format=readout_format,
        layout=widgets.Layout(width=width),  # Apply optional width
    )


def calculate_rolling_distances(deltas, window_size=5):
    if len(deltas) < window_size:
        # Not enough data to compute rolling statistics
        return np.nan

    # Calculate rolling sums of the distances traveled
    rolling_sums = np.convolve(deltas, np.ones(window_size), mode='valid')
    return rolling_sums.mean()  # Return the average of these rolling sums

def calculate_rolling_metrics(speeds, window_size=5):
    if len(speeds) < window_size:
        # Not enough data to compute rolling statistics
        return {
            'rolling_mean': np.nan,
            'rolling_median': np.nan,
            'rolling_max': np.nan,
            'rolling_min': np.nan,
            'rolling_std': np.nan
        }

    rolling_mean = np.convolve(speeds, np.ones(window_size)/window_size, mode='valid').mean()
    rolling_median = np.median(np.convolve(speeds, np.ones(window_size)/window_size, mode='valid'))
    rolling_max = np.max(np.convolve(speeds, np.ones(window_size)/window_size, mode='valid'))
    rolling_min = np.min(np.convolve(speeds, np.ones(window_size)/window_size, mode='valid'))
    rolling_std = np.std(np.convolve(speeds, np.ones(window_size)/window_size, mode='valid'))

    return {
        'rolling_mean': rolling_mean,
        'rolling_median': rolling_median,
        'rolling_max': rolling_max,
        'rolling_min': rolling_min,
        'rolling_std': rolling_std
    }


def calculate_track_metrics_rolling(group, window_size=5):
    group = group.sort_values('POSITION_T')
    positions = group[['POSITION_X', 'POSITION_Y', 'POSITION_Z']].values
    times = group['POSITION_T'].values

    # Track Duration
    duration = times[-1] - times[0]

    # Speeds and distances calculation
    deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    time_diffs = np.diff(times)
    time_diffs[time_diffs == 0] = 1e-10  # To avoid division by zero
    speeds = deltas / time_diffs

    # Calculate rolling metrics for speeds
    rolling_metrics = calculate_rolling_metrics(speeds, window_size)

    # Calculate average rolling total distance traveled
    average_rolling_distance = calculate_rolling_distances(deltas, window_size)

    return pd.Series({
        'Mean Speed Rolling': rolling_metrics['rolling_mean'],
        'Median Speed Rolling': rolling_metrics['rolling_median'],
        'Max Speed Rolling': rolling_metrics['rolling_max'],
        'Min Speed Rolling Rolling': rolling_metrics['rolling_min'],
        'Speed Standard Deviation Rolling': rolling_metrics['rolling_std'],
        'Total Distance Traveled Rolling': average_rolling_distance
    })


def calculate_directionality(group):

    group = group.sort_values('POSITION_T')
    start_point = group.iloc[0][['POSITION_X', 'POSITION_Y', 'POSITION_Z']].to_numpy()
    end_point = group.iloc[-1][['POSITION_X', 'POSITION_Y', 'POSITION_Z']].to_numpy()

    euclidean_distance = np.linalg.norm(end_point - start_point)

    deltas = np.linalg.norm(np.diff(group[['POSITION_X', 'POSITION_Y', 'POSITION_Z']].values, axis=0), axis=1)
    total_path_length = deltas.sum()

    D = euclidean_distance / total_path_length if total_path_length != 0 else 0

    return pd.Series({'Directionality': D})


def calculate_rolling_directionality(group, window_size=5):
    # Ensure the group is sorted by time
    group = group.sort_values('POSITION_T')

    # Initialize directionality list to store results for each window
    directionalities = []

    # Loop over the DataFrame using a rolling window
    for start in range(len(group) - window_size + 1):
        window = group.iloc[start:start + window_size]
        if len(window) < 2:
            directionalities.append(np.nan)  # Not enough points to define direction
            continue

        start_point = window.iloc[0][['POSITION_X', 'POSITION_Y', 'POSITION_Z']].to_numpy()
        end_point = window.iloc[-1][['POSITION_X', 'POSITION_Y', 'POSITION_Z']].to_numpy()
        euclidean_distance = np.linalg.norm(end_point - start_point)
        path_length = np.linalg.norm(np.diff(window[['POSITION_X', 'POSITION_Y', 'POSITION_Z']].values, axis=0), axis=1).sum()
        D = euclidean_distance / path_length if path_length != 0 else 0
        directionalities.append(D)

    # Return the average directionality for the track if there are valid directionalities calculated
    if directionalities:
        return pd.Series({'Directionality Rolling': np.nanmean(directionalities)})
    else:
        return pd.Series({'Directionality Rolling': np.nan})

def calculate_tortuosity(group):
    group = group.sort_values('POSITION_T')

    # Apply spatial calibration to the coordinates
    calibrated_coords = group[['POSITION_X', 'POSITION_Y', 'POSITION_Z']].values

    start_point = calibrated_coords[0]
    end_point = calibrated_coords[-1]

    # Calculating Euclidean distance in 3D between start and end points
    euclidean_distance = np.linalg.norm(end_point - start_point)

    # Calculating the total path length in 3D
    deltas = np.linalg.norm(np.diff(calibrated_coords, axis=0), axis=1)
    total_path_length = deltas.sum()

    # Calculating Tortuosity
    T = total_path_length / euclidean_distance if euclidean_distance != 0 else 0

    return pd.Series({'Tortuosity': T})

def calculate_rolling_tortuosity(group, window_size=5):
    # Ensure the group is sorted by time
    group = group.sort_values('POSITION_T')

    # Initialize tortuosity list to store results for each window
    tortuosities = []

    # Loop over the DataFrame using a rolling window
    for start in range(len(group) - window_size + 1):
        window = group.iloc[start:start + window_size]
        if len(window) < 2:
            tortuosities.append(np.nan)  # Not enough points to define a path
            continue

        # Apply spatial calibration to the coordinates
        calibrated_coords = window[['POSITION_X', 'POSITION_Y', 'POSITION_Z']].values

        start_point = calibrated_coords[0]
        end_point = calibrated_coords[-1]

        # Calculating Euclidean distance in 3D between start and end points
        euclidean_distance = np.linalg.norm(end_point - start_point)

        # Calculating the total path length in 3D
        deltas = np.linalg.norm(np.diff(calibrated_coords, axis=0), axis=1)
        total_path_length = deltas.sum()

        # Calculating Tortuosity
        T = total_path_length / euclidean_distance if euclidean_distance != 0 else 0
        tortuosities.append(T)

    # Return the average tortuosity for the track if there are valid tortuosities calculated
    if tortuosities:
        return pd.Series({'Tortuosity Rolling': np.nanmean(tortuosities)})
    else:
        return pd.Series({'Tortuosity Rolling': np.nan})

def calculate_total_turning_angle(group):
    group = group.sort_values('POSITION_T')
    directions = group[['POSITION_X', 'POSITION_Y', 'POSITION_Z']].diff().dropna()
    total_turning_angle = 0

    for i in range(1, len(directions)):
        dir1 = directions.iloc[i - 1]
        dir2 = directions.iloc[i]

        if np.linalg.norm(dir1) == 0 or np.linalg.norm(dir2) == 0:
            continue

        cos_angle = np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.degrees(np.arccos(cos_angle))
        total_turning_angle += angle

    return pd.Series({'Total Turning Angle': total_turning_angle})

def calculate_rolling_total_turning_angle(group, window_size=5):
    group = group.sort_values('POSITION_T')

    # Initialize the total turning angle list for each window
    rolling_turning_angles = []

    # Loop over the DataFrame using a rolling window
    for start in range(len(group) - window_size + 1):
        window = group.iloc[start:start + window_size]
        if len(window) < 2:
            rolling_turning_angles.append(np.nan)  # Not enough points to define a path
            continue

        directions = window[['POSITION_X', 'POSITION_Y', 'POSITION_Z']].diff().dropna()
        window_turning_angle = 0

        for i in range(1, len(directions)):
            dir1 = directions.iloc[i - 1]
            dir2 = directions.iloc[i]

            if np.linalg.norm(dir1) == 0 or np.linalg.norm(dir2) == 0:
                continue

            cos_angle = np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.degrees(np.arccos(cos_angle))
            window_turning_angle += angle

        rolling_turning_angles.append(window_turning_angle)

    # Calculate the average of the turning angles for all windows
    average_turning_angle = np.nanmean(rolling_turning_angles) if rolling_turning_angles else np.nan

    return pd.Series({'Total Turning Angle Rolling': average_turning_angle})
    

def calculate_spatial_coverage(group):
    group = group.sort_values('POSITION_T')

    # Apply spatial calibration to the coordinates
    calibrated_coords = group[['POSITION_X', 'POSITION_Y', 'POSITION_Z']].values  # Ensure .values is added here

    # Drop rows with NaN values in the coordinates
    calibrated_coords = calibrated_coords[~np.isnan(calibrated_coords).any(axis=1)]

    # Check the variance of Z coordinates
    z_variance = np.var(calibrated_coords[:, 2])

    if z_variance == 0:  # If variance of Z is 0, calculate 2D spatial coverage
        if len(calibrated_coords) < 3:  # Need at least 3 points for a 2D convex hull
            return pd.Series({'Spatial Coverage': 0})

        try:
            coords_2d = calibrated_coords[:, :2]  # Use only X and Y coordinates
            hull_2d = ConvexHull(coords_2d, qhull_options='QJ')  # 'QJ' joggles the input to avoid precision errors
            spatial_coverage = hull_2d.volume  # Area of the convex hull in 2D
        except Exception as e:
            print(f"Error calculating 2D spatial coverage: {e}")
            spatial_coverage = 0
    else:  # If variance of Z is not 0, calculate 3D spatial coverage
        if len(calibrated_coords) < 4:  # Need at least 4 points for a 3D convex hull
            return pd.Series({'Spatial Coverage': 0})

        try:
            hull = ConvexHull(calibrated_coords, qhull_options='QJ')  # 'QJ' joggles the input to avoid precision errors
            spatial_coverage = hull.volume  # Volume of the convex hull in 3D
        except Exception as e:
            print(f"Error calculating 3D spatial coverage: {e}")
            spatial_coverage = 0

    return pd.Series({'Spatial Coverage': spatial_coverage})
    
def calculate_rolling_spatial_coverage(group, window_size=5):
    group = group.sort_values('POSITION_T')
    spatial_coverages = []

    for start in range(len(group) - window_size + 1):
        window = group.iloc[start:start + window_size]
        calibrated_coords = window[['POSITION_X', 'POSITION_Y', 'POSITION_Z']].dropna().values

        if len(calibrated_coords) < 3:  # Not enough points to form a convex hull in 2D
            continue

        try:
            if np.var(calibrated_coords[:, 2]) == 0 and len(calibrated_coords) >= 3:  # Calculate 2D hull if Z variance is zero
                hull = ConvexHull(calibrated_coords[:, :2], qhull_options='QJ')
                spatial_coverages.append(hull.volume)  # Area in 2D
            elif len(calibrated_coords) >= 4:  # Calculate 3D hull if enough points and Z variance is non-zero
                hull = ConvexHull(calibrated_coords, qhull_options='QJ')
                spatial_coverages.append(hull.volume)  # Volume in 3D
        except Exception as e:
            print(f"Error in calculating hull: {e}")

    # Average spatial coverage over the track
    average_spatial_coverage = np.nanmean(spatial_coverages) if spatial_coverages else 0
    return pd.Series({'Spatial Coverage Rolling': average_spatial_coverage})

def compute_morphological_metrics(spots_df, metrics):
    # Compute mean, median, std, min, and max for each metric
    mean_df = spots_df.groupby('Unique_ID')[metrics].mean(numeric_only=True).add_prefix('MEAN_')
    median_df = spots_df.groupby('Unique_ID')[metrics].median(numeric_only=True).add_prefix('MEDIAN_')
    std_df = spots_df.groupby('Unique_ID')[metrics].std(numeric_only=True).add_prefix('STD_')
    min_df = spots_df.groupby('Unique_ID')[metrics].min(numeric_only=True).add_prefix('MIN_')
    max_df = spots_df.groupby('Unique_ID')[metrics].max(numeric_only=True).add_prefix('MAX_')

    # Concatenate the computed metrics into a single dataframe
    metrics_df = pd.concat([mean_df, median_df, std_df, min_df, max_df], axis=1)

    return metrics_df

def check_metrics_availability(spots_df, required_metrics):
    # Identify available metrics in the DataFrame
    available_metrics = [metric for metric in required_metrics if metric in spots_df.columns]
    return available_metrics
