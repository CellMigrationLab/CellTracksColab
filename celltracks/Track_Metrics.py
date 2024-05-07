import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from tqdm.notebook import tqdm
import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np

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
    time_diffs[time_diffs == 0] = 1e-10  # Avoid division by zero
    speeds = deltas / time_diffs

    return pd.Series({
        'Track Duration': duration,
        'Mean Speed': speeds.mean(),
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

def filter_on_button_click(button):
    global filtered_and_smoothed_df
    metric_filters = {
        'Track Duration': duration_slider.value,
        'Mean Speed': mean_speed_slider.value,
        'Max Speed': max_speed_slider.value,
        'Min Speed': min_speed_slider.value,
        'Total Distance Traveled': total_distance_slider.value,
    }
    with output:
        clear_output(wait=True)
        filtered_and_smoothed_df, metrics_summary_df = optimized_filter_and_smooth_tracks(
            merged_spots_df,
            metric_filters,
            smoothing_neighbors=smoothing_slider.value,
            global_metrics_df=global_metrics_df  
        )
        # Save parameters
        params_file_path = os.path.join(Results_Folder, "filter_smoothing_parameters.csv")
        save_filter_smoothing_params(
            params_file_path,
            smoothing_slider.value,
            duration_slider.value,
            mean_speed_slider.value,
            max_speed_slider.value,
            min_speed_slider.value,
            total_distance_slider.value
        )
        print("Filtering and Smoothing Done")

def calculate_directionality(group):

    group = group.sort_values('POSITION_T')
    start_point = group.iloc[0][['POSITION_X', 'POSITION_Y', 'POSITION_Z']].to_numpy()
    end_point = group.iloc[-1][['POSITION_X', 'POSITION_Y', 'POSITION_Z']].to_numpy()

    euclidean_distance = np.linalg.norm(end_point - start_point)

    deltas = np.linalg.norm(np.diff(group[['POSITION_X', 'POSITION_Y', 'POSITION_Z']].values, axis=0), axis=1)
    total_path_length = deltas.sum()

    D = euclidean_distance / total_path_length if total_path_length != 0 else 0

    return pd.Series({'Directionality': D})


