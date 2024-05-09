import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors
import matplotlib.cm as cm


def plot_track_coordinates(filename, merged_spots_df, Results_Folder, display_plots=True):
    if filename:
        # Filter the DataFrame based on the selected filename
        filtered_df = merged_spots_df[merged_spots_df['File_name'] == filename]

        plt.figure(figsize=(10, 8))
        for unique_id in filtered_df['Unique_ID'].unique():
            unique_df = filtered_df[filtered_df['Unique_ID'] == unique_id].sort_values(by='POSITION_T')
            plt.plot(unique_df['POSITION_X'], unique_df['POSITION_Y'], marker='o', linestyle='-', markersize=2)

        plt.xlabel('POSITION_X')
        plt.ylabel('POSITION_Y')
        plt.title(f'Coordinates for {filename}')
        plt.savefig(f"{Results_Folder}/Tracks/Tracks_{filename}.pdf")
        plt.gca().invert_yaxis()
        if display_plots:  # Only display plots if explicitly requested
            plt.show()
        else:
            plt.close()  # Close the plot to prevent it from displaying
    else:
        print("No valid filename selected")


def plot_origin_normalized_coordinates_FOV(filename, merged_spots_df, Results_Folder, display_plots=True):
    if filename:
        # Filter the DataFrame based on the selected filename
        filtered_df = merged_spots_df[merged_spots_df['File_name'] == filename]

        plt.figure(figsize=(10, 8))

        # Group by Unique_ID to work with each track individually
        for unique_id in filtered_df['Unique_ID'].unique():
            unique_df = filtered_df[filtered_df['Unique_ID'] == unique_id].sort_values(by='POSITION_T')

            # Normalize starting point to (0, 0)
            start_x = unique_df.iloc[0]['POSITION_X']
            start_y = unique_df.iloc[0]['POSITION_Y']
            normalized_x = unique_df['POSITION_X'] - start_x
            normalized_y = unique_df['POSITION_Y'] - start_y

            # Plot the normalized track without adding to the legend
            plt.plot(normalized_x, normalized_y, marker='o', linestyle='-', markersize=2)

        plt.xlabel('Normalized POSITION_X')
        plt.ylabel('Normalized POSITION_Y')
        plt.title(f'Origin-Normalized Tracks for {filename}')
        plt.savefig(f"{Results_Folder}/Tracks/Origin_Normalized_Tracks_{filename}.pdf")
        if display_plots:  # Only display plots if explicitly requested
            plt.show()
        else:
            plt.close()  # Close the plot to prevent it from displaying
    else:
        print("No valid filename selected")


def plot_origin_normalized_coordinates_condition_repeat(condition, repeat, merged_spots_df, Results_Folder,
                                                        display_plots=True):
    # Filter the DataFrame based on the selected condition and repeat
    filtered_df = merged_spots_df[(merged_spots_df['Condition'] == condition) &
                                  (merged_spots_df['Repeat'] == repeat)]

    if not filtered_df.empty:
        plt.figure(figsize=(10, 8))

        # Iterate over each unique track ID in the filtered DataFrame
        for unique_id in filtered_df['Unique_ID'].unique():
            track_df = filtered_df[filtered_df['Unique_ID'] == unique_id].sort_values(by='POSITION_T')

            # Normalize starting point to (0, 0)
            start_x = track_df.iloc[0]['POSITION_X']
            start_y = track_df.iloc[0]['POSITION_Y']
            normalized_x = track_df['POSITION_X'] - start_x
            normalized_y = track_df['POSITION_Y'] - start_y

            # Plot the normalized track
            plt.plot(normalized_x, normalized_y, marker='o', linestyle='-', markersize=2)

        plt.xlabel('Normalized POSITION_X')
        plt.ylabel('Normalized POSITION_Y')
        plt.title(f'Origin-Normalized Tracks for Condition: {condition}, Repeat: {repeat}')

        if display_plots:
            plt.show()
        else:
            plt.close()

        # Optionally save the plot
        plot_filename = f"Condition_{condition}_Repeat_{repeat}.pdf"
        plt.savefig(os.path.join(Results_Folder, "Tracks", plot_filename))
    else:
        print("No data available for the selected condition and repeat.")


def plot_origin_normalized_coordinates_condition(condition, merged_spots_df, Results_Folder, display_plots=True):
    # Filter the DataFrame based on the selected condition
    filtered_df = merged_spots_df[(merged_spots_df['Condition'] == condition)]

    if not filtered_df.empty:
        plt.figure(figsize=(10, 8))

        # Iterate over each unique track ID in the filtered DataFrame
        for unique_id in filtered_df['Unique_ID'].unique():
            track_df = filtered_df[filtered_df['Unique_ID'] == unique_id].sort_values(by='POSITION_T')

            # Normalize starting point to (0, 0)
            start_x = track_df.iloc[0]['POSITION_X']
            start_y = track_df.iloc[0]['POSITION_Y']
            normalized_x = track_df['POSITION_X'] - start_x
            normalized_y = track_df['POSITION_Y'] - start_y

            # Plot the normalized track
            plt.plot(normalized_x, normalized_y, marker='o', linestyle='-', markersize=2)

        plt.xlabel('Normalized POSITION_X')
        plt.ylabel('Normalized POSITION_Y')
        plt.title(f'Origin-Normalized Tracks for Condition: {condition}')

        if display_plots:
            plt.show()
        else:
            plt.close()

        # Optionally save the plot
        plot_filename = f"Condition_{condition}_All_Repeat.pdf"
        plt.savefig(os.path.join(Results_Folder, "Tracks", plot_filename))
        print(f"Plot saved as {plot_filename} in {Results_Folder}/Tracks/")
    else:
        print("No data available for the selected condition.")


def plot_migration_vectors(filename, merged_spots_df, Results_Folder, display_plots):
    # Filter data for the selected field of view
    fov_df = merged_spots_df[merged_spots_df['File_name'] == filename]

    # Set up the plot
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Initialize list to store vector magnitudes for color coding
    magnitudes = []
    coordinates = []

    # Collect data for all vectors
    for unique_id in fov_df['Unique_ID'].unique():
        track_df = fov_df[fov_df['Unique_ID'] == unique_id]
        track_df = track_df.sort_values(by='POSITION_T')

        start_x = track_df.iloc[0]['POSITION_X']
        start_y = track_df.iloc[0]['POSITION_Y']
        end_x = track_df.iloc[-1]['POSITION_X']
        end_y = track_df.iloc[-1]['POSITION_Y']

        vector_x = end_x - start_x
        vector_y = end_y - start_y
        magnitude = np.sqrt(vector_x ** 2 + vector_y ** 2)

        magnitudes.append(magnitude)
        coordinates.append((start_x, start_y, vector_x, vector_y))

    # Normalize magnitude for color mapping and determine arrow size scaling
    norm = mcolors.Normalize(vmin=min(magnitudes), vmax=max(magnitudes))
    cmap = cm.magma  # Choose a colormap
    scale_factor = np.mean(magnitudes) * 0.08  # Scale factor for arrow size

    # Plot each vector with color coding and scaled arrow size
    for (start_x, start_y, vector_x, vector_y), magnitude in zip(coordinates, magnitudes):
        color = cmap(norm(magnitude))
        ax.arrow(start_x, start_y, vector_x, vector_y,
                 head_width=scale_factor * magnitude / np.mean(magnitudes),
                 head_length=scale_factor * magnitude / np.mean(magnitudes) * 2,  # Twice the width for visibility
                 fc=color, ec=color)

    plt.title(f'Migration Vectors for {filename}')
    plt.xlabel('POSITION_X')
    plt.ylabel('POSITION_Y')
    plt.grid(True)
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.savefig(f"{Results_Folder}/Tracks/Vectors_Tracks_{filename}.pdf")
    if display_plots:  # Only display plots if explicitly requested
        plt.show()
    else:
        plt.close()  # Close the plot to prevent it from displaying
        
def plot_coordinates_side_by_side(filename, merged_spots_df, filtered_and_smoothed_df, Results_Folder):
    if filename:
        # Filter the DataFrames based on the selected filename
        raw_df = merged_spots_df[merged_spots_df['File_name'] == filename]
        processed_df = filtered_and_smoothed_df[filtered_and_smoothed_df['File_name'] == filename]

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Invert y-axis for both plots (optional, based on your preference)
        # axes[0].invert_yaxis()
        # axes[1].invert_yaxis()

        # Create a colormap to ensure consistent colors across tracks
        unique_ids = raw_df['Unique_ID'].unique()
        colormap = plt.get_cmap('tab20')

        # Calculate data ranges for both DataFrames
        raw_x_min, raw_x_max = raw_df['POSITION_X'].min(), raw_df['POSITION_X'].max()
        raw_y_min, raw_y_max = raw_df['POSITION_Y'].min(), raw_df['POSITION_Y'].max()

        processed_x_min, processed_x_max = processed_df['POSITION_X'].min(), processed_df['POSITION_X'].max()
        processed_y_min, processed_y_max = processed_df['POSITION_Y'].min(), processed_df['POSITION_Y'].max()

        # Combine data ranges for setting axis limits (consider margins if needed)
        x_min = min(raw_x_min, processed_x_min) - 0.1  # Add margin (adjust as needed)
        x_max = max(raw_x_max, processed_x_max) + 0.1  # Add margin (adjust as needed)
        y_min = min(raw_y_min, processed_y_min) - 0.1  # Add margin (adjust as needed)
        y_max = max(raw_y_max, processed_y_max) + 0.1  # Add margin (adjust as needed)

        # Plot Raw Data with shared limits
        for idx, unique_id in enumerate(unique_ids):
            unique_data = raw_df[raw_df['Unique_ID'] == unique_id].sort_values(by='POSITION_T')
            color_val = colormap(idx % 20 / 20)
            axes[0].plot(unique_data['POSITION_X'], unique_data['POSITION_Y'],
                         color=color_val, marker='o', linestyle='-', markersize=2)
        axes[0].set_title(f'Raw Coordinates for {filename}')
        axes[0].set_xlabel('POSITION_X')
        axes[0].set_ylabel('POSITION_Y')
        axes[0].set_xlim(x_min, x_max)
        axes[0].set_ylim(y_min, y_max)

        # Plot Filtered & Smoothed Data with shared limits
        for idx, unique_id in enumerate(unique_ids):
            unique_data = processed_df[processed_df['Unique_ID'] == unique_id].sort_values(by='POSITION_T')
            color_val = colormap(idx % 20 / 20)
            axes[1].plot(unique_data['POSITION_X'], unique_data['POSITION_Y'],
                         color=color_val, marker='o', linestyle='-', markersize=2)
        axes[1].set_title(f'Filtered & Smoothed Coordinates for {filename}')
        axes[1].set_xlabel('POSITION_X')
        axes[1].set_ylabel('POSITION_Y')
        axes[1].set_xlim(x_min, x_max)
        axes[1].set_ylim(y_min, y_max)

        plt.tight_layout()
        plt.savefig(f"{Results_Folder}/Tracks/Filtered_tracks_{filename}.pdf")
        plt.show()
    else:
        print("No valid filename selected")
