import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def plot_track_coordinates(filename, merged_spots_df, Results_Folder, display_plots=True):
    if not filename:
        print("No valid filename selected")
        return

    tracks_dir = os.path.join(Results_Folder, "Tracks")
    os.makedirs(tracks_dir, exist_ok=True)

    filtered_df = merged_spots_df[merged_spots_df["File_name"] == filename]
    if filtered_df.empty:
        print(f"No data available for filename: {filename}")
        return

    plt.figure(figsize=(10, 8))
    for unique_id in filtered_df["Unique_ID"].unique():
        unique_df = filtered_df[filtered_df["Unique_ID"] == unique_id].sort_values(by="POSITION_T")
        if unique_df.empty:
            continue
        plt.plot(unique_df["POSITION_X"], unique_df["POSITION_Y"], marker="o", linestyle="-", markersize=2)

    plt.xlabel("POSITION_X")
    plt.ylabel("POSITION_Y")
    plt.title(f"Coordinates for {filename}")

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()

    plt.savefig(os.path.join(tracks_dir, f"Tracks_{filename}.pdf"))

    if display_plots:
        plt.show()
    else:
        plt.close()


def plot_origin_normalized_coordinates_FOV(
    filename, merged_spots_df, Results_Folder, x_scale=0, y_scale=0, display_plots=True
):
    if not filename:
        print("No valid filename selected")
        return

    tracks_dir = os.path.join(Results_Folder, "Tracks")
    os.makedirs(tracks_dir, exist_ok=True)

    filtered_df = merged_spots_df[merged_spots_df["File_name"] == filename]
    if filtered_df.empty:
        print(f"No data available for filename: {filename}")
        return

    plt.figure(figsize=(10, 8))

    for unique_id in filtered_df["Unique_ID"].unique():
        unique_df = filtered_df[filtered_df["Unique_ID"] == unique_id].sort_values(by="POSITION_T")
        if unique_df.empty:
            continue

        start_x = unique_df.iloc[0]["POSITION_X"]
        start_y = unique_df.iloc[0]["POSITION_Y"]
        normalized_x = unique_df["POSITION_X"] - start_x
        normalized_y = unique_df["POSITION_Y"] - start_y

        plt.plot(normalized_x, normalized_y, marker="o", linestyle="-", markersize=2)

    plt.xlabel("Normalized POSITION_X")
    plt.ylabel("Normalized POSITION_Y")
    plt.title(f"Origin-Normalized Tracks for {filename}")

    if x_scale != 0:
        plt.xlim(-x_scale, x_scale)
    if y_scale != 0:
        plt.ylim(-y_scale, y_scale)

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()

    plt.savefig(os.path.join(tracks_dir, f"Origin_Normalized_Tracks_{filename}.pdf"))

    if display_plots:
        plt.show()
    else:
        plt.close()


def plot_origin_normalized_coordinates_condition_repeat(
    condition, repeat, merged_spots_df, Results_Folder, x_scale=0, y_scale=0, display_plots=True
):
    tracks_dir = os.path.join(Results_Folder, "Tracks")
    os.makedirs(tracks_dir, exist_ok=True)

    filtered_df = merged_spots_df[
        (merged_spots_df["Condition"] == condition) & (merged_spots_df["Repeat"] == repeat)
    ]
    if filtered_df.empty:
        print("No data available for the selected condition and repeat.")
        return

    plt.figure(figsize=(10, 8))

    for unique_id in filtered_df["Unique_ID"].unique():
        track_df = filtered_df[filtered_df["Unique_ID"] == unique_id].sort_values(by="POSITION_T")
        if track_df.empty:
            continue

        start_x = track_df.iloc[0]["POSITION_X"]
        start_y = track_df.iloc[0]["POSITION_Y"]
        normalized_x = track_df["POSITION_X"] - start_x
        normalized_y = track_df["POSITION_Y"] - start_y

        plt.plot(normalized_x, normalized_y, marker="o", linestyle="-", markersize=2)

    plt.xlabel("Normalized POSITION_X")
    plt.ylabel("Normalized POSITION_Y")
    plt.title(f"Origin-Normalized Tracks for Condition: {condition}, Repeat: {repeat}")

    if x_scale != 0:
        plt.xlim(-x_scale, x_scale)
    if y_scale != 0:
        plt.ylim(-y_scale, y_scale)

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()

    plt.savefig(os.path.join(tracks_dir, f"Origin_Normalized_Tracks_{condition}_{repeat}.pdf"))

    if display_plots:
        plt.show()
    else:
        plt.close()


def plot_origin_normalized_coordinates_condition(
    condition, merged_spots_df, Results_Folder, x_scale=0, y_scale=0, display_plots=True
):
    tracks_dir = os.path.join(Results_Folder, "Tracks")
    os.makedirs(tracks_dir, exist_ok=True)

    filtered_df = merged_spots_df[merged_spots_df["Condition"] == condition]
    if filtered_df.empty:
        print("No data available for the selected condition.")
        return

    plt.figure(figsize=(10, 8))

    for unique_id in filtered_df["Unique_ID"].unique():
        track_df = filtered_df[filtered_df["Unique_ID"] == unique_id].sort_values(by="POSITION_T")
        if track_df.empty:
            continue

        start_x = track_df.iloc[0]["POSITION_X"]
        start_y = track_df.iloc[0]["POSITION_Y"]
        normalized_x = track_df["POSITION_X"] - start_x
        normalized_y = track_df["POSITION_Y"] - start_y

        plt.plot(normalized_x, normalized_y, marker="o", linestyle="-", markersize=2)

    plt.xlabel("Normalized POSITION_X")
    plt.ylabel("Normalized POSITION_Y")
    plt.title(f"Origin-Normalized Tracks for Condition: {condition}")

    if x_scale != 0:
        plt.xlim(-x_scale, x_scale)
    if y_scale != 0:
        plt.ylim(-y_scale, y_scale)

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()

    plt.savefig(os.path.join(tracks_dir, f"Origin_Normalized_Tracks_{condition}.pdf"))

    if display_plots:
        plt.show()
    else:
        plt.close()


def plot_migration_vectors(filename, merged_spots_df, Results_Folder, display_plots):
    if not filename:
        print("No valid filename selected")
        return

    tracks_dir = os.path.join(Results_Folder, "Tracks")
    os.makedirs(tracks_dir, exist_ok=True)

    fov_df = merged_spots_df[merged_spots_df["File_name"] == filename]
    if fov_df.empty:
        print(f"No data available for filename: {filename}")
        return

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    magnitudes = []
    coordinates = []

    for unique_id in fov_df["Unique_ID"].unique():
        track_df = fov_df[fov_df["Unique_ID"] == unique_id].sort_values(by="POSITION_T")
        if track_df.empty:
            continue

        start_x = track_df.iloc[0]["POSITION_X"]
        start_y = track_df.iloc[0]["POSITION_Y"]
        end_x = track_df.iloc[-1]["POSITION_X"]
        end_y = track_df.iloc[-1]["POSITION_Y"]

        vector_x = end_x - start_x
        vector_y = end_y - start_y
        magnitude = float(np.sqrt(vector_x**2 + vector_y**2))

        magnitudes.append(magnitude)
        coordinates.append((start_x, start_y, vector_x, vector_y))

    if len(magnitudes) == 0:
        print(f"No tracks available for filename: {filename}")
        plt.close()
        return

    mean_mag = float(np.mean(magnitudes))
    if mean_mag == 0:
        mean_mag = 1e-12

    vmin, vmax = float(min(magnitudes)), float(max(magnitudes))
    if vmin == vmax:
        # Avoid degenerate normalization
        eps = 1e-12 if vmin == 0 else abs(vmin) * 1e-12
        vmin -= eps
        vmax += eps

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.magma
    scale_factor = mean_mag * 0.08

    for (start_x, start_y, vector_x, vector_y), magnitude in zip(coordinates, magnitudes):
        color = cmap(norm(magnitude))
        hw = scale_factor * (magnitude / mean_mag)
        hl = scale_factor * (magnitude / mean_mag) * 2
        ax.arrow(start_x, start_y, vector_x, vector_y, head_width=hw, head_length=hl, fc=color, ec=color)

    plt.title(f"Migration Vectors for {filename}")
    plt.xlabel("POSITION_X")
    plt.ylabel("POSITION_Y")
    plt.grid(True)
    plt.axis("equal")

    ax.invert_yaxis()

    plt.savefig(os.path.join(tracks_dir, f"Vectors_Tracks_{filename}.pdf"))

    if display_plots:
        plt.show()
    else:
        plt.close()

        
def plot_coordinates_side_by_side(filename, merged_spots_df, filtered_and_smoothed_df, Results_Folder):
    if not filename:
        print("No valid filename selected")
        return

    # Filter the DataFrames based on the selected filename
    raw_df = merged_spots_df[merged_spots_df['File_name'] == filename]
    processed_df = filtered_and_smoothed_df[filtered_and_smoothed_df['File_name'] == filename]

    # Guard against empty dataframes
    if raw_df.empty or processed_df.empty:
        print("No data available for the selected filename in one or both dataframes.")
        return

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Create a colormap to ensure consistent colors across tracks
    unique_ids = raw_df['Unique_ID'].unique()
    colormap = plt.get_cmap('tab20')

    # Calculate data ranges for both DataFrames
    raw_x_min, raw_x_max = raw_df['POSITION_X'].min(), raw_df['POSITION_X'].max()
    raw_y_min, raw_y_max = raw_df['POSITION_Y'].min(), raw_df['POSITION_Y'].max()

    processed_x_min, processed_x_max = processed_df['POSITION_X'].min(), processed_df['POSITION_X'].max()
    processed_y_min, processed_y_max = processed_df['POSITION_Y'].min(), processed_df['POSITION_Y'].max()

    # Combine data ranges for setting axis limits (consider margins if needed)
    margin = 0.1
    x_min = min(raw_x_min, processed_x_min) - margin
    x_max = max(raw_x_max, processed_x_max) + margin
    y_min = min(raw_y_min, processed_y_min) - margin
    y_max = max(raw_y_max, processed_y_max) + margin

    # --- Plot Raw Data with shared limits ---
    for idx, unique_id in enumerate(unique_ids):
        unique_data = raw_df[raw_df['Unique_ID'] == unique_id].sort_values(by='POSITION_T')
        color_val = colormap((idx % 20) / 20)
        axes[0].plot(
            unique_data['POSITION_X'],
            unique_data['POSITION_Y'],
            color=color_val,
            marker='o',
            linestyle='-',
            markersize=2
        )

    axes[0].set_title(f'Raw Coordinates for {filename}')
    axes[0].set_xlabel('POSITION_X')
    axes[0].set_ylabel('POSITION_Y')
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)
    axes[0].invert_yaxis()  

    # --- Plot Filtered & Smoothed Data with shared limits ---
    for idx, unique_id in enumerate(unique_ids):
        unique_data = processed_df[processed_df['Unique_ID'] == unique_id].sort_values(by='POSITION_T')
        color_val = colormap((idx % 20) / 20)
        axes[1].plot(
            unique_data['POSITION_X'],
            unique_data['POSITION_Y'],
            color=color_val,
            marker='o',
            linestyle='-',
            markersize=2
        )

    axes[1].set_title(f'Filtered & Smoothed Coordinates for {filename}')
    axes[1].set_xlabel('POSITION_X')
    axes[1].set_ylabel('POSITION_Y')
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)
    axes[1].invert_yaxis()  # IMPORTANT: invert after setting ylim

    plt.tight_layout()
    plt.savefig(f"{Results_Folder}/Tracks/Filtered_tracks_{filename}.pdf")
    plt.show()

