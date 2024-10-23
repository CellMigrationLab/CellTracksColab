import os
import re
import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
import requests
import zipfile
import time
import gzip

from .xml_loader import load_and_populate_from_TM_XML


def replace_inf_with_nan(df, df_name):
    """
    Replaces all infinite values (positive or negative infinity) in the DataFrame with NaN
    and prints a message for each column where infinities are found.

    Args:
    df (pd.DataFrame): DataFrame to replace inf values.
    df_name (str): The name of the DataFrame as a string, used for printing.
    
    Returns:
    pd.DataFrame: DataFrame with infinity values replaced by NaN.
    """
    # Check for positive and negative infinity
    inf_columns = df.columns[(df == np.inf).any() | (df == -np.inf).any()].tolist()
    
    # Print message for each column that contains infinity values
    if inf_columns:
        for col in inf_columns:
            inf_count = ((df[col] == np.inf) | (df[col] == -np.inf)).sum()
            print(f"Column '{col}' in {df_name} contains {inf_count} infinity values. Replacing with NaN.")
    
    # Replace inf and -inf with NaN
    return df.replace([np.inf, -np.inf], np.nan)

def check_for_nans(df, df_name):
    """
    Checks the given DataFrame for NaN values and prints the count for each column containing NaNs.
    It first converts infinite values to NaNs before the check.

    Args:
    df (pd.DataFrame): DataFrame to be checked for NaN values.
    df_name (str): The name of the DataFrame as a string, used for printing.
    """
    # Replace infinity with NaN before checking for NaN values
    df = replace_inf_with_nan(df, df_name)
    
    # Check if the DataFrame has any NaN values and print a warning if it does.
    nan_columns = df.columns[df.isna().any()].tolist()

    if nan_columns:
        for col in nan_columns:
            nan_count = df[col].isna().sum()
            print(f"Column '{col}' in {df_name} contains {nan_count} NaN values.")
    else:
        print(f"No NaN values found in {df_name}.")

def save_parameters(params, file_path, param_type):
    # Convert params dictionary to a DataFrame for human readability
    new_params_df = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])
    new_params_df['Type'] = param_type

    if os.path.exists(file_path):
        # Read existing file
        existing_params_df = pd.read_csv(file_path)

        # Merge the new parameters with the existing ones
        # Update existing parameters or append new ones
        updated_params_df = pd.merge(existing_params_df, new_params_df,
                                     on=['Type', 'Parameter'],
                                     how='outer',
                                     suffixes=('', '_new'))

        # If there's a new value, update it, otherwise keep the old value
        updated_params_df['Value'] = updated_params_df['Value_new'].combine_first(updated_params_df['Value'])

        # Drop the temporary new value column
        updated_params_df.drop(columns='Value_new', inplace=True)
    else:
        # Use new parameters DataFrame directly if file doesn't exist
        updated_params_df = new_params_df

    # Save the updated DataFrame to CSV
    updated_params_df.to_csv(file_path, index=False)

def save_dataframe_with_progress(df, path, desc="Saving", chunk_size=50000):
    """Save a DataFrame with a progress bar and gzip compression."""

    # Estimating the number of chunks based on the provided chunk size
    num_chunks = int(len(df) / chunk_size) + 1

    # Create a tqdm instance for progress tracking
    with tqdm(total=len(df), unit="rows", desc=desc) as pbar:
        # Open the file for writing with gzip compression
        with gzip.open(path, "wt") as f:
            # Write the header once at the beginning
            df.head(0).to_csv(f, index=False)

            for chunk in np.array_split(df, num_chunks):
                chunk.to_csv(f, mode="a", header=False, index=False)
                pbar.update(len(chunk))

def download_test_datasets(path_extracted_dir, url, local_zip_file=''):
    local_zip_file = os.path.join(path_extracted_dir, "T_cell_dataset.zip")

    # Check if the extracted directory exists
    if os.path.exists(path_extracted_dir):
        print(
            f"Dataset already downloaded in {path_extracted_dir}")  ##TODO: should we just say that there's something there already?
        print(f" Please remove this file if you want to download the data again.")

    # Check if the ZIP file already exists
    print(local_zip_file)
    if not os.path.exists(local_zip_file):
        print("Downloading test dataset")
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            # Create the extracted directory if it doesn't exist
            os.makedirs(path_extracted_dir, exist_ok=True)

            # Calculate the total file size for the progress bar
            total_size = int(response.headers.get('content-length', 0))

            # Create a tqdm progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                # Download and save the content with progress tracking
                with open(local_zip_file, "wb") as file:
                    for data in response.iter_content(chunk_size=1024):
                        pbar.update(len(data))
                        file.write(data)

            print("Test dataset downloaded successfully.")

    # Extract the contents of the zip file to the specified directory
    with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
        zip_ref.extractall(path_extracted_dir)
    print("Test dataset extracted successfully.")

def populate_columns(df, filepath):
    # Extract the parts of the file path
    path_parts = os.path.normpath(filepath).split(os.sep)

    if len(path_parts) < 3:
        # if there are not enough parts in the path to extract folder and parent folder
        print(f"Error: Cannot extract parent folder and folder from the filepath: {filepath}")
        return df

    # Assuming that the file is located at least two levels deep in the directory structure
    folder_name = path_parts[-2]  # The folder name is the second last part of the path
    parent_folder_name = path_parts[-3]  # The parent folder name is the third last part of the path

    filename_without_extension = os.path.splitext(os.path.basename(filepath))[0]
    df['File_name_raw'] = remove_suffix(filename_without_extension)
    df['Condition'] = parent_folder_name  # Populate 'Condition' with the parent folder name
    df['experiment_nb'] = folder_name  # Populate 'Repeat' with the folder name
    df['File_name'] = parent_folder_name + "_" + folder_name + "_" + remove_suffix(filename_without_extension)

    return df

def find_calibration_units(filepath, line=3):
    k = 0
    for row in open(filepath):
        k += 1
        if k > line:
            return row[37:43]

def filter_and_delete_short_tracks(merged_spots_df, merged_tracks_df):
    """
    Filters out tracks with fewer than two spots from the given spots and tracks DataFrames.

    Parameters:
    - merged_spots_df: DataFrame containing all spots data, each row is a spot,
                       and 'Unique_ID' is used to identify tracks.
    - merged_tracks_df: DataFrame containing all tracks data, each row is a track,
                        and 'Unique_ID' must correspond to the IDs in the spots DataFrame.

    Returns:
    - filtered_spots_df: DataFrame containing only spots from tracks with two or more spots.
    - filtered_tracks_df: DataFrame containing only tracks that have two or more spots.

    Note: This function assumes that both DataFrames are properly aligned and that the
    'Unique_ID' column is correctly formatted.
    """

    # Print initial status
    print("Starting to filter tracks with fewer than two spots...")

    # Filter out tracks with fewer than 2 spots
    valid_ids = merged_spots_df.groupby('Unique_ID').filter(lambda x: len(x) >= 2)['Unique_ID'].unique()
    print(f"Found {len(valid_ids)} tracks with two or more spots.")

    filtered_spots_df = merged_spots_df[merged_spots_df['Unique_ID'].isin(valid_ids)]
    filtered_tracks_df = merged_tracks_df[merged_tracks_df['Unique_ID'].isin(valid_ids)]

    print("Filtering complete. Returning filtered DataFrames.")

    return filtered_spots_df, filtered_tracks_df

def generate_repeat(group):
    unique_experiment_nbs = sorted(group['experiment_nb'].unique())
    experiment_nb_to_repeat = {experiment_nb: i + 1 for i, experiment_nb in enumerate(unique_experiment_nbs)}
    group['Repeat'] = group['experiment_nb'].map(experiment_nb_to_repeat)
    return group

def sort_and_generate_repeat(merged_df):
    merged_df.sort_values(['Condition', 'experiment_nb'], inplace=True)
    merged_df = merged_df.groupby('Condition', group_keys=False).apply(generate_repeat)
    return merged_df

def frame_estimation(df, col_file = "File_name", t_col="POSITION_T"):

    files = np.unique(df[col_file])
    new_df = None
    df["FRAME"] = 0
    for f in files:
        aux = df.loc[lambda df: df[col_file] == f]
        # normalisation of the time dimension to obtain the frame number
        t0 = np.min(aux[t_col])
        time_points = np.unique(aux[t_col])
        frame_rate = np.min(time_points[1:]-time_points[:-1])
        frames = (1/frame_rate)*(aux[t_col]-t0)
        # Update values of FRAME. Copying is recommended to avoid panda dataframes visualisation
        aux1 = aux.copy()
        aux1.loc[frames.index, "FRAMES"] = frames.values
        #Build the new dataset
        if new_df is None:
            new_df = aux1
        else:
            new_df = pd.concat([new_df, aux1]).reset_index(drop=True)
    return new_df

def remove_suffix(filename):
    suffixes_to_remove = ["-tracks", "-spots"]
    for suffix in suffixes_to_remove:
        if filename.endswith(suffix):
            filename = filename[:-len(suffix)]
            break
    return filename

def validate_tracks_df(df):
    """Validate the tracks dataframe for necessary columns and data types."""
    required_columns = ['TRACK_ID']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Column '{col}' missing in tracks dataframe.")
            return False

    return True

def validate_spots_df(df, data_dims="2D"):
    """Validate the spots dataframe for necessary columns and data types, and clean NaN values from TRACK_ID."""
    if data_dims == "2D":
        required_columns = ['TRACK_ID', 'POSITION_X', 'POSITION_Y', 'POSITION_T']
    else:
        required_columns = ['TRACK_ID', 'POSITION_X', 'POSITION_Y', 'POSITION_Z', 'POSITION_T']

    # Check for required columns
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Column '{col}' missing in spots dataframe.")
            return False

    # Check for NaN in TRACK_ID
    if df['TRACK_ID'].isnull().any():
        print("Warning: NaN values found in TRACK_ID column.")
        # Find filenames associated with NaN TRACK_IDs
        filenames_with_nan = df[df['TRACK_ID'].isnull()]['File_name'].unique()
        for filename in filenames_with_nan:
            print(f"Removing rows with NaN in TRACK_ID for file: {filename}")

        # Remove rows where TRACK_ID is NaN
        initial_row_count = len(df)
        df.dropna(subset=['TRACK_ID'], inplace=True)
        final_row_count = len(df)
        print(f"Rows removed: {initial_row_count - final_row_count}")

    return True

def check_unique_id_match(df1, df2):
    df1_ids = set(df1['Unique_ID'])
    df2_ids = set(df2['Unique_ID'])

    # Check if the IDs in the two dataframes match
    if df1_ids == df2_ids:
        print("The Unique_ID values in both dataframes match perfectly!")
    else:
        missing_in_df1 = df2_ids - df1_ids
        missing_in_df2 = df1_ids - df2_ids

        if missing_in_df1:
            print(
                f"There are {len(missing_in_df1)} Unique_ID values present in the second dataframe but missing in the first.")
            print("Examples of these IDs are:", list(missing_in_df1)[:5])

        if missing_in_df2:
            print(
                f"There are {len(missing_in_df2)} Unique_ID values present in the first dataframe but missing in the second.")
            print("Examples of these IDs are:", list(missing_in_df2)[:5])

def check_metadata(df, df_name="DataFrame", metadata_columns=['Condition', 'experiment_nb', 'Repeat']):
    """
    Checks the metadata columns that are expected to have identical values for each filename.
    For example: the file track_01.xml should only belong to one experimental repeat and condition.
    The column names depend on the dataframe being tested.

    Parameters:
    df (pd.DataFrame): The DataFrame to be checked.
    df_name (str): The name of the DataFrame (for display purposes).
    metadata_columns (list): List of columns expected to have identical values for each filename.

    Returns:
    None
    """
    inconsistencies = []
    
    for name, group in df.groupby('File_name'):
        for col in metadata_columns:
            unique_values = group[col].unique()
            if len(unique_values) > 1:
                inconsistencies.append({
                    'File_name': name,
                    'Column': col,
                    'Values': unique_values
                })

    if not inconsistencies:
        print(f"{df_name} has consistent metadata.")
    else:
        print(f"{df_name} has inconsistencies in the metadata. Please check the output for details.")
        for inconsistency in inconsistencies:
            print(f"Inconsistency found in {df_name} for file: {inconsistency['File_name']} in column: {inconsistency['Column']}")
            print(f"Values found: {inconsistency['Values']}\n")

def automatic_column_mapping(ref_columns, data_columns):
    aux = data_columns.copy()
    values = []
    # Match columns having some characters in commonn
    for c in ref_columns:
        if aux is not None:
            key = c[-1]
            if key=="D":
                key = "ID"
            match = [i for i in aux if i.__contains__(key)]
            if len(match)>0:
                values.append(match[0])
                aux.remove(match[0])

    # Check that all the needed columns are matched and if not add by default the next one in order
    if len(values)<len(ref_columns):
        for i in range(len(ref_columns) - len(values)):
            values.append(aux[i])
    return dict(zip(ref_columns, values))

def load_and_populate(Folder_path, file_pattern, skiprows=None, usecols=None, chunksize=100000, check_calibration=False, row=3):
    df_list = []
    pattern = re.compile(file_pattern)  # Compile the file pattern to a regex object
    files_to_process = []
    if skiprows is None:
        header_len = 1
    else:
        header_len = len(skiprows)

    # First, list all the files we'll be processing
    for dirpath, dirnames, filenames in os.walk(Folder_path):
        for filename in filenames:
            if pattern.match(filename):  # Check if the filename matches the file pattern
                filepath = os.path.join(dirpath, filename)
                files_to_process.append(filepath)

    # Metadata list used to check for correct loading of rows
    metadata_list = []

    # Create a tqdm instance for progress tracking
    for filepath in tqdm(files_to_process, desc="Processing Files"):

        # Add to the metadata list
        if check_calibration:
            calibration_units = find_calibration_units(filepath, line=row)
            metadata_list.append({
                'filename': os.path.basename(filepath),
                'expected_rows': sum(1 for row in open(filepath, encoding = "ISO-8859-1")) - (1+header_len),
                # Get the expected number of rows in the file (subtracting header rows)
                'file_size': os.path.getsize(filepath),  # Get file size
                'calibration_units': calibration_units
            })
        else:
            metadata_list.append({
                'filename': os.path.basename(filepath),
                'expected_rows': sum(1 for row in open(filepath, encoding = "ISO-8859-1")) - (1+header_len),
                # Get the expected number of rows in the file (subtracting header rows)
                'file_size': os.path.getsize(filepath)  # Get file size
            })
        # Load the data in chunksizes to avoid memory colapse
        chunked_reader = pd.read_csv(filepath, skiprows=skiprows, usecols=usecols, chunksize=chunksize, encoding = "ISO-8859-1")
        for chunk in chunked_reader:
            processed_chunk = populate_columns(chunk, filepath)
            df_list.append(processed_chunk)

    if not df_list:  # if df_list is empty, return an empty DataFrame
        print(f"No files found with pattern: {file_pattern}")
        return pd.DataFrame()

    else:

        merged_df = pd.concat(df_list, ignore_index=True)

        # Verify the total rows in the merged dataframe matches the total expected rows from metadata
        total_expected_rows = sum(item['expected_rows'] for item in metadata_list)
        if check_calibration:
            calibration_units = [item['calibration_units'] for item in metadata_list]

            if len(np.unique(calibration_units)) > 1:
                print(f'Warning: The data is calibrated using different units: {np.unique(calibration_units)}')
            else:
                print(f'The data is calibrated using {np.unique(calibration_units)[0]} as units.')

        if len(merged_df) != total_expected_rows:
            print(f"Warning: Mismatch in total rows. Expected {total_expected_rows}, found {len(merged_df)} in the merged dataframe.")
        else:
            print(f"Success: The processed dataframe matches the metadata. Total rows: {len(merged_df)}")
        return merged_df


class TrackingData:

    def __init__(self):
        # ---------Parameters---------#
        # self.test_data_url = "https://zenodo.org/record/8420011/files/T_Cells_spots_only.zip?download=1"
        self.parent_dir = os.getcwd().split("/Notebook")[
            0]  # We want the parent dir to the github repo. Sometimes the notebook may run somewhere else.
        self.Folder_path = os.path.join(self.parent_dir, "Tracks")
        self.Results_Folder = os.path.join(self.parent_dir, "Results")
        self.skiprows = None # Rows to skip if a TrackMate CSV file was given. It's a list, e.g., [1, 2, 3]
        self.usecols = None  # One could choose to load only specific columns, for example: self.usecols = [1, 2, 3] or self.usecols = ['ID', 'X', 'Y', 'T']
        self.file_format = "csv"  # CSV is the format by default but it could be XML file
        self.data_type = "Custom"  # One of the followings ["Custom", "TrackMate Table", "TrackMate Files"]
        self.data_dims = "2D" # Either "2D" or "3D"
        self.Spot_table = None # path to TrackMate spot table. If needed, update before loading the data.
        self.Track_table = None # path to TrackMate spot table. If needed, update before loading the data.
        # ----------------------------#

    def DownloadTestData(self, test_data_url):
        
        self.test_data_url = test_data_url
        
        # Define the path to the ready to use tracking data
        local_zip_file = os.path.join(self.parent_dir, "Test_dataset", "T_cell_dataset.zip")

        # Download and extract the test data
        download_test_datasets(self.Folder_path, self.test_data_url, local_zip_file=local_zip_file)

    def __column_mapping__(self):

        # If 2D is selected and POSITION_Z exists, delete it
        if self.data_dims == '2D' and 'POSITION_Z' in self.spots_data.columns:
            self.spots_data = self.spots_data.drop('POSITION_Z', axis=1)
        column_mapping = {data_col: col for col, data_col in self.dim_mapping.items()}
        self.spots_data = self.spots_data.rename(columns=column_mapping)
        print("Columns Renamed!")
        self.spots_data = sort_and_generate_repeat(self.spots_data)
        save_dataframe_with_progress(self.spots_data, os.path.join(self.Results_Folder, 'merged_Spots.csv.gz'),
                                     desc="Saving Spots")

    def CalibrateUnits(self, x_cal=1., y_cal=1., z_cal=1., t_cal=1., spatial_calibration_unit="pixel", time_unit="second"):

        # Update the calibratio values to get back to them
        self.x_cal = x_cal
        self.y_cal = y_cal
        self.z_cal = z_cal
        self.spatial_calibration_unit = spatial_calibration_unit
        self.t_cal = t_cal
        self.time_unit = time_unit

        # Update the merged_spots_df columns with calibration values
        self.spots_data['POSITION_X'] = self.spots_data['POSITION_X'] * x_cal
        self.spots_data['POSITION_Y'] = self.spots_data['POSITION_Y'] * y_cal
        if self.data_type == "3D":
            self.spots_data['POSITION_Z'] = self.spots_data['POSITION_Z'] * z_cal

        # Update the temporal column with calibration value
        self.spots_data['POSITION_T'] = self.spots_data['POSITION_T'] * t_cal

        save_dataframe_with_progress(self.spots_data,
                                     os.path.join(self.Results_Folder, 'merged_Spots_calibrated.csv.gz'), desc="Saving Spots")

        print(
            f"Spatial Calibration saved: X={x_cal}, Y={y_cal}, Z={z_cal} in {spatial_calibration_unit}")
        print(f"Temporal Calibration saved: {t_cal} per frame in {time_unit}")

    def __load_csv__(self):
        # Load Tracking data in memory
        file_pattern = f'.*\.{self.file_format}$'
        merged_spots_df = load_and_populate(self.Folder_path, file_pattern, skiprows=self.skiprows,
                                            usecols=self.usecols, check_calibration=False)
        print(f"Tracking data loaded in memory.")

        merged_spots_df = sort_and_generate_repeat(merged_spots_df)

        save_dataframe_with_progress(merged_spots_df, os.path.join(self.Results_Folder, 'merged_Spots.csv.gz'),
                                     desc="Saving Spots")

        self.spots_data = merged_spots_df

        self.dim_mapping = automatic_column_mapping(self.ref_cols, list(merged_spots_df.columns))
        print(f"Data columns automatically mapped as: {self.dim_mapping}.")

    def __create_tracks_csv(self):
        self.spots_data['Unique_ID'] = self.spots_data ['File_name'] + "_" + self.spots_data ['TRACK_ID'].astype(str)
        save_dataframe_with_progress(self.spots_data, os.path.join(self.Results_Folder, 'merged_Spots.csv.gz'),
                                     desc="Saving Spots")

        # Extracting unique Unique_ID values from merged_spots_df
        unique_ids = self.spots_data['Unique_ID'].drop_duplicates().reset_index(drop=True)

        # Creating merged_tracks_df with only the unique Unique_ID values
        merged_tracks_df = pd.DataFrame(unique_ids, columns=['Unique_ID'])
        print("Create the merged_tracks_df to store track parameters")
        # Specify the columns you want to merge
        columns_to_merge = ['Unique_ID', 'TRACK_ID', 'File_name', 'Condition', 'experiment_nb', 'File_name_raw', 'Repeat']

        # Filter to only include the desired columns
        filtered_df = self.spots_data[columns_to_merge].drop_duplicates(subset='Unique_ID')

        # Find the overlapping columns between the two DataFrames, excluding the merging key
        overlapping_columns = merged_tracks_df.columns.intersection(filtered_df.columns).drop('Unique_ID')

        # Drop the overlapping columns from the left DataFrame
        merged_tracks_df.drop(columns=overlapping_columns, inplace=True)

        # Merge the filtered df_directionality back into the original DataFrame
        merged_tracks_df = pd.merge(merged_tracks_df, filtered_df, on='Unique_ID', how='left')
        # Check that there are no duplicate files with different metadata
        metadata_columns = ['Condition', 'experiment_nb', 'Repeat']
        check_metadata(merged_tracks_df, df_name="Tracks Dataframe", metadata_columns=metadata_columns)

        self.tracks_data = merged_tracks_df

    def __load_trackmate_xml__(self):

        # Load Tracking data in memory
        merged_spots_df, merged_tracks_df = load_and_populate_from_TM_XML(self.Folder_path)

        print(f"Tracking data loaded in memory.")

        if not validate_tracks_df(merged_tracks_df):
            print("Error: Validation failed for merged tracks dataframe.")
        else:
            merged_tracks_df = sort_and_generate_repeat(merged_tracks_df)
            merged_tracks_df['Unique_ID'] = merged_tracks_df['File_name'] + "_" + merged_tracks_df['TRACK_ID'].astype(
                str)

        print(f"These are its column names:{merged_spots_df.columns}")
        merged_spots_df = sort_and_generate_repeat(merged_spots_df)

        if not validate_spots_df(merged_spots_df, data_dims=self.data_dims):
            print("Error: Validation failed for merged spots dataframe.")
        else:
            merged_spots_df = sort_and_generate_repeat(merged_spots_df)
            merged_spots_df['TRACK_ID'] = merged_spots_df['TRACK_ID'].astype(int)
            merged_spots_df['Unique_ID'] = merged_spots_df['File_name'] + "_" + merged_spots_df['TRACK_ID'].astype(str)
            merged_spots_df.dropna(subset=['POSITION_X', 'POSITION_Y', 'POSITION_Z'], inplace=True)
            # filter out short tracks (2 data points)
            merged_spots_df, merged_tracks_df = filter_and_delete_short_tracks(merged_spots_df, merged_tracks_df)

        # Check that there are no duplicate files with different metadata
        metadata_columns = ['Condition', 'experiment_nb', 'Repeat']
        check_metadata(merged_tracks_df, df_name="Tracks Dataframe", metadata_columns=metadata_columns)
        check_metadata(merged_spots_df, df_name="Spots Dataframe", metadata_columns=metadata_columns)

        save_dataframe_with_progress(merged_tracks_df, os.path.join(self.Results_Folder, 'merged_Tracks.csv.gz'),
                                     desc="Saving Tracks")
        save_dataframe_with_progress(merged_spots_df, os.path.join(self.Results_Folder, 'merged_Spots.csv.gz'),
                                     desc="Saving Spots")
        self.spots_data = merged_spots_df
        self.tracks_data = merged_tracks_df

    def __load_trackmate_csv__(self):
        # Trackmate is composed of tracks and spots
        # Load the tracking info data in memory
        file_pattern = f'.*tracks.*\.{self.file_format}$'
        merged_tracks_df = load_and_populate(self.Folder_path, file_pattern, skiprows=self.skiprows,
                                             usecols=self.usecols, check_calibration=True)
        print(f"Tracking data loaded in memory.")
        print(f"These are its column names:{merged_tracks_df.columns}")

        if not validate_tracks_df(merged_tracks_df):
            print("Error: Validation failed for merged tracks dataframe.")
        else:
            merged_tracks_df = sort_and_generate_repeat(merged_tracks_df)
            merged_tracks_df['Unique_ID'] = merged_tracks_df['File_name'] + "_" + merged_tracks_df['TRACK_ID'].astype(
                str)
            save_dataframe_with_progress(merged_tracks_df, os.path.join(self.Results_Folder, 'merged_Tracks.csv.gz'),
                                         desc="Saving Tracks")

        # Load the spots data info in memory
        file_pattern = f'.*spots.*\.{self.file_format}$'
        merged_spots_df = load_and_populate(self.Folder_path, file_pattern, skiprows=self.skiprows,
                                            usecols=self.usecols)

        if not validate_spots_df(merged_spots_df, data_dims=self.data_dims):
            print("Error: Validation failed for merged spots dataframe.")
        else:
            merged_spots_df = sort_and_generate_repeat(merged_spots_df)
            merged_spots_df['TRACK_ID'] = merged_spots_df['TRACK_ID'].astype(int)
            merged_spots_df['Unique_ID'] = merged_spots_df['File_name'] + "_" + merged_spots_df['TRACK_ID'].astype(str)
            merged_spots_df.dropna(subset=['POSITION_X', 'POSITION_Y', 'POSITION_Z'], inplace=True)
            # filter out short tracks (2 data points)
            merged_spots_df, merged_tracks_df = filter_and_delete_short_tracks(merged_spots_df, merged_tracks_df)

        # Check that there are no duplicate files with different metadata
        metadata_columns = ['Condition', 'experiment_nb', 'Repeat']
        check_metadata(merged_tracks_df, df_name="Tracks Dataframe", metadata_columns=metadata_columns)
        check_metadata(merged_spots_df, df_name="Spots Dataframe", metadata_columns=metadata_columns)
        save_dataframe_with_progress(merged_spots_df, os.path.join(self.Results_Folder, 'merged_Spots.csv.gz'),
                                     desc="Saving Spots")
        save_dataframe_with_progress(merged_tracks_df, os.path.join(self.Results_Folder, 'merged_Tracks.csv.gz'),
                                     desc="Saving Tracks")
        # Store the values
        self.spots_data = merged_spots_df
        self.tracks_data = merged_tracks_df


    def __load_celltrackscolab_table(self):
        print("Loading track table file....")
        merged_tracks_df = pd.read_csv(os.path.join(self.Folder_path, self.Track_table), low_memory=False)
        if not validate_tracks_df(merged_tracks_df):
            print("Error: Validation failed for loaded tracks dataframe.")
        # Check that there are no duplicate files with different metadata
        metadata_columns = ['Condition', 'experiment_nb', 'Repeat']
        check_metadata(merged_tracks_df, df_name="Tracks Dataframe", metadata_columns=metadata_columns)
        self.tracks_data = merged_tracks_df

        print("Loading spot table file....")
        merged_spots_df = pd.read_csv(os.path.join(self.Folder_path, self.Spot_table), low_memory=False)
        if not validate_spots_df(merged_spots_df, data_dims=self.data_dims):
            print("Error: Validation failed for loaded spots dataframe.")
        # Check that there are no duplicate files with different metadata
        check_metadata(merged_spots_df, df_name="Spots Dataframe", metadata_columns=metadata_columns)
        self.spots_data = merged_spots_df
        # filter out short tracks (2 data points)
        merged_spots_df, merged_tracks_df = filter_and_delete_short_tracks(self.spots_data, self.tracks_data)
        self.spots_data = merged_spots_df
        self.tracks_data = merged_tracks_df


    def LoadTrackingData(self):

        if self.data_dims=="2D":
            self.ref_cols = ['TRACK_ID', 'POSITION_X', 'POSITION_Y', 'POSITION_T']
        else:
            self.ref_cols = ['TRACK_ID', 'POSITION_X', 'POSITION_Y', 'POSITION_Z', 'POSITION_T']

        start_time = time.time()
        # The following loading commands have redundancy in code but it's more modular this way and faster for now.

        if self.data_type == "TrackMate Files" and self.file_format.__contains__("csv"):

            self.skiprows = [1, 2, 3]
            self.__load_trackmate_csv__()

        elif self.data_type == "TrackMate Files" and self.file_format.__contains__("xml"):

            self.__load_trackmate_xml__()

        elif self.data_type == "Custom" and self.file_format.__contains__("csv"):

            self.__load_csv__()

        elif self.data_type == "Custom" and self.file_format.__contains__("xml"):

            self.__load_trackmate_xml__()

        elif self.data_type == "CellTracksColab":

            self.__load_celltrackscolab_table()

        if self.data_type == "TrackMate Files" or self.data_type == "CellTracksColab":
            if self.data_dims == "2D":
                self.dim_mapping = {'TRACK_ID': 'ID',
                                    'POSITION_X': 'POSITION_X',
                                    'POSITION_Y': 'POSITION_Y',
                                    'POSITION_T': 'POSITION_T'}
            elif self.data_dims == "3D":
                self.dim_mapping = {'TRACK_ID': 'ID',
                                    'POSITION_X': 'POSITION_X',
                                    'POSITION_Y': 'POSITION_Y',
                                    'POSITION_Z': 'POSITION_Z',
                                    'POSITION_T': 'POSITION_T'}
            print(f"Data columns mapped as: {self.dim_mapping}.")

        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Dataset processing completed in {elapsed_time:.2f} seconds.")

    def CompileTrackingData(self):

        # Only run this if dimensions were properly mapped in self.dim_mapping

        if self.data_type == "Custom" and self.file_format.__contains__("csv"):
            self.__column_mapping__()
            self.__create_tracks_csv()
            self.spots_data = frame_estimation(self.spots_data, col_file="File_name", t_col="POSITION_T")

        check_unique_id_match(self.spots_data, self.tracks_data)
        # Check that there are no duplicate files with different metadata
        metadata_columns = ['Condition', 'experiment_nb', 'Repeat']
        check_metadata(self.spots_data, df_name="Spots DataFrame", metadata_columns=metadata_columns)
        check_metadata(self.tracks_data, df_name="Tracks Dataframe", metadata_columns=metadata_columns)
        # filter out short tracks (2 data points)
        merged_spots_df, merged_tracks_df = filter_and_delete_short_tracks(self.spots_data, self.tracks_data)
        self.spots_data = merged_spots_df
        self.tracks_data = merged_tracks_df
        # Save the DataFrame with the selected columns merged
        save_dataframe_with_progress(self.spots_data, os.path.join(self.Results_Folder, 'merged_Spots.csv.gz'),
                                     desc="Saving Spots")
        save_dataframe_with_progress(self.tracks_data, os.path.join(self.Results_Folder, 'merged_Tracks.csv.gz'),
                                     desc="Saving Tracks")
